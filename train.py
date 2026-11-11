"""Train BharatNanoGPT from scratch (random init — no pre-trained weights).

Usage:
    python train.py --config config/train_tiny.py        # CPU smoke test
    python train.py --config config/train_bharat.py      # full run (GPU)

Writes checkpoints to <out_dir>/ckpt.pt (latest) and
<out_dir>/ckpt_<iter>.pt at each eval, plus <out_dir>/train_log.jsonl
(iter, train_loss, val_loss) for the loss curve.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_gpt  # noqa: E402


def load_config(path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device):
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, cfg):
    if it < cfg["warmup_iters"]:
        return cfg["learning_rate"] * it / max(1, cfg["warmup_iters"])
    if it > cfg["lr_decay_iters"]:
        return cfg["min_lr"]
    decay_ratio = (it - cfg["warmup_iters"]) / max(1, cfg["lr_decay_iters"] - cfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/train_tiny.py")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init-from", default=None,
                    help="resume / continue training from this checkpoint "
                         "(weights only; a fresh cosine LR schedule is used "
                         "for the new run, so --max-iters also resets "
                         "lr_decay_iters)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_iters:
        cfg["max_iters"] = args.max_iters
        cfg["lr_decay_iters"] = args.max_iters
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    torch.manual_seed(cfg["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | config={args.config}")

    for name in ("train.bin", "val.bin"):
        if not os.path.exists(os.path.join(cfg["data_dir"], name)):
            print(f"Missing {cfg['data_dir']}/{name}. Run: python -m data.prepare --config {args.config}")
            sys.exit(1)
    train_data = np.memmap(os.path.join(cfg["data_dir"], "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(cfg["data_dir"], "val.bin"), dtype=np.uint16, mode="r")
    print(f"Data: train={len(train_data):,} tokens, val={len(val_data):,} tokens")
    if len(val_data) <= cfg["block_size"] or len(train_data) <= cfg["block_size"]:
        print(f"ERROR: corpus too small for block_size={cfg['block_size']} "
              f"(val has only {len(val_data):,} tokens). This happens when "
              "Wikipedia returned HTTP 429 and prepare fell back to the "
              "313-line local corpus. Do NOT train on this. Fix: re-run "
              "prepare with --from-text <prebuilt corpus.txt> (see COLAB.md), "
              "or wait and retry the Wikipedia download.")
        sys.exit(2)

    # Tokenizer vocab must match the config's vocab_size.
    tok_path = "tokenizer/tokenizer.json"
    if os.path.exists(tok_path):
        from tokenizer import load_tokenizer

        tok = load_tokenizer(tok_path)
        if tok.vocab_size != cfg["vocab_size"]:
            print(f"WARNING: tokenizer vocab {tok.vocab_size} != config vocab "
                  f"{cfg['vocab_size']}. Re-run: python -m data.prepare --config {args.config}")

    model = build_gpt(cfg).to(device)
    iter_offset = 0
    if args.init_from:
        init = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init["model"])
        iter_offset = int(init.get("iter", 0)) + 1
        print(f"Resumed weights from {args.init_from} "
              f"(was iter {init.get('iter')}, val loss {init.get('val_loss')}); "
              f"logging continues from iter {iter_offset}")
    print(f"Model: {model.num_parameters():,} parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )

    os.makedirs(cfg["out_dir"], exist_ok=True)
    log_path = os.path.join(cfg["out_dir"], "train_log.jsonl")
    # Append so a resumed run does not erase the earlier phase's log.
    log_f = open(log_path, "a", encoding="utf-8")
    best_val = float("inf")
    t0 = time.time()
    for it in range(cfg["max_iters"]):
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"] - 1:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            rec = {"iter": it + iter_offset, "train_loss": losses["train"], "val_loss": losses["val"],
                   "elapsed_s": round(time.time() - t0, 1)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"iter {it}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f} (ppl {math.exp(losses['val']):.1f})")
            ckpt = {"model": model.state_dict(), "config": cfg, "iter": it + iter_offset,
                    "val_loss": losses["val"]}
            torch.save(ckpt, os.path.join(cfg["out_dir"], "ckpt.pt"))
            torch.save(ckpt, os.path.join(cfg["out_dir"], f"ckpt_{it + iter_offset}.pt"))
            best_val = min(best_val, losses["val"])
        lr = get_lr(it, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        xb, yb = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        if it % cfg["log_interval"] == 0:
            print(f"iter {it}: loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"{time.time() - t0:.0f}s elapsed")
    log_f.close()
    print(f"Done. Best val loss {best_val:.4f}. Checkpoint: {cfg['out_dir']}/ckpt.pt")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
