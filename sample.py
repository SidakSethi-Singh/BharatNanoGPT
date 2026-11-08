"""Generate text with a trained BharatNanoGPT checkpoint (local only).

No LLM API is used here — the output comes from our own trained weights.

Usage:
    python sample.py --checkpoint out/tiny/ckpt.pt --prompt "राम सुबह उठकर"
    python sample.py --checkpoint out/tiny/ckpt.pt --prompt "bhai kal exam hai" --temperature 0.8 --top-k 40 --max-new-tokens 80
    python sample.py --prompt "The capital of India is" --start "The capital of India is"  # (prompt is the start)
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_gpt  # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_gpt(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg, ckpt.get("iter", "?"), ckpt.get("val_loss")


def generate_text(checkpoint_path, prompt, max_new_tokens=80, temperature=0.8,
                  top_k=40, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer("tokenizer/tokenizer.json")
    model, cfg, it, val_loss = load_model(checkpoint_path, device)
    ids = tok.encode(prompt)
    if not ids:
        ids = [tok.bos_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    text = tok.decode(out[0].tolist())
    return {
        "text": text,
        "prompt": prompt,
        "model": "BharatNanoGPT (local, from scratch)",
        "checkpoint_iter": it,
        "val_loss": val_loss,
        "latency_ms": int((time.time() - t0) * 1000),
        "tokens_generated": max_new_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/tiny/ckpt.pt")
    ap.add_argument("--prompt", default="राम सुबह उठकर")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}. Train first: python train.py --config config/train_tiny.py")
        sys.exit(1)
    res = generate_text(args.checkpoint, args.prompt, args.max_new_tokens,
                        args.temperature, args.top_k)
    print(f"[local BharatNanoGPT | ckpt iter {res['checkpoint_iter']} | "
          f"{res['latency_ms']} ms]")
    print(res["text"])


if __name__ == "__main__":
    main()
