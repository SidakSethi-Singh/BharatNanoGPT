"""Evaluate a trained BharatNanoGPT checkpoint — local only, no API.

  - val perplexity over the held-out set
  - sample generations for fixed Hindi / Hinglish / English prompts
  - a simple repetition check (share of repeated n-grams in samples)

No LLM API is used anywhere here. The model is judged by its own
perplexity and by reading its samples — see README for the honest limits.

Usage:
    python evaluate.py --checkpoint out/tiny/ckpt.pt
    python evaluate.py --checkpoint out/tiny/ckpt.pt --num-samples 6
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_gpt  # noqa: E402
from sample import generate_text  # noqa: E402

PROMPTS = [
    "राम सुबह उठकर",
    "bhai kal exam hai aur mujhe",
    "The capital of India is",
    "भारत एक बड़ा देश है",
    "mujhe kal subah 9 baje meeting hai",
    "क्रिकेट भारत का सबसे लोकप्रिय खेल है",
]


def perplexity(checkpoint_path, device, eval_iters=20, batch_size=32):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_gpt(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    val = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
    block = cfg["block_size"]
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            ix = torch.randint(len(val) - block, (batch_size,))
            x = torch.stack([torch.from_numpy(val[i:i + block].astype(np.int64)) for i in ix]).to(device)
            y = torch.stack([torch.from_numpy(val[i + 1:i + 1 + block].astype(np.int64)) for i in ix]).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    mean_loss = float(np.mean(losses))
    return mean_loss, math.exp(mean_loss)


def repetition_rate(text):
    """Share of repeated word bigrams — a crude coherence sanity check."""
    words = text.split()
    if len(words) < 4:
        return 0.0
    bigrams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
    return 1.0 - len(set(bigrams)) / len(bigrams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/tiny/ckpt.pt")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss, ppl = perplexity(args.checkpoint, device)
    print(f"Val loss: {loss:.4f} | val perplexity: {ppl:.1f}")

    prompts = PROMPTS[: max(1, args.num_samples)]
    samples = []
    for p in prompts:
        res = generate_text(args.checkpoint, p, args.max_new_tokens,
                            args.temperature, args.top_k, device=device)
        res["repetition_rate"] = round(repetition_rate(res["text"]), 3)
        samples.append(res)
        print(f"\nPrompt: {p!r}\n  -> {res['text']!r}\n  repetition: {res['repetition_rate']}")

    os.makedirs("results", exist_ok=True)
    with open("results/eval_samples.jsonl", "a", encoding="utf-8") as f:
        for s in samples:
            # NOTE: **s must come FIRST — sample.py's dict also carries a
            # "val_loss" key (the checkpoint's stored value), and the freshly
            # computed loss below must win.
            f.write(json.dumps({**s, "checkpoint": args.checkpoint, "val_loss": loss,
                                "perplexity": ppl}, ensure_ascii=False) + "\n")
    print("\nSamples logged to results/eval_samples.jsonl")


if __name__ == "__main__":
    main()
