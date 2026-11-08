"""Train the BharatNanoGPT tokenizer on data/corpus.txt.

Usage:
    python -m tokenizer.train_tokenizer --vocab-size 2048 --out tokenizer/tokenizer.json
    python -m tokenizer.train_tokenizer --config config/train_tiny.py

If the `tokenizers` package is missing, this falls back to a character-level
tokenizer and says so loudly (see tokenizer/__init__.py). It never silently
pretends BPE worked.
"""
import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import train_tokenizer  # noqa: E402


def load_config(path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--out", default="tokenizer/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    if args.config:
        args.vocab_size = load_config(args.config).get(
            "tokenizer_vocab_size", args.vocab_size
        )
    if not os.path.exists(args.corpus):
        print(f"Corpus not found: {args.corpus}. Run: python -m data.prepare")
        sys.exit(1)

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    # Train on lines (each line is a roughly independent sentence/paragraph).
    lines = [ln for ln in text.splitlines() if ln.strip()]
    print(f"Training tokenizer on {len(lines)} lines, vocab_size={args.vocab_size} ...")
    tok, fallback = train_tokenizer(lines, vocab_size=args.vocab_size)
    tok.save(args.out)
    print(f"Saved tokenizer ({tok.kind}) to {args.out} "
          f"| vocab_size={tok.vocab_size} | fallback={fallback}")
    for sample in ["नमस्ते भाई, कैसे हो?", "bhai kal exam hai", "The capital of India is"]:
        ids = tok.encode(sample)
        print(f"  {sample!r} -> {ids[:12]}... -> {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
