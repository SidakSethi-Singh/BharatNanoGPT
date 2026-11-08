# MLAI.md — Project conventions for the ML.ai coding agent

This file is read by **ML.ai** (https://ml.ai, an AI coding agent by Pixis.ai
that works in VS Code and Cursor, as a standalone desktop app, from the
terminal, and as a GitHub PR Reviewer). It tells the agent how this project
is built, so it can plan tasks, edit code, run tests, fix issues, and debug
without breaking the core claim of the project.

## What this project is
BharatNanoGPT — a tiny GPT trained **from scratch** (Karpathy nanoGPT-style)
on a small Hindi–English / Hinglish corpus. No pre-trained weights, no LLM
API at inference. The model generates from its own trained weights only.

## Stack
- Python 3.11+ (tested on 3.12). Pin versions in `requirements.txt`
  (Sid's LawSathi style: pinned, reproducible).
- PyTorch 2.x (CPU wheel for local work; CUDA wheel only on Colab GPU).
- Hugging Face `tokenizers` for BPE. If it is unavailable, use the
  character-level fallback in `tokenizer/__init__.py` and say so loudly —
  never silently pretend BPE worked.
- Streamlit for the demo UI (`app.py`). pytest for tests.

## File layout
```
model.py                 # GPT, Block, CausalSelfAttention (from scratch)
config/train_bharat.py   # full ~14M model (Colab T4)
config/train_tiny.py     # tiny ~4M model (CPU smoke test)
data/prepare.py          # corpus build + tokenise -> data/train.bin, data/val.bin
data/corpus/             # hand-written Hindi / English / Hinglish sentences
data/SOURCES.md          # licences + provenance for every corpus source
tokenizer/               # BPE tokenizer (train_tokenizer.py)
train.py / sample.py / evaluate.py
app.py                   # Streamlit demo (local only)
tests/                   # pytest: model forward, tokenizer round-trip, data prep
docs/ML_AI_USAGE.md      # how ML.ai was used on this project
ml-ai-sessions/          # Sid's exported ML.ai session reports go here
```

## Rules for the agent (do not break these)
1. **Never use an LLM API for generation.** `sample.py`, `app.py`, and
   `train.py` must work fully offline after training, with no API key.
   Do not add an API client, a hosted-model comparison, or a judge that
   calls an external model. (An earlier aimlapi.com integration was added
   in error and removed — see `docs/ML_AI_USAGE.md`. Do not re-add it.
   The only remaining mentions of that service are historical notes in
   the docs explaining the removal, not code.)
2. **Never load pre-trained weights.** Every parameter must be randomly
   initialised and learned by `train.py`. Do not add a Hugging Face
   `from_pretrained` call.
3. **Keep the corpus legal and traceable.** Only public-domain /
   permissively licensed sources, recorded in `data/SOURCES.md`. No
   scraped copyrighted books, no social-media chats, no personal data.
4. **Verify before claiming done.** After any code change:
   - `python -m pytest -q` must pass.
   - A training run must show loss decreasing (check `out/*/train_log.jsonl`).
   - `python sample.py` must produce output for a Hindi, a Hinglish, and
     an English prompt. Read the samples — do not claim coherence you did
     not see.
5. **Record real results only.** Never invent training losses, perplexity,
   or sample outputs. Put actual measured numbers in `README.md` and
   label the full 15M Colab run as *not yet run* until it has been run.
6. **Small and hackathon-ready.** Do not download huge datasets or large
   models. Keep the default corpus small enough to prepare in minutes.

## How to run (the commands the agent should use)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m data.prepare --config config/train_tiny.py
python train.py --config config/train_tiny.py --max-iters 600
python sample.py --prompt "राम सुबह उठकर"
python evaluate.py --checkpoint out/tiny/ckpt.pt
python -m pytest -q
streamlit run app.py
```
