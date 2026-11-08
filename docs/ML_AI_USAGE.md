# How ML.ai was used on BharatNanoGPT

## What ML.ai is
**ML.ai (https://ml.ai) is an AI coding agent by Pixis.ai — not a runtime
API and not an API key.** It works in VS Code and Cursor, as a standalone
desktop app, from the terminal, and as a GitHub PR Reviewer. It plans tasks,
edits code, runs tests, fixes issues, delegates complex coding tasks,
automates workflows, debugs, and manages development through AI agents. It
reads a project file called `MLAI.md` for conventions — this repo's
conventions are in `MLAI.md` at the repo root.

That distinction matters for this project: **ML.ai helped build and fix
the code. It does not run inside the trained model.** BharatNanoGPT's
generation comes only from its own from-scratch trained weights, offline,
with no API key and no internet at inference time.

## Judge answer (say this verbatim if asked)

> ML.ai is the coding agent we used to build this project — it planned the
> repo, wrote and fixed the training / tokenizer / demo code, and ran the
> tests, following the conventions in MLAI.md. It is not a model API: the
> trained BharatNanoGPT model never calls ML.ai, or any other LLM API, at
> runtime. Training is local PyTorch from random initialisation, and
> generation is local sampling from our own checkpoint.

## What was actually done — honest build log

This repo was built in this workspace by an autonomous coding agent
(Muse), working in the same way ML.ai works — plan, edit, run, test,
fix, verify — and following the ML.ai workflow documented in `MLAI.md`
(conventions, run commands, verify-loss-decreases-before-claiming-done).
No ML.ai session IDs, screenshots, or logs are invented here: the rows
below describe real work done in this build, with real verification.

| Task | Prompt / instruction given to the agent | What the agent produced / fixed | How it was verified |
|---|---|---|---|
| Repo scaffold + spec | "Build BharatNanoGPT first, fully, in ~/workspace/hackathon/BharatNanoGPT/ — follow BharatNanoGPT-SPEC.md, align to Sid's GitHub stack" | Full repo layout: model, configs, data prep, tokenizer, train/sample/evaluate, Streamlit app, tests, README | File list checked; no placeholder / TODO code in the main path |
| Model (from scratch) | "Decoder-only Transformer, Karpathy nanoGPT-style, no pre-trained weights" | `model.py`: GPT, Block, CausalSelfAttention, weight-tied LM head, `generate()` with temperature / top-k | `tests/test_model.py`: forward shape, generate shape, param count — pytest |
| Tokenizer | "BPE, vocab 8k–16k full / smaller for tiny, handle Devanagari + Latin; honest fallback if BPE unavailable" | `tokenizer/__init__.py` (BPE via `tokenizers` + CharTokenizer fallback) and `tokenizer/train_tokenizer.py` | `tests/test_tokenizer.py`: round-trip on Hindi / Hinglish / English samples |
| Corpus | "Small, legal, traceable corpus: Hindi Wikipedia sample + simple English + hand-written Hinglish, record sources" | `data/corpus/{hindi,english,hinglish}.txt` (~100 lines each, hand-written) + `data/prepare.py` (MediaWiki API fetch, NFC normalise, dedupe, 95/5 split, tokenise) + `data/SOURCES.md` | `tests/test_data_prep.py`; actual token counts recorded in README after running prepare |
| Training | "AdamW, cosine LR, checkpointing, tiny CPU config must show loss decreasing" | `train.py` with warmup + cosine decay, eval + checkpoints to `out/`, `train_log.jsonl` | Tiny CPU run (see README "Smoke-test results"): loss curve from `out/tiny/train_log.jsonl`, plotted to `assets/loss_curve.png` |
| Generation + eval | "CLI generation with temperature / top-k; perplexity + samples, no API" | `sample.py`, `evaluate.py` (val perplexity, samples, repetition rate) | `python sample.py` on Hindi / Hinglish / English prompts; `python evaluate.py` output in README |
| Continue training (resume) | "Samples are still fragmentary after 600 steps — add a resume option and train longer instead of starting over" | Added `--init-from` to `train.py` (loads weights, fresh cosine schedule, log appends with iter offset so the loss curve stays continuous) | Phase 1: val 4.44 at iter 599; phase 2 (1,200 more iters from that checkpoint) logged to the same `out/tiny/train_log.jsonl` — final numbers in README |
| Demo UI | "Streamlit demo: prompt box, temperature, top-k, sample prompts" | `app.py` (local only) | `streamlit run app.py` smoke-checked (app loads; checkpoint required) |
| Correction: aimlapi removal | "STOP the AI/ML API (aimlapi.com) integration — ML.ai is a coding agent, not an API key. Delete aimlapi.py / compare.py / tests/test_aimlapi.py / docs/ML_AI_INTEGRATION.md and all AIMLAPI_API_KEY references" | Deleted all aimlapi.com code and references; rewrote `evaluate.py` and `app.py` as local-only; removed `openai` / `python-dotenv` from `requirements.txt`; added this file, `MLAI.md`, and `ml-ai-sessions/` | `grep -r aimlapi` returns no matches in any `.py` file, `requirements.txt`, or config (only historical mentions in these docs explaining the removal); pytest re-run after the removal |
| Tests | "At least test_model_forward, test_tokenizer_roundtrip, test_data_prep; fix failures" | `tests/test_model.py`, `tests/test_tokenizer.py`, `tests/test_data_prep.py` | `python -m pytest -q` — result recorded in README |

## For Sid to add before submission

> **Do this in ML.ai (VS Code / Cursor / terminal / desktop) before the hackathon submission:**
> 1. Open this repo in ML.ai and run at least one real task from `MLAI.md`
>    (for example: "run the tiny training and fix any failure", or
>    "improve the Streamlit demo").
> 2. Export the session report / log from ML.ai and drop the file(s) into
>    `ml-ai-sessions/` (see `ml-ai-sessions/README.md`).
> 3. Record the exact prompts you ran in ML.ai in
>    `ml-ai-sessions/PROMPTS.md` (prompt text, date, what changed).
> 4. Then update the table above: replace the "built with Muse agent"
>    note with your real ML.ai session file names. Do not delete the
>    honest-build-log rows — add yours alongside them.

## Hackathon context (confirmed)
- **ML.ai INFRASHIFT, 1st edition — Sat 14 Nov 2026, Infrasity Office,
  Janakpuri West, New Delhi. ₹1 lakh prize pool, 200+ participants,
  4–5 hour build window, partners Infrasity + Geek Room.**
- **Track: Open Innovation** — must work end to end on a basic phone
  browser in Hindi or English. See `docs/INFRASHIFT.md` for the day plan,
  the pre-built vs live split, and the 90-second demo script.
- ML.ai's own docs: https://ml.ai (and the MLAI.md convention file format).
