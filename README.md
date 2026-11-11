# BharatNanoGPT 🇮🇳

A tiny GPT **trained from scratch** (Karpathy nanoGPT-style) on a small
Hindi–English / Hinglish corpus. No pre-trained weights, no LLM API at
inference — generation comes only from our own trained weights, offline.

Built for **ML.ai INFRASHIFT, Sat 14 Nov 2026** (Infrasity Office,
Janakpuri West, New Delhi) — track: **Open Innovation**. The demo must work
end to end on a basic phone browser in Hindi or English. See
`docs/INFRASHIFT.md` for the day plan and the 90-second demo script.

## The pitch (30 seconds)

Every other team calls an API. We trained this — a tiny GPT, from scratch,
on Hindi and Hinglish. It runs offline on a phone, with no API key and no
per-token cost, so a Hindi-medium creator can use it in their own language.

**Buyer + price (proposed, not launched):** Hindi-medium coaching /
content creators and small ed-tech teams in India — **₹499/month per
creator, ₹4,999/year per small team, free for students.**

## What is in this repo

| File | What it does |
|---|---|
| `model.py` | Decoder-only Transformer from scratch (GPT, Block, CausalSelfAttention, weight-tied LM head, `generate()` with temperature / top-k) |
| `config/train_tiny.py` | Tiny config, ~4M params (4 layers, 256 embd, vocab 2048, block 128, 600 iters) — CPU smoke test |
| `config/train_bharat.py` | Full config, ~14M params (6 layers, 384 embd, vocab 8192, block 256, 6000 iters) — free Colab T4 |
| `tokenizer/` | BPE tokenizer (`tokenizers`, vocab 2048 tiny / 8192 full) for Devanagari + Latin, with a character-level fallback |
| `data/` | Corpus build + tokenise (`prepare.py`): hand-written Hindi / English / Hinglish + Hindi & English Wikipedia extracts (CC BY-SA 4.0). Sources in `data/SOURCES.md` |
| `train.py` / `sample.py` / `evaluate.py` | AdamW + cosine LR training with checkpoints; CLI generation; val perplexity + samples + repetition rate |
| `app.py` | Streamlit demo — mobile-first: single column, no sidebar, big full-width buttons, Hindi + English sample prompts on the first screen |
| `tests/` | pytest: model forward, tokenizer round-trip, data prep |
| `MLAI.md` | Conventions for the ML.ai coding agent |
| `docs/ML_AI_USAGE.md` | How ML.ai was used on this project (honest build log) |
| `docs/INFRASHIFT.md` | Hackathon day plan, pre-built vs live split, demo script |
| `ml-ai-sessions/` | Sid's real exported ML.ai session reports go here (see its README) |

## Corpus (measured)

- Hand-written, in-repo: `data/corpus/hindi.txt` 106 lines, `data/corpus/english.txt` 106 lines, `data/corpus/hinglish.txt` 101 lines.
- Plus Hindi + English Wikipedia extracts fetched by `data/prepare.py` (MediaWiki API, plain text, citation markers stripped).
- Built corpus: `data/corpus.txt`, 1,139 lines, 386,695 bytes.
- Tokenised (BPE, vocab 2048): `data/train.bin` 114,266 tokens, `data/val.bin` 6,014 tokens (95/5 split, uint16 ids).

## Training — status (Sun 8 Nov 2026, verified, CPU)

**Tiny CPU smoke test is finished** (`config/train_tiny.py`, started 07:46 UTC).

Phase 1 — 600 iters from scratch, ~33 min on 4 CPU threads:

| iter | train loss | val loss |
|---|---|---|
| 0 | 7.67 | 7.68 |
| 100 | 5.20 | 5.36 |
| 200 | 4.37 | 4.77 |
| 300 | 3.94 | 4.58 |
| 400 | 3.59 | 4.51 |
| 500 | 3.33 | 4.46 |
| 599 | 3.21 | 4.44 |

Phase 2 — resumed from iter 599 (`--init-from out/tiny/ckpt.pt`, 300 more
iters, fresh cosine schedule) to see if samples improved: train loss fell
to 1.83 by iter 900, but **val loss rose to 4.90** — the model was
memorising the 120k-token corpus, not generalising. Training was stopped
at iter 900. The loss curve (`assets/loss_curve.png`, from
`out/tiny/train_log.jsonl`) shows this honestly: train keeps falling,
val bottoms out at iter 600 and then rises.

**Demo checkpoint is `out/tiny/ckpt.pt` = iter 600 (best val loss 4.38,
~15 MB)** — the later checkpoints were deliberately *not* used, because
their val loss is worse. `python evaluate.py` on it: **val loss 4.45,
perplexity 85.3** (random-batch variation across eval runs: 4.38–4.45).

**Full 14M run (Colab T4) — RAN 8 Nov 2026, but overfitted; checkpoints lost.**
On 8 Nov the full run (`config/train_bharat.py`, 6000 iters, ~53 min on a
free T4) reached iter 5999 and printed `Done`:

- Train loss 7.67 → **0.0236** (the model memorised the corpus)
- Val loss 7.68 → **9.88**, perplexity **20,876**; best val loss **6.57**
  (early in the run) — worse than the tiny model's best val loss 4.38 /
  perplexity 85.3
- Samples mixed memorised Wikipedia fragments into the Hindi prompt
  instead of generating coherent Hindi (e.g. `राम सुबह उठकर स्कूल गया।`
  followed by an English music-dictionary fragment)

The 120,280-token corpus is too small for a ~14M model, and 6000 iters
was too many. **The 14 checkpoints (742 MB) were then LOST**: after a
disconnect, Reconnect gave a new Colab runtime with an empty `/content`
— the files only ever lived on that temporary machine. Nothing was
downloaded in time. The numbers above survive from screenshots, so they
are reported honestly here; the checkpoints themselves are gone.

**Retrain (Colab T4) — RAN 11 Nov 2026, 1000 iters, ~10.5 min.**
`config/train_bharat.py --max-iters 1000`, data train=90,802 / val=4,779
tokens (BPE vocab 8192 tokenisation of the 120,280-token corpus):

- Train loss 9.12 → **0.581**; val loss 9.12 → **6.30** at iter 999
- **Best val loss 5.6645 at iter 500 (perplexity 288.5)** — the best
  checkpoint is `out/bharat/ckpt_500.pt`, not the final `ckpt.pt`
- Much better than the 6000-iter run (best 6.57, final 9.88, ppl 20,876),
  but still worse than the tiny model (best val 4.38, ppl 85.3), so the
  demo checkpoint stays the tiny one. Samples are still fragmentary
  (e.g. `राम सुबह उठकर` → Hindi fragments mixed with English text) —
  the corpus is still too small for a 14M model to learn coherent Hindi.
- Checkpoints were copied to Google Drive
  (`MyDrive/BharatNanoGPT-retrain-1000/`) right after training, so this
  run cannot be lost the way the 6000-iter run was.

**Demo checkpoint stays `out/tiny/ckpt.pt` (iter 600, val 4.38)** until a
better full-model checkpoint exists.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m data.prepare            # build corpus + train.bin / val.bin
python train.py --config config/train_tiny.py   # tiny CPU run (~4M params)
python sample.py --checkpoint out/tiny/ckpt.pt --prompt "राम सुबह उठकर"
python sample.py --checkpoint out/tiny/ckpt.pt --prompt "bhai kal exam hai" --temperature 0.8 --top-k 40
python evaluate.py --checkpoint out/tiny/ckpt.pt
streamlit run app.py              # phone-friendly demo
```

## Tests (measured)

```bash
python -m pytest -q
# 9 passed — verified 8 Nov 2026 from the repo root with .venv (Python 3.12, torch 2.5.1+cpu)
```

## Results (measured, not claimed)

| Run | Train loss | Val loss | Perplexity | Samples |
|---|---|---|---|---|
| Tiny CPU (~4M, 600 iters) | 7.67 → 3.21 | 7.68 → **4.38** (best, iter 600) | **85.3** | See below — fragmentary, honest |
| Phase 2 resume (+300 iters) | → 1.83 | **rose to 4.90** (overfit, stopped) | — | Worse val; not used for demo |
| Full Colab T4 (~14M, 6000 iters, 8 Nov) | → **0.0236** | **9.88** (best 6.57, overfit) | **20,876** | Memorised Wikipedia fragments; checkpoints lost on reconnect — see above |
| Full retrain T4 (~14M, 1000 iters, 11 Nov) | 9.12 → 0.581 | **6.30** (best **5.66** at iter 500) | **288.5** (best) | Still fragmentary; best ckpt is `ckpt_500.pt`, saved to Drive |

Sample outputs from the demo checkpoint (iter 600, temp 0.8, top-k 40 —
exactly what `python sample.py` prints; read them before judging):

```
राम सुबह उठकर → "राम सुबह उठकर मिपॉर्ड क्बी मा रूपा द्पिया गया, ..."
bhai kal exam hai aur mujhe → "bhai kal exam hai aur mujhe bahut bii hai. ..."
The capital of India is → "The capital of India is of averorrown, and then ..."
भारत एक बड़ा देश है → "भारत एक बड़ा देश है और के शक्रम है, ..."
```

**Honest read:** the model learned real Hindi/Hinglish/English words and
short phrases, and the loss fell from 7.67 to 3.21 — the pipeline genuinely
trains. But at 120k tokens and ~4M params it does **not** yet produce
coherent full sentences; completions are fragmentary. That is the real
result of this smoke test, and it is why the full 14M run on a bigger
corpus (2–5M tokens, Colab T4) is the next step — not a claim that the
tiny model is already good. The demo's value is transparency: judges can
see the data, the loss curve, and the checkpoint behind every output.

## ML.ai — what it is and is not

**ML.ai (https://ml.ai) is a coding agent, not an API key and not a model
API.** It works in VS Code / Cursor, as a desktop app, from the terminal,
and as a GitHub PR reviewer; it reads `MLAI.md` for project conventions.
ML.ai helped build and fix this code. It never runs inside the trained
model — BharatNanoGPT generates only from its own weights, offline, with
no API key. Full honest log: `docs/ML_AI_USAGE.md`.

> **Still to do before submission:** Sid runs ML.ai in VS Code / Cursor
> on at least 2 modules of this repo and exports the session reports into
> `ml-ai-sessions/` (prompts + what changed + verification). That folder is
> empty right now — an empty, labelled folder, not hidden evidence. See
> `ml-ai-sessions/README.md`.

An earlier AI/ML API (aimlapi.com) integration was added in error and
**removed** — no `AIMLAPI_API_KEY`, no compare/judge mode. See
`docs/ML_AI_USAGE.md`.

## Limitations (say these to judges)

- Small corpus (~120k tokens) + small model (~4M tiny / ~14M full) =
  limited knowledge, occasional gibberish or repetition. That is the
  honest trade-off of training from scratch on a laptop/Colab budget.
- The tiny checkpoint is a smoke test to prove the pipeline end to end;
  the full 14M run (8 Nov) overfitted and its checkpoints were lost, so
  the tiny checkpoint stays the demo checkpoint until the ~1000-iter
  retrain (with Drive saving) produces a better val loss.
- Hindi generation quality depends on the corpus mix; Devanagari + Latin
  share one BPE vocabulary (2048 tiny / 8192 full).

## Licence

MIT — see `LICENSE`.
