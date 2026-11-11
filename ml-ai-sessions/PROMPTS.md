# PROMPTS.md — ML.ai sessions (BharatNanoGPT)

Status: **NOT YET RUN** — prepared 2026-11-11 as a fill-in template. No session
below is claimed as done. Fill in each row only after running ML.ai yourself
and exporting the session report into this folder. Use this only if
BharatNanoGPT becomes the main submission; otherwise use NyayaAgent's copy.

## Suggested sessions (at least 2 modules before 14 Nov 2026)

### Session 1 — Training fix (suggested)
- **Date:** —
- **Prompt (copy-paste):** "Run the tiny training and fix any failure"
- **What changed:** —
- **Verification:** `python -m pytest -q` → — passed; loss before/after: —
- **Exported session file:** — (save as e.g. `2026-11-12-train-fix-session.md`)

### Session 2 — Demo / sampling (suggested)
- **Date:** —
- **Prompt (copy-paste):** "Make the Streamlit demo (`app.py`) work cleanly on a basic phone browser: single column, large tap targets, Hindi + English prompts on the first screen"
- **What changed:** —
- **Verification:** `python -m pytest -q` → — passed; demo opened on phone browser: yes/no
- **Exported session file:** — (save as e.g. `2026-11-12-demo-session.md`)

## How to run each session (click-by-click)
1. Open VS Code / Cursor → Open Folder → `workspace/hackathon/BharatNanoGPT/`.
2. Open the ML.ai extension (ml.ai coding agent — not an API key).
3. Paste one prompt above, let ML.ai propose the change.
4. Review every diff yourself before accepting.
5. Run `python -m pytest -q` in the terminal (from the repo root, venv active).
6. Export the ML.ai session report and save it in this folder.
7. Fill in the row above, then add the same date/prompt/change/result to the
   table in `docs/ML_AI_USAGE.md`.

## What NOT to put here
- Do not invent session IDs, screenshots, or logs.
- Do not paste API keys or secrets. ML.ai needs no API key in this repo.
- Do not put aimlapi.com / AI/ML API material here — that was a different
  service, added in error and removed.
