# ml-ai-sessions/ — Sid's real ML.ai session evidence

This folder is intentionally empty in the repo as built. It is where Sid
adds genuine evidence of using **ML.ai (https://ml.ai)** — the AI coding
agent — on this project, so judges can see real use, not a claim.

## What to put here (before submission)
1. **Exported ML.ai session reports / logs** — export a session from ML.ai
   (VS Code / Cursor extension, desktop app, or terminal) and save the
   file here, e.g. `2026-11-08-train-fix-session.md`.
2. **`PROMPTS.md`** — the exact prompts you ran in ML.ai, with dates, e.g.:
   - Prompt: "Run the tiny training and fix any failure"
   - What changed: (files / diff summary)
   - Verification: (pytest result, loss before/after)
3. **Before/after diffs** (optional but strong for judges): `git diff`
   excerpts or screenshots showing what ML.ai changed and what you reviewed.

## What NOT to put here
- Do not invent session IDs, screenshots, or logs.
- Do not paste API keys or secrets. ML.ai needs no API key in this repo —
  the trained model runs offline and never calls ML.ai at runtime.
- Do not put aimlapi.com / AI/ML API material here — that was a different
  service, added in error and removed. See `docs/ML_AI_USAGE.md`.

## How judges should read this folder
If this folder is still empty at submission time, the honest statement is
in `docs/ML_AI_USAGE.md`: the repo was built by an autonomous coding agent
in this workspace following the ML.ai workflow in `MLAI.md`, and Sid's own
ML.ai sessions are to be added here. An empty folder is not hidden — it is
labelled, with instructions, so nobody mistakes it for evidence.
