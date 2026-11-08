# BharatNanoGPT at ML.ai INFRASHIFT

## Event
- **Event:** ML.ai INFRASHIFT, 1st edition — hosted by ML.ai
- **Date:** Saturday 14 November 2026, one-day
- **Venue:** Infrasity Office, Janakpuri West, New Delhi, India
- **Partners:** Infrasity + Geek Room
- **Prize pool:** ₹1 lakh | **Participants:** 200+ students & developers
- **Build window:** 4 to 5 hours — judged on whether it actually works, not on the pitch

## Track
**Open Innovation** — "open to any AI agent that solves a defined problem
for people in India, and it must work end to end on a basic phone browser
in Hindi or English."

BharatNanoGPT fits because: Hindi/Hinglish users need text completion
without paying per-token API costs, and the model runs offline on a basic
phone browser — no API key, no internet at demo time.

## What is pre-built vs what is built live (4–5h window)

The 4–5 hour window is **not** enough to train a model from scratch on the
day. So the honest split is:

| Pre-built before 14 Nov (this repo, done now) | Built / done live on the day |
|---|---|
| Tiny checkpoint trained + verified (`out/tiny/ckpt.pt`, ~4M params) | Load the checkpoint, verify the demo runs on a real phone |
| Corpus + tokenizer (`data/corpus.txt`, `tokenizer/tokenizer.json`) | Fix phone-layout bugs with ML.ai (record sessions in `ml-ai-sessions/`) |
| `train.py` / `sample.py` / `evaluate.py` / tests, all passing | Re-run eval, fix any bug judges hit, rehearse the 90-sec pitch |
| Streamlit demo (`app.py`), mobile-first | Final demo run-through on a real phone, submit on Unstop |
| README, business case, `MLAI.md`, `docs/ML_AI_USAGE.md` | Polish README / business case if judges ask questions |

**Checkpoint strategy (say this if asked):** the demo loads a checkpoint
that was pre-trained before the event, so it starts instantly on a phone.
Training is fully reproducible — `python -m data.prepare && python train.py`
re-trains it from scratch — but nobody claims that training happens live
in 5 minutes. The loss curve and training log in `out/tiny/` are the proof
it was trained; the demo is the proof it works.

## Day plan (14 Nov, 4–5h)
| Time | Do |
|---|---|
| 0:00–0:30 | Load pre-trained checkpoint, verify demo runs on a phone browser |
| 0:30–2:30 | Live fixes + polish with ML.ai (record sessions), fix phone layout bugs |
| 2:30–3:30 | Eval run, README / business case check, rehearse 90-sec pitch |
| 3:30–4:30 | Final demo run-through on a real phone, submit on Unstop, buffer |

## 90-second demo script

**English (say this, ~90 sec):**
1. (0–10s) "Every other team calls an API. We trained this — a tiny GPT,
   from scratch, on Hindi and Hinglish."
2. (10–35s) On a phone, tap the Hindi prompt: "राम सुबह उठकर" — the local
   model completes it live. Then the Hinglish one: "bhai kal exam hai".
3. (35–55s) Show the loss curve dropping and the checkpoint: "4 million
   parameters, trained on our own corpus, runs offline on this phone."
4. (55–75s) "No API key, no per-token cost. A Hindi-medium creator can use
   it offline, in their own language."
5. (75–90s) Business case: "Buyer: Hindi-medium coaching and content
   creators and small ed-tech teams in India. Price: ₹499 a month per
   creator, ₹4,999 a year per small team, free for students. This is a
   proposed price — the product is the demo you just saw working."

**हिंदी (same script, ~90 sec):**
1. (0–10s) "बाकी सब API बुलाते हैं। हमने ये खुद train किया है — एक छोटा
   GPT, हिंदी और Hinglish पर, शुरू से।"
2. (10–35s) फोन पर हिंदी prompt दबाएँ: "राम सुबह उठकर" — मॉडल live पूरा
   करता है। फिर Hinglish: "bhai kal exam hai"।
3. (35–55s) Loss curve दिखाएँ: "40 लाख parameters, अपने डेटा पर train
   हुआ, फोन पर offline चलता है।"
4. (55–75s) "कोई API key नहीं, कोई per-token खर्च नहीं। हिंदी-medium
   creator इसे offline, अपनी भाषा में इस्तेमाल कर सकता है।"
5. (75–90s) Business case: "खरीदार: भारत के हिंदी-medium coaching और
   content creators और छोटे ed-tech teams। कीमत: ₹499 प्रति माह प्रति
   creator, ₹4,999 प्रति वर्ष प्रति छोटी team, students के लिए free।
   यह proposed कीमत है — product वही demo है जो आपने अभी चलते देखा।"

## Phone-demo checklist (before judging)
- [ ] Demo opens on a basic phone browser (Chrome, no desktop needed)
- [ ] Hindi + English sample prompts visible on the first screen
- [ ] Buttons are big enough to tap (min 56px, full width)
- [ ] Single column, no sidebar, no horizontal scrolling
- [ ] Generation works with mobile data / offline after load
- [ ] Business case slide/line ready at the end of the demo
