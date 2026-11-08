# Data sources — BharatNanoGPT

Every line in `data/corpus.txt` comes from one of the sources below.
No scraped copyrighted books, no Instagram / WhatsApp chats, no personal data.

## 1. In-repo hand-written corpus (always included, offline-safe)
- `data/corpus/hindi.txt` — ~100 original Hindi sentences, written for this project (everyday life, India, cricket, school, festivals).
- `data/corpus/english.txt` — ~100 original English sentences, parallel in spirit to the Hindi file (not a word-for-word translation).
- `data/corpus/hinglish.txt` — ~100 original Hinglish sentences (Roman-script Hindi-English code-mix, everyday conversation).
- Licence: created for this project, MIT (same as the repo).

## 2. Wikipedia extracts (CC BY-SA 4.0, fetched at prepare time)
- Hindi Wikipedia — https://hi.wikipedia.org — articles on the topics listed in `data/prepare.py` (`HI_TOPICS`: भारत, दिल्ली, क्रिकेट, …).
- English Wikipedia — https://en.wikipedia.org — matching topics (`EN_TOPICS`).
- Fetched via the MediaWiki API (`action=query&prop=extracts&explaintext=1`), plain text only, citation markers and URLs stripped by `clean_text()`.
- Licence: CC BY-SA 4.0 — attribution: Wikipedia contributors. If you redistribute the corpus, keep this attribution.
- If the download fails (offline / no internet), `prepare.py` skips it gracefully and builds the corpus from the in-repo files only. The smoke test in this repo was run on the combined corpus; see README for the exact token counts.

## 3. What is NOT in the corpus
- No IIT Bombay / Dakshina / OSCAR data was downloaded for the smoke test (listed in the spec as options for the full run; they need a licence check per dataset before use).
- No copyrighted books, song lyrics, or social-media scrapes.
