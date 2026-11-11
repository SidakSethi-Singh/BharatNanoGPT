"""Build BharatNanoGPT's corpus and tokenised train/val splits.

Sources (see data/SOURCES.md for licences):
  1. In-repo hand-written corpus: data/corpus/{hindi,english,hinglish}.txt
     — always included, always works offline.
  2. Wikipedia extracts (CC BY-SA): Hindi + English articles on curated
     topics, fetched via the MediaWiki API. Skipped gracefully offline
     (--no-download), in which case only the in-repo corpus is used.

Pipeline: fetch -> NFC normalise -> strip markup/URLs -> dedupe lines ->
shuffle (seeded) -> 95/5 train/val split -> BPE tokenise -> train.bin / val.bin
(uint16 token ids, the nanoGPT convention).

Usage:
    python -m data.prepare                      # default: 120+120 wiki articles
    python -m data.prepare --no-download        # in-repo corpus only
    python -m data.prepare --config config/train_tiny.py
    python -m data.prepare --hi-articles 400 --en-articles 400   # bigger corpus
"""
import argparse
import importlib.util
import json
import os
import random
import re
import sys
import unicodedata
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HI_TOPICS = [
    "भारत", "दिल्ली", "मुंबई", "क्रिकेट", "हिंदी", "गंगा", "ताज महल",
    "महात्मा गांधी", "रामायण", "महाभारत", "दीवाली", "होली", "केरल",
    "राजस्थान", "हिमालय", "शिक्षा", "किसान", "रेलवे", "खेल", "संगीत",
    "विज्ञान", "इतिहास", "भोजन", "त्योहार", "नदी", "पहाड़", "समुद्र",
    "मंदिर", "किताब", "बच्चे",
]
EN_TOPICS = [
    "India", "Delhi", "Mumbai", "Cricket", "Hindi", "Ganges", "Taj Mahal",
    "Mahatma Gandhi", "Ramayana", "Mahabharata", "Diwali", "Holi", "Kerala",
    "Rajasthan", "Himalayas", "Education in India", "Agriculture in India",
    "Indian Railways", "Sport in India", "Music of India", "Science",
    "History of India", "Indian cuisine", "Festivals", "River", "Mountain",
    "Ocean", "Temple", "Book", "Children",
]


def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)  # citation markers
    text = re.sub(r"[ \t]+", " ", text)
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if len(ln) < 20:  # drop headings / fragments
            continue
        if ln.startswith("==") or ln.startswith("Category:"):
            continue
        lines.append(ln)
    return lines


def _api(url, timeout=20, retries=3):
    # Wikipedia rate-limits shared Colab IPs with HTTP 429. Retry with
    # backoff instead of failing every topic at once.
    import time as _time
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BharatNanoGPT/1.0 (hackathon; contact: sid)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            _time.sleep(2 * (attempt + 1))
    raise last


def wiki_titles(lang, topics, per_topic=4):
    """Top article titles for each topic via MediaWiki search."""
    titles = []
    for topic in topics:
        q = urllib.parse.quote(topic)
        url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search"
               f"&srsearch={q}&srlimit={per_topic}&format=json")
        try:
            data = _api(url)
            for hit in data.get("query", {}).get("search", []):
                titles.append(hit["title"])
        except Exception as e:  # noqa: BLE001 - offline / API hiccup: skip topic
            print(f"  search failed for {topic!r}: {e}")
    # dedupe, preserve order
    return list(dict.fromkeys(titles))


def wiki_extracts(lang, titles):
    """Plain-text extracts, batched 20 titles per request."""
    texts = []
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        q = urllib.parse.quote("|".join(batch))
        url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts"
               f"&explaintext=1&titles={q}&format=json")
        try:
            data = _api(url)
            for page in data.get("query", {}).get("pages", {}).values():
                ext = page.get("extract", "")
                if ext:
                    texts.append(ext)
        except Exception as e:  # noqa: BLE001
            print(f"  extracts failed for batch {i}: {e}")
    return texts


def download_wikipedia(hi_articles, en_articles):
    lines = []
    for lang, topics, limit in (("hi", HI_TOPICS, hi_articles), ("en", EN_TOPICS, en_articles)):
        print(f"Fetching {lang} Wikipedia ...")
        titles = wiki_titles(lang, topics, per_topic=8)[:limit]
        print(f"  {len(titles)} titles")
        for text in wiki_extracts(lang, titles):
            lines.extend(clean_text(text))
        print(f"  {len(lines)} total lines so far")
    return lines


def load_local_corpus(corpus_dir):
    lines = []
    for name in ("hindi.txt", "english.txt", "hinglish.txt"):
        path = os.path.join(corpus_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                lines.extend(clean_text(f.read()))
            print(f"  local {name}: included")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--out-text", default="data/corpus.txt")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--from-text", default=None,
                    help="Use this plain-text file as the corpus directly "
                         "(one line per sample), skipping Wikipedia. Use it "
                         "when Wikipedia returns HTTP 429, e.g. the prebuilt "
                         "corpus.txt from the repo's Colab run.")
    ap.add_argument("--hi-articles", type=int, default=120)
    ap.add_argument("--en-articles", type=int, default=120)
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.config:
        spec = importlib.util.spec_from_file_location("cfg", args.config)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args.vocab_size = mod.config.get("tokenizer_vocab_size", args.vocab_size)

    if args.from_text:
        with open(args.from_text, encoding="utf-8") as f:
            lines = clean_text(f.read())
        wiki_lines = 0
        print(f"  using prebuilt corpus file {args.from_text}: {len(lines)} lines")
    else:
        lines = load_local_corpus(args.corpus_dir)
        wiki_lines = 0
    if not args.no_download and not args.from_text:
        try:
            wiki = download_wikipedia(args.hi_articles, args.en_articles)
            wiki_lines = len(wiki)
            lines.extend(wiki)
        except Exception as e:  # noqa: BLE001 - offline: fall back to local only
            print(f"Wikipedia download failed ({e}); using in-repo corpus only.")

    # Dedupe (keep first occurrence), then seeded shuffle.
    seen, deduped = set(), []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            deduped.append(ln)
    random.Random(args.seed).shuffle(deduped)
    print(f"Corpus: {len(deduped)} unique lines "
          f"({wiki_lines} from Wikipedia, {len(deduped) - wiki_lines} local)")

    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(deduped) + "\n")

    # Tokenise with the project's own tokenizer (BPE, or char fallback).
    from tokenizer import train_tokenizer

    tok, fallback = train_tokenizer(deduped, vocab_size=args.vocab_size)
    if fallback:
        print("WARNING: character-level fallback tokenizer in use (see tokenizer/).")
    tok.save("tokenizer/tokenizer.json")
    print(f"Tokenizer ({tok.kind}) vocab_size={tok.vocab_size} saved to tokenizer/tokenizer.json")

    import numpy as np

    ids = []
    for ln in deduped:
        ids.extend(tok.encode(ln))
        ids.append(tok.eos_id)
    ids = np.array(ids, dtype=np.uint16 if tok.vocab_size < 65536 else np.int32)
    n_val = max(1, int(0.05 * len(ids)))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    train_ids.tofile("data/train.bin")
    val_ids.tofile("data/val.bin")
    print(f"Tokens: total={len(ids):,} train={len(train_ids):,} val={len(val_ids):,}")
    if len(ids) < 20000:
        print("WARNING: corpus is too small to train properly "
              f"({len(ids):,} tokens). Wikipedia probably returned HTTP 429. "
              "Do NOT train the full config on this — re-run prepare with "
              "--from-text <prebuilt corpus.txt>, or wait and retry Wikipedia.")
    print("Wrote data/train.bin, data/val.bin, data/corpus.txt")


if __name__ == "__main__":
    main()
