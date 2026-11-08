import os

from data.prepare import clean_text, load_local_corpus


def test_clean_text_normalises_and_drops_short_lines():
    lines = clean_text("नमस्ते भाई, कैसे हो? https://example.com [1]\nshort\n== Heading ==")
    assert lines == ["नमस्ते भाई, कैसे हो?"]
    assert "https://example.com" not in lines[0]


def test_local_corpus_loads():
    lines = load_local_corpus("data/corpus")
    assert len(lines) >= 200, f"expected >=200 local lines, got {len(lines)}"
    assert any("bhai kal exam" in ln for ln in lines)
    assert any("भारत" in ln for ln in lines)


def test_corpus_files_exist():
    for name in ("hindi.txt", "english.txt", "hinglish.txt"):
        assert os.path.exists(f"data/corpus/{name}")
