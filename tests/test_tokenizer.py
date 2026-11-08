from tokenizer import CharTokenizer, train_tokenizer


def test_tokenizer_roundtrip_bpe_or_char():
    texts = ["नमस्ते भाई, कैसे हो?", "bhai kal exam hai", "The capital of India is"]
    tok, fallback = train_tokenizer(texts, vocab_size=256)
    for t in texts:
        ids = tok.encode(t)
        assert len(ids) > 0
        # Round-trip must preserve the text for the char fallback, and for
        # BPE it must preserve it too on this small in-domain sample.
        assert tok.decode(ids) == t, (t, tok.decode(ids))


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer.train(["नमस्ते", "hello bhai"])
    assert tok.decode(tok.encode("नमस्ते")) == "नमस्ते"
    assert tok.decode(tok.encode("hello")) == "hello"


def test_tokenizer_save_load(tmp_path):
    tok, _ = train_tokenizer(["राम सुबह उठकर", "bhai kal exam hai"], vocab_size=256)
    path = str(tmp_path / "tok.json")
    tok.save(path)
    from tokenizer import load_tokenizer

    tok2 = load_tokenizer(path)
    assert tok2.decode(tok2.encode("राम सुबह")) == "राम सुबह"
