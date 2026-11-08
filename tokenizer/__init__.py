"""BPE tokenizer for BharatNanoGPT, built on Hugging Face `tokenizers`.

Two implementations live here:
1. BPETokenizer — real byte-level BPE trained on our corpus (default).
2. CharTokenizer — character-level fallback, used ONLY when `tokenizers`
   is not installed. It is honest about what it is: train_tokenizer.py
   prints a clear warning and records the fallback in the tokenizer
   metadata, so nobody can mistake a char tokenizer for BPE.

Both expose the same interface: encode(text) -> list[int],
decode(ids) -> str, vocab_size, save(path), load(path).
"""
import json
import os

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class CharTokenizer:
    """Character-level fallback. Deterministic, no dependencies."""

    kind = "char"

    def __init__(self, chars=None):
        self.chars = sorted(set(chars)) if chars else []
        self.stoi = {ch: i + len(SPECIAL_TOKENS) for i, ch in enumerate(self.chars)}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.stoi[tok] = i
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @classmethod
    def train(cls, texts, vocab_size=None):
        chars = set()
        for t in texts:
            chars.update(t)
        return cls(sorted(chars))

    def encode(self, text):
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        out = []
        for i in ids:
            if i in (self.bos_id, self.eos_id, self.pad_id):
                continue
            out.append(self.itos.get(i, ""))
        return "".join(out)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"kind": "char", "chars": self.chars, "vocab_size": self.vocab_size},
                f, ensure_ascii=False,
            )

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["chars"])
        return tok


class BPETokenizer:
    """Byte-level BPE tokenizer (Hugging Face `tokenizers`)."""

    kind = "bpe"

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.unk_id = tokenizer.token_to_id("<unk>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    @classmethod
    def train(cls, texts, vocab_size=8192):
        from tokenizers import Tokenizer, decoders
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        # Without a ByteLevel decoder, decode() returns raw byte mojibake
        # (e.g. 'à¤°...' instead of 'राम'). This line is the fix.
        tok.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=True)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._tok.save(path)

    @classmethod
    def load(cls, path):
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))


def load_tokenizer(path):
    """Load a tokenizer saved by train_tokenizer.py (BPE .json or char .json)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer not found: {path}. Run: python -m tokenizer.train_tokenizer")
    with open(path, encoding="utf-8") as f:
        head = json.load(f)
    if isinstance(head, dict) and head.get("kind") == "char":
        return CharTokenizer.load(path)
    return BPETokenizer.load(path)


def train_tokenizer(texts, vocab_size=8192):
    """Train BPE if `tokenizers` is available, else fall back to char-level.

    Returns (tokenizer, used_fallback: bool). The caller must log the
    fallback loudly — never silently pretend BPE worked.
    """
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("WARNING: `tokenizers` not installed — using character-level "
              "fallback tokenizer. Install tokenizers for real BPE: "
              "pip install tokenizers")
        return CharTokenizer.train(texts), True
    return BPETokenizer.train(texts, vocab_size=vocab_size), False
