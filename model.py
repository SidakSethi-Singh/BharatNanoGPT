"""BharatNanoGPT model — decoder-only Transformer, Karpathy nanoGPT-style.

Written from scratch for this project (inspired by, not copied from, nanoGPT).
No pre-trained weights are loaded anywhere in this file: every parameter is
randomly initialised and learned by train.py on our own corpus.

Architecture: token embedding + learned position embedding, N x Block
(CausalSelfAttention + MLP + LayerNorms), final LayerNorm + LM head
(weight-tied to the token embedding).
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        # One fused projection for q, k, v (nanoGPT style).
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Flash-attention path when available; falls back to a manual
        # causal implementation on older PyTorch / CPU-only builds.
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:  # pragma: no cover - manual fallback
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]),
                ln_f=nn.LayerNorm(n_embd),
            )
        )
        # Weight tying: LM head shares weights with the token embedding.
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation. idx: (B, T) LongTensor of token ids."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_gpt(config):
    """Build a GPT from a config dict (see config/train_bharat.py)."""
    return GPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config.get("dropout", 0.0),
    )
