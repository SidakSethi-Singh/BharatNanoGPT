import torch

from model import GPT, build_gpt


def test_model_forward():
    cfg = dict(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    model = build_gpt(cfg)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and loss.item() > 0


def test_model_generate_shape():
    model = GPT(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model.eval()
    x = torch.tensor([[10, 20, 30]])
    out = model.generate(x, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 8)


def test_num_parameters_positive():
    model = build_gpt(dict(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64))
    assert model.num_parameters() > 0
