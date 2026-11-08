# BharatNanoGPT — tiny CPU smoke-test config (~4M parameters).
# This is the config actually trained and verified in this repo (see README
# "Smoke-test results"). Runs on CPU in a few minutes.
# Parameter count (approx): embeddings 2048*256 + 128*256 = 0.56M,
# 4 blocks x ~0.79M = 3.2M, total ~3.7M (reported as ~4M / "approx 5M" class).
config = dict(
    # model
    vocab_size=2048,
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.0,
    # training
    batch_size=32,
    learning_rate=1e-3,
    max_iters=600,
    warmup_iters=50,
    lr_decay_iters=600,
    min_lr=1e-4,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    eval_interval=100,
    eval_iters=20,
    log_interval=20,
    # data / output
    data_dir="data",
    out_dir="out/tiny",
    seed=1337,
    # tokenizer
    tokenizer_vocab_size=2048,
)
