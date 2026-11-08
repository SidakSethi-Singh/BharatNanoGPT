# BharatNanoGPT — full config (~14M parameters).
# Train on a free Colab T4 GPU. NOT run to completion on the CPU smoke test;
# see README.md "Full 15M run (Colab)" for instructions.
# Parameter count (approx): embeddings 8192*384 + 256*384 = 3.2M,
# 6 blocks x ~1.77M = 10.6M, total ~14M (reported as ~15M in the pitch).
config = dict(
    # model
    vocab_size=8192,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.0,
    # training
    batch_size=64,
    learning_rate=3e-4,
    max_iters=6000,
    warmup_iters=200,
    lr_decay_iters=6000,
    min_lr=3e-5,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    eval_interval=500,
    eval_iters=100,
    log_interval=50,
    # data / output
    data_dir="data",
    out_dir="out/bharat",
    seed=1337,
    # tokenizer
    tokenizer_vocab_size=8192,
)
