## Part 4 - GPT

Material from [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY). The lecture builds a character-level, decoder-only transformer step by step, trained on [tiny shakespeare](input.txt).

This is the basic GPT from the lecture. The GPT-2 reproduction and my other GPT experiments are in XXX.

### What's here

The scripts follow the lecture's progression, with each one building on the last:

- [bigram.py](bigram.py) - the baseline: a bigram language model with token and position embeddings.
- [v2.py](v2.py) - adds masked multi-head self-attention, a feed-forward layer, and residual connections, grouped into transformer blocks.
- [v3.py](v3.py) - the full model from the end of the lecture: adds LayerNorm, dropout and a configurable number of blocks, and scales up to 6 layers, 6 heads, a 384-dimensional embedding and a 256-character context. It uses CUDA or Apple's MPS when available, and realistically needs one of them.
- [attention.py](attention.py) - a standalone `Head` / `MultiHeadAttention` with every size passed in as an argument, used to check the batched implementation against the separate-heads one.

Each script trains and then samples text:

```bash
uv run python src/zero_to_hero/part_04_gpt/v3.py
```

### Notebooks

In [notebooks/04_gpt](../../../notebooks/04_gpt):

- `gpt_lectures/gpt_1.ipynb` - follow-along notes for the lecture, from the data and the bigram model through to self-attention. It also has notes on the rest of the architecture (the rest of the code is in the scripts), decoder-only vs encoder-decoder, and pretraining vs fine-tuning.
- `gpt_intuition/` - small notebooks I wrote to understand pieces of attention:
  - `pos_embedding.ipynb` - why attention needs positional embeddings: without them, reordering the tokens just reorders the attention scores.
  - `embedding_add.ipynb` - what adding (rather than concatenating) token and position embeddings does inside an attention head.
  - `multi_attention_test.ipynb` - checks that computing all heads with one batched matrix multiply gives the same result as `attention.py`'s list of separate heads.
