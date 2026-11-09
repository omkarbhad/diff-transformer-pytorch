# diff-transformer-pytorch

A from-scratch PyTorch implementation of **differential attention** from
[Differential Transformer](https://arxiv.org/abs/2410.05258) (Ye et al., ICLR 2025), with a
parameter-matched standard Transformer next to it so the two can be compared fairly.

This is an unofficial implementation written from the paper. The authors' reference code is at
<https://aka.ms/Diff-Transformer>.

## The idea

Softmax attention puts some weight on tokens that do not matter ("attention noise"). Differential
attention computes **two** softmax attention maps and subtracts one from the other, so weight that
both maps share cancels out:

```
DiffAttn(X) = ( softmax(Q1 K1ᵀ / √d) − λ · softmax(Q2 K2ᵀ / √d) ) V
```

`λ` is a learnable scalar per layer. The paper re-parameterises it as
`exp(λq1·λk1) − exp(λq2·λk2) + λ_init`, with `λ_init = 0.8 − 0.6·exp(−0.3·(l − 1))` for layer `l`.
Each head output then goes through an RMSNorm and is scaled by `(1 − λ_init)`.

The paper uses half as many heads (`d_model / 2d`), because every head holds two query/key groups.
That keeps parameters and compute equal to the baseline.

## Install

```bash
git clone https://github.com/omkarbhad/diff-transformer-pytorch
cd diff-transformer-pytorch
pip install -e ".[dev]"
```

## Usage

```python
import torch
from diff_transformer import DiffConfig, DiffTransformerLM

cfg = DiffConfig(vocab_size=256, d_model=256, n_layers=4, head_dim=32)   # differential=True
model = DiffTransformerLM(cfg)

idx = torch.randint(0, 256, (2, 64))
logits, loss = model(idx, targets=idx.roll(-1, 1))

out = model.generate(idx[:, :8], max_new_tokens=32, temperature=0.8)     # uses a KV cache

baseline = DiffTransformerLM(DiffConfig(vocab_size=256, d_model=256, n_layers=4, head_dim=32,
                                        differential=False))             # standard attention
```

The attention operator is also available on its own:

```python
from diff_transformer import diff_attention
# q, k: (batch, heads, 2, seq, d)   v: (batch, heads, seq, 2d)   lam: scalar tensor
out = diff_attention(q, k, v, lam, mask)
```

## What is implemented

| Paper | Here |
| --- | --- |
| Eq. 1, two softmax maps subtracted | `diff_attention` |
| Eq. 2, `λ` re-parameterisation and `λ_init` schedule | `DiffAttention.lam`, `lambda_init` |
| Eq. 3, per-head RMSNorm and `(1 − λ_init)` scale | `DiffAttention.forward` |
| Eq. 4-5, pre-RMSNorm and SwiGLU layout | `Block` |
| `h = d_model / 2d` heads | `DiffConfig.n_heads` |

Also included: rotary embeddings, tied embeddings, and a KV cache for generation.

## Tests

```bash
pytest -q
```

24 tests. The main ones:

- Differential attention equals eq. 1 written out naively, one head at a time.
- With `λ = 0` it reduces to ordinary attention.
- `λ` follows eq. 2, the head norm and `(1 − λ_init)` scale follow eq. 3, and `λ` receives gradients.
- No token sees the future (changing later tokens does not change earlier logits).
- Decoding with the KV cache gives the same logits as a full forward pass, including a chunk after a cached prefix.
- The model can overfit a single sequence, and the parameter difference to the baseline is exactly the `λ` and head-norm weights.

I checked that the tests fail when I break the code on purpose: flip the sign of `λ`, remove the
causal mask, ignore the cache offset in the rotary embedding, or drop the `(1 − λ_init)` scale.

## A small experiment

`examples/train_shakespeare.py` trains a character-level model on tiny Shakespeare, with either
attention. Same seed, same data order, same schedule.

```bash
python examples/train_shakespeare.py --attn diff
python examples/train_shakespeare.py --attn standard
```

Settings: 4 layers, `d_model` 128, `head_dim` 32, context 128, batch 32, 2,000 steps, learning rate
2e-3 with warmup and cosine decay. About 0.86M parameters. Validation loss at the end, 3 seeds:

| Attention | Seed 0 | Seed 1 | Seed 2 | Mean (nats) | Mean (bits/char) |
| --- | --- | --- | --- | --- | --- |
| Standard | 1.5041 | 1.4991 | 1.5077 | **1.5036** | **2.169** |
| Differential | 1.5382 | 1.5230 | 1.5352 | 1.5321 | 2.210 |

**At this scale, differential attention is slightly worse than the baseline in all three seeds.**
That does not contradict the paper. The paper trains models from 830M to 13B parameters, and 3B
models on 1T tokens, and reports that differential attention matches the baseline with about 65%
of the parameters or training tokens. This run has only 2 differential heads, 2,000 steps and no tuning of the learning rate. It
says nothing either way about large models. A larger run (more layers, more steps) is the next thing
to try.

## Layout

```
diff_transformer/
  config.py       DiffConfig
  attention.py    DiffAttention, standard Attention, RoPE, KV cache, causal mask
  model.py        Block, DiffTransformerLM, generate
tests/            24 tests
examples/         train_shakespeare.py
```

## Citation

```bibtex
@inproceedings{ye2025differential,
    title     = {Differential Transformer},
    author    = {Tianzhu Ye and Li Dong and Yuqing Xia and Yutao Sun and Yi Zhu and Gao Huang and Furu Wei},
    booktitle = {International Conference on Learning Representations},
    year      = {2025},
    url       = {https://arxiv.org/abs/2410.05258}
}
```

## License

MIT
