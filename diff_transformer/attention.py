"""Differential attention (Ye et al., 2024, https://arxiv.org/abs/2410.05258) and a standard baseline.

Differential attention computes two softmax attention maps and subtracts one from the other,
so that attention mass that both maps put on irrelevant tokens ("attention noise") cancels:

    DiffAttn(X) = (softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))) V

Layout used here, per head: q, k are (b, h, 2, t, d) -- the 2 is the pair (Q1, Q2) -- and v is
(b, h, t, 2d). Both attention classes share one interface so that the model can swap them.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiffConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def precompute_rope(head_dim: int, max_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start: int = 0) -> torch.Tensor:
    """Rotary embedding on the last dim; the sequence axis is dim -2. Works for any leading dims."""
    t = x.shape[-2]
    c = cos[start : start + t].to(x.dtype)
    s = sin[start : start + t].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class KVCache:
    """Keys and values of one layer. Both are concatenated along the sequence axis (dim -2)."""

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[-2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.k = k if self.k is None else torch.cat([self.k, k], dim=-2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=-2)
        return self.k, self.v


def causal_mask(t: int, s: int, device) -> Optional[torch.Tensor]:
    """Boolean (t, s) mask for t new queries over s keys, the last t of which are the queries' own
    positions. Query i may see key j if j <= s - t + i. A single new query sees everything."""
    if t == 1:
        return None
    return torch.arange(s, device=device)[None, :] <= torch.arange(s - t, s, device=device)[:, None]


def lambda_init(layer_idx: int) -> float:
    """Paper eq. after (2): lambda_init = 0.8 - 0.6 * exp(-0.3 * (l - 1)), with l counted from 1."""
    return 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))


def diff_attention(q, k, v, lam, mask=None):
    """Paper eq. (1). q, k: (b, h, 2, t|s, d); v: (b, h, s, 2d); lam: scalar tensor."""
    a1 = F.scaled_dot_product_attention(q[:, :, 0], k[:, :, 0], v, attn_mask=mask)
    a2 = F.scaled_dot_product_attention(q[:, :, 1], k[:, :, 1], v, attn_mask=mask)
    return a1 - lam.to(a1.dtype) * a2


class DiffAttention(nn.Module):
    """Multi-head differential attention (paper Fig. 2).

    - h = d_model / 2d heads; each head has two query/key groups of size d and a value of size 2d.
    - lambda is one learnable scalar per layer, shared by all heads, re-parameterised as
      exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init  (paper eq. 2).
    - Each head output goes through RMSNorm and is scaled by (1 - lambda_init) (paper eq. 3).
    """

    def __init__(self, cfg: DiffConfig, layer_idx: int):
        super().__init__()
        d = cfg.head_dim
        self.h, self.d = cfg.n_heads, d
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.lambda_q1 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_init = lambda_init(layer_idx)
        self.head_norm = RMSNorm(2 * d, cfg.norm_eps)

    def lam(self) -> torch.Tensor:
        l1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum().float())
        l2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum().float())
        return l1 - l2 + self.lambda_init

    def forward(self, x, cos, sin, cache: Optional[KVCache] = None):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.h, 2, self.d).permute(0, 2, 3, 1, 4)  # (b, h, 2, t, d)
        k = self.wk(x).view(b, t, self.h, 2, self.d).permute(0, 2, 3, 1, 4)
        v = self.wv(x).view(b, t, self.h, 2 * self.d).transpose(1, 2)        # (b, h, t, 2d)

        start = cache.length if cache is not None else 0
        q, k = apply_rope(q, cos, sin, start), apply_rope(k, cos, sin, start)
        if cache is not None:
            k, v = cache.update(k, v)

        mask = causal_mask(t, k.shape[-2], x.device)
        out = diff_attention(q, k, v, self.lam(), mask)                      # (b, h, t, 2d)
        out = self.head_norm(out) * (1.0 - self.lambda_init)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


class Attention(nn.Module):
    """Standard multi-head causal attention with h = d_model / d heads (the paper's baseline)."""

    def __init__(self, cfg: DiffConfig, layer_idx: int):
        super().__init__()
        self.h, self.d = cfg.n_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: Optional[KVCache] = None):
        b, t, _ = x.shape
        q, k, v = (w(x).view(b, t, self.h, self.d).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        start = cache.length if cache is not None else 0
        q, k = apply_rope(q, cos, sin, start), apply_rope(k, cos, sin, start)
        if cache is not None:
            k, v = cache.update(k, v)
        mask = causal_mask(t, k.shape[-2], x.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))
