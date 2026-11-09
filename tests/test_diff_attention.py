import math

import torch
import torch.nn.functional as F

from diff_transformer import DiffAttention, DiffConfig, diff_attention, lambda_init
from diff_transformer.attention import apply_rope, precompute_rope


def test_lambda_init_schedule():
    assert math.isclose(lambda_init(1), 0.2)  # 0.8 - 0.6
    assert math.isclose(lambda_init(2), 0.8 - 0.6 * math.exp(-0.3))
    assert lambda_init(1) < lambda_init(2) < lambda_init(12) < 0.8


def test_matches_paper_formula_written_naively():
    """Eq. (1), computed with explicit matrices, one head and one sequence at a time."""
    b, h, t, d = 2, 3, 7, 8
    q, k = torch.randn(b, h, 2, t, d), torch.randn(b, h, 2, t, d)
    v, lam = torch.randn(b, h, t, 2 * d), torch.tensor(0.37)
    mask = torch.arange(t)[None, :] <= torch.arange(t)[:, None]
    got = diff_attention(q, k, v, lam, mask)
    for bi in range(b):
        for hi in range(h):
            q1, q2, k1, k2 = q[bi, hi, 0], q[bi, hi, 1], k[bi, hi, 0], k[bi, hi, 1]
            a1, a2 = (q1 @ k1.T) / math.sqrt(d), (q2 @ k2.T) / math.sqrt(d)
            a1, a2 = a1.masked_fill(~mask, -torch.inf), a2.masked_fill(~mask, -torch.inf)
            want = (a1.softmax(-1) - lam * a2.softmax(-1)) @ v[bi, hi]
            torch.testing.assert_close(got[bi, hi], want, atol=1e-5, rtol=1e-5)


def test_lambda_zero_is_plain_attention():
    q, k, v = torch.randn(1, 2, 2, 5, 8), torch.randn(1, 2, 2, 5, 8), torch.randn(1, 2, 5, 16)
    got = diff_attention(q, k, v, torch.tensor(0.0))
    want = F.scaled_dot_product_attention(q[:, :, 0], k[:, :, 0], v)
    torch.testing.assert_close(got, want)


def test_lambda_reparameterisation():
    cfg = DiffConfig(d_model=64, head_dim=16)
    attn = DiffAttention(cfg, layer_idx=3)
    with torch.no_grad():
        for p in (attn.lambda_q1, attn.lambda_k1, attn.lambda_q2, attn.lambda_k2):
            p.zero_()
        attn.lambda_q1.fill_(1.0)
        attn.lambda_k1.fill_(0.5)  # dot = 16 * 0.5 = 8
    want = math.exp(8.0) - math.exp(0.0) + lambda_init(3)
    assert math.isclose(attn.lam().item(), want, rel_tol=1e-5)


def test_lambda_gets_gradient():
    cfg = DiffConfig(vocab_size=20, d_model=64, n_layers=2, head_dim=16, max_seq_len=16)
    from diff_transformer import DiffTransformerLM
    m = DiffTransformerLM(cfg)
    idx = torch.randint(0, 20, (2, 8))
    _, loss = m(idx, idx)
    loss.backward()
    for name, p in m.named_parameters():
        if "lambda_" in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_rope_is_relative():
    """Scores after rotary depend only on the position gap, so shifting both by the same amount changes nothing."""
    cos, sin = precompute_rope(16, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    def score(pq, pk):
        return (apply_rope(q, cos, sin, pq) * apply_rope(k, cos, sin, pk)).sum()
    torch.testing.assert_close(score(10, 4), score(30, 24), atol=1e-4, rtol=1e-4)


def test_head_norm_and_scale_match_paper_eq3():
    """Module output == RMSNorm per head, times (1 - lambda_init), then the output projection."""
    cfg = DiffConfig(d_model=64, head_dim=16, max_seq_len=16)
    attn = DiffAttention(cfg, layer_idx=2)
    x = torch.randn(2, 6, 64)
    cos, sin = precompute_rope(16, 16, 10000.0)

    b, t, h, d = 2, 6, cfg.n_heads, 16
    q = attn.wq(x).view(b, t, h, 2, d).permute(0, 2, 3, 1, 4)
    k = attn.wk(x).view(b, t, h, 2, d).permute(0, 2, 3, 1, 4)
    v = attn.wv(x).view(b, t, h, 2 * d).transpose(1, 2)
    mask = torch.arange(t)[None, :] <= torch.arange(t)[:, None]
    heads = diff_attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, attn.lam(), mask)
    heads = heads * torch.rsqrt(heads.pow(2).mean(-1, keepdim=True) + cfg.norm_eps) * (1 - lambda_init(2))
    want = attn.wo(heads.transpose(1, 2).reshape(b, t, -1))
    torch.testing.assert_close(attn(x, cos, sin), want, atol=1e-5, rtol=1e-5)
