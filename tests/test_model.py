import pytest
import torch

from diff_transformer import DiffConfig, DiffTransformerLM, new_cache
from conftest import make


def test_head_count_halves():
    assert DiffConfig(d_model=64, head_dim=16, differential=False).n_heads == 4
    assert DiffConfig(d_model=64, head_dim=16, differential=True).n_heads == 2


def test_param_counts_match_baseline():
    """Paper: h = d_model / 2d keeps parameters equal. The only extra weights are lambda and the head norm."""
    diff, base = make(True), make(False)
    c = diff.cfg
    extra = c.n_layers * (4 * c.head_dim + 2 * c.head_dim)
    assert diff.num_params() - base.num_params() == extra


def test_invalid_config():
    with pytest.raises(ValueError):
        DiffConfig(d_model=100, head_dim=16)


def test_shapes_and_loss(model):
    idx = torch.randint(0, 50, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, 50) and loss.ndim == 0 and torch.isfinite(loss)


def test_no_future_leakage(model):
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, 8:] = torch.randint(0, 50, (4,))
    la, _ = model(a)
    lb, _ = model(b)
    torch.testing.assert_close(la[:, :8], lb[:, :8], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:])


def test_kv_cache_matches_full_forward(model):
    idx = torch.randint(0, 50, (2, 20))
    full, _ = model(idx)
    cache = new_cache(model.cfg)
    steps = [model(idx[:, i : i + 1], cache=cache)[0] for i in range(20)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-4, rtol=1e-4)


def test_chunk_after_cached_prefix(model):
    idx = torch.randint(0, 50, (2, 24))
    full, _ = model(idx)
    cache = new_cache(model.cfg)
    model(idx[:, :10], cache=cache)
    tail, _ = model(idx[:, 10:], cache=cache)
    torch.testing.assert_close(tail, full[:, 10:], atol=1e-4, rtol=1e-4)


def test_greedy_generate_matches_naive_loop(model):
    prompt = torch.randint(0, 50, (1, 5))
    got = model.generate(prompt, 12, temperature=0)
    idx = prompt
    for _ in range(12):
        logits, _ = model(idx)
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(got, idx)


def test_length_limit(model):
    with pytest.raises(ValueError):
        model(torch.randint(0, 50, (1, 65)))
    assert model.generate(torch.randint(0, 50, (1, 60)), 100, temperature=0).shape[1] == 64


@pytest.mark.parametrize("differential", [True, False])
def test_overfits_one_sequence(differential):
    m = make(differential).train()
    idx = torch.randint(0, 50, (1, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        _, loss = m(idx, idx.roll(-1, 1))
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first
