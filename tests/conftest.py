import pytest
import torch

from diff_transformer import DiffConfig, DiffTransformerLM


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def make(differential=True, **kw):
    cfg = DiffConfig(vocab_size=50, d_model=64, n_layers=3, head_dim=16, max_seq_len=64, differential=differential, **kw)
    return DiffTransformerLM(cfg).eval()


@pytest.fixture(params=[True, False], ids=["diff", "baseline"])
def model(request):
    return make(request.param)
