"""Model configuration."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DiffConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    head_dim: int = 32            # "d" in the paper: the head size of the *baseline* Transformer
    differential: bool = True     # False builds the parameter-matched standard Transformer
    ffn_dim: Optional[int] = None  # default: 8/3 * d_model, rounded up to a multiple of 64
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = (int(8 / 3 * self.d_model) + 63) // 64 * 64
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        width = 2 * self.head_dim if self.differential else self.head_dim
        if self.d_model % width:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by {width}")

    @property
    def n_heads(self) -> int:
        # The paper halves the head count (h = d_model / 2d), because each differential head
        # holds two query/key groups. This keeps parameters and FLOPs equal to the baseline.
        return self.d_model // (2 * self.head_dim if self.differential else self.head_dim)

    def to_dict(self) -> dict:
        return asdict(self)
