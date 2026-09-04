import math

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import nn

from transformer.functions import attention


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head self-attention for an encoder.

    Projects the input into `h` heads of width `d_k = d_model // h`, attends
    within each head, then merges the heads and mixes them with `w_o`.
    """

    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        if d_model % h != 0:
            raise ValueError(  # noqa: NAR001
                f"d_model={d_model} must be divisible by h={h}"
            )
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        scale = 1.0 / math.sqrt(d_model)  # noqa: NAR001
        self.w_q = nn.Parameter(data=torch.randn(h, d_model, self.d_k) * scale)  # noqa: NAR001
        self.w_k = nn.Parameter(data=torch.randn(h, d_model, self.d_k) * scale)  # noqa: NAR001
        self.w_v = nn.Parameter(data=torch.randn(h, d_model, self.d_k) * scale)  # noqa: NAR001
        self.w_o = nn.Parameter(data=torch.randn(d_model, d_model) * scale)  # noqa: NAR001

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        batch: Float[torch.Tensor, "batch seq d_model"],
        mask: Bool[torch.Tensor, "#batch #h #seq seq"] | None = None,
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        """Attend `batch` over itself, optionally hiding keys marked False.

        Args:
            batch: Input sequences.
            mask: Passed through to `attention`. Leading axes may be 1 and
                broadcast, so `(batch, 1, 1, seq)` is a per-sequence padding
                mask and `(1, 1, seq, seq)` a shared causal mask.
        """
        batch_size, n, d_model = batch.shape
        q = batch.unsqueeze(dim=1) @ self.w_q
        k = batch.unsqueeze(dim=1) @ self.w_k
        v = batch.unsqueeze(dim=1) @ self.w_v
        concat = attention(q=q, k=k, v=v, mask=mask)  # (batch, h, seq, d_k)
        merged = concat.transpose(dim0=1, dim1=2).reshape(batch_size, n, d_model)  # noqa: NAR001
        return merged @ self.w_o
