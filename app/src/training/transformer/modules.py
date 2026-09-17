"""Encoder building blocks: multi-head self-attention and a pre-norm block.

Shapes use the axis vocabulary in this package's README: B batch, L sequence
length, d_model embedding width, h heads, d_k = d_model // h.
"""

import math
from typing import Any, Optional

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import nn

from training.transformer.constants import D_MODEL, B, H, L
from training.transformer.functions import attention


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head self-attention for an encoder, with an optional padding mask.

    Projects the input into `h` heads of width `d_k = d_model // h`, optionally
    applies QK norm, attends within each head, then merges the heads and mixes
    them with `w_o`. In and out are both (B, L, d_model).

    ```
                        batch (B, L, d_model)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
          @ w_q             @ w_k             @ w_v      (h, d_model, d_k)
              ▼                 ▼                 │
           q_norm            k_norm               │      RMSNorm over d_k,
              │                 │                 │      only if qk_norm
              ▼                 ▼                 ▼
            q (B,h,L,d_k)     k (B,h,L,d_k)     v (B,h,L,d_k)
              └────────────────►│◄────────────────┘
                                ▼
                     attention(q, k, v, mask=pad_mask)
                                │  out (B, h, L, d_k)
                                ▼
                     transpose(1, 2) + reshape          merge the heads
                                │  (B, L, d_model)
                                ▼
                             @ w_o                      mix across heads
                                │
                                ▼  (B, L, d_model)
    ```

    The three projections are one batched matmul each, not `h` separate ones:
    `batch.unsqueeze(1)` is (B, 1, L, d_model) and broadcasts against the (h,
    d_model, d_k) weight, so the head axis appears without a reshape. The
    per-head outputs only become a single vector at `w_o` — attention itself
    never mixes across heads.
    """

    def __init__(self, d_model: int, h: int, qk_norm: bool) -> None:
        """Raises ValueError if `d_model` is not divisible by `h`.

        Args:
            qk_norm: RMSNorm each query and key over `d_k`, per head, before the
                dot product, so attention scores stay bounded as weights grow.
        """
        super().__init__()
        if d_model % h != 0:
            raise ValueError(  # noqa: NAR001
                f"d_model={d_model} must be divisible by h={h}"
            )
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        scale = 1.0 / math.sqrt(d_model)  # noqa: NAR001
        self.w_q = nn.Parameter(
            data=torch.randn(h, d_model, self.d_k) * scale  # noqa: NAR001
        )
        self.w_k = nn.Parameter(
            data=torch.randn(h, d_model, self.d_k) * scale  # noqa: NAR001
        )
        self.w_v = nn.Parameter(
            data=torch.randn(h, d_model, self.d_k) * scale  # noqa: NAR001
        )
        self.w_o = nn.Parameter(
            data=torch.randn(d_model, d_model) * scale  # noqa: NAR001
        )
        self.q_norm = nn.RMSNorm(normalized_shape=self.d_k) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(normalized_shape=self.d_k) if qk_norm else nn.Identity()

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        batch: Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
        pad_mask: Optional[Bool[torch.Tensor, f"#{B} #{H} #{L} {L}"]],
    ) -> tuple[
        Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
        Float[torch.Tensor, f"{B} {H} {L} {L}"],
    ]:
        """Apply multi-head self-attention to `batch`.

        Args:
            batch: Token embeddings.
            pad_mask: Passed straight to `attention` as its `mask`; True marks a
                visible key. Build it with `transformer.mask.padding_keep_mask`.
                None attends everywhere.

        Returns:
            (out, weights): the mixed output, and each head's post-softmax
            attention weights as returned by `attention`.
        """
        batch_size, n, d_model = batch.shape
        q = self.q_norm(batch.unsqueeze(dim=1) @ self.w_q)  # noqa: NAR001
        k = self.k_norm(batch.unsqueeze(dim=1) @ self.w_k)  # noqa: NAR001
        v = batch.unsqueeze(dim=1) @ self.w_v
        out, weights = attention(q=q, k=k, v=v, mask=pad_mask)
        concat = out.transpose(dim0=1, dim1=2).reshape(  # noqa: NAR001
            batch_size, n, d_model
        )
        return concat @ self.w_o, weights


class EncoderBlock(nn.Module):
    """One pre-norm encoder block: self-attention, then a position-wise FFN.

    Each sublayer is wrapped in a residual connection, with the norm on the
    sublayer branch and dropout on the sublayer output. In and out are both
    (B, L, d_model), so blocks stack by composition.

    ```
    x ──┬────────────────────────────► + ──┬──────────────────────────► + ──►
        │                              ▲   │                           ▲
        └─ attn_in_norm ─ attn ─ drop ─┘   └─ ffn_norm ─ ffn ─ drop ────┘
    ```

    The FFN is the same two-layer MLP applied to every position independently:

    ```
    (B, L, d_model) ─► Linear ─► ReLU ─► Linear ─► (B, L, d_model)
                        d_ff = d_ff_multiplier * d_model
    ```

    Pre-norm, not post-norm: `x` reaches every add untouched, so there is a
    path from input to output that passes through no norm at all and gradients
    reach early blocks undamped. The trade is that block outputs grow with
    depth, which a final norm outside the stack is expected to absorb.

    Attention is the only place positions talk to each other; the norms, the
    FFN and the dropout all act per position.
    """

    def __init__(self,
                 d_model: int, 
                 h: int,
                 qk_norm: bool = False,
                 d_ff_multiplier: int = 4,
                 dropout: float = 0.1,
                 norm: type[nn.LayerNorm] | type[nn.RMSNorm] = nn.LayerNorm ,
                 ) -> None:
        """Build the block's two sublayers and their norms.

        Args:
            qk_norm: Forwarded to `MultiHeadAttentionLayer`; RMSNorms queries
                and keys per head before the dot product.
            d_ff_multiplier: FFN hidden width as a multiple of `d_model`.
            dropout: One `nn.Dropout` shared by both sublayer outputs. Shared
                rather than duplicated because a Dropout holds no state — it
                samples a fresh mask per call.
            norm: Norm class to instantiate for both sublayers. `nn.RMSNorm`
                drops the mean-centering and the bias, which is cheaper and
                usually just as stable here.

        Raises:
            ValueError: If `d_model` is not divisible by `h`.
        """
        super().__init__()

        self.attn_in_norm = norm(normalized_shape = d_model)

        self.attn = MultiHeadAttentionLayer(d_model=d_model, h = h, qk_norm = qk_norm)


        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff_multiplier* d_model),
            nn.ReLU(),
            nn.Linear(in_features=d_ff_multiplier* d_model, out_features=d_model),
        )
        self.ffn_norm = norm(normalized_shape = d_model)
        self.dropout = nn.Dropout(p = dropout)


    def forward(
            self, 
            batch: Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
            pad_mask: Optional[Bool[torch.Tensor, f"#{B} #{H} #{L} {L}"]]
        ) -> tuple[
            Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
            Float[torch.Tensor, f"{B} {H} {L} {L}"],
        ]:
        """Run `batch` through the attention sublayer, then the FFN sublayer.

        Args:
            batch: Token embeddings.
            pad_mask: Passed through to the attention sublayer; True marks a
                visible key. Build it with `transformer.mask.padding_keep_mask`.
                None attends everywhere.

        Returns:
            (out, weights): the block output, same shape as `batch`, and the
            attention sublayer's weights.
        """
        attn_in_normalized = self.attn_in_norm(batch)
        attn, weights = self.attn(
            batch = attn_in_normalized,
            pad_mask = pad_mask
        )
        batch = batch + self.dropout(attn) # attn layer is completed
        ffn_in_normalized = self.ffn_norm(batch)
        ffn_output = self.ffn(ffn_in_normalized)
        return batch + self.dropout(ffn_output), weights

    

