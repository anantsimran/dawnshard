"""Scaled dot-product attention, implemented from scratch.

Operates on pre-split heads: callers reshape (B, L, d_model) into
(B, h, L, d_k) before calling and merge the heads afterwards.
"""

import math
from typing import Optional

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped

from training.transformer.constants import D_K, B, H, L


@jaxtyped(typechecker=beartype)
def attention(
    q: Float[torch.Tensor, f"{B} {H} {L} {D_K}"],
    k: Float[torch.Tensor, f"{B} {H} {L} {D_K}"],
    v: Float[torch.Tensor, f"{B} {H} {L} d_v"],
    mask: Optional[Bool[torch.Tensor, f"#{B} #{H} #{L} {L}"]],
) -> tuple[Float[torch.Tensor, f"{B} {H} {L} d_v"], Float[torch.Tensor, f"{B} {H} {L} {L}"]]:
    """Attend q over k/v, optionally restricting which keys each query sees.

    Args:
        q: Queries, one row per position that needs a context vector.
        k: Keys, matched against queries to produce relevance scores.
        v: Values, mixed together using the resulting attention weights.
        mask: True marks a (query, key) pair as visible, False forbids it.
            Leading axes may be 1 and will broadcast: (1, 1, L, L) for a
            causal mask shared across the batch, (B, 1, 1, L) for a
            per-sequence padding mask. None means fully bidirectional.

    Returns:
        (out, weights): the attended values, and the post-softmax weights that
        produced them (each query row sums to 1).

    Forbidden scores are set to -inf so that softmax assigns them exactly zero
    weight. A query row that is entirely masked has no finite score to normalise
    against and yields NaN, so never mask every key for a live query.

    Scaling q before the matmul rather than dividing the scores after it touches
    B*h*L*d_k elements instead of B*h*L*L.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])  # noqa: NAR001
    attn_scores = (q * scale) @ k.mT
    if mask is not None:
        attn_scores.masked_fill_(mask=~mask, value=-torch.inf)
    attn_normalized = torch.softmax(input=attn_scores, dim=-1)
    return attn_normalized @ v, attn_normalized
