"""Scaled dot-product attention, implemented from scratch.

Operates on pre-split heads: callers reshape (batch, seq, d_model) into
(batch, h, seq, d_k) before calling and merge the heads afterwards.
"""

import math

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def attention(
    q: Float[torch.Tensor, "batch h seq d_k"],
    k: Float[torch.Tensor, "batch h seq d_k"],
    v: Float[torch.Tensor, "batch h seq d_v"],
    mask: Bool[torch.Tensor, "#batch #h #seq seq"] | None = None,
) -> Float[torch.Tensor, "batch h seq d_v"]:
    """Attend q over k/v, optionally restricting which keys each query sees.

    Args:
        q: Queries, one row per position that needs a context vector.
        k: Keys, matched against queries to produce relevance scores.
        v: Values, mixed together using the resulting attention weights.
        mask: True marks a (query, key) pair as visible, False forbids it.
            Leading axes may be 1 and will broadcast: (1, 1, seq, seq) for a
            causal mask shared across the batch, (batch, 1, 1, seq) for a
            per-sequence padding mask. None means fully bidirectional.

    Forbidden scores are set to -inf so that softmax assigns them exactly zero
    weight. A query row that is entirely masked has no finite score to normalise
    against and yields NaN, so never mask every key for a live query.

    Scaling q before the matmul rather than dividing the scores after it touches
    batch*h*seq*d_k elements instead of batch*h*seq*seq.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])  # noqa: NAR001
    attn_scores = (q * scale) @ k.mT
    if mask is not None:
        attn_scores.masked_fill_(mask=~mask, value=-torch.inf)
    return torch.softmax(input=attn_scores, dim=-1) @ v
