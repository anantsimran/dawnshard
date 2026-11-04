"""Boolean attention masks for `transformer.functions.attention`.

Every mask uses attention's convention: True = the (query, key) pair is visible. They
index only the key axis, or the (query, key) pair, and leave the other axes as size 1
so they broadcast over the scores (B, h, L, L). Combine them with `&`.

`document_ids`, `document_keep_mask`, and `document_positions` serve packed rows,
where several documents separated by EOS share one sequence: they keep attention
inside each document and restart positions at each document's first token.
"""

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from training.transformer.constants import D_MODEL, B, L


@jaxtyped(typechecker=beartype)
def padding_keep_mask(
    token_ids: Int[torch.Tensor, f"{B} {L}"], pad_id: int
) -> Bool[torch.Tensor, f"{B} 1 1 {L}"]:
    """Mark real (non-pad) key positions.

    Only key columns are masked. A pad position's own query row stays visible,
    because masking a whole row yields NaN after softmax.

    Args:
        token_ids: Token ids, padded with `pad_id`.
        pad_id: The id reserved for padding.

    Returns:
        True where the key is a real token. The size-1 axes broadcast over heads
        and query positions.
    """
    return (token_ids != pad_id).unsqueeze(dim=1).unsqueeze(dim=1)


@jaxtyped(typechecker=beartype)
def causal_keep_mask(
    seq_len: int, device: torch.device | None = None
) -> Bool[torch.Tensor, f"1 1 {L} {L}"]:
    """Let each query see only keys at its own position or earlier.

    Args:
        seq_len: Sequence length L; queries and keys index the same sequence.
        device: Where to allocate the mask; must match the scores it is used with.

    Returns:
        Lower-triangular mask, True at (i, j) iff j <= i. The size-1 axes broadcast
        over batch and heads.
    """
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)  # noqa: NAR001
    return torch.tril(input=ones).unsqueeze(dim=0).unsqueeze(dim=0)


@jaxtyped(typechecker=beartype)
def pad_masked_mean(
    attn: Float[torch.Tensor, f"{B} {L} {D_MODEL}"], mask: Bool[torch.Tensor, f"{B} {L}"]
) -> Float[torch.Tensor, f"{B} {D_MODEL}"]:
    """Average `attn` over the sequence axis, counting only real positions.

    Args:
        attn: Per-position outputs to pool.
        mask: True where the position is a real token, False where it is padding.

    Returns:
        The mean over positions where `mask` is True. A row with no real tokens
        comes back as zeros rather than NaN.
    """
    keep = mask.unsqueeze(dim=-1)
    return attn.masked_fill(mask=~keep, value=0.0).sum(dim=1) / keep.sum(dim=1).clamp(min=1)


@jaxtyped(typechecker=beartype)
def document_ids(
    token_ids: Int[torch.Tensor, f"{B} {L}"], eos_id: int
) -> Int[torch.Tensor, f"{B} {L}"]:
    """Number the documents packed into each row: 0, then +1 after every `eos_id`.

    The EOS belongs to the document it ends, so a row `[a, b, EOS, c]` is `[0, 0, 0, 1]`.
    """
    is_eos = (token_ids == eos_id).long()
    return is_eos.cumsum(dim=1) - is_eos


@jaxtyped(typechecker=beartype)
def document_keep_mask(
    doc_ids: Int[torch.Tensor, f"{B} {L}"],
) -> Bool[torch.Tensor, f"{B} 1 {L} {L}"]:
    """Causal and block-diagonal: each query sees earlier keys of its own document only.

    Packing documents back to back saves the padding, but plain causal attention lets
    a document read the end of the one before it. This mask removes those pairs. The
    diagonal is always kept, so no query row is fully masked.
    """
    same = doc_ids.unsqueeze(dim=2) == doc_ids.unsqueeze(dim=1)
    return (same & causal_keep_mask(seq_len=doc_ids.shape[1], device=doc_ids.device)[0]).unsqueeze(
        dim=1
    )


@jaxtyped(typechecker=beartype)
def document_positions(doc_ids: Int[torch.Tensor, f"{B} {L}"]) -> Int[torch.Tensor, f"{B} {L}"]:
    """Position of each token within its own document, restarting at 0 for each."""
    index = torch.arange(end=doc_ids.shape[1], device=doc_ids.device).expand_as(other=doc_ids)
    starts = torch.ones_like(input=doc_ids, dtype=torch.bool)
    starts[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    first = torch.where(condition=starts, input=index, other=torch.zeros_like(input=index))
    return index - first.cummax(dim=1).values
