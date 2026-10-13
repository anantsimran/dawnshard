"""Probes that record attention from a model during `run_epoch`."""

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class HasAttentionMap(Protocol):
    """A model whose `forward` stores each block's (B, h, L, S) weights, by kind.

    Keys name the kind of attention (`"encoder"`, `"decoder_self"`, ...); each value
    holds one tensor per block, in block order.
    """

    attention_map: dict[str, list[torch.Tensor]]


def save_attention_maps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    predicted: torch.Tensor,
    epoch: int,
    batch_index: int,
    seed: Optional[int],
    *,
    out_dir: Path,
    num_sentences: int,
) -> None:
    """Save the first batch's attention maps for one epoch to `out_dir`.

    A probe for `EpochSpec`: bind `out_dir` and `num_sentences` with
    `functools.partial`. `seed` is part of the probe signature and unused here,
    because saving a tensor draws no randomness. Acts only on batch 0 and writes
    `epoch_<NNN>.pt` holding `{"token_ids": (n, ...), "weights": {kind: [(n, h, L, S)
    per block]}}` on the cpu, for the first `num_sentences` rows. `token_ids` is the
    input as the model took it: `(n, L)`, or `(n, 2, L)` for a model that takes
    source and target stacked.

    Raises:
        TypeError: If `model` has no `attention_map`.
    """
    if batch_index != 0:
        return
    if not isinstance(model, HasAttentionMap):  # noqa: NAR001
        raise TypeError(  # noqa: NAR001
            f"save_attention_maps needs a model with attention_map, got {type(model)}"  # noqa: NAR001
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        obj={
            "token_ids": input_tensor[:num_sentences].cpu(),
            "weights": {
                kind: [weights[:num_sentences].cpu() for weights in blocks]
                for kind, blocks in model.attention_map.items()
            },
        },
        f=out_dir / f"epoch_{epoch:03d}.pt",
    )
