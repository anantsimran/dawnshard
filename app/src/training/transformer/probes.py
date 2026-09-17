"""Probes that record attention from a model during `run_epoch`."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class HasAttentionMap(Protocol):
    """A model whose `forward` stores each block's (B, h, L, L) weights in block order."""

    attention_map: list[torch.Tensor]


def save_attention_maps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    predicted: torch.Tensor,
    epoch: int,
    batch_index: int,
    *,
    out_dir: Path,
    num_sentences: int,
) -> None:
    """Save the first batch's attention maps for one epoch to `out_dir`.

    A probe for `EpochSpec`: bind `out_dir` and `num_sentences` with
    `functools.partial`. Acts only on batch 0 and writes
    `epoch_<NNN>.pt` holding `{"token_ids": (n, L), "weights": [(n, h, L, L) per
    block]}` on the cpu, for the first `num_sentences` rows.

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
            "weights": [weights[:num_sentences].cpu() for weights in model.attention_map],
        },
        f=out_dir / f"epoch_{epoch:03d}.pt",
    )
