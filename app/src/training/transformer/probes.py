"""Probes that record attention from a model during `run_epoch`, and their reader.

A model exposes its last forward pass's weights as `attention_maps`, an
`AttentionProbePerEpoch` that groups each block's weights by kind.
`save_attention_maps` writes one `AttentionCapture` per block, kind, and epoch, and
`load_attention_captures` reads a run's captures back.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch
from jaxtyping import Float, Int
from torch import nn
from training.transformer.constants import B, H, L, S


@dataclass
class AttentionProbePerEpoch:
    """Every block's attention weights from one forward pass, grouped by kind.

    Attributes:
        map: Each kind of attention the model has (`"encoder"`, or a decoder's
            `"decoder_self"` and `"decoder_cross"`) mapped to its blocks' detached
            `(B, h, L, S)` weights. The list index is the block number. Empty until
            the first `forward`.
    """

    map: dict[str, list[Float[torch.Tensor, f"{B} {H} {L} {S}"]]] = field(default_factory=dict)


@dataclass(frozen=True)
class AttentionCapture:
    """One block's weights as `save_attention_maps` saved them, with epoch and inputs.

    Attributes:
        epoch: 1-based epoch of the validation pass it came from.
        token_ids: The input as the model took it, cut to the saved sentences:
            `(n, L)`, or `(n, 2, L)` for a model that takes source and target stacked.
        kind: Which attention this is, a key of `AttentionProbePerEpoch.map`.
        block: 0-based index of the block within its stack.
        weights: `(n, h, L, S)` weights, cut to the same `n` sentences and moved to
            the cpu; `L` is the query axis.
    """

    epoch: int
    token_ids: Int[torch.Tensor, "..."]
    kind: str
    block: int
    weights: Float[torch.Tensor, f"{B} {H} {L} {S}"]


@runtime_checkable
class HasAttentionMaps(Protocol):
    """A model whose `forward` stores each block's attention weights, by kind."""

    attention_maps: AttentionProbePerEpoch


def attention_capture_path(out_dir: Path, epoch: int, kind: str, block: int) -> Path:
    """Where the capture of `kind` attention in `block` at `epoch` lives.

    `<out_dir>/epoch_<NNN>/<kind>_block_<i>.pt`. This is the one place the layout
    is spelled out.
    """
    return out_dir / f"epoch_{epoch:03d}" / f"{kind}_block_{block}.pt"


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
    """Save the first batch's attention maps for one epoch, one file per map.

    A probe for `EpochSpec`: bind `out_dir` and `num_sentences` with
    `functools.partial`. `seed` is part of the probe signature and unused here,
    because saving a tensor draws no randomness. Acts only on batch 0 and writes one
    `AttentionCapture` per block and kind in `model.attention_maps`, at
    `attention_capture_path`, for the first `num_sentences` rows.

    Each file holds the capture as a plain dict of tensors, strings, and ints rather
    than the pickled dataclass, so `torch.load`'s default `weights_only=True` reads it
    without allowlisting a class. Read it back with `load_attention_captures`.

    Raises:
        TypeError: If `model` has no `attention_maps`.
    """
    if batch_index != 0:
        return
    if not isinstance(model, HasAttentionMaps):  # noqa: NAR001
        raise TypeError(  # noqa: NAR001
            f"save_attention_maps needs a model with attention_maps, got {type(model)}"  # noqa: NAR001
        )
    token_ids = input_tensor[:num_sentences].cpu()
    for kind, blocks in model.attention_maps.map.items():
        for block, weights in enumerate(blocks):  # noqa: NAR001
            path = attention_capture_path(out_dir=out_dir, epoch=epoch, kind=kind, block=block)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                obj={
                    "epoch": epoch,
                    "token_ids": token_ids,
                    "kind": kind,
                    "block": block,
                    "weights": weights[:num_sentences].cpu(),
                },
                f=path,
            )


def load_attention_captures(probe_dir: Path) -> list[AttentionCapture]:
    """Return every capture `save_attention_maps` wrote to `probe_dir`.

    Sorted by epoch, then kind, then block number, so `block_10` follows `block_9`.
    """
    captures = []
    for path in probe_dir.glob(pattern="epoch_*/*.pt"):
        saved = torch.load(f=path)
        captures.append(  # noqa: NAR001
            AttentionCapture(
                epoch=saved["epoch"],
                token_ids=saved["token_ids"],
                kind=saved["kind"],
                block=saved["block"],
                weights=saved["weights"],
            )
        )
    return sorted(captures, key=lambda c: (c.epoch, c.kind, c.block))  # noqa: NAR001
