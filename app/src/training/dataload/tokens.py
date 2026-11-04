"""
tokens.py
Language-model windows over a flat file of uint32 token ids.

The file is the whole corpus tokenized once, documents concatenated, read through a
memmap, so the loader never sees text and never holds the corpus in memory. Window i
is tokens `[i*L, (i+1)*L]`: its first L ids are the input and its last L the target,
so position l predicts the token after it.

uint32, not uint16: uint16 holds ids up to 65,535, and a larger vocabulary would wrap
silently into wrong tokens.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TOKEN_DTYPE = np.uint32


def write_tokens(path: Path, ids: list[int]) -> None:
    """Write `ids` as a flat uint32 file.

    Raises:
        ValueError: If an id does not fit in uint32.

    Side effects:
        Creates `path`'s parent and overwrites `path`.
    """
    array = np.asarray(ids, dtype=np.int64)  # noqa: NAR001
    if array.size and (array.min() < 0 or array.max() > np.iinfo(TOKEN_DTYPE).max):  # noqa: NAR001
        raise ValueError("token ids must fit in uint32")  # noqa: NAR001
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(TOKEN_DTYPE).tofile(path)  # noqa: NAR001


class TokenWindows(Dataset[torch.Tensor]):
    """Non-overlapping `seq_len + 1` windows of a uint32 token file, as int64 tensors."""

    def __init__(self, path: Path, seq_len: int) -> None:
        """Memmap the file; nothing is read until an item is."""
        self.tokens = np.memmap(filename=path, dtype=TOKEN_DTYPE, mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        """Whole windows in the file; the last partial one is dropped."""
        return (len(self.tokens) - 1) // self.seq_len  # noqa: NAR001

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return window `index`, `(seq_len + 1,)` int64."""
        start = index * self.seq_len
        window = self.tokens[start : start + self.seq_len + 1].astype(np.int64)  # noqa: NAR001
        return torch.from_numpy(window)  # noqa: NAR001


def lm_collate(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack windows into `(X (B, L), Y (B*L,))`: Y is X shifted left by one, flattened.

    Flattened row-major, so row `b*L + l` of Y lines up with a model that flattens its
    `(B, L, vocab)` logits the same way.
    """
    batch = torch.stack(tensors=rows)
    return batch[:, :-1], batch[:, 1:].reshape(-1)  # noqa: NAR001
