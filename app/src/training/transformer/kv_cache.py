"""Key/value cache for incremental decoding with `DecoderBlock.forward_step`.

In a causal decoder, position t's keys and values depend only on tokens 0..t, so they
never change once computed. Caching them turns step t from a forward over all t
positions into a forward over one, which is the difference between O(n^2) and O(n)
token-forwards for an n-token output. Cross-attention keys and values depend only on
the source, so they are computed once per source and reused at every step.

Shapes use the package's axis names: B batch (beams flattened into it), h heads, d_k
width per head, S source length.
"""

from typing import Optional

import torch

type KeyValue = tuple[torch.Tensor, torch.Tensor]


def kv_cache_bytes(
    num_layers: int, batch_size: int, h: int, d_k: int, positions: int, dtype: torch.dtype
) -> int:
    """Return the bytes a cache of `positions` keys and values per row occupies.

    2 (K and V) x layers x rows x heads x d_k x positions x bytes per element. Per
    token and per row that is `2 * num_layers * h * d_k * itemsize`: the number that
    decides how many tokens fit in memory at serving time.
    """
    itemsize = torch.empty(size=(), dtype=dtype).element_size()
    return 2 * num_layers * batch_size * h * d_k * positions * itemsize


class KVCache:
    """Self-attention K/V preallocated to `max_len`, plus cross-attention K/V per layer.

    Self-attention rows live in two `(num_layers, B, h, max_len, d_k)` tensors written
    in place, so a step never reallocates. `lengths[layer]` counts the positions
    written for each layer; a step appends to every layer in turn, so between steps
    they agree. Cross-attention K/V are `(B, h, S, d_k)` per layer, stored on the
    first step and read on every later one, unless `cache_cross` is off.

    Beam search forks and prunes rows; `reorder` gathers every tensor's batch rows by
    index so row i keeps following the hypothesis it now belongs to.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        h: int,
        d_k: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_cross: bool = True,
    ) -> None:
        """Allocate the self-attention tensors; cross-attention fills on first use.

        Args:
            batch_size: Rows decoded together: sentences x beams.
            max_len: Most target positions any row will feed, BOS included.
            cache_cross: Keep cross-attention K/V across steps. Off recomputes them
                every step, which isolates what caching them saves.
        """
        shape = (num_layers, batch_size, h, max_len, d_k)
        self.keys = torch.zeros(size=shape, dtype=dtype, device=device)
        self.values = torch.zeros(size=shape, dtype=dtype, device=device)
        self.lengths = [0] * num_layers
        self.cache_cross = cache_cross
        self.cross_kv: list[Optional[KeyValue]] = [None] * num_layers

    @property
    def length(self) -> int:
        """Positions every layer has cached: the offset of the next token to feed."""
        return min(self.lengths)  # noqa: NAR001

    def append(self, layer: int, kv: KeyValue) -> KeyValue:
        """Write `kv`'s new positions for `layer` and return all cached ones.

        Args:
            kv: `(k, v)`, each `(B, h, t, d_k)`, for the `t` newest positions.

        Returns:
            `(k, v)` views of positions `0 .. lengths[layer] + t`.

        Raises:
            ValueError: If the write would pass `max_len`.

        Side effects:
            Advances `lengths[layer]` by `t`.
        """
        new_keys, new_values = kv
        start = self.lengths[layer]
        end = start + new_keys.shape[2]
        if end > self.keys.shape[3]:
            raise ValueError(f"cache holds {self.keys.shape[3]} positions, step needs {end}")  # noqa: NAR001
        self.keys[layer, :, :, start:end] = new_keys
        self.values[layer, :, :, start:end] = new_values
        self.lengths[layer] = end
        return self.keys[layer, :, :, :end], self.values[layer, :, :, :end]

    def cross(self, layer: int) -> Optional[KeyValue]:
        """Return `layer`'s stored cross-attention `(k, v)`, or None before the first step."""
        return self.cross_kv[layer]

    def store_cross(self, layer: int, kv: KeyValue) -> None:
        """Keep `kv` as `layer`'s cross-attention K/V, if `cache_cross` is on.

        Side effects:
            Sets `cross_kv[layer]`; does nothing when `cache_cross` is off.
        """
        if self.cache_cross:
            self.cross_kv[layer] = kv

    def reorder(self, index: torch.Tensor) -> None:
        """Make row i a copy of row `index[i]`, in every cached tensor.

        Args:
            index: `(B,)` int64 row indices, such as the parent beam of each new
                hypothesis. Repeats fork a row; omissions drop one.

        Side effects:
            Replaces `keys`, `values`, and every stored cross K/V with gathered copies.
        """
        self.keys = self.keys.index_select(dim=1, index=index)
        self.values = self.values.index_select(dim=1, index=index)
        self.cross_kv = [
            None
            if kv is None
            else (kv[0].index_select(dim=0, index=index), kv[1].index_select(dim=0, index=index))
            for kv in self.cross_kv
        ]

    def nbytes(self) -> int:
        """Return the bytes the cache's tensors occupy, measured from the tensors."""
        tensors = [self.keys, self.values] + [t for kv in self.cross_kv if kv for t in kv]
        return sum(t.numel() * t.element_size() for t in tensors)  # noqa: NAR001
