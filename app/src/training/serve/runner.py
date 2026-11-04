"""One forward over a ragged batch, with the KV cache in fixed-size blocks.

The KV pool is two tensors per layer, `(num_blocks * block_size, kv_heads, head_dim)`:
slot `block * block_size + offset` holds one token's key or value. A batch is flat:
every request's new tokens are concatenated, with their positions and the slot each
one's K/V goes to, so prefill chunks and single decode tokens ride in one forward.

```
requests:   A (prefill chunk, 5 tokens)   B (decode, 1)   C (decode, 1)
input_ids:  [a0 a1 a2 a3 a4 | b9 | c3]      positions: [0 1 2 3 4 | 9 | 3]
per layer:  q, k, v = projections(input_ids rows)
            pool.write(k, v, slot_mapping)                 new K/V into their slots
            for each request: gather its blocks → K, V of its whole context
                              attention(q rows, K, V, causal from its offset)
logits:     only at each request's last new token          (R, vocab)
```

The model supplies `forward_paged(input_ids, positions, attend)`; `attend(layer, q, k,
v)` is this module's closure that writes and gathers, so the model knows nothing about
blocks and the engine knows nothing about the model's layers. Attention gathers each
request's blocks into a contiguous tensor and calls SDPA: correct on any device, and
the place a paged-attention kernel (FlashInfer) would drop in.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

type Attend = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class PagedModel(Protocol):
    """What the runner needs from a model."""

    def forward_paged(
        self, input_ids: torch.Tensor, positions: torch.Tensor, attend: Attend
    ) -> torch.Tensor:
        """Return `(T, d_model)` hidden states for the flat batch."""
        ...

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return `(R, vocab)` logits for `(R, d_model)` hidden states."""
        ...


@dataclass(frozen=True)
class PagedBatch:
    """A flat batch: new tokens of every scheduled request, in schedule order."""

    input_ids: torch.Tensor  # (T,)
    positions: torch.Tensor  # (T,)
    slot_mapping: torch.Tensor  # (T,)
    query_lens: list[int]  # new tokens per request
    context_lens: list[int]  # tokens in the cache per request after this step
    block_tables: list[list[int]]


def build_batch(items: list[tuple[list[int], int, int, list[int]]], block_size: int) -> PagedBatch:
    """Build a `PagedBatch` from `(tokens, computed, num_new, block_table)` per request."""
    ids, positions, slots, query_lens, context_lens, tables = [], [], [], [], [], []
    for tokens, computed, num_new, table in items:
        span = range(computed, computed + num_new)  # noqa: NAR001
        ids += tokens[computed : computed + num_new]
        positions += list(span)  # noqa: NAR001
        slots += [table[p // block_size] * block_size + p % block_size for p in span]
        query_lens.append(num_new)  # noqa: NAR001
        context_lens.append(computed + num_new)  # noqa: NAR001
        tables.append(table)  # noqa: NAR001
    as_long = lambda values: torch.tensor(data=values, dtype=torch.int64)  # noqa: E731
    return PagedBatch(
        input_ids=as_long(ids),  # noqa: NAR001
        positions=as_long(positions),  # noqa: NAR001
        slot_mapping=as_long(slots),  # noqa: NAR001
        query_lens=query_lens,
        context_lens=context_lens,
        block_tables=tables,
    )  # noqa: NAR001


class KVPool:
    """Every layer's K and V slots, written by slot index and read by block table."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """Allocate the whole pool up front: its size is the engine's KV budget."""
        shape = (num_layers, num_blocks * block_size, kv_heads, head_dim)
        self.k, self.v = torch.zeros(size=shape, dtype=dtype), torch.zeros(size=shape, dtype=dtype)
        self.block_size = block_size

    def nbytes(self) -> int:
        """Bytes the pool holds: 2 x layers x slots x kv_heads x head_dim x itemsize."""
        return 2 * self.k.numel() * self.k.element_size()

    def gather(
        self, layer: int, table: list[int], length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V `(length, kv_heads, head_dim)` of one request, from its block table."""
        slots = torch.tensor(
            data=[
                table[p // self.block_size] * self.block_size + p % self.block_size
                for p in range(length)  # noqa: NAR001
            ]
        )  # noqa: NAR001
        return self.k[layer, slots], self.v[layer, slots]


def make_attend(pool: KVPool, batch: PagedBatch) -> Attend:
    """Return the closure the model calls per layer: write new K/V, attend over each context."""

    def attend(layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """`q`, `k`, `v`: `(T, heads, head_dim)` for the batch's new tokens; returns q's output."""
        pool.k[layer, batch.slot_mapping], pool.v[layer, batch.slot_mapping] = k, v
        outputs, start = [], 0
        for query_len, context, table in zip(  # noqa: NAR001
            batch.query_lens, batch.context_lens, batch.block_tables
        ):  # noqa: NAR001
            keys, values = pool.gather(layer=layer, table=table, length=context)
            query = q[start : start + query_len].transpose(dim0=0, dim1=1)  # (h, ql, dk)
            # Query i sits at position context - query_len + i and sees keys up to it.
            mask = torch.ones(query_len, context, dtype=torch.bool).tril(  # noqa: NAR001
                diagonal=context - query_len
            )  # noqa: NAR001
            out = F.scaled_dot_product_attention(
                query=query,
                key=keys.transpose(dim0=0, dim1=1),
                value=values.transpose(dim0=0, dim1=1),
                attn_mask=mask,
                enable_gqa=True,
            )
            outputs.append(out.transpose(dim0=0, dim1=1))  # noqa: NAR001
            start += query_len
        return torch.cat(tensors=outputs)

    return attend


def execute(model: PagedModel, pool: KVPool, batch: PagedBatch) -> torch.Tensor:
    """Run the batch; return `(R, vocab)` logits at each request's last new token."""
    hidden = model.forward_paged(
        input_ids=batch.input_ids,
        positions=batch.positions,
        attend=make_attend(pool=pool, batch=batch),
    )
    last = torch.tensor(data=batch.query_lens).cumsum(dim=0) - 1
    return model.logits(hidden=hidden[last])
