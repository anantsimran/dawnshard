"""Tensor parallelism by hand: split each matmul across the ranks of one TP group.

Two ways to split `y = x @ W.T` with W of shape (out, in):

```
column parallel: W split by output features      row parallel: W split by input features

      x (…, in) ── replicated on every rank                  x_r (…, in/tp) ── each rank's slice
          │                                                       │
          ▼                                                       ▼
   copy_to_tp ── identity fwd, all-reduce grad bwd         @ W_r.T   (out, in/tp)
          │                                                       │  partial sum (…, out)
          ▼                                                       ▼
   @ W_r.T   (out/tp, in)                                  reduce_from_tp: all-reduce fwd,
          │                                                       │    identity bwd
          │                                                       │
          ▼  y_r (…, out/tp): complete outputs, no comm           ▼  y (…, out), replicated
```

A column-parallel layer followed by a row-parallel one needs exactly one all-reduce
in forward (after the row layer) and one in backward (before the column layer): the
column layer's outputs are complete for its slice of features, so anything elementwise
in between (SwiGLU, attention per head) runs with no communication. This is
Megatron-LM's "f" (`copy_to_tp`) and "g" (`reduce_from_tp`) pair.

Vocabulary parallelism splits the embedding and the LM head by vocabulary rows. The
cross-entropy then works on vocabulary shards: ranks exchange the per-row max, the
per-row sum of exponentials, and the target logit, three numbers per token, so the
full `(tokens, vocab)` logits never exist on one rank.

Sequence parallelism replaces the all-reduce around a norm region with a
reduce-scatter (into sequence shards) and an all-gather (back to full sequences): the
same bytes on the wire, but the norm's activations are stored once per rank, not
replicated, cutting that memory by the TP degree.

Every layer here takes an already-built dense weight and keeps its rank's shard, so a
model is initialized once in full, then sharded: per-shard random init would break
parity with the dense model and look like a math bug.
"""

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn


class _CopyToTP(torch.autograd.Function):
    """Identity forward; all-reduce of the gradient backward (Megatron's f)."""

    @staticmethod
    def forward(ctx, x, group):  # noqa: NAR002
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):  # noqa: NAR002
        grad = grad.contiguous()
        dist.all_reduce(tensor=grad, group=ctx.group)
        return grad, None


class _ReduceFromTP(torch.autograd.Function):
    """All-reduce forward; identity backward (Megatron's g)."""

    @staticmethod
    def forward(ctx, x, group):  # noqa: NAR002
        x = x.contiguous()
        dist.all_reduce(tensor=x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):  # noqa: NAR002
        return grad, None


class _ReduceScatterSequence(torch.autograd.Function):
    """Reduce-scatter along dim 1 (sequence) forward; all-gather backward."""

    @staticmethod
    def forward(ctx, x, group):  # noqa: NAR002
        ctx.group = group
        world = dist.get_world_size(group=group)
        chunks = [c.contiguous() for c in x.chunk(chunks=world, dim=1)]
        out = torch.empty_like(input=chunks[0])
        dist.reduce_scatter(output=out, input_list=chunks, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):  # noqa: NAR002
        return _all_gather_dim1(x=grad, group=ctx.group), None


class _GatherSequence(torch.autograd.Function):
    """All-gather along dim 1 (sequence) forward; reduce-scatter backward."""

    @staticmethod
    def forward(ctx, x, group):  # noqa: NAR002
        ctx.group = group
        return _all_gather_dim1(x=x, group=group)

    @staticmethod
    def backward(ctx, grad):  # noqa: NAR002
        world = dist.get_world_size(group=ctx.group)
        chunks = [c.contiguous() for c in grad.chunk(chunks=world, dim=1)]
        out = torch.empty_like(input=chunks[0])
        dist.reduce_scatter(output=out, input_list=chunks, group=ctx.group)
        return out, None


def _all_gather_dim1(x: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Concatenate every rank's `x` along dim 1."""
    parts = [torch.empty_like(input=x) for _ in range(dist.get_world_size(group=group))]  # noqa: NAR001
    dist.all_gather(tensor_list=parts, tensor=x.contiguous(), group=group)
    return torch.cat(tensors=parts, dim=1)


def copy_to_tp(x: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Enter a TP region: `x` unchanged, its gradient summed over the group."""
    return _CopyToTP.apply(x, group)  # noqa: NAR001


def reduce_from_tp(x: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Leave a TP region: sum the ranks' partial results."""
    return _ReduceFromTP.apply(x, group)  # noqa: NAR001


def scatter_sequence(x: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Leave a TP region into sequence shards: reduce-scatter `(B, L, d)` into `(B, L/tp, d)`."""
    return _ReduceScatterSequence.apply(x, group)  # noqa: NAR001


def gather_sequence(x: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Enter a TP region from sequence shards: all-gather `(B, L/tp, d)` into `(B, L, d)`."""
    return _GatherSequence.apply(x, group)  # noqa: NAR001


def shard_rows(weight: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """This rank's contiguous block of rows of `weight`.

    Raises:
        ValueError: If the row count does not divide by the group size.
    """
    world, rank = dist.get_world_size(group=group), dist.get_rank(group=group)
    if weight.shape[0] % world:
        raise ValueError(f"{weight.shape[0]} rows do not divide over {world} TP ranks")  # noqa: NAR001
    return weight.chunk(chunks=world, dim=0)[rank].clone()


class CopyToTP(nn.Module):
    """Mark the start of a TP region: identity forward, gradient all-reduced backward.

    In and out: the same tensor. Put it after a replicated computation whose output
    feeds column-parallel work, such as the final norm before a vocabulary-parallel head.

    ```
    x ──► copy_to_tp ──► x        (backward: grad ──► all-reduce ──► grad)
    ```

    `forward` holds no state.
    """

    def __init__(self, group: Optional[dist.ProcessGroup]) -> None:
        """Remember the TP group."""
        super().__init__()
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: NAR002
        return copy_to_tp(x=x, group=self.group)


class ColumnParallelLinear(nn.Module):
    """`y_r = x @ W_r.T`: this rank's slice of the output features.

    In: `(…, in)` replicated on every rank. Out: `(…, out/tp)`, this rank's features.

    ```
    x (…, in) ──► copy_to_tp ──► @ weight.T ──► (…, out/tp)
                  (all-reduce        weight (out/tp, in): rows of the dense weight
                   grad in bwd)
    ```

    With `sequence_parallel`, the input arrives as sequence shards `(B, L/tp, in)` and
    `gather_sequence` replaces `copy_to_tp`; its backward reduce-scatters the gradient.

    `forward` holds no state.
    """

    def __init__(
        self, dense: nn.Linear, group: Optional[dist.ProcessGroup], sequence_parallel: bool = False
    ) -> None:
        """Keep this rank's rows of `dense.weight`; `dense` must be bias-free."""
        super().__init__()
        assert dense.bias is None
        self.group, self.sequence_parallel = group, sequence_parallel
        self.weight = nn.Parameter(data=shard_rows(weight=dense.weight.detach(), group=group))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: NAR002
        x = (
            gather_sequence(x=x, group=self.group)
            if self.sequence_parallel
            else copy_to_tp(x=x, group=self.group)
        )
        return x @ self.weight.T


class RowParallelLinear(nn.Module):
    """`y = sum_r x_r @ W_r.T`: partial sums over input-feature slices, all-reduced.

    In: `(…, in/tp)`, this rank's slice of the input features (what a column-parallel
    layer produced). Out: `(…, out)` replicated, or `(B, L/tp, out)` sequence shards
    with `sequence_parallel`.

    ```
    x_r (…, in/tp) ──► @ weight.T ──► partial (…, out) ──► reduce_from_tp ──► (…, out)
                        weight (out, in/tp): columns       (or scatter_sequence)
    ```

    `forward` holds no state.
    """

    def __init__(
        self, dense: nn.Linear, group: Optional[dist.ProcessGroup], sequence_parallel: bool = False
    ) -> None:
        """Keep this rank's columns of `dense.weight`; `dense` must be bias-free."""
        super().__init__()
        assert dense.bias is None
        self.group, self.sequence_parallel = group, sequence_parallel
        self.weight = nn.Parameter(
            data=shard_rows(weight=dense.weight.detach().T, group=group).T.contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: NAR002
        partial = x @ self.weight.T
        if self.sequence_parallel:
            return scatter_sequence(x=partial, group=self.group)
        return reduce_from_tp(x=partial, group=self.group)


class VocabParallelEmbedding(nn.Module):
    """Embedding whose vocabulary rows are split across the TP group.

    In: `(B, L)` token ids, the same on every rank. Out: `(B, L, d_model)` replicated.

    ```
    ids ──► local = ids - vocab_start ──► mask ids outside [start, end) ──► weight[local]
                                                                              │ zeros where masked
                                                                              ▼
                                                                       reduce_from_tp ──► (B, L, d)
    ```

    Exactly one rank owns each id, so the sum over ranks is the dense lookup. The
    same shard serves as the tied LM head: `hidden @ weight.T` gives this rank's slice
    of the logits, which `vocab_parallel_cross_entropy` consumes without gathering.

    `forward` holds no state.
    """

    def __init__(self, dense: nn.Embedding, group: Optional[dist.ProcessGroup]) -> None:
        """Keep this rank's rows of `dense.weight`."""
        super().__init__()
        self.group = group
        self.weight = nn.Parameter(data=shard_rows(weight=dense.weight.detach(), group=group))
        self.vocab_start = dist.get_rank(group=group) * self.weight.shape[0]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # noqa: NAR002
        local = ids - self.vocab_start
        outside = (local < 0) | (local >= self.weight.shape[0])
        out = self.weight[local.masked_fill(mask=outside, value=0)]
        out = out.masked_fill(mask=outside.unsqueeze(dim=-1), value=0.0)
        return reduce_from_tp(x=out, group=self.group)


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Cross-entropy over vocabulary-sharded logits; see the module docstring."""

    @staticmethod
    def forward(ctx, logits, target, vocab_start, ignore_index, group):  # noqa: NAR002
        ctx.dtype = logits.dtype
        if logits.dtype in (torch.float16, torch.bfloat16):  # never reduce in half precision
            logits = logits.float()
        row_max = logits.max(dim=-1).values
        dist.all_reduce(tensor=row_max, op=dist.ReduceOp.MAX, group=group)
        shifted = logits - row_max.unsqueeze(dim=-1)
        exp = shifted.exp()
        sum_exp = exp.sum(dim=-1)
        dist.all_reduce(tensor=sum_exp, group=group)
        local = target - vocab_start
        mine = (local >= 0) & (local < logits.shape[-1])
        index = local.clamp(min=0, max=logits.shape[-1] - 1)
        target_logit = shifted.gather(dim=-1, index=index.unsqueeze(dim=-1)).squeeze(dim=-1) * mine
        dist.all_reduce(tensor=target_logit, group=group)
        valid = target != ignore_index
        loss = (sum_exp.log() - target_logit) * valid
        count = valid.sum().clamp(min=1)
        ctx.save_for_backward(exp / sum_exp.unsqueeze(dim=-1), index, mine & valid, valid)  # noqa: NAR001
        ctx.count = count
        return loss.sum() / count

    @staticmethod
    def backward(ctx, grad):  # noqa: NAR002
        softmax, index, hit, valid = ctx.saved_tensors
        grad_logits = softmax * valid.unsqueeze(dim=-1)
        grad_logits.scatter_add_(
            dim=-1,
            index=index.unsqueeze(dim=-1),
            src=-hit.unsqueeze(dim=-1).to(dtype=grad_logits.dtype),
        )
        return (grad_logits * grad / ctx.count).to(dtype=ctx.dtype), None, None, None, None


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start: int,
    group: Optional[dist.ProcessGroup],
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean cross-entropy from this rank's logit columns `[vocab_start, vocab_start + V/tp)`.

    Args:
        logits: `(N, V/tp)`, this rank's slice of every row's logits.
        target: `(N,)` global class ids, the same on every rank.

    Returns:
        The same scalar on every rank, equal to `F.cross_entropy` on the full logits.
    """
    return _VocabParallelCrossEntropy.apply(logits, target, vocab_start, ignore_index, group)  # noqa: NAR001


class VocabParallelCrossEntropy(nn.Module):
    """`nn.Module` wrapper so the loop's `criterion(predicted, target)` works under TP.

    In: `(N, V/tp)` local logits and `(N,)` targets. Out: the scalar mean loss.

    ```
    logits_r ──► max ──all-reduce MAX──► sum exp ──all-reduce──► target logit ──all-reduce──► loss
    ```

    `forward` holds no state.
    """

    def __init__(self, embedding: VocabParallelEmbedding) -> None:
        """Read the vocabulary offset and group from the tied embedding shard."""
        super().__init__()
        self.vocab_start, self.group = embedding.vocab_start, embedding.group

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: NAR002
        return vocab_parallel_cross_entropy(
            logits=logits, target=target, vocab_start=self.vocab_start, group=self.group
        )
