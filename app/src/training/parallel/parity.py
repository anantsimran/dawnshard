"""Step-restart parity: compare a parallel implementation's updates with a reference's.

A loss-parity test runs both implementations freely and compares losses. That mixes
two things: real bugs, and floating-point noise from a different summation order,
which training amplifies until the runs drift apart. A per-step tensor check avoids
the drift, but it sees one step at a time, so a bug whose error each step is smaller
than the noise passes every step and still adds up.

These functions support a third check. Every step restarts the candidate from the
reference's state, so per-step errors cannot compound; each step records the error
in the update, `(after_cand - before_cand) - (after_ref - before_ref)`; and
`drift_test` asks whether those errors point the same way across steps. Reordering
noise is zero-mean and nearly orthogonal from step to step, so its sum grows like
the square root of the step count. A biased bug adds a fixed direction every step,
so its sum grows linearly. The test needs no tolerance, because it compares the
errors with each other rather than with a threshold.

```
step k:  reference state ──► candidate.load ──► candidate step ──► Δ_cand
               │                                                      │
               └────────────► reference step ──► Δ_ref ───────────────┤
                                                                      ▼
                                            error_k = Δ_cand - Δ_ref (this rank's shard)
after K steps: gram[i, j] = Σ_ranks <error_i, error_j> ──► drift_test ──► ratio, p_value
```
"""

from collections.abc import Sequence

import torch
import torch.distributed as dist


def flatten_shards(
    tensors: Sequence[torch.Tensor], replicated: list[bool], world_size: int
) -> torch.Tensor:
    """Concatenate this rank's `tensors` into one fp32 vector for global dot products.

    Summing the local dot products of these vectors over ranks gives the dot product
    of the full, unsharded tensors: a sharded tensor contributes its own slice, and a
    replicated tensor, held whole by every rank, is scaled by `1 / sqrt(world_size)`
    so the sum counts it once.

    Args:
        replicated: Per tensor, whether every rank holds the whole tensor.
        world_size: Number of ranks the sharded tensors are split over.
    """
    scale = world_size**-0.5
    return torch.cat(  # noqa: NAR001
        [
            t.detach().float().flatten() * (scale if whole else 1.0)
            for t, whole in zip(tensors, replicated, strict=True)  # noqa: NAR001
        ]
    )


def tensor_dots(a: torch.Tensor, b: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """Return `<a_t, b_t>` per tensor `t`, summed over ranks, in float64.

    `a` and `b` come from `flatten_shards`, and `sizes` is this rank's element count
    per tensor, so the split recovers each tensor's slice. One `all_reduce` covers
    all tensors; the sum of the result is the global dot product.
    """
    pairs = zip(a.split(split_size=sizes), b.split(split_size=sizes), strict=True)  # noqa: NAR001
    local = torch.stack(tensors=[torch.dot(x, y) for x, y in pairs]).double()  # noqa: NAR001
    if dist.is_initialized():
        dist.all_reduce(tensor=local)
    return local


def drift_test(errors: list[torch.Tensor], num_flips: int, seed: int) -> tuple[float, float]:
    """Return the drift ratio of per-step `errors` and its sign-flip p-value.

    The drift ratio is `||Σ_k e_k||² / Σ_k ||e_k||²`. Independent zero-mean errors
    give about 1; a bias that keeps the same direction gives about `1 + (K - 1) ρ`,
    where ρ is the share of each step's error energy that the bias carries.

    The p-value is a randomization test. If each `e_k` is independent and its
    distribution symmetric about zero, flipping the sign of any of them leaves the
    joint distribution unchanged, so the observed ratio is exchangeable with the
    ratios of `num_flips` random sign patterns. Rejecting when the p-value is at most
    α then has false-alarm probability at most α, for any K and any noise scale.

    Args:
        errors: This rank's per-step error vectors, from `flatten_shards`.
        num_flips: Random sign patterns drawn for the null distribution.
        seed: Seed of those patterns, the same on every rank.

    Returns:
        `(ratio, p_value)`, the same on every rank.
    """
    stacked = torch.stack(tensors=errors).double()
    gram = stacked @ stacked.T
    if dist.is_initialized():
        dist.all_reduce(tensor=gram)
    trace = gram.trace()
    ratio = gram.sum() / trace
    generator = torch.Generator().manual_seed(seed)  # noqa: NAR001
    shape = (num_flips, len(errors))  # noqa: NAR001
    signs = torch.randint(low=0, high=2, size=shape, generator=generator)
    signs = (2 * signs - 1).double()
    null = ((signs @ gram) * signs).sum(dim=1) / trace
    # Ties count against rejecting; the tolerance keeps rounding from breaking them.
    p_value = (1 + (null >= ratio * (1 - 1e-9)).sum().item()) / (num_flips + 1)
    return ratio.item(), p_value
