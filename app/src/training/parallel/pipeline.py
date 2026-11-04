"""Pipeline parallelism by hand: stages, schedules as data, a simulator, an executor.

A model is cut into `p` contiguous stages, one per rank, and each batch into `m`
micro-batches. A schedule is plain data: for every stage, the ordered list of actions
`("F", i)` (forward micro-batch i) and `("B", i)` (backward micro-batch i). Keeping it
as data means it can be validated, simulated, and printed before any process runs.

```
GPipe, p = 3, m = 4 (time →; F = forward, B = backward, . = idle)
stage 0: F0 F1 F2 F3 .  .  .  .  .  .  B3 B2 B1 B0
stage 1: .  F0 F1 F2 F3 .  .  .  .  B3 B2 B1 B0 .
stage 2: .  .  F0 F1 F2 F3 B3 B2 B1 B0 .  .  .  .

1F1B, p = 3, m = 4: after p - s warm-up forwards, stage s alternates one forward with
one backward, so at most p - s micro-batches are in flight on it.
```

- **The bubble is (p - 1) / (m + p - 1)** for both: the first micro-batch must cross
  every stage before the last one has work. 1F1B does not shrink it; it caps the
  activations each stage holds at p micro-batches instead of m.
- **Stages are balanced by cost, not by layer count.** `partition` splits a list of
  per-layer costs into contiguous stages minimizing the most expensive one. A large
  tied LM head (about six decoder blocks of FLOPs in Qwen3-1.7B) gets a stage with
  correspondingly fewer blocks.

`simulate` replays a schedule with per-action times and returns each stage's timeline
and the measured bubble. `run_stage` executes one stage's actions on a real model
slice, with point-to-point sends of activations forward and gradients backward.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

type Action = tuple[str, int]


def gpipe(num_stages: int, num_microbatches: int) -> list[list[Action]]:
    """All forwards, then all backwards in reverse micro-batch order, on every stage."""
    forward = [("F", i) for i in range(num_microbatches)]  # noqa: NAR001
    backward = [("B", i) for i in reversed(range(num_microbatches))]  # noqa: NAR001
    return [forward + backward for _ in range(num_stages)]  # noqa: NAR001


def one_f_one_b(num_stages: int, num_microbatches: int) -> list[list[Action]]:
    """Warm up with p - s forwards on stage s, alternate F and B, then drain backwards."""
    schedule = []
    for stage in range(num_stages):  # noqa: NAR001
        warmup = min(num_stages - stage, num_microbatches)  # noqa: NAR001
        actions: list[Action] = [("F", i) for i in range(warmup)]  # noqa: NAR001
        next_forward, next_backward = warmup, 0
        while next_backward < num_microbatches:
            actions.append(("B", next_backward))  # noqa: NAR001
            next_backward += 1
            if next_forward < num_microbatches:
                actions.append(("F", next_forward))  # noqa: NAR001
                next_forward += 1
        schedule.append(actions)  # noqa: NAR001
    return schedule


def validate(schedule: list[list[Action]], num_microbatches: int, max_in_flight: int) -> None:
    """Raise if any stage runs B(i) before F(i), skips an action, or holds too many activations.

    Raises:
        ValueError: Describing the first violation.
    """
    for stage, actions in enumerate(schedule):  # noqa: NAR001
        expected = {(kind, i) for kind in "FB" for i in range(num_microbatches)}  # noqa: NAR001
        if sorted(actions) != sorted(expected) or len(actions) != len(expected):  # noqa: NAR001
            raise ValueError(f"stage {stage} does not run every F and B exactly once")  # noqa: NAR001
        done_forward: set[int] = set()
        in_flight = 0
        for kind, i in actions:
            if kind == "F":
                done_forward.add(i)  # noqa: NAR001
                in_flight += 1
            elif i not in done_forward:
                raise ValueError(f"stage {stage} runs B{i} before F{i}")  # noqa: NAR001
            else:
                in_flight -= 1
            if in_flight > max_in_flight:
                raise ValueError(f"stage {stage} holds {in_flight} > {max_in_flight} micro-batches")  # noqa: NAR001


@dataclass(frozen=True)
class Timeline:
    """Per-stage `(kind, microbatch, start, end)` events and the resulting bubble."""

    events: list[list[tuple[str, int, float, float]]]
    makespan: float
    bubble: float


def simulate(
    schedule: list[list[Action]], forward_time: list[float], backward_time: list[float]
) -> Timeline:
    """Replay `schedule` with per-stage action times, honoring cross-stage dependencies.

    F(i) on stage s waits for F(i) on stage s-1; B(i) on stage s waits for B(i) on
    stage s+1 (the last stage's B(i) waits for its own F(i)). The bubble is the idle
    share of `stages x makespan`.
    """
    p = len(schedule)  # noqa: NAR001
    finish: dict[tuple[str, int, int], float] = {}
    cursor, clock = [0] * p, [0.0] * p
    events: list[list[tuple[str, int, float, float]]] = [[] for _ in range(p)]  # noqa: NAR001
    while any(cursor[s] < len(schedule[s]) for s in range(p)):  # noqa: NAR001
        progressed = False
        for s in range(p):  # noqa: NAR001
            if cursor[s] == len(schedule[s]):  # noqa: NAR001
                continue
            kind, i = schedule[s][cursor[s]]
            dep = (
                ("F", s - 1, i)
                if kind == "F" and s > 0
                else ("B", s + 1, i)
                if kind == "B" and s < p - 1
                else None
            )
            if dep is not None and dep not in finish:
                continue
            start = max(clock[s], finish.get(dep, 0.0) if dep else 0.0)  # noqa: NAR001
            end = start + (forward_time[s] if kind == "F" else backward_time[s])
            finish[(kind, s, i)] = clock[s] = end
            events[s].append((kind, i, start, end))  # noqa: NAR001
            cursor[s] += 1
            progressed = True
        if not progressed:
            raise ValueError("schedule deadlocks")  # noqa: NAR001
    makespan = max(clock)  # noqa: NAR001
    busy = sum(end - start for stage in events for _, _, start, end in stage)  # noqa: NAR001
    return Timeline(events=events, makespan=makespan, bubble=1 - busy / (p * makespan))


def theoretical_bubble(num_stages: int, num_microbatches: int) -> float:
    """(p - 1) / (m + p - 1), the idle share when every action takes the same time."""
    return (num_stages - 1) / (num_microbatches + num_stages - 1)


def partition(costs: list[float], num_stages: int) -> list[range]:
    """Split layers with `costs` into `num_stages` contiguous ranges minimizing the largest sum.

    Dynamic programming over (layers used, stages used); exact for contiguous splits.
    """
    n = len(costs)  # noqa: NAR001
    prefix = [0.0]
    for cost in costs:
        prefix.append(prefix[-1] + cost)  # noqa: NAR001
    inf = float("inf")  # noqa: NAR001
    best = [[inf] * (num_stages + 1) for _ in range(n + 1)]  # noqa: NAR001
    cut = [[0] * (num_stages + 1) for _ in range(n + 1)]  # noqa: NAR001
    best[0][0] = 0.0
    for end in range(1, n + 1):  # noqa: NAR001
        for k in range(1, num_stages + 1):  # noqa: NAR001
            for start in range(k - 1, end):  # noqa: NAR001
                value = max(best[start][k - 1], prefix[end] - prefix[start])  # noqa: NAR001
                if value < best[end][k]:
                    best[end][k], cut[end][k] = value, start
    ranges, end = [], n
    for k in range(num_stages, 0, -1):  # noqa: NAR001
        start = cut[end][k]
        ranges.append(range(start, end))  # noqa: NAR001
        end = start
    return ranges[::-1]


def run_stage(
    stage: nn.Module,
    actions: list[Action],
    microbatches: list[torch.Tensor],
    targets: list[torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rank: int,
    num_stages: int,
    activation_shape: tuple[int, ...],
) -> list[float]:
    """Execute one stage's actions; return the losses on the last stage (empty elsewhere).

    Stage 0 reads `microbatches`; others receive activations from `rank - 1`. Sends are
    asynchronous and all waited for before returning. The last
    stage computes `loss_fn(out, target) / m` (the 1/m makes the summed gradients equal
    one batch's) and starts backward; earlier stages receive the gradient from `rank + 1`.
    Gradients accumulate in `stage`'s parameters; the caller steps the optimizer.

    Args:
        activation_shape: Shape of the tensor passed between stages, for receives.
    """
    m, last = len(microbatches), rank == num_stages - 1  # noqa: NAR001
    inputs: dict[int, torch.Tensor] = {}
    outputs: dict[int, torch.Tensor] = {}
    losses: list[float] = []
    pending: list[tuple[dist.Work, torch.Tensor]] = []
    for kind, i in actions:
        if kind == "F":
            x = (
                microbatches[i]
                if rank == 0
                else _recv(shape=activation_shape, src=rank - 1).requires_grad_()
            )
            inputs[i], outputs[i] = x, stage(x)  # noqa: NAR001
            if last:
                loss = loss_fn(outputs[i], targets[i]) / m  # noqa: NAR001
                losses.append(loss.item())  # noqa: NAR001
                outputs[i] = loss
            else:
                pending.append(_isend(tensor=outputs[i].detach().contiguous(), dst=rank + 1))  # noqa: NAR001
        else:
            grad = None if last else _recv(shape=outputs[i].shape, src=rank + 1)
            outputs.pop(i).backward(gradient=grad)  # noqa: NAR001
            x = inputs.pop(i)  # noqa: NAR001
            if rank > 0:
                assert x.grad is not None
                pending.append(_isend(tensor=x.grad.contiguous(), dst=rank - 1))  # noqa: NAR001
    for work, _ in pending:
        work.wait()
    return losses


def _isend(tensor: torch.Tensor, dst: int) -> tuple[dist.Work, torch.Tensor]:
    """Start sending `tensor` to `dst`; return the handle and the tensor, kept alive until waited.

    Sends are asynchronous: under 1F1B two neighbouring stages can each be sending to
    the other at the same moment (an activation forward, a gradient backward), and two
    blocking sends that wait for their receives would deadlock.
    """
    work = dist.isend(tensor=tensor, dst=dst)
    assert work is not None
    return work, tensor


def _recv(shape: tuple[int, ...] | torch.Size, src: int) -> torch.Tensor:
    """Receive a float tensor of `shape` from rank `src`."""
    buffer = torch.empty(size=shape)
    dist.recv(tensor=buffer, src=src)
    return buffer
