# Observability in the Training Loop

**Revision question:** When an epoch is slow, is the delay in loading data, launching work, or running kernels on the accelerator?

An epoch total tells you *that* training is slow. A step timeline gives you the first split: **wait for the next batch → run the step → record the result**. A profiler can then show where time went inside that step. The distinction matters because accelerator work is often queued asynchronously.

**Reading map:** Start with [what is already measured](#whats-already-there), then [the timing trap](#the-async-gotcha-that-invalidates-naive-timing). [Profiler versus structured telemetry](#profiler-vs-slog--why-theyre-not-the-same-tool) gives the tool choice. The four paths below are design sketches for adding detail, with [the recommendation](#recommendation) as a quick summary.

## What's Already There

`train_loop.py` ships with epoch-level telemetry out of the box:

- `epoch_duration_seconds` — `time.perf_counter()` brackets the training pass for an epoch, before validation ([train_loop.py:154-169](../../app/src/training/train/train_loop.py#L154-L169))
- `collect_system_metrics()` — CPU%, RAM used, and GPU memory; CUDA peak stats reset at epoch start and may also include validation before collection ([environment.py:8-18](../../app/src/training/train/environment.py#L8-L18))
- Three output sinks: loguru (human-readable), wandb (optional), history JSON (machine-readable)

So the gap is everything **below the epoch boundary**: per-step throughput, the forward/backward/optimizer split, and whether the accelerator is idling while the DataLoader does CPU work.

### Follow a slow epoch backward

Suppose epoch duration rises but loss is unchanged. First compare **samples per second** across epochs; an epoch total alone may have changed because the number of batches changed. If throughput fell, split a representative step into data wait and step elapsed time. A long wait points toward loading or preprocessing; a long step points toward computation, synchronization, or host overhead. Only a profiler timeline can distinguish these inside a busy step or prove the accelerator was idle.

Two different tools fill different parts of that gap — and conflating them is the source of most confusion in this space.

______________________________________________________________________

## Profiler vs. Slog — Why They're Not the Same Tool

| | Profiler | Slog (structured telemetry) |
|---|---|---|
| **Answers** | Where did my ms go this step? | How is this run trending? |
| **Granularity** | Op/kernel level | Coarse counters (loss, samples/sec, data-wait) |
| **Overhead** | Noticeable during capture | Low when sampled |
| **Run duration** | Bounded window (a few steps) | Always on |
| **Output** | Chrome trace / flamegraph | JSON lines |

A profiler is a microscope. Structured telemetry is a flight recorder. The first diagnoses a short window; the second shows trends across a run. Use the measurements together: a trend tells you *when* to profile, and a trace helps explain *why* that trend appeared.

______________________________________________________________________

## The Async Gotcha That Invalidates Naive Timing

CUDA and MPS work can be **asynchronous**. A `time.perf_counter()` around `loss.backward()` may measure launch time rather than completed device work. Without a synchronization point, that duration cannot be read as kernel compute time.

```python
# MISLEADING on an asynchronous device — may measure launch time only
t0 = time.perf_counter()
loss.backward()
elapsed = time.perf_counter() - t0   # could be microseconds while GPU still runs
```

```python
# CUDA wall-clock section timing — synchronize on both sides
torch.cuda.synchronize()
t0 = time.perf_counter()
loss.backward()
torch.cuda.synchronize()   # blocks until GPU finishes
elapsed = time.perf_counter() - t0
```

For MPS, the corresponding barrier is `torch.mps.synchronize()`; `torch.cuda.synchronize()` applies only to CUDA. PyTorch's profiler uses device timing so you can inspect a trace without placing manual barriers around every operation.

In this repository, `train_step()` calls `loss.item()` after `optimizer.step()`. Reading a device scalar on the CPU synchronizes the queued work needed for that value, so `epoch_duration_seconds` is a meaningful wall-clock measure of the training pass. It already includes the cost of those per-step synchronizations. For a new fine-grained section timer, use appropriate device synchronization, knowing that barriers can reduce throughput.

______________________________________________________________________

## Path 1 — `torch.profiler` Scheduled Window

Concept sketch: profile a bounded window around the `run_epoch` batch loop. Include `ProfilerActivity.CUDA` only when CUDA is in use; other devices need their supported profiler activity and timing path.

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for batch_index, batch in enumerate(loader):
        batch_metrics = step_fn(state, config, batch, epoch, batch_index)
        prof.step()
```

Gives you an operator and kernel timeline, including the data-load-versus-compute split when the loader is marked in the trace. Device events account for asynchronous execution; manual `perf_counter()` sections do not.

**Trade-off:** heavyweight — unusable to leave on. A flag-gated diagnostic run, not production telemetry.

**Right when:** you have a concrete "why is this slow / why does it OOM" question.

______________________________________________________________________

## Path 2 — Lightweight Step-Level Slog (recommended)

Concept sketch: time the wait for a batch separately from the step, then report a compact record every N steps. This fits the existing `run_epoch`/`EpochSpec` split: `EpochSpec` can carry a reporting interval, while `run_epoch` owns the timing. The snippets below describe a possible extension; the current `EpochSpec` does not yet have `log_every_n_steps`.

```python
@dataclass
class EpochSpec:
    device: torch.device
    metrics_type: type[Metrics]
    calculate_metrics: MetricsCalculateFn
    accumulate_metrics: MetricsAccumulateFn
    reduce_metrics: MetricsReduceFn
    log_every_n_steps: int = 50       # 0 = disabled
```

Inside `run_epoch`:

```python
step_index = 0
data_wait_start = time.perf_counter()

for batch in loader:
    data_wait_seconds = time.perf_counter() - data_wait_start

    step_start = time.perf_counter()
    batch_metrics = step_fn(state, config, batch, epoch, step_index)
    # train_step() already calls loss.item(), synchronizing the loss read
    step_seconds = time.perf_counter() - step_start

    config.accumulate_metrics(accumulator, batch_metrics)

    if config.log_every_n_steps and step_index % config.log_every_n_steps == 0:
        samples_per_second = batch_metrics.count / (data_wait_seconds + step_seconds)
        logger.info(
            "step {:>5} | loss {:.4f} | {:.0f} samples/s | data wait {:.1f}ms",
            step_index, batch_metrics.loss, samples_per_second, data_wait_seconds * 1000,
        )

    step_index += 1
    data_wait_start = time.perf_counter()
```

The data-wait pattern measures time from the end of one step to the arrival of the next batch. A high value suggests a loading or preprocessing bottleneck, though host-side work in that gap also counts. The displayed throughput includes both the wait and the step; `count / step_seconds` would answer a different question: throughput *while inside the step*.

**Trade-off:** low logging overhead when sampled, but the current `train_step()` already synchronizes once per batch through `loss.item()`. This measurement will not tell you the forward-versus-backward split.

**Right when:** you want a continuous flight recorder and trend lines without the profiler's weight.

______________________________________________________________________

## Path 3 — Manual Section Timing in `train_step`

Concept sketch: time forward and backward separately with device-appropriate synchronization. The CUDA-only snippet below omits optimizer timing and the project's full step signature for clarity.

```python
def train_step(state, config, batch):
    x, y = (tensor.to(device=config.device) for tensor in batch)
    state.optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t_fwd = time.perf_counter()
    predicted = state.model(x)
    torch.cuda.synchronize()
    fwd_seconds = time.perf_counter() - t_fwd

    loss = state.criterion(predicted, y)

    torch.cuda.synchronize()
    t_bwd = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    bwd_seconds = time.perf_counter() - t_bwd

    state.optimizer.step()
    return loss.item(), y.size(dim=0), {"fwd": fwd_seconds, "bwd": bwd_seconds}
```

**Trade-off:** barriers around every section can measurably slow training. They also make the step interface carry timing data. This sketch gives only two numbers and no kernel detail.

**Right when:** you just want a rough forward/backward ratio and don't mind the overhead.

______________________________________________________________________

## Path 4 — A `Telemetry` Callback Protocol

Define a protocol and inject it via `EpochSpec`, unifying the three sinks you already have:

```python
from typing import Protocol

class Telemetry(Protocol):
    def on_batch_end(self, step: int, loss: float, n: int, elapsed: float) -> None: ...
    def on_epoch_end(self, epoch: int, record: dict) -> None: ...

class NullTelemetry:
    def on_batch_end(self, *args, **kwargs): pass
    def on_epoch_end(self, *args, **kwargs): pass

class JsonlTelemetry:
    def on_batch_end(self, step, loss, n, elapsed):
        print(json.dumps({"step": step, "loss": loss, "samples_per_s": n / elapsed}))
    def on_epoch_end(self, epoch, record):
        print(json.dumps({"epoch": epoch, **record}))
```

`fit` stops knowing about wandb specifically — `WandbTelemetry` becomes just another impl.

**Trade-off:** a clean extension seam, but it touches the working wandb path and adds an abstraction to learn. It becomes useful when several signals or sinks need the same lifecycle events.

**Right when:** you expect to keep adding sinks/signals and want one extension point.

______________________________________________________________________

## Recommendation

Treat them as two separate jobs:

1. **Always-on slog → Path 2.** Slots into the seam already built (`EpochSpec` callbacks), stays cheap via sampling, reveals the one thing epoch telemetry hides — throughput and DataLoader stall.
1. **Diagnostic profiler → Path 1, flag-gated.** Keep as an opt-in window for a few steps when something's slow. Don't entangle it with the always-on path.

Hold Path 4 in reserve until several signals or sinks need the same events. Use Path 3 only for a short diagnostic when a profiler is unavailable or unnecessary.

______________________________________________________________________

## Common Confusions

| Easy to mix up | What to remember |
|---|---|
| Host elapsed time versus device compute time | A host clock can stop while queued accelerator work is still running. Synchronize or inspect device events for section timing. |
| Step throughput versus end-to-end throughput | Excluding data wait can make compute look fast while the training run stays slow. State which interval the denominator covers. |
| Low GPU utilization versus a known cause | Utilization shows a symptom. Data-wait timing and a profiler trace help locate the gap. |
| Logging interval versus synchronization interval | Sampling log output does not remove synchronizations that already happen every step through `loss.item()`. |

## Check Your Recall

1. Why can a host timer around `backward()` report a short duration while the accelerator is still busy?
1. If data wait is 80 ms and step time is 20 ms, what does `count / step_seconds` hide?
1. When should you move from sampled step telemetry to a profiler trace?

<details markdown="1">
<summary>Answers</summary>

1. The host may only have queued asynchronous kernels. A device barrier or profiler event is needed to account for their completion.
1. It excludes the 80 ms spent waiting for data, so it overstates end-to-end throughput by a factor of five in this example.
1. When a trend shows a slow step or memory problem but coarse timing cannot identify which operation, kernel, or gap is responsible.

</details>

**One-minute recap:** Epoch timing tells you the size of a problem; sampled data-wait and step timing tell you where to look; a short profiler trace shows the operations behind it. On asynchronous devices, interpret host timers only after accounting for synchronization, including the `loss.item()` call already present in this loop.
