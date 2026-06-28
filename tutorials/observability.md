# Observability in the Training Loop

## What's Already There

`train_loop.py` ships with epoch-level telemetry out of the box:

- `epoch_duration_seconds` — `time.perf_counter()` brackets the entire epoch ([train_loop.py:186-196](../app/src/train/train_loop.py#L186-L196))
- `_collect_system_metrics()` — CPU%, RAM used, GPU peak memory; CUDA peak stats reset per epoch ([train_loop.py:101-115](../app/src/train/train_loop.py#L101-L115))
- Three output sinks: loguru (human-readable), wandb (optional), history JSON (machine-readable)

So the gap is everything **below the epoch boundary**: per-step throughput, the forward/backward/optimizer split, and whether the GPU is idling while the DataLoader does CPU work.

Two different tools fill different parts of that gap — and conflating them is the source of most confusion in this space.

______________________________________________________________________

## Profiler vs. Slog — Why They're Not the Same Tool

| | Profiler | Slog (structured telemetry) |
|---|---|---|
| **Answers** | Where did my ms go this step? | How is this run trending? |
| **Granularity** | Op/kernel level | Coarse counters (loss, samples/sec, data-wait) |
| **Overhead** | High | Near-zero |
| **Run duration** | Bounded window (a few steps) | Always on |
| **Output** | Chrome trace / flamegraph | JSON lines |

A profiler is a microscope. A slog is a flight recorder. You want both eventually, but they live in different places and have opposite cost profiles. Building them independently is the right call.

______________________________________________________________________

## The Async Gotcha That Invalidates Naive Timing

CUDA and MPS kernels are **asynchronous**. A `time.perf_counter()` around `loss.backward()` measures *launch time*, not *compute time* — the kernel may not have finished. The number is garbage.

```python
# WRONG — measures launch time only
t0 = time.perf_counter()
loss.backward()
elapsed = time.perf_counter() - t0   # could be microseconds while GPU still runs
```

```python
# CORRECT — sync before measuring
torch.cuda.synchronize()
t0 = time.perf_counter()
loss.backward()
torch.cuda.synchronize()   # blocks until GPU finishes
elapsed = time.perf_counter() - t0
```

The existing `epoch_duration_seconds` is accidentally correct because `loss.item()` at every step forces a sync. But any new fine-grained timing you add needs explicit syncs — and syncs themselves cost throughput. This is the central tension in all the options below.

______________________________________________________________________

## Path 1 — `torch.profiler` Scheduled Window

Wrap the `run_epoch` batch loop in `torch.profiler.profile` with a schedule:

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for batch in loader:
        loss, n = step_fn(state, config, batch)
        prof.step()
```

Gives you a flamegraph of every kernel, the data-load-vs-compute timeline, and a memory timeline. Handles the async sync problem for you.

**Trade-off:** heavyweight — unusable to leave on. A flag-gated diagnostic run, not production telemetry.

**Right when:** you have a concrete "why is this slow / why does it OOM" question.

______________________________________________________________________

## Path 2 — Lightweight Step-Level Slog (recommended)

Add optional timing inside `run_epoch`: measure data-wait (time between batches) vs step time, emit a dict every N steps. Fits the existing functional style — pass a `step_telemetry` callback through `RuntimeConfig` exactly like `accumulate_loss` already is.

```python
@dataclass
class RuntimeConfig:
    device: torch.device
    accumulate_loss: AccumulateFn
    compute_reduced_loss: ReduceFn
    log_every_n_steps: int = 50       # 0 = disabled
```

Inside `run_epoch`:

```python
step_index = 0
data_wait_start = time.perf_counter()

for batch in loader:
    data_wait_seconds = time.perf_counter() - data_wait_start

    step_start = time.perf_counter()
    batch_loss, batch_size = step_fn(state, config, batch)
    # loss.item() in step_fn already synced the GPU
    step_seconds = time.perf_counter() - step_start

    config.accumulate_loss(accumulator, batch_loss, batch_size)

    if config.log_every_n_steps and step_index % config.log_every_n_steps == 0:
        samples_per_second = batch_size / step_seconds
        logger.info(
            "step {:>5} | loss {:.4f} | {:.0f} samples/s | data wait {:.1f}ms",
            step_index, batch_loss, samples_per_second, data_wait_seconds * 1000,
        )

    step_index += 1
    data_wait_start = time.perf_counter()
```

The data-wait pattern (measure time from end of last step to start of next) is how you detect DataLoader stalls without any special instrumentation.

**Trade-off:** one sync per sampled step for honest GPU timing; at every-50-steps the cost rounds to zero. Won't tell you forward-vs-backward split.

**Right when:** you want a continuous flight recorder and trend lines without the profiler's weight.

______________________________________________________________________

## Path 3 — Manual Section Timing in `train_step`

Break `train_step` into forward / backward / optimizer sections with explicit syncs:

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

**Trade-off:** a sync per section *every step* measurably slows training. Pollutes the clean `train_step` signature. Only gives you two numbers with no kernel detail.

**Right when:** you just want a rough forward/backward ratio and don't mind the overhead.

______________________________________________________________________

## Path 4 — A `Telemetry` Callback Protocol

Define a protocol and inject it via `RuntimeConfig`, unifying the three sinks you already have:

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

**Trade-off:** the cleanest extension seam, but it touches the working wandb path. Mild over-engineering if you only need one new signal. CLAUDE.md §2 applies here — don't build this until you have a second concrete signal to add.

**Right when:** you expect to keep adding sinks/signals and want one extension point.

______________________________________________________________________

## Recommendation

Treat them as two separate jobs:

1. **Always-on slog → Path 2.** Slots into the seam already built (`RuntimeConfig` callbacks), stays cheap via sampling, reveals the one thing epoch telemetry hides — throughput and DataLoader stall.
1. **Diagnostic profiler → Path 1, flag-gated.** Keep as an opt-in window for a few steps when something's slow. Don't entangle it with the always-on path.

Hold Path 4 in reserve — reach for it only when you have a *second* concrete signal to add. Path 3 has Path 2's sync cost without its continuity, so skip it.
