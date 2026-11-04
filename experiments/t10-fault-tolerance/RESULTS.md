# T10: results

CPU only: FSDP2 on 2 gloo ranks (4 for the straggler study), a tiny Qwen3, the Multi30k
token shards from T6 through `StatefulLoader`. Command:
`uv run python app/src/training/experiments/fault_tolerance.py all`.

## Bitwise resume and resharding ([resume.csv](resume.csv))

Three runs from the same seed: 24 steps uninterrupted; 12 steps, checkpoint, a fresh job
resumes to 24 on 2 ranks; the same step-12 checkpoint resumed on 1 rank. Per step, the
global loss (mean over ranks) and a hash of the global batch:

| Steps | Resumed on 2 ranks vs uninterrupted | Resumed on 1 rank vs uninterrupted |
|---|---|---|
| 13-24 | loss and batch bitwise identical, every step | batch identical every step; loss equal within 1e-6 |

The 1-rank losses differ in the seventh digit because the gradient sum is reduced in a
different order on one rank; the data and the weights at the resume step are the same.
The gate holds: resume is bitwise on the same world size.

## Save cost ([save.csv](save.csv))

FSDP on 2 ranks, AdamW state populated:

| Model | Method | Blocking | Total | Rank-0 memory growth |
|---|---|---|---|---|
| 0.22M | rank-0 gather + `torch.save` | 0.16 s | 0.16 s | 1.3 MB |
| 0.22M | DCP | 0.15 s | 0.15 s | 1.3 MB |
| 0.22M | async DCP | 0.05 s | 0.23 s | 0.7 MB |
| 13.8M | rank-0 gather + `torch.save` | 1.07 s | 1.07 s | 34.1 MB |
| 13.8M | DCP | 0.35 s | 0.35 s | 2.0 MB |
| 13.8M | async DCP | 0.37 s | 1.84 s | 45.0 MB |

- **DCP is 3x faster than gathering to rank 0 at 13.8M parameters and has no rank-0
  spike** (2 MB vs 34 MB). The gap grows with the model: the gather is serial and the
  full state must fit on one rank.
- **Async only helps when writing is slower than copying.** On CPU with a local disk
  the copy to the staging buffer is the whole cost, so async blocks as long as DCP and
  holds a full staging copy (45 MB). On GPUs writing to network storage, the copy is a
  device-to-host transfer and the write is the slow part; that is where the "seconds to
  the snapshot copy" prediction applies.

## Goodput against checkpoint interval ([goodput.csv](goodput.csv))

Simulated 2-hour runs, MTBF 20 minutes (5 seeded failure schedules, 1-10 failures
each, identical across intervals), restart 60 s; mean over schedules:

| Interval | Sync saves (30 s) | Async saves (3 s) |
|---|---|---|
| 1 min | 0.622 | 0.879 |
| 1.4 min (async Young/Daly) | | **0.886** |
| 2 min | 0.736 | 0.886 |
| 4.5 min (sync Young/Daly) | **0.766** | |
| 5 min | 0.768 | 0.822 |
| 10 min | 0.679 | 0.702 |
| 30 min | 0.573 | 0.576 |

The Young/Daly interval lands at the optimum in both modes and the curve is flat around
it (sync: 0.766 at 4.5 min, 0.768 at 5 min). Async saves move the optimum from 4.5 to
1.4 minutes and raise goodput from 0.77 to 0.89. The first-order prediction
`1 - (delta/tau + (tau/2 + R)/MTBF)` matches within 0.01 near the optimum and breaks
down once the interval approaches the MTBF.

## Chaos under `torchrun --max-restarts` ([chaos.csv](chaos.csv))

The worker trains 60 steps, saving every 10, while `scripts/chaos.py` SIGKILLs a worker
at seeded times and `torchrun` restarts both ranks:

| Failure at | Step reached | Resumed from | Steps lost | Kill to first new step |
|---|---|---|---|---|
| 16.1 s | 51 | 50 | 1 | 8.9 s |

Step goodput 0.984 (one step computed twice in 61). Recovery is dominated by
starting two Python processes and importing torch; reload is a fraction of it.
Earlier runs with a 15-second MTBF surfaced the two failure modes in DESIGN.md: a
nine-minute hang under the default collective timeout, and a job that exhausted its
restart budget when kills landed during process-group initialization.

## Straggler ([straggler.csv](straggler.csv))

4 ranks, rank 2 slowed by 30 ms per step, compute and all-reduce wait timed per step:
`StragglerDetector` flagged `[2]` and nothing else.

## Not run here

The save-time table for 0.6B and 1.7B on GPUs with local NVMe and object storage, the
2-hour GPU runs under chaos, and TorchFT: they need the 4-GPU box.
