# T2: results

Measured on the development container (4 vCPU Linux, no Docker daemon, no GPU, egress
blocked by a proxy). Only the subprocess runtime could run here; runc, gVisor and the
agent rows need the Linux VM and the vLLM GPU in the ticket's setup.

Task list: 200 MBPP problems sampled with seed 0, ids in [task_ids.txt](task_ids.txt).

## Harness throughput and correctness (oracle mode)

Oracle mode submits each task's reference solution, so it measures the harness and
verifier without a model.

| Run | Runtime | Concurrency | Injected infra faults | Scored | Dropped | Retried | Pass rate | Tasks/hour |
|---|---|---|---|---|---|---|---|---|
| `826ae82d` | subprocess | 4 | 0% | 200 | 0 | 0 | 1.000 | 191,746 |
| `8819c8ce` | subprocess | 4 | 10% | 199 | 1 | 17 | 1.000 | 176,406 |

- **The verifier passes every reference solution.** The first run passed 199/200:
  MBPP 40 returns a `collections.Counter`, which the first trusted-type rule rejected.
  The rule is now an exact-type allowlist; see [DESIGN.md](DESIGN.md).
- **Faults do not leak into rewards.** With 10% of sandbox runs failing like
  infrastructure, 17 episodes were retried and one failed twice and was dropped. Every
  scored episode still has reward 1, the same distribution as the fault-free run.
- **Replay:** all 200 saved episodes of `826ae82d` replay with identical rewards
  ([replay.csv](replay.csv)).

Sandbox start on the subprocess runtime is a `mkdtemp`: p50 0.04 ms, p99 0.17 ms
([startup.csv](startup.csv)). Tasks/hour here is bounded by Python start-up per
verification, not by the sandbox.

## Probes (subprocess runtime)

| Probe | Expected | Observed | Contained |
|---|---|---|---|
| network egress | refused or unreachable | timed out | yes, but only because this container's egress is proxied |
| host file read | canary not visible | read the host canary | **no** |
| `/proc` snooping | own processes only | saw 89 host processes | **no** |
| memory bomb | MemoryError or OOM kill | MemoryError (RLIMIT_AS) | yes |
| fork bomb | pids limit stops it | skipped: root has no pid limit on the host | not run |

This is the ticket's v0 point: the subprocess runtime limits resources but isolates
nothing.

## Not run here

| Ticket item | Needs |
|---|---|
| runc and gVisor rows: startup p50/p99, pytest slowdown, tasks/hour | Docker daemon with `runsc` registered |
| Agent pass rate, false-reward rate under a real model | GPU with vLLM serving Qwen3-1.7B |
| Firecracker or Kubernetes `RuntimeClass` at 64 concurrent episodes | nested virtualization or the T3 cluster |
| Fleet sizing for 10,000 concurrent episodes | the runc and gVisor tasks/hour per vCPU above |

Commands, from the repo root:

```bash
uv run python app/src/training/experiments/sandbox_benchmark.py oracle --runtime runsc
```

```bash
uv run python app/src/training/experiments/sandbox_benchmark.py agent --runtime runsc --base-url http://<vllm-host>:8000
```

```bash
uv run python app/src/training/experiments/sandbox_benchmark.py probes --runtime runsc
```
