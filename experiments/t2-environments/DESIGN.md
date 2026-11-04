# T2: environments, agent harness and sandbox, design notes

This ticket adds the `env` package
([app/src/training/env/](../../app/src/training/env/)): a Gymnasium-shaped
environment for a coding agent, sandbox runtimes behind one interface, a verifier the
agent cannot reach, an infra-versus-task failure classifier with retries, an agent
loop, and a probe suite. The benchmark command is
[sandbox_benchmark.py](../../app/src/training/experiments/sandbox_benchmark.py); the
measured numbers are in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## One `Sandbox` interface, runtimes behind it

`start`, `run(files, command, limits)`, `reset`, `destroy`. `run` never raises for
anything the program did; it returns a `RunResult` with `timed_out`, `oom_killed`
and `start_failed`, so the classifier decides what a failure means.

- **Pros.** The environment, verifier and benchmark are runtime-agnostic: switching
  from subprocess to runc to gVisor is one flag. Tests run the real subprocess
  runtime and check the Docker runtime's exact commands through an injected runner.
- **Cons.** The lowest common denominator: features one runtime has (snapshots,
  checkpoint/restore in gVisor) are not reachable through the interface.

**Alternatives considered.** The Docker SDK for Python instead of the CLI: typed,
but an extra dependency and no gain for a handful of calls. A Kubernetes pod per
episode: the right shape at fleet scale (T3's `RuntimeClass`), far too slow to
start for the single-node benchmark.

## Docker: one long-lived container, `docker exec` per command

The container runs `sleep infinity`; programs run as `nobody` via `docker exec` in a
tmpfs `/work`, with no network, a read-only root, cgroup limits, all capabilities
dropped and `no-new-privileges`. OOM kills come from the cgroup's `memory.events`
counter, read before and after each command.

- **Pros.** A crashing or killed program never kills the sandbox, so the warm pool's
  reset is two `exec`s instead of a container start. The OOM counter distinguishes
  "the program exceeded its limit" (a task failure) from "something killed it" (an
  infra failure), which an exit code of 137 alone cannot.
- **Cons.** Processes of consecutive commands share a container; `reset` has to
  kill them (`kill -9 -1` as `nobody`) and wipe `/work`. The per-command timeout
  relies on coreutils `timeout` inside the image, backed by a host-side timeout.

**Alternative considered.** A fresh container per command (`docker run --rm`):
simplest isolation between commands, but pays container start on every
`run_python`.

## Subprocess runtime is kept, as the unsafe baseline

rlimits on address space, CPU and file size, `start_new_session` and a
process-group SIGTERM then SIGKILL. It contains timeouts, memory bombs and output
floods. It does not contain host file reads, `/proc` snooping or a fork bomb as root
(RLIMIT_NPROC does not apply to root), and the probe table shows exactly that. The
fork-bomb probe is skipped on it rather than run.

## The verifier: hash, fresh sandbox, nonce, trusted comparisons

Tests are stored read-only and re-hashed before every run; each submission runs in a
new sandbox the agent never touched. Inside it, a nonce read from stdin is printed
only after the last assert, and `assert a == b` is rewritten to a check that refuses
values whose exact type is not an allowlisted builtin or standard-library value type.

- **Pros.** Catches edited tests, `sys.exit(0)` / `os._exit(0)` before the asserts,
  and objects whose `__eq__` always returns True, including ones nested in a list or
  with a spoofed `__module__`.
- **Cons.** The allowlist can reject a legitimate answer of an unlisted type (the
  first version used "module is builtins" and rejected MBPP 40's
  `collections.Counter`, which the oracle run caught). A solution that searches the
  harness's memory for the nonce still wins; in-process checks are defense in depth.

**Alternatives considered.** Running tests in a separate process that talks to the
solution over a pipe: defeats the nonce search, but every MBPP test calls the
function directly, so it would mean serializing arguments and results. Pytest as the
runner: familiar output, but its plugin and conftest machinery is more surface for
the solution to tamper with.

## Failures: classify, retry under one id, drop rather than score

`classify` labels a run infra (start failed, or SIGKILL with neither a timeout nor a
sandbox OOM) or task. Episode code raises `InfraError` for infra, including any
model-server HTTP error. `run_with_retries` retries under the same episode id and
records one ledger row per id; an episode that fails twice is dropped, never scored.

- **Pros.** Infra noise cannot become a zero reward; a retry cannot be counted twice.
- **Cons.** A program that is killed by the host for being too large (beyond its own
  limits) is retried rather than blamed; with a sandbox memory limit in place this
  should not happen, and if it does the drop count shows it.

## Episode state is data

`EpisodeState` holds the messages, the raw tool calls as sent (malformed ones too),
the rewards, the seed and the ids, and round-trips through JSON. `replay` feeds the
calls back into a fresh environment. This is what async RL needs: a trainer can
receive episodes as data, and a retry is a replay up to the failed step.

## The agent loop uses the standard library

`chat_completion` posts to any OpenAI-compatible `/v1/chat/completions` endpoint with
`urllib`, temperature 0 and the episode seed. No client dependency; the cost is no
streaming and no automatic retry, which the ledger handles instead.

## What was not built

Firecracker and the Kubernetes `RuntimeClass` (v7) need nested virtualization or the
T3 cluster. The runc and gVisor runtimes are implemented and command-tested but were
not run: this machine has no Docker daemon. The agent mode needs a GPU running vLLM.

## Routes not to take

- **Scoring a run that the infrastructure killed.** A timeout from an overloaded host
  scored 0 teaches the policy that correct code is wrong. Classify, retry under the same
  episode id, drop if it fails again.
- **Letting the verifier run inside the agent's sandbox, or trusting test files the
  agent could edit.** Hash the tests, keep them outside, run them in a fresh sandbox.
- **Trusting exit code 0.** `sys.exit(0)` before the asserts passes; require the nonce.
- **Trusting `==`.** An object whose `__eq__` returns True passes every assert; check
  exact types, and do not trust a class's `__module__`.
- **The subprocess runtime for untrusted code.** It limits resources and isolates
  nothing; the probe table shows the host file read and `/proc`.
- **Mounting the Docker socket or `--privileged`.** Either is host root.
- **Killing only the parent process on timeout.** Its children live on as zombies or
  keep running; kill the process group.
- **Unbounded stdout.** One `print` loop can fill host memory; cap the output.
- **Episodes that are not serializable.** Without saved state there is no replay, no
  retry under the same id, and no way to ship episodes to an asynchronous trainer.
