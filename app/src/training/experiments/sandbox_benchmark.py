"""
sandbox_benchmark.py
Run MBPP coding episodes through `env` under one sandbox runtime, and measure it.

Modes:

- `oracle`: submit each task's reference solution. No model is needed, so this
  measures the harness alone: tasks/hour, the verifier's pass rate on known-good
  code, and, with `--inject-faults`, whether infra faults leak into rewards.
- `agent`: an LLM agent on an OpenAI-compatible server (`--base-url`, `--model`)
  solves each task through `run_python` and `submit`.
- `replay`: re-run the episodes a previous run saved and check the rewards match.
- `probes`: run the adversarial probe suite against the runtime.
- `startup`: time `start()` for `--tasks` sandboxes; p50 and p99.

Results append to experiments/t2-environments/results.csv (probes, replay and startup
to their own CSVs);
episodes are saved to app/history/episodes/<run_id>/ for replay.

    uv run python app/src/training/experiments/sandbox_benchmark.py oracle --runtime subprocess
"""

import argparse
import csv
import json
import random
import statistics
import tempfile
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from training.constants import HISTORY_PATH, REPO_ROOT
from training.dataload.mbpp import load_mbpp_tasks
from training.env.agent import chat_completion, run_episode
from training.env.environment import Environment, EpisodeState, Task, replay
from training.env.failures import EpisodeLedger, run_with_retries
from training.env.probes import probes, run_probe
from training.env.sandbox import (
    SIGKILL_EXIT_CODE,
    DockerSandbox,
    Limits,
    RunResult,
    Sandbox,
    SubprocessSandbox,
    WarmPool,
)
from training.env.tools import SUBMIT
from training.env.verify import Verifier

RESULTS_DIR = REPO_ROOT / "experiments" / "t2-environments"
EPISODES_DIR = HISTORY_PATH / "episodes"
IMAGE = "python:3.12-slim"
LIMITS = Limits()


class FaultySandbox:
    """Wrap a sandbox so each `run` fails like infrastructure with probability `rate`.

    Half the faults are a sandbox that did not start, half a SIGKILL from outside the
    sandbox's limits; both must be classified infra and retried, never scored.
    """

    def __init__(self, inner: Sandbox, rate: float, rng: random.Random) -> None:
        """Wrap `inner`; draw faults from `rng`."""
        self.inner, self.rate, self.rng = inner, rate, rng
        self.python = inner.python

    def start(self) -> None:
        """Start the wrapped sandbox."""
        self.inner.start()

    def run(
        self, files: dict[str, str], command: list[str], limits: Limits, stdin: str = ""
    ) -> RunResult:
        """Fail with probability `rate`, else run the wrapped sandbox."""
        if self.rng.random() < self.rate:
            if self.rng.random() < 0.5:
                return RunResult(
                    exit_code=-1, stdout="", stderr="injected", duration_s=0.0, start_failed=True
                )
            return RunResult(exit_code=SIGKILL_EXIT_CODE, stdout="", stderr="", duration_s=0.0)
        return self.inner.run(files=files, command=command, limits=limits, stdin=stdin)

    def reset(self) -> None:
        """Reset the wrapped sandbox."""
        self.inner.reset()

    def destroy(self) -> None:
        """Destroy the wrapped sandbox."""
        self.inner.destroy()


def sandbox_factory(runtime: str, fault_rate: float, seed: int) -> Callable[[], Sandbox]:
    """Return a factory for `runtime` sandboxes, wrapped in `FaultySandbox` if `fault_rate`."""
    rng = random.Random(x=seed)  # noqa: NAR001

    def make() -> Sandbox:
        """Build one unstarted sandbox."""
        inner: Sandbox = (
            SubprocessSandbox()
            if runtime == "subprocess"
            else DockerSandbox(runtime=runtime, image=IMAGE, limits=LIMITS)
        )
        return FaultySandbox(inner=inner, rate=fault_rate, rng=rng) if fault_rate else inner

    return make


def build_environment_factory(
    tasks: list[dict[str, str]], factory: Callable[[], Sandbox], pool_size: int, tests_dir: Path
) -> Callable[[], Environment]:
    """Register every task's tests once; return a factory of environments sharing a pool."""
    verifier = Verifier(tests_dir=tests_dir, sandbox_factory=factory, limits=LIMITS)
    for task in tasks:
        verifier.register(task_id=task["task_id"], tests=task["tests"])
    pool = WarmPool(factory=factory, size=pool_size)
    by_id = {
        t["task_id"]: Task(task_id=t["task_id"], prompt=t["prompt"], tests=t["tests"])
        for t in tasks
    }
    return lambda: Environment(
        tasks=by_id, acquire=pool.acquire, release=pool.release, verifier=verifier, limits=LIMITS
    )


def oracle_episode(env: Environment, task: dict[str, str], seed: int, episode_id: str) -> float:
    """Submit the reference solution as the only action; return the reward."""
    env.reset(task_id=task["task_id"], seed=seed, episode_id=episode_id)
    call = {"id": "oracle", "name": SUBMIT, "arguments": json.dumps(obj={"code": task["solution"]})}
    return env.step(tool_call=call).reward


def run_episodes(
    tasks: list[dict[str, str]],
    make_env: Callable[[], Environment],
    episode: Callable[[Environment, dict[str, str], int, str], float],
    concurrency: int,
    run_dir: Path,
) -> tuple[EpisodeLedger, float]:
    """Run one episode per task, retrying infra failures; return the ledger and wall seconds.

    Side effects:
        Writes each finished episode's state to `run_dir/<episode_id>.json`.
    """
    ledger = EpisodeLedger.empty()
    run_dir.mkdir(parents=True, exist_ok=True)

    def one(index: int) -> None:
        """Run, retry, and save the episode for `tasks[index]`."""
        task, episode_id, env = tasks[index], f"{tasks[index]['task_id']}-{index}", make_env()

        def attempt() -> float:
            """One attempt; a fresh reset each time, under the same episode id."""
            return episode(env, task, index, episode_id)  # noqa: NAR001

        run_with_retries(episode_id=episode_id, episode=attempt, ledger=ledger)
        (run_dir / f"{episode_id}.json").write_text(data=env.state.to_json(), encoding="utf-8")

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(one, range(len(tasks))))  # noqa: NAR001
    return ledger, time.perf_counter() - start


def append_row(path: Path, row: dict) -> None:
    """Append `row` to the CSV at `path`, writing the header on first use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(file=path, mode="a", newline="") as handle:
        writer = csv.DictWriter(f=handle, fieldnames=list(row))  # noqa: NAR001
        if is_new:
            writer.writeheader()
        writer.writerow(rowdict=row)


def summarize(args: argparse.Namespace, ledger: EpisodeLedger, seconds: float, run_id: str) -> dict:
    """Return the results row for one oracle or agent run."""
    scored = len(ledger.rewards)  # noqa: NAR001
    return {
        "run_id": run_id,
        "mode": args.mode,
        "runtime": args.runtime,
        "tasks": args.tasks,
        "concurrency": args.concurrency,
        "fault_rate": args.inject_faults,
        "scored": scored,
        "dropped": len(ledger.dropped),  # noqa: NAR001
        "retried": sum(1 for count in ledger.attempts.values() if count > 1),  # noqa: NAR001
        "pass_rate": round(sum(ledger.rewards.values()) / max(scored, 1), 4),  # noqa: NAR001
        "tasks_per_hour": round(scored * 3600 / seconds),  # noqa: NAR001
        "wall_s": round(seconds, 1),  # noqa: NAR001
    }


def replay_run(run_dir: Path, make_env: Callable[[], Environment]) -> dict:
    """Replay every saved episode in `run_dir`; return how many reproduced their rewards."""
    paths = sorted(run_dir.glob(pattern="*.json"))  # noqa: NAR001
    states = [EpisodeState.from_json(text=path.read_text()) for path in paths]
    matches = sum(replay(state=state, env=make_env()) == state.rewards for state in states)  # noqa: NAR001
    return {"run_id": run_dir.name, "mode": "replay", "episodes": len(states), "matching": matches}  # noqa: NAR001


def startup_times(factory: Callable[[], Sandbox], count: int) -> dict:
    """Start and destroy `count` sandboxes; return p50 and p99 start time in ms."""
    times = []
    for _ in range(count):  # noqa: NAR001
        sandbox = factory()
        start = time.perf_counter()
        sandbox.start()
        times.append(1000 * (time.perf_counter() - start))  # noqa: NAR001
        sandbox.destroy()
    cuts = statistics.quantiles(data=times, n=100)
    return {
        "mode": "startup",
        "count": count,
        "p50_ms": round(cuts[49], 2),  # noqa: NAR001
        "p99_ms": round(cuts[98], 2),  # noqa: NAR001
    }


def run_probes(args: argparse.Namespace, factory: Callable[[], Sandbox]) -> None:
    """Run the probe suite on one sandbox and append each outcome to probes.csv.

    Side effects:
        Writes a host canary file in a temporary directory and deletes it after.
    """
    with tempfile.TemporaryDirectory() as canary_dir:
        canary = Path(canary_dir) / "canary.txt"  # noqa: NAR001
        canary.write_text(data=f"host-secret-{uuid.uuid4()}", encoding="utf-8")  # noqa: NAR001
        sandbox = factory()
        sandbox.start()
        for probe in probes(canary_path=str(canary)):  # noqa: NAR001
            if args.runtime == "subprocess" and not probe.host_safe:
                append_row(
                    path=RESULTS_DIR / "probes.csv",
                    row={
                        "runtime": args.runtime,
                        "probe": probe.name,
                        "expected": probe.expected,
                        "observed": "skipped: no pid limit on the host",
                        "contained": False,
                    },
                )
                continue
            outcome = run_probe(probe=probe, sandbox=sandbox, limits=LIMITS)
            row = {
                "runtime": args.runtime,
                "probe": outcome.name,
                "expected": outcome.expected,
                "observed": outcome.observed,
                "contained": outcome.contained,
            }
            append_row(path=RESULTS_DIR / "probes.csv", row=row)
            print(row)  # noqa: NAR001
        sandbox.destroy()


def parse_args() -> argparse.Namespace:
    """Parse the command line; see the module docstring."""
    parser = argparse.ArgumentParser(description="Benchmark sandboxed MBPP episodes.")
    parser.add_argument("mode", choices=["oracle", "agent", "replay", "probes", "startup"])  # noqa: NAR001
    parser.add_argument("--runtime", default="subprocess", choices=["subprocess", "runc", "runsc"])  # noqa: NAR001
    parser.add_argument("--tasks", type=int, default=200)  # noqa: NAR001
    parser.add_argument("--seed", type=int, default=0)  # noqa: NAR001
    parser.add_argument("--concurrency", type=int, default=4)  # noqa: NAR001
    parser.add_argument("--pool-size", type=int, default=4)  # noqa: NAR001
    parser.add_argument("--inject-faults", type=float, default=0.0)  # noqa: NAR001
    parser.add_argument("--base-url", default="http://localhost:8000")  # noqa: NAR001
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")  # noqa: NAR001
    parser.add_argument("--run-dir", type=Path, help="Saved run to replay.")  # noqa: NAR001
    return parser.parse_args()


def main() -> None:
    """Dispatch to the mode named on the command line."""
    args = parse_args()
    factory = sandbox_factory(runtime=args.runtime, fault_rate=args.inject_faults, seed=args.seed)
    if args.mode == "probes":
        run_probes(args=args, factory=factory)
        return
    if args.mode == "startup":
        append_row(
            path=RESULTS_DIR / "startup.csv",
            row={"runtime": args.runtime, **startup_times(factory=factory, count=args.tasks)},
        )
        return
    tasks = load_mbpp_tasks(num_tasks=args.tasks, seed=args.seed)
    with tempfile.TemporaryDirectory() as tests_dir:
        make_env = build_environment_factory(
            tasks=tasks,
            factory=factory,
            pool_size=args.pool_size,
            tests_dir=Path(tests_dir),  # noqa: NAR001
        )
        if args.mode == "replay":
            row = replay_run(run_dir=args.run_dir, make_env=make_env)
            append_row(path=RESULTS_DIR / "replay.csv", row=row)
            print(row)  # noqa: NAR001
            return
        else:
            run_id = str(uuid.uuid4())[:8]  # noqa: NAR001
            complete = partial(chat_completion, args.base_url, args.model)  # noqa: NAR001
            episode = (
                oracle_episode
                if args.mode == "oracle"
                else (
                    lambda env, task, seed, episode_id: run_episode(
                        env=env,
                        task_id=task["task_id"],
                        seed=seed,
                        episode_id=episode_id,
                        complete=complete,
                    )[0]
                )
            )
            ledger, seconds = run_episodes(
                tasks=tasks,
                make_env=make_env,
                episode=episode,
                concurrency=args.concurrency,
                run_dir=EPISODES_DIR / run_id,
            )
            row = summarize(args=args, ledger=ledger, seconds=seconds, run_id=run_id)
        append_row(path=RESULTS_DIR / "results.csv", row=row)
        print(row)  # noqa: NAR001


if __name__ == "__main__":
    main()
