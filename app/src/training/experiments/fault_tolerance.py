"""
fault_tolerance.py
Checkpointing and fault tolerance on CPU: bitwise resume, resharding, save cost,
goodput under a seeded failure schedule, and straggler detection.

`worker` is a training job that survives being killed: a tiny Qwen3 under FSDP2 (2 gloo
ranks) on the Multi30k token shards from the data-loading experiment, read through
`StatefulLoader`, checkpointed by `Checkpointer` every `--every` steps. On start it
loads the newest committed checkpoint, so `torchrun --max-restarts` turns a killed rank
into a restart from the last commit. Rank 0 appends an event log (start, step,
saved) that the goodput accounting reads.

Studies (each writes under experiments/t10-fault-tolerance/):

- `resume`: 24 steps uninterrupted; 12 steps, then a fresh job resumes to 24; the same
  resume on 1 rank instead of 2. Per-step global losses and batch hashes compared.
- `save`: blocking save time and rank-0 memory growth for rank-0 gather, DCP, and
  async DCP, at two model sizes.
- `chaos`: `scripts/chaos.py` kills a worker at seeded times while `torchrun` restarts
  it; lost steps and recovery time per failure, and goodput, from the event log.
- `goodput`: simulated two-hour runs under one seeded failure schedule, intervals from
  1 to 60 minutes, against `expected_goodput` and the Young/Daly optimum.
- `straggler`: 4 ranks, one slowed on purpose; `StragglerDetector` must flag only it.

    uv run python app/src/training/experiments/fault_tolerance.py all
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
from torch import nn
from training.common import bpe
from training.constants import HISTORY_PATH, REPO_ROOT
from training.dataload.streaming import StatefulLoader, StreamingTokens, TokenSource
from training.model.qwen3 import QWEN3_SMALL, QWEN3_TINY, Qwen3, Qwen3Block
from training.parallel.fsdp import ParallelismSpec, apply_fsdp
from training.train.checkpoint import (
    Checkpointer,
    expected_goodput,
    save_full_on_rank0,
    young_daly_interval,
)
from training.train.model import TrainState
from training.train.straggler import StragglerDetector

RESULTS_DIR = REPO_ROOT / "experiments" / "t10-fault-tolerance"
SHARDS_DIR = REPO_ROOT / "app" / "datasets" / "cache" / "multi30k_shards"
WORK_DIR = HISTORY_PATH / "fault_tolerance"
COLLECTIVE_TIMEOUT_S = 30
CONFIG = replace(QWEN3_TINY, vocab_size=bpe.FIRST_MERGE_ID + 2000 + 42)  # noqa: NAR001


def log_event(path: Path, **event) -> None:
    """Append one JSON event with a wall-clock timestamp (rank 0 only)."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(file=path, mode="a") as handle:
            handle.write(json.dumps(obj={"time": time.time(), **event}) + "\n")  # noqa: NAR001


def build_state(world: int, rank: int) -> TrainState:
    """Tiny Qwen3 from a fixed seed, sharded with FSDP2 when distributed, plus a stateful loader."""
    torch.manual_seed(seed=0)
    model: nn.Module = Qwen3(config=CONFIG)
    if world > 1:
        model = apply_fsdp(
            model=model,
            block_type=Qwen3Block,
            spec=ParallelismSpec(param_dtype=None, reduce_dtype=None),
            device_type="cpu",
        )
    loader = StatefulLoader(
        dataset=StreamingTokens(
            source=TokenSource(root=str(SHARDS_DIR)),  # noqa: NAR001
            seq_len=32,
            global_batch=8,
            seed=0,
            rank=rank,
            world_size=world,
        )
    )  # noqa: NAR001
    return TrainState(
        model=model,
        optimizer=torch.optim.AdamW(params=model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        loader=loader,
    )


def worker(args: argparse.Namespace) -> None:
    """Train to `--steps`, resuming from the newest commit and saving every `--every` steps."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:  # noqa: NAR001
        # A short timeout turns a peer that died mid-collective into an error torchrun
        # restarts, instead of a hang that waits out the 30-minute default.
        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=COLLECTIVE_TIMEOUT_S))
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    state = build_state(world=world, rank=rank)
    checkpointer = Checkpointer(
        root=Path(args.root),  # noqa: NAR001
        async_save=args.async_save,
        config={"steps": args.steps},  # noqa: NAR001
    )  # noqa: NAR001
    step = checkpointer.load_latest(state=state) or 0
    log_event(path=Path(args.events), kind="start", step=step, world=world)  # noqa: NAR001
    assert state.loader is not None
    for x, y in state.loader:
        step += 1
        loss = state.criterion(state.model(x), y)  # noqa: NAR001
        state.optimizer.zero_grad()
        loss.backward()
        state.optimizer.step()
        global_loss = loss.detach().clone()
        rows = x.clone()
        if dist.is_initialized():
            dist.all_reduce(tensor=global_loss)
            global_loss /= world
            parts = [torch.empty_like(input=rows) for _ in range(world)]  # noqa: NAR001
            dist.all_gather(tensor_list=parts, tensor=rows)
            rows = torch.cat(tensors=parts)
        log_event(
            path=Path(args.events),  # noqa: NAR001
            kind="step",
            step=step,
            loss=global_loss.item(),
            batch=hashlib.sha256(rows.numpy().tobytes()).hexdigest()[:12],  # noqa: NAR001
        )  # noqa: NAR001
        time.sleep(args.step_sleep)  # noqa: NAR001
        if step % args.every == 0:
            checkpointer.save(state=state, step=step)
            log_event(path=Path(args.events), kind="saved", step=step)  # noqa: NAR001
        if step == args.steps:
            break
    checkpointer.wait()
    if dist.is_initialized():
        dist.destroy_process_group()


def launch(
    root: Path,
    events: Path,
    steps: int,
    every: int,
    nproc: int,
    max_restarts: int = 0,
    step_sleep: float = 0.0,
    async_save: bool = False,
) -> subprocess.Popen:
    """Start `worker` under `torchrun` (or alone for one process)."""
    args = [
        "worker",
        f"--root={root}",
        f"--events={events}",
        f"--steps={steps}",
        f"--every={every}",
        f"--step-sleep={step_sleep}",
    ] + (["--async-save"] if async_save else [])
    script = str(Path(__file__))  # noqa: NAR001
    command = (
        [sys.executable, script, *args]
        if nproc == 1
        else [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            f"--max-restarts={max_restarts}",
            script,
            *args,
        ]
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "app" / "src")}  # noqa: NAR001
    log = open(file=events.with_suffix(suffix=".log"), mode="a")  # the launcher's own output
    return subprocess.Popen(args=command, env=env, stdout=log, stderr=subprocess.STDOUT)


def read_events(path: Path) -> list[dict]:
    """The event log as dicts, in order."""
    return [json.loads(s=line) for line in path.read_text().splitlines()] if path.exists() else []


def fresh(name: str) -> tuple[Path, Path]:
    """An empty checkpoint root and event-log path for one run."""
    root = WORK_DIR / name
    shutil.rmtree(path=root, ignore_errors=True)
    root.mkdir(parents=True)
    return root / "ckpt", root / "events.jsonl"


def resume_study() -> list[dict]:
    """Uninterrupted vs resumed (2 ranks) vs resumed on 1 rank: per-step loss and batch hash."""
    runs = {}
    root, events = fresh(name="uninterrupted")
    launch(root=root, events=events, steps=24, every=100, nproc=2).wait()
    runs["uninterrupted"] = {e["step"]: e for e in read_events(path=events) if e["kind"] == "step"}
    root, events = fresh(name="resumed")
    launch(root=root, events=events, steps=12, every=12, nproc=2).wait()
    snapshot = root.parent / "ckpt_at_12"
    shutil.copytree(src=root, dst=snapshot)
    launch(root=root, events=events, steps=24, every=100, nproc=2).wait()
    runs["resumed_2"] = {e["step"]: e for e in read_events(path=events) if e["kind"] == "step"}
    _, events_1 = fresh(name="resharded")
    launch(root=snapshot, events=events_1, steps=24, every=100, nproc=1).wait()
    runs["resumed_1"] = {e["step"]: e for e in read_events(path=events_1) if e["kind"] == "step"}
    rows = []
    for step in range(10, 25):  # noqa: NAR001
        base, two, one = (
            runs["uninterrupted"][step],
            runs["resumed_2"].get(step),  # noqa: NAR001
            runs["resumed_1"].get(step),  # noqa: NAR001
        )  # noqa: NAR001
        rows.append(  # noqa: NAR001
            {
                "step": step,
                "loss_uninterrupted": base["loss"],
                "loss_resumed_2_ranks": two["loss"] if two else "",
                "loss_resumed_1_rank": one["loss"] if one else "",
                "bitwise_equal_2": two is not None
                and two["loss"] == base["loss"]
                and two["batch"] == base["batch"],
                "batch_equal_1": one is not None and one["batch"] == base["batch"],
            }
        )  # noqa: NAR001
    return rows


def save_study() -> list[dict]:
    """Blocking save time and rank-0 RSS growth: rank-0 gather vs DCP vs async DCP."""
    rows = []
    for name, config in (
        ("tiny", CONFIG),
        ("small", replace(QWEN3_SMALL, vocab_size=CONFIG.vocab_size)),  # noqa: NAR001
    ):  # noqa: NAR001
        for method in ("rank0", "dcp", "async"):
            root, _ = fresh(name=f"save_{name}_{method}")
            out = root.parent / "result.json"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(Path(__file__)),  # noqa: NAR001
                "save-worker",
                f"--method={method}",
                f"--root={root}",
                f"--config={name}",
                f"--out={out}",
            ]  # noqa: NAR001
            subprocess.run(
                args=command,
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "app" / "src")},  # noqa: NAR001
                check=True,
                capture_output=True,
            )  # noqa: NAR001
            rows.append({"model": name, "method": method, **json.loads(s=out.read_text())})  # noqa: NAR001
    return rows


def save_worker(args: argparse.Namespace) -> None:
    """One save under FSDP on 2 ranks; rank 0 writes timing and its RSS growth to `--out`."""
    dist.init_process_group(backend="gloo")
    config = CONFIG if args.config == "tiny" else replace(QWEN3_SMALL, vocab_size=CONFIG.vocab_size)  # noqa: NAR001
    torch.manual_seed(seed=0)
    model = apply_fsdp(
        model=Qwen3(config=config),
        block_type=Qwen3Block,
        spec=ParallelismSpec(param_dtype=None, reduce_dtype=None),
        device_type="cpu",
    )
    optimizer = torch.optim.AdamW(params=model.parameters())
    model(torch.zeros(1, 8, dtype=torch.long)).sum().backward()  # noqa: NAR001
    optimizer.step()
    state = TrainState(model=model, optimizer=optimizer, criterion=nn.CrossEntropyLoss())
    process = psutil.Process()
    rss, start = process.memory_info().rss, time.perf_counter()
    checkpointer = Checkpointer(root=Path(args.root), async_save=args.method == "async")  # noqa: NAR001
    if args.method == "rank0":
        save_full_on_rank0(state=state, path=Path(args.root) / "full.pt")  # noqa: NAR001
    else:
        checkpointer.save(state=state, step=1)
    blocking = time.perf_counter() - start
    peak = process.memory_info().rss - rss
    checkpointer.wait()
    total = time.perf_counter() - start
    params = sum(p.numel() for p in Qwen3(config=config).parameters())  # noqa: NAR001
    if dist.get_rank() == 0:
        Path(args.out).write_text(  # noqa: NAR001
            data=json.dumps(
                obj={
                    "parameters": params,
                    "blocking_s": round(blocking, 3),  # noqa: NAR001
                    "total_s": round(total, 3),  # noqa: NAR001
                    "rank0_rss_growth_mb": round(peak / 2**20, 1),  # noqa: NAR001
                }
            )
        )  # noqa: NAR001
    dist.destroy_process_group()


def failure_schedule(seed: int, mtbf_s: float, duration_s: float) -> list[float]:
    """Seeded failure times: exponential gaps with mean `mtbf_s`, the same for every interval."""
    rng, now, times = random.Random(x=seed), 0.0, []  # noqa: NAR001
    while True:
        now += rng.expovariate(lambd=1 / mtbf_s)
        if now >= duration_s:
            return times
        times.append(now)  # noqa: NAR001


def simulate_goodput(
    interval_s: float,
    checkpoint_s: float,
    restart_s: float,
    failures: list[float],
    duration_s: float,
) -> float:
    """Useful work / wall clock when checkpointing every `interval_s` under `failures`.

    Work accumulates; every `interval_s` of work a checkpoint blocks for `checkpoint_s`
    and commits it. A failure throws away uncommitted work and costs `restart_s`.
    """
    clock, committed, work, failures = 0.0, 0.0, 0.0, list(failures)  # noqa: NAR001
    while clock < duration_s:
        next_save = clock + (interval_s - (work - committed))
        next_failure = failures[0] if failures else math.inf
        if next_failure < next_save:
            clock, work = next_failure + restart_s, committed
            failures.pop(0)  # noqa: NAR001
        else:
            work += next_save - clock
            clock = next_save + checkpoint_s
            committed = work
    return min(committed, duration_s) / clock  # noqa: NAR001


def goodput_study() -> list[dict]:
    """Two-hour runs under one failure schedule (MTBF 20 min), sync and async saves."""
    duration, mtbf, restart = 7200.0, 1200.0, 60.0
    rows = []
    for seed in range(5):  # noqa: NAR001
        failures = failure_schedule(seed=seed, mtbf_s=mtbf, duration_s=duration)
        for mode, delta in (("sync", 30.0), ("async", 3.0)):
            optimum = young_daly_interval(checkpoint_s=delta, mtbf_s=mtbf)
            for interval in sorted(  # noqa: NAR001
                {60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, round(optimum, 1)}  # noqa: NAR001
            ):
                rows.append(  # noqa: NAR001
                    {
                        "seed": seed,
                        "mode": mode,
                        "interval_s": interval,
                        "young_daly_s": round(optimum, 1),  # noqa: NAR001
                        "failures": len(failures),  # noqa: NAR001
                        "simulated_goodput": round(  # noqa: NAR001
                            simulate_goodput(
                                interval_s=interval,
                                checkpoint_s=delta,
                                restart_s=restart,
                                failures=failures,
                                duration_s=duration,
                            ),
                            4,
                        ),
                        "predicted_goodput": round(  # noqa: NAR001
                            expected_goodput(
                                interval_s=interval,
                                checkpoint_s=delta,
                                mtbf_s=mtbf,
                                restart_s=restart,
                            ),
                            4,
                        ),
                    }
                )  # noqa: NAR001
    return rows


def chaos_study(seed: int = 3) -> list[dict]:
    """Kill workers at seeded times under `torchrun --max-restarts`; account goodput."""
    root, events = fresh(name="chaos")
    job = launch(
        root=root, events=events, steps=60, every=10, nproc=2, max_restarts=10, step_sleep=0.25
    )
    chaos = subprocess.Popen(
        args=[
            sys.executable,
            str(REPO_ROOT / "scripts" / "chaos.py"),  # noqa: NAR001
            f"--pid={job.pid}",
            f"--seed={seed}",
            "--mtbf=25",
            "--duration=60",
            "--warmup=12",
        ],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "app" / "src")},  # noqa: NAR001
        stdout=subprocess.PIPE,
        text=True,
    )  # noqa: NAR001
    job.wait()
    kills = [
        float(line.split()[1])  # noqa: NAR001
        for line in (chaos.communicate()[0] or "").splitlines()
        if line.startswith("killed")  # noqa: NAR001
    ]  # noqa: NAR001
    return account(events=read_events(path=events), kills=kills)


def account(events: list[dict], kills: list[float]) -> list[dict]:
    """Per failure: steps lost (reached - last committed) and recovery time; then the total."""
    rows, start = [], events[0]["time"]
    starts = [e for e in events if e["kind"] == "start"]
    for kill in kills:
        previous = [e for e in starts if e["time"] < kill][-1]
        restart = next((e for e in starts if e["time"] > kill), None)  # noqa: NAR001
        if restart is None:
            continue
        # Only steps since the attempt that was killed count; a kill during a restart
        # interrupts no new work and loses nothing beyond the restart itself.
        since = [e for e in events if previous["time"] <= e["time"] < kill and e["kind"] == "step"]
        reached = max((e["step"] for e in since), default=previous["step"])  # noqa: NAR001
        first_after = next(e for e in events if e["kind"] == "step" and e["time"] > restart["time"])  # noqa: NAR001
        rows.append(  # noqa: NAR001
            {
                "failure_at_s": round(kill - start, 2),  # noqa: NAR001
                "step_reached": reached,
                "resumed_from_step": restart["step"],
                "steps_lost": reached - restart["step"],
                "restart_and_reload_s": round(restart["time"] - kill, 2),  # noqa: NAR001
                "to_first_step_s": round(first_after["time"] - kill, 2),  # noqa: NAR001
            }
        )  # noqa: NAR001
    steps = [e for e in events if e["kind"] == "step"]
    wall = steps[-1]["time"] - start
    useful = len({e["step"] for e in steps}) / max(len(steps), 1)  # noqa: NAR001
    rows.append(  # noqa: NAR001
        {
            "failure_at_s": "total",
            "step_reached": steps[-1]["step"],
            "resumed_from_step": "",
            "steps_lost": len(steps) - len({e["step"] for e in steps}),  # noqa: NAR001
            "restart_and_reload_s": round(wall, 2),  # noqa: NAR001
            "to_first_step_s": f"step goodput {useful:.3f}",
        }
    )  # noqa: NAR001
    return rows


def straggler_study() -> list[dict]:
    """4 gloo ranks, rank 2 slowed by 30 ms per step; return which ranks the detector flags."""
    root, _ = fresh(name="straggler")
    out = root.parent / "flagged.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(Path(__file__)),  # noqa: NAR001
        "straggler-worker",
        f"--out={out}",
    ]  # noqa: NAR001
    subprocess.run(
        args=command,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "app" / "src")},  # noqa: NAR001
        check=True,
        capture_output=True,
    )  # noqa: NAR001
    return [json.loads(s=out.read_text())]


def straggler_worker(args: argparse.Namespace) -> None:
    """Time compute and the gradient all-reduce per step; gather to rank 0 and run the detector."""
    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(seed=0)
    model, detector = Qwen3(config=CONFIG), StragglerDetector(num_ranks=world)
    for _ in range(20):  # noqa: NAR001
        start = time.perf_counter()
        model(torch.randint(low=0, high=CONFIG.vocab_size, size=(4, 32))).sum().backward()  # noqa: NAR001
        if rank == 2:
            time.sleep(0.03)  # noqa: NAR001
        computed = time.perf_counter()
        for p in model.parameters():
            dist.all_reduce(tensor=p.grad)  # type: ignore[arg-type]
        waited = time.perf_counter()
        timings: list = [None] * world
        dist.all_gather_object(object_list=timings, obj=(computed - start, waited - computed))
        detector.record(compute_s=[t[0] for t in timings], wait_s=[t[1] for t in timings])
    if rank == 0:
        Path(args.out).write_text(  # noqa: NAR001
            data=json.dumps(obj={"slowed_rank": 2, "flagged": detector.stragglers()})
        )  # noqa: NAR001
    dist.destroy_process_group()


def write_csv(name: str, rows: list[dict]) -> None:
    """Overwrite `RESULTS_DIR/<name>.csv`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(file=RESULTS_DIR / f"{name}.csv", mode="w", newline="") as handle:
        writer = csv.DictWriter(f=handle, fieldnames=list(rows[0]))  # noqa: NAR001
        writer.writeheader()
        writer.writerows(rowdicts=rows)


def main() -> None:
    """Dispatch: the two workers run under torchrun; the studies drive them."""
    parser = argparse.ArgumentParser(description="Checkpointing and fault-tolerance studies.")
    parser.add_argument(  # noqa: NAR001
        "command",
        choices=[
            "worker",
            "save-worker",
            "straggler-worker",
            "resume",
            "save",
            "chaos",
            "goodput",
            "straggler",
            "all",
        ],
    )  # noqa: NAR001
    for flag, kind in (
        ("--root", str),
        ("--events", str),
        ("--out", str),
        ("--method", str),
        ("--config", str),
    ):
        parser.add_argument(flag, type=kind)  # noqa: NAR001
    parser.add_argument("--steps", type=int, default=24)  # noqa: NAR001
    parser.add_argument("--every", type=int, default=12)  # noqa: NAR001
    parser.add_argument("--step-sleep", type=float, default=0.0)  # noqa: NAR001
    parser.add_argument("--async-save", action="store_true")  # noqa: NAR001
    args = parser.parse_args()
    workers = {"worker": worker, "save-worker": save_worker, "straggler-worker": straggler_worker}
    if args.command in workers:
        workers[args.command](args)  # noqa: NAR001
        return
    studies = {
        "resume": resume_study,
        "save": save_study,
        "chaos": chaos_study,
        "goodput": goodput_study,
        "straggler": straggler_study,
    }
    for name, study in studies.items():
        if args.command in (name, "all"):
            rows = study()
            write_csv(name=name, rows=rows)
            print(name, *rows[:8], sep="\n")  # noqa: NAR001


if __name__ == "__main__":
    main()
