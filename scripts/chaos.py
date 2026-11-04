"""Kill training workers at seeded times, so every checkpoint setting faces the same failures.

Watches a launcher process (`torchrun`, or anything that restarts its children) and,
at each time of a seeded failure schedule, SIGKILLs one of its child worker processes,
chosen with the same seed. The schedule is a pure function of `--seed`, `--mtbf`, and
`--duration`, so runs with different checkpoint intervals can be compared failure for
failure. Prints one line per kill: `killed <unix time> <pid>`.

    uv run python scripts/chaos.py --pid <torchrun pid> --seed 0 --mtbf 600 --duration 7200

To kill a pod or an instance instead, replace `kill_one` with a call to the launch
backend's `cancel` for that job.
"""

import argparse
import os
import random
import signal
import time

import psutil
from training.experiments.fault_tolerance import failure_schedule


def kill_one(launcher: psutil.Process, rng: random.Random) -> int | None:
    """SIGKILL one worker under `launcher` (a grandchild of the launcher's agent); return its pid."""
    workers = [p for p in launcher.children(recursive=True) if "worker" in " ".join(p.cmdline())]
    if not workers:
        return None
    victim = rng.choice(seq=sorted(workers, key=lambda p: p.pid))
    os.kill(victim.pid, signal.SIGKILL)  # noqa: NAR001
    return victim.pid


def main() -> None:
    """Follow the schedule until the launcher exits or the schedule ends."""
    parser = argparse.ArgumentParser(description="Seeded failure injection.")
    parser.add_argument("--pid", type=int, required=True)  # noqa: NAR001
    parser.add_argument("--seed", type=int, default=0)  # noqa: NAR001
    parser.add_argument("--mtbf", type=float, required=True, help="Mean seconds between failures.")  # noqa: NAR001
    parser.add_argument("--duration", type=float, required=True)  # noqa: NAR001
    parser.add_argument("--warmup", type=float, default=0.0, help="Seconds before the schedule starts.")  # noqa: NAR001
    args = parser.parse_args()
    launcher, rng = psutil.Process(pid=args.pid), random.Random(x=args.seed)  # noqa: NAR001
    start = time.time() + args.warmup
    for at in failure_schedule(seed=args.seed, mtbf_s=args.mtbf, duration_s=args.duration):
        time.sleep(max(0.0, start + at - time.time()))  # noqa: NAR001
        if not launcher.is_running():
            return
        pid = kill_one(launcher=launcher, rng=rng)
        if pid is not None:
            print(f"killed {time.time()} {pid}", flush=True)  # noqa: NAR001


if __name__ == "__main__":
    main()
