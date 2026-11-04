"""
serving_platform.py
Judge routing, admission, autoscaling, and canary policies on one seeded open-loop
trace, with a discrete-event model of four identical LLM replicas.

Each simulated replica runs up to `SLOTS` requests at once (continuous batching), queues
the rest FIFO, and keeps an LRU prefix cache of `CACHE_PREFIXES` system prompts. A
request's time to first token is its queueing delay plus prefill of the prompt tokens
not already cached; it then decodes at a fixed inter-token latency. The routing code
under test is the real `serve.router` policies and admission; only the replicas are
simulated, so policy differences are not hidden by GPU noise.

Writes CSVs and a burst plot under experiments/t9-serving-platform/:

    uv run python app/src/training/experiments/serving_platform.py
"""

import csv
import heapq
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from training.constants import REPO_ROOT
from training.serve.autoscale import ScalingPolicy, Signals, decide, scale_up_legs
from training.serve.canary import CanaryController, Decision
from training.serve.router import (
    Policy,
    Replica,
    RouteRequest,
    admit,
    least_outstanding,
    power_of_two,
    prefix_bounded_load,
    round_robin,
)

RESULTS_DIR = REPO_ROOT / "experiments" / "t9-serving-platform"
REPLICAS, SLOTS, CACHE_PREFIXES = 4, 16, 6
PREFILL_S_PER_TOKEN, ITL_S = 0.00025, 0.030
TTFT_SLO_S = 1.0  # written down before the first run, with ITL p99 < 60 ms (met by ITL_S)
DOLLARS_PER_REPLICA_HOUR = 0.7068  # GKE g2-standard-4 (L4), on demand


@dataclass(frozen=True)
class SimRequest:
    """One request of the trace."""

    rid: int
    arrival: float
    prefix: Optional[int]  # shared system prompt id, or None
    prefix_tokens: int
    prompt_tokens: int
    output_tokens: int


def make_trace(seed: int, rate: float = 6.0, duration: float = 240.0) -> list[SimRequest]:
    """Poisson arrivals, 3x faster from 60 s to 120 s; 30% share one of 20 1,000-token prefixes."""
    rng, now, trace = random.Random(x=seed), 0.0, []  # noqa: NAR001
    while now < duration:
        now += rng.expovariate(lambd=rate * (3.0 if 60.0 <= now < 120.0 else 1.0))
        shared = rng.random() < 0.3
        trace.append(  # noqa: NAR001
            SimRequest(
                rid=len(trace),  # noqa: NAR001
                arrival=now,
                prefix=rng.randrange(20) if shared else None,  # noqa: NAR001
                prefix_tokens=1000 if shared else 0,
                prompt_tokens=rng.randint(a=50, b=500),
                output_tokens=rng.randint(a=50, b=400),
            )
        )  # noqa: NAR001
    return trace


@dataclass
class SimReplica:
    """Running finish-time heap, FIFO queue, and LRU prefix cache of one replica."""

    running: list[float] = field(default_factory=list)
    queue: deque = field(default_factory=deque)
    cache: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    lookups: int = 0

    def outstanding(self) -> int:
        """Running plus queued."""
        return len(self.running) + len(self.queue)  # noqa: NAR001

    def start(self, request: SimRequest, at: float, ttft: dict[int, float]) -> None:
        """Begin `request` at time `at`; record its TTFT; update the prefix cache."""
        cached = request.prefix is not None and request.prefix in self.cache
        if request.prefix is not None:
            self.lookups, self.hits = self.lookups + 1, self.hits + int(cached)  # noqa: NAR001
            self.cache[request.prefix] = None
            self.cache.move_to_end(key=request.prefix)
            if len(self.cache) > CACHE_PREFIXES:  # noqa: NAR001
                self.cache.popitem(last=False)
        prefill = (
            request.prompt_tokens + (0 if cached else request.prefix_tokens)
        ) * PREFILL_S_PER_TOKEN
        ttft[request.rid] = at + prefill - request.arrival
        heapq.heappush(self.running, at + prefill + request.output_tokens * ITL_S)  # noqa: NAR001

    def advance(self, until: float, ttft: dict[int, float]) -> None:
        """Finish requests due by `until`, starting queued ones into the freed slots."""
        while self.running and self.running[0] <= until:
            done = heapq.heappop(self.running)  # noqa: NAR001
            if self.queue:
                self.start(request=self.queue.popleft(), at=done, ttft=ttft)


def simulate(trace: list[SimRequest], policy: Policy, admission: bool, seed: int) -> dict:
    """Route `trace` with `policy`; return goodput, tails, hit rate, imbalance, and $/M tokens."""
    replicas, ttft, rejected, imbalance = [SimReplica() for _ in range(REPLICAS)], {}, 0, []  # noqa: NAR001
    rng = random.Random(x=seed)  # noqa: NAR001
    mean_service = 0.3 * 1000 * PREFILL_S_PER_TOKEN + 275 * PREFILL_S_PER_TOKEN + 225 * ITL_S
    for request in trace:
        for replica in replicas:
            replica.advance(until=request.arrival, ttft=ttft)
        snapshot = [
            Replica(name=str(i), outstanding=r.outstanding())  # noqa: NAR001
            for i, r in enumerate(replicas)  # noqa: NAR001
        ]  # noqa: NAR001
        loads = [r.outstanding for r in snapshot]
        imbalance.append(max(loads) / max(np.mean(loads), 1e-9))  # noqa: NAR001
        index = policy(  # noqa: NAR001
            snapshot,
            RouteRequest(
                prompt=f"system-{request.prefix}"
                if request.prefix is not None
                else f"user-{request.rid}"
            ),
            rng,
        )  # noqa: NAR001
        if index is None or (
            admission
            and not admit(
                replica=snapshot[index],
                service_time_s=mean_service,
                slots=SLOTS,
                ttft_slo_s=TTFT_SLO_S,
            )
        ):
            rejected += 1
            continue
        target = replicas[index]
        if len(target.running) < SLOTS:  # noqa: NAR001
            target.start(request=request, at=request.arrival, ttft=ttft)
        else:
            target.queue.append(request)  # noqa: NAR001
    for replica in replicas:
        replica.advance(until=float("inf"), ttft=ttft)  # noqa: NAR001
    duration = trace[-1].arrival
    served = [r for r in trace if r.rid in ttft]
    tokens_per_s = sum(r.output_tokens for r in served) / duration  # noqa: NAR001
    good = sum(1 for r in served if ttft[r.rid] < TTFT_SLO_S)  # noqa: NAR001
    return {
        "goodput_rps": round(good / duration, 3),  # noqa: NAR001
        "ttft_p99_s": round(float(np.percentile(a=list(ttft.values()), q=99)), 2),  # noqa: NAR001
        "prefix_hit_rate": round(  # noqa: NAR001
            sum(r.hits for r in replicas) / max(sum(r.lookups for r in replicas), 1),  # noqa: NAR001
            3,  # noqa: NAR001
        ),  # noqa: NAR001
        "mean_imbalance": round(float(np.mean(imbalance)), 2),  # noqa: NAR001
        "rejection_rate": round(rejected / len(trace), 3),  # noqa: NAR001
        "dollars_per_M_output_tokens": round(  # noqa: NAR001
            REPLICAS * DOLLARS_PER_REPLICA_HOUR / (tokens_per_s * 3600 / 1e6), 3
        ),  # noqa: NAR001
    }


POLICIES: dict[str, tuple] = {
    "v0 round robin": (round_robin, False),
    "v1 least outstanding": (lambda: least_outstanding, False),
    "v2 power of two": (lambda: power_of_two, False),
    "v3 prefix + bounded load (1.25)": (lambda: prefix_bounded_load(epsilon=0.25), False),
    "v4 v3 + admission control": (lambda: prefix_bounded_load(epsilon=0.25), True),
}


def policy_table(seeds: tuple[int, ...] = (0, 1, 2)) -> list[dict]:
    """Every policy on the same three seeded traces; mean and range of each metric."""
    rows = []
    for name, (make, admission) in POLICIES.items():
        runs = [
            simulate(trace=make_trace(seed=s), policy=make(), admission=admission, seed=s)
            for s in seeds
        ]
        row: dict = {"policy": name}
        for key in runs[0]:
            values = [run[key] for run in runs]
            row[key] = f"{np.mean(values):.3f} ({min(values)}-{max(values)})"  # noqa: NAR001
        rows.append(row)  # noqa: NAR001
        print(row)  # noqa: NAR001
    return rows


def random_vs_two_choices(num_bins: int = 100, num_balls: int = 10_000, seed: int = 0) -> dict:
    """Max load of balls into bins, one random choice against the better of two."""
    rng = random.Random(x=seed)  # noqa: NAR001
    one, two = [0] * num_bins, [0] * num_bins
    for _ in range(num_balls):  # noqa: NAR001
        one[rng.randrange(num_bins)] += 1  # noqa: NAR001
        a, b = rng.randrange(num_bins), rng.randrange(num_bins)  # noqa: NAR001
        two[a if two[a] <= two[b] else b] += 1
    return {"mean": num_balls / num_bins, "max_one_choice": max(one), "max_two_choices": max(two)}  # noqa: NAR001


def canary_demo(seed: int = 0) -> list[str]:
    """Roll out a version whose TTFT is 1.5 s worse; return the controller's log."""
    rng, controller = random.Random(x=seed), CanaryController(min_requests=30)  # noqa: NAR001
    for i in range(5000):  # noqa: NAR001
        version = controller.route(request_id=f"r{i}")
        ttft = rng.expovariate(lambd=1 / 0.4) + (1.5 if version == "canary" else 0.0)
        if controller.observe(version=version, met_slo=ttft < TTFT_SLO_S) in (
            Decision.ROLLBACK,
            Decision.DONE,
        ):
            controller.log.append(f"stopped after {i + 1} requests")  # noqa: NAR001
            break
    return controller.log


def autoscale_demo() -> list[dict]:
    """Drive `decide` with a queue that triples during a burst; return each decision with legs."""
    policy, replicas, last, rows = ScalingPolicy(min_replicas=4, max_replicas=12), 4, -1e9, []
    legs = scale_up_legs(
        stamps={
            "decided": 0.0,
            "node_ready": 150.0,
            "image_pulled": 200.0,
            "weights_loaded": 207.0,
            "ready": 240.0,
        }
    )
    for now in range(0, 600, 15):  # noqa: NAR001
        waiting = 60.0 if 60 <= now < 300 else 4.0
        target = decide(
            policy=policy,
            signals=Signals(replicas=replicas, waiting=waiting, kv_utilization=0.5),
            now=float(now),  # noqa: NAR001
            last_change=last,
        )  # noqa: NAR001
        if target != replicas:
            rows.append(  # noqa: NAR001
                {
                    "t_s": now,
                    "from": replicas,
                    "to": target,
                    "capacity_arrives_s": now + sum(legs.values()) if target > replicas else now,  # noqa: NAR001
                    **{f"leg_{k}_s": v for k, v in legs.items()},
                }
            )  # noqa: NAR001
            replicas, last = target, float(now)  # noqa: NAR001
    return rows


def burst_plot(seed: int = 0) -> None:
    """Plot TTFT over arrival time for round robin and the best policy on one trace.

    Side effects:
        Writes burst.png and burst.csv.
    """
    trace = make_trace(seed=seed)
    fig, ax = plt.subplots(figsize=(9, 4))
    rows = []
    for name, make, admission in (
        ("round robin", round_robin, False),
        ("prefix + bounded load + admission", lambda: prefix_bounded_load(epsilon=0.25), True),
    ):
        ttft: dict[int, float] = {}
        replicas = [SimReplica() for _ in range(REPLICAS)]  # noqa: NAR001
        policy, rng = make(), random.Random(x=seed)  # noqa: NAR001
        for request in trace:
            for replica in replicas:
                replica.advance(until=request.arrival, ttft=ttft)
            snapshot = [
                Replica(name=str(i), outstanding=r.outstanding())  # noqa: NAR001
                for i, r in enumerate(replicas)  # noqa: NAR001
            ]  # noqa: NAR001
            index = policy(  # noqa: NAR001
                snapshot,
                RouteRequest(
                    prompt=f"system-{request.prefix}"
                    if request.prefix is not None
                    else f"user-{request.rid}"
                ),
                rng,
            )  # noqa: NAR001
            if index is None or (
                admission
                and not admit(
                    replica=snapshot[index], service_time_s=7.0, slots=SLOTS, ttft_slo_s=TTFT_SLO_S
                )
            ):
                continue
            target = replicas[index]
            (
                target.start(request=request, at=request.arrival, ttft=ttft)
                if len(target.running) < SLOTS  # noqa: NAR001
                else target.queue.append(request)  # noqa: NAR001
            )  # noqa: NAR001
        for replica in replicas:
            replica.advance(until=float("inf"), ttft=ttft)  # noqa: NAR001
        points = sorted((trace[rid].arrival, value) for rid, value in ttft.items())  # noqa: NAR001
        ax.plot([p[0] for p in points], [p[1] for p in points], label=name, linewidth=0.8)  # noqa: NAR001
        rows += [
            {"policy": name, "arrival_s": round(a, 2), "ttft_s": round(t, 3)}  # noqa: NAR001
            for a, t in points  # noqa: NAR001
        ]  # noqa: NAR001
    ax.axvspan(xmin=60, xmax=120, color="grey", alpha=0.15, label="3x burst")
    ax.axhline(y=TTFT_SLO_S, color="black", linestyle=":", label="TTFT SLO")
    ax.set(
        xlabel="arrival time (s)",
        ylabel="TTFT (s)",
        yscale="log",
        title="TTFT through a 3x burst, 4 replicas",
    )
    ax.legend(fontsize=8)
    fig.savefig(fname=RESULTS_DIR / "burst.png", bbox_inches="tight", dpi=130)
    write_csv(name="burst", rows=rows)


def write_csv(name: str, rows: list[dict]) -> None:
    """Overwrite `RESULTS_DIR/<name>.csv`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(file=RESULTS_DIR / f"{name}.csv", mode="w", newline="") as handle:
        writer = csv.DictWriter(f=handle, fieldnames=list(rows[0]))  # noqa: NAR001
        writer.writeheader()
        writer.writerows(rowdicts=rows)


def main() -> None:
    """Run every study and write its outputs."""
    write_csv(name="policies", rows=policy_table())
    write_csv(name="two_choices", rows=[random_vs_two_choices()])
    log = canary_demo()
    write_csv(name="canary", rows=[{"event": line} for line in log])
    write_csv(name="autoscale", rows=autoscale_demo())
    burst_plot()
    print(log, random_vs_two_choices(), autoscale_demo(), sep="\n")  # noqa: NAR001


if __name__ == "__main__":
    main()
