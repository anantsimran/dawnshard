"""Route requests across model replicas: pure policies, admission control, and a proxy.

Every policy is a pure function `pick(replicas, request, rng) -> index or None`: it
reads a snapshot of replica state and returns where to send the request, so policies
are unit-tested and simulated without a cluster, and the proxy only gathers state and
forwards bytes.

```
request ──► admission (predicted wait > SLO? → 429) ──► policy.pick(snapshot) ──► replica
                                                              ▲
             replica /metrics (outstanding, KV utilization) ──┘  polled; the router keeps
                                                                  nothing it cannot rebuild
```

The policies, in order of sophistication:

- **Round robin**: ignores load; one slow request backs up its replica's queue.
- **Least outstanding**: the replica with the fewest in-flight requests; needs fresh,
  global state, and herds onto one replica when that state is stale.
- **Power of two choices**: sample two replicas, take the less loaded. Almost as good
  as least outstanding with far less coordination, and two routers with the same
  stale view rarely pick the same replica.
- **Prefix hash with bounded load**: hash the prompt's first block onto a ring of
  replicas so requests with the same system prompt land where its KV is cached; skip
  any replica above `(1 + epsilon) x` the mean load, walking the ring, so a popular
  prefix cannot pile onto one replica (consistent hashing with bounded loads).
- **Adapter aware**: prefer a replica that already holds the request's LoRA adapter,
  under the same load bound.
"""

import bisect
import hashlib
import json
import math
import random
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

PREFIX_CHARS = 256  # characters of the prompt that name its prefix-cache key
VIRTUAL_NODES = 64


@dataclass
class Replica:
    """A snapshot of one replica, as read from its metrics.

    Attributes:
        outstanding: Requests running or waiting on the replica.
        kv_utilization: Share of its KV cache in use, 0 to 1.
        adapters: LoRA adapters it currently holds.
    """

    name: str
    outstanding: int = 0
    kv_utilization: float = 0.0
    adapters: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RouteRequest:
    """What a policy may look at."""

    prompt: str
    adapter: Optional[str] = None


type Policy = Callable[[list[Replica], RouteRequest, random.Random], Optional[int]]


def round_robin() -> Policy:
    """Return a round-robin policy; its only state is a counter."""
    counter = iter(range(10**18))  # noqa: NAR001
    return lambda replicas, request, rng: next(counter) % len(replicas)  # noqa: NAR001


def least_outstanding(
    replicas: list[Replica], request: RouteRequest, rng: random.Random
) -> Optional[int]:
    """The replica with the fewest in-flight requests; ties broken at random."""
    lowest = min(r.outstanding for r in replicas)  # noqa: NAR001
    return rng.choice(seq=[i for i, r in enumerate(replicas) if r.outstanding == lowest])  # noqa: NAR001


def power_of_two(
    replicas: list[Replica], request: RouteRequest, rng: random.Random
) -> Optional[int]:
    """Sample two distinct replicas; return the less loaded (the first on a tie)."""
    if len(replicas) == 1:  # noqa: NAR001
        return 0
    a, b = rng.sample(population=range(len(replicas)), k=2)  # noqa: NAR001
    return a if replicas[a].outstanding <= replicas[b].outstanding else b


def prefix_key(prompt: str) -> int:
    """Stable 64-bit key of the prompt's leading characters."""
    return int.from_bytes(hashlib.sha256(prompt[:PREFIX_CHARS].encode()).digest()[:8], "big")  # noqa: NAR001


def _ring(replicas: list[Replica]) -> tuple[list[int], list[int]]:
    """Consistent-hash ring: `VIRTUAL_NODES` points per replica, sorted."""
    points = sorted(  # noqa: NAR001
        (prefix_key(prompt=f"{r.name}#{v}"), i)
        for i, r in enumerate(replicas)  # noqa: NAR001
        for v in range(VIRTUAL_NODES)  # noqa: NAR001
    )  # noqa: NAR001
    return [p for p, _ in points], [i for _, i in points]


def bounded_load_cap(replicas: list[Replica], epsilon: float) -> int:
    """Most requests a replica may hold: ceil((1 + epsilon) x the mean, counting this one).

    The ceiling is what guarantees a replica under the cap always exists (the least
    loaded one), so the bound sheds no traffic by itself; without it, four replicas at
    load 1 get a cap of 1.56 and every placement is refused.
    """
    total = sum(r.outstanding for r in replicas) + 1  # noqa: NAR001
    return math.ceil((1 + epsilon) * total / len(replicas))  # noqa: NAR001


def prefix_bounded_load(epsilon: float = 0.25) -> Policy:
    """Prefix-hash routing with a load bound; see the module docstring."""

    def pick(replicas: list[Replica], request: RouteRequest, rng: random.Random) -> Optional[int]:
        """Walk the ring from the prefix's point to the first replica under the cap."""
        keys, owners = _ring(replicas=replicas)
        cap = bounded_load_cap(replicas=replicas, epsilon=epsilon)
        start = bisect.bisect(keys, prefix_key(prompt=request.prompt))  # noqa: NAR001
        for step in range(len(keys)):  # noqa: NAR001
            index = owners[(start + step) % len(keys)]  # noqa: NAR001
            if replicas[index].outstanding + 1 <= cap:
                return index
        return None

    return pick


def adapter_aware(epsilon: float = 0.25) -> Policy:
    """Prefer the least-loaded replica holding the request's adapter, under the load bound."""

    def pick(replicas: list[Replica], request: RouteRequest, rng: random.Random) -> Optional[int]:
        """Resident and under the cap first; otherwise the least loaded overall."""
        cap = bounded_load_cap(replicas=replicas, epsilon=epsilon)
        resident = [
            i
            for i, r in enumerate(replicas)  # noqa: NAR001
            if request.adapter in r.adapters and r.outstanding + 1 <= cap
        ]  # noqa: NAR001
        pool = resident or range(len(replicas))  # noqa: NAR001
        return min(pool, key=lambda i: replicas[i].outstanding)  # noqa: NAR001

    return pick


def predicted_wait(replica: Replica, service_time_s: float, slots: int) -> float:
    """Seconds a new request waits on `replica`: requests ahead of it beyond the free slots."""
    queued = max(0, replica.outstanding - slots + 1)  # noqa: NAR001
    return queued * service_time_s / slots


def admit(replica: Replica, service_time_s: float, slots: int, ttft_slo_s: float) -> bool:
    """Accept only if the predicted wait leaves the TTFT SLO reachable; else shed with 429."""
    return predicted_wait(replica=replica, service_time_s=service_time_s, slots=slots) <= ttft_slo_s


def read_metrics(text: str) -> tuple[int, float]:
    """`(outstanding, kv_utilization)` from a vLLM Prometheus `/metrics` body."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):  # noqa: NAR001
            name, _, value = line.rpartition(" ")  # noqa: NAR001
            values[name.split("{")[0]] = values.get(name.split("{")[0], 0.0) + float(value)  # noqa: NAR001
    outstanding = values.get("vllm:num_requests_running", 0.0) + values.get(  # noqa: NAR001
        "vllm:num_requests_waiting", 0.0
    )  # noqa: NAR001
    kv = values.get("vllm:kv_cache_usage_perc", values.get("vllm:gpu_cache_usage_perc", 0.0))  # noqa: NAR001
    return int(outstanding), kv  # noqa: NAR001


class Router:
    """An HTTP proxy in front of replicas: polls their metrics, admits, picks, forwards."""

    def __init__(
        self,
        urls: list[str],
        policy: Policy,
        service_time_s: float,
        slots: int,
        ttft_slo_s: float,
        seed: int = 0,
    ) -> None:
        """Track one `Replica` per base URL."""
        self.urls, self.policy = urls, policy
        self.replicas = [Replica(name=url) for url in urls]
        self.service_time_s, self.slots, self.ttft_slo_s = service_time_s, slots, ttft_slo_s
        self.rng, self.lock = random.Random(x=seed), threading.Lock()  # noqa: NAR001

    def refresh(self) -> None:
        """Rebuild every replica's state from its `/metrics`; an unreachable one looks full."""
        for replica, url in zip(self.replicas, self.urls):  # noqa: NAR001
            try:
                with urllib.request.urlopen(url=f"{url}/metrics", timeout=2) as response:
                    replica.outstanding, replica.kv_utilization = read_metrics(
                        text=response.read().decode()
                    )
            except OSError:
                replica.outstanding = 10**6

    def route(self, request: RouteRequest) -> Optional[int]:
        """Pick a replica and count the request on it; None means reject with 429."""
        with self.lock:
            index = self.policy(self.replicas, request, self.rng)  # noqa: NAR001
            if index is None or not admit(
                replica=self.replicas[index],
                service_time_s=self.service_time_s,
                slots=self.slots,
                ttft_slo_s=self.ttft_slo_s,
            ):
                return None
            self.replicas[index].outstanding += 1
            return index

    def handler(self) -> type[BaseHTTPRequestHandler]:
        """An HTTP handler that proxies POSTs to the chosen replica, streaming the reply."""
        router = self

        class Handler(BaseHTTPRequestHandler):
            """Forward one request; 429 when admission refuses it."""

            def do_POST(self) -> None:  # noqa: N802
                """Route, forward, and stream the replica's response back unchanged."""
                body = self.rfile.read(int(self.headers["Content-Length"]))  # noqa: NAR001
                payload = json.loads(s=body)
                index = router.route(
                    request=RouteRequest(
                        prompt=payload.get("prompt", ""),  # noqa: NAR001
                        adapter=payload.get("model"),  # noqa: NAR001
                    )
                )  # noqa: NAR001
                if index is None:
                    self.send_error(code=429, message="predicted wait exceeds the SLO")
                    return
                upstream = urllib.request.Request(
                    url=f"{router.urls[index]}{self.path}",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(url=upstream, timeout=600) as response:
                        self.send_response(code=response.status)
                        self.send_header(
                            keyword="Content-Type", value=response.headers["Content-Type"]
                        )
                        self.end_headers()
                        for chunk in iter(lambda: response.read1(65536), b""):  # noqa: NAR001
                            self.wfile.write(chunk)  # noqa: NAR001
                            self.wfile.flush()
                finally:
                    with router.lock:
                        router.replicas[index].outstanding -= 1

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                """Silence per-request logging."""

        return Handler

    def serve(self, host: str, port: int) -> ThreadingHTTPServer:
        """Return a proxy server ready for `serve_forever()`."""
        return ThreadingHTTPServer((host, port), self.handler())  # noqa: NAR001
