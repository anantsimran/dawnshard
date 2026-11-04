"""Seeded traces, open-loop replay, and the latency metrics serving is judged by.

A trace is a list of requests with arrival times drawn from a Poisson process (and
optional bursts), prompt and output lengths, and, for prefix-heavy traffic, a share
of requests that start with one of a few long shared system prompts. The trace is a
pure function of its seed, so every engine and every version replays the same one.

Replay is open-loop: each request is sent at its arrival time whether or not earlier
ones have finished. A closed loop (send the next when one returns) slows down with
the server and hides exactly the queueing tails a serving system must be judged on.

Metrics, per request, from timestamps:

- **TTFT**, time to first token: arrival to first token.
- **ITL**, inter-token latency: gaps between consecutive tokens.
- **Goodput**: requests per second that met the SLO (TTFT and p99 ITL both under).
"""

import json
import random
import threading
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TraceRequest:
    """One request of a trace: when it arrives, its prompt, how many tokens to generate."""

    arrival: float
    prompt: str
    max_tokens: int


def make_trace(
    seed: int,
    num_requests: int,
    rate: float,
    prompt_words: tuple[int, int] = (20, 200),
    output_tokens: tuple[int, int] = (16, 128),
    shared_prefix_share: float = 0.0,
    num_prefixes: int = 20,
    prefix_words: int = 300,
    burst: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> list[TraceRequest]:
    """Return a seeded trace.

    Args:
        rate: Mean arrivals per second.
        shared_prefix_share: Fraction of requests that start with one of `num_prefixes`
            shared system prompts of `prefix_words` words.
        burst: `(start_s, end_s, multiplier)`: arrivals in that window come
            `multiplier` times faster.
    """
    rng = random.Random(x=seed)  # noqa: NAR001
    vocabulary = ["the", "a", "model", "token", "cache", "batch", "serve", "über", "日本"]

    def words(n: int) -> str:
        """`n` random words, some multi-byte."""
        return " ".join(rng.choice(seq=vocabulary) for _ in range(n))  # noqa: NAR001

    prefixes = [f"System {i}: " + words(prefix_words) for i in range(num_prefixes)]  # noqa: NAR001
    trace, now = [], 0.0
    for _ in range(num_requests):  # noqa: NAR001
        in_burst = burst[0] <= now < burst[1]
        now += rng.expovariate(lambd=rate * (burst[2] if in_burst else 1.0))
        prompt = words(rng.randint(a=prompt_words[0], b=prompt_words[1]))  # noqa: NAR001
        if rng.random() < shared_prefix_share:
            prompt = rng.choice(seq=prefixes) + " " + prompt
        trace.append(  # noqa: NAR001
            TraceRequest(
                arrival=now,
                prompt=prompt,
                max_tokens=rng.randint(a=output_tokens[0], b=output_tokens[1]),
            )
        )  # noqa: NAR001
    return trace


@dataclass
class RequestRecord:
    """Timestamps of one replayed request, in seconds from the start of the replay."""

    arrival: float
    token_times: list[float] = field(default_factory=list)


def summarize(
    records: list[RequestRecord], duration: float, ttft_slo: float = 0.5, itl_slo: float = 0.05
) -> dict:
    """TTFT p50/p99, ITL p99, output tokens/s, and goodput from per-request timestamps."""
    done = [r for r in records if r.token_times]
    ttft = [r.token_times[0] - r.arrival for r in done]
    gaps = [b - a for r in done for a, b in zip(r.token_times, r.token_times[1:])]  # noqa: NAR001

    def meets(record: RequestRecord) -> bool:
        """Whether one request met both SLOs."""
        own = np.diff(record.token_times)  # noqa: NAR001
        p99 = float(np.percentile(a=own, q=99)) if len(own) else 0.0  # noqa: NAR001
        return record.token_times[0] - record.arrival < ttft_slo and p99 < itl_slo

    def pct(values: list[float], q: int) -> float | None:
        """The q-th percentile, rounded, or None for no values."""
        return round(float(np.percentile(a=values, q=q)), 4) if values else None  # noqa: NAR001

    return {
        "requests": len(records),  # noqa: NAR001
        "completed": len(done),  # noqa: NAR001
        "ttft_p50_s": pct(values=ttft, q=50),  # noqa: NAR001
        "ttft_p99_s": pct(values=ttft, q=99),  # noqa: NAR001
        "itl_p99_s": pct(values=gaps, q=99),  # noqa: NAR001
        "output_tokens_per_s": round(sum(len(r.token_times) for r in done) / duration, 1),  # noqa: NAR001
        "goodput_rps": round(sum(meets(record=r) for r in done) / duration, 3),  # noqa: NAR001
    }


def replay_http(
    trace: list[TraceRequest], url: str, ignore_eos: bool = True
) -> tuple[list[RequestRecord], float]:
    """Replay `trace` open-loop against an OpenAI-compatible streaming endpoint.

    Every engine is driven with the same fields: greedy, `max_tokens` from the trace,
    and `ignore_eos` so all of them generate the same number of tokens.
    """
    records = [RequestRecord(arrival=r.arrival) for r in trace]
    start = time.perf_counter()

    def send(index: int) -> None:
        """Wait for the arrival time, stream the completion, stamp each chunk."""
        request = trace[index]
        time.sleep(max(0.0, request.arrival - (time.perf_counter() - start)))  # noqa: NAR001
        body = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": ignore_eos,
        }
        http = urllib.request.Request(
            url=url,
            data=json.dumps(obj=body).encode(),
            headers={"Content-Type": "application/json"},
        )  # noqa: NAR001
        with urllib.request.urlopen(url=http, timeout=600) as response:
            for line in response:
                if line.startswith(b"data: {"):  # noqa: NAR001
                    records[index].token_times.append(time.perf_counter() - start)  # noqa: NAR001

    threads = [threading.Thread(target=send, args=(i,)) for i in range(len(trace))]  # noqa: NAR001
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return records, time.perf_counter() - start


def replay_engine(trace: list[TraceRequest], engine, encode) -> tuple[list[RequestRecord], float]:  # noqa: NAR002
    """Replay `trace` open-loop against an in-process engine, without HTTP.

    The engine steps whenever it has work; a request is added once the wall clock
    passes its arrival time, between steps, as the API's engine thread would see it.
    Same settings as `replay_http`: greedy, `ignore_eos`, `max_tokens` from the trace.

    Args:
        engine: A `serve.engine.Engine`.
        encode: Prompt text to token ids.
    """
    from training.serve.sampler import SamplingParams

    records = [RequestRecord(arrival=r.arrival) for r in trace]
    index_of: dict[str, int] = {}
    start, next_arrival = time.perf_counter(), 0
    while next_arrival < len(trace) or engine.has_work():  # noqa: NAR001
        now = time.perf_counter() - start
        while next_arrival < len(trace) and trace[next_arrival].arrival <= now:  # noqa: NAR001
            request = trace[next_arrival]
            params = SamplingParams(max_tokens=request.max_tokens, ignore_eos=True)
            index_of[engine.add_request(prompt=encode(request.prompt), params=params)] = (  # noqa: NAR001
                next_arrival  # noqa: NAR001
            )
            next_arrival += 1
        if not engine.has_work():
            time.sleep(max(0.0, trace[next_arrival].arrival - now))  # noqa: NAR001
            continue
        for event in engine.step():
            records[index_of[event.request_id]].token_times.append(event.time - start)  # noqa: NAR001
    return records, time.perf_counter() - start
