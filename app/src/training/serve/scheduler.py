"""Iteration-level scheduling: which requests run this step, and how many tokens each.

Every engine step the scheduler builds one ragged batch under two limits, a token
budget (`max_tokens_per_step`) and a sequence cap (`max_num_seqs`). Prefill and decode
are the same thing here, "tokens this request still needs computed": a running
request in decode needs 1, a request in prefill needs the rest of its prompt, cut to
`chunk_size` so one long prompt cannot stall every decode behind it (chunked prefill).

```
step:   running requests (decode, 1 token each, oldest first)
        └─► then waiting requests, oldest first, while budget, seq cap, and blocks allow
             └─► allocate blocks (prefix-cache hits skip those tokens)
out:    [(request, num_tokens), ...]      one forward over all of them
```

Requests join and leave between steps (continuous batching), so a finished request's
slot is refilled on the next step instead of idling until the longest request in a
static batch ends. `static_batching=True` turns that off, as the baseline.

When a decode needs a new block and none is free, the most recently admitted running
request is preempted by recompute: its blocks are freed, its progress reset, and it
goes back to the front of the waiting queue with its generated tokens kept, to be
re-prefilled later. Recompute costs FLOPs; swapping to host memory would cost PCIe
bandwidth and a second memory pool.
"""

import enum
from collections import deque
from dataclasses import dataclass, field

from training.serve.block_manager import BlockManager


class State(enum.Enum):
    """Where a request is in its life."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    """One generation request and its progress.

    Attributes:
        computed: Tokens of `tokens` whose KV is in the cache.
    """

    request_id: str
    prompt: list[int]
    max_new_tokens: int
    output: list[int] = field(default_factory=list)
    state: State = State.WAITING
    computed: int = 0
    preemptions: int = 0

    @property
    def tokens(self) -> list[int]:
        """Prompt plus everything generated so far."""
        return self.prompt + self.output

    @property
    def remaining(self) -> int:
        """Tokens still to compute before the next one can be sampled."""
        return len(self.tokens) - self.computed  # noqa: NAR001


@dataclass(frozen=True)
class SchedulerConfig:
    """The step's limits."""

    max_num_seqs: int = 64
    max_tokens_per_step: int = 512
    chunk_size: int = 256
    static_batching: bool = False


class Scheduler:
    """Picks each step's batch; owns the waiting queue and the running list."""

    def __init__(self, config: SchedulerConfig, blocks: BlockManager) -> None:
        """Start empty."""
        self.config, self.blocks = config, blocks
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.preemptions = 0

    def add(self, request: Request) -> None:
        """Queue a new request."""
        self.waiting.append(request)  # noqa: NAR001

    def has_work(self) -> bool:
        """Whether any request is waiting or running."""
        return bool(self.waiting or self.running)  # noqa: NAR001

    def _preempt_latest(self) -> None:
        """Free the most recently admitted running request and requeue it first."""
        victim = self.running.pop()
        self.blocks.free_sequence(seq_id=victim.request_id)
        victim.state, victim.computed = State.WAITING, 0
        victim.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(victim)  # noqa: NAR001

    def schedule(self) -> list[tuple[Request, int]]:
        """Return this step's `(request, num_tokens)` pairs within every limit.

        Side effects:
            Admits, allocates, and may preempt requests.
        """
        budget, batch = self.config.max_tokens_per_step, []
        for request in list(self.running):  # noqa: NAR001
            if request not in self.running:
                continue  # preempted while making room for an earlier one
            want = min(request.remaining, self.config.chunk_size, budget)  # noqa: NAR001
            while want and not self.blocks.can_append(
                seq_id=request.request_id, num_tokens=request.computed + want
            ):
                if self.running[-1] is request:
                    self._preempt_latest()
                    want = 0
                else:
                    self._preempt_latest()
            if want:
                self.blocks.append(seq_id=request.request_id, num_tokens=request.computed + want)
                batch.append((request, want))  # noqa: NAR001
                budget -= want
        admit = not (self.config.static_batching and self.running)
        while admit and self.waiting and budget and len(self.running) < self.config.max_num_seqs:  # noqa: NAR001
            request = self.waiting[0]
            try:
                cached = self.blocks.allocate(seq_id=request.request_id, prompt=request.tokens)
            except MemoryError:
                break
            self.waiting.popleft()
            request.state, request.computed = State.RUNNING, min(cached, len(request.tokens) - 1)  # noqa: NAR001
            want = min(request.remaining, self.config.chunk_size, budget)  # noqa: NAR001
            self.running.append(request)  # noqa: NAR001
            batch.append((request, want))  # noqa: NAR001
            budget -= want
        return batch

    def finish(self, request: Request) -> None:
        """Retire `request` and free its blocks.

        Side effects:
            Marks it finished and removes it from `running`.
        """
        request.state = State.FINISHED
        self.running.remove(request)  # noqa: NAR001
        self.blocks.free_sequence(seq_id=request.request_id)
