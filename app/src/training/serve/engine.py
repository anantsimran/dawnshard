"""The engine loop: schedule, run one forward, sample, update; repeat.

```
add_request ──► Scheduler.waiting
step():  batch = scheduler.schedule()            continuous batching, budgets, preemption
         logits = runner.execute(model, pool, …)  one ragged forward over paged KV
         for each request whose prompt is now fully computed:
             token = sample(logits[row], params)   per-request params and generator
             text  = detokenizer → stop checker    UTF-8-safe, stop strings across chunks
             finish on EOS, max_tokens, or a stop string; free its blocks
         return events (request id, token, text, finished, timestamp)
```

A request in prefill whose chunk does not reach the end of its prompt gets no token
this step: its logits are not the next token's. Full blocks are committed to the
prefix cache after every step, so a request arriving later reuses them even while
the first request is still running.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Optional

import torch
from training.serve.block_manager import BlockManager
from training.serve.runner import KVPool, PagedModel, build_batch, execute
from training.serve.sampler import IncrementalDetokenizer, SamplingParams, StopChecker, sample
from training.serve.scheduler import Request, Scheduler, SchedulerConfig


@dataclass(frozen=True)
class Event:
    """One step's output for one request."""

    request_id: str
    token: int
    text: str
    finished: bool
    time: float


@dataclass
class _Tracked:
    """Per-request sampling state the scheduler does not need."""

    params: SamplingParams
    generator: Optional[torch.Generator]
    detokenizer: IncrementalDetokenizer
    stops: StopChecker


class Engine:
    """A single-model inference engine over a paged KV cache."""

    def __init__(
        self,
        model: PagedModel,
        pool: KVPool,
        blocks: BlockManager,
        config: SchedulerConfig,
        eos_id: int,
        token_bytes: Callable[[int], bytes],
        max_model_len: int,
    ) -> None:
        """Wire the model, the pool, the block manager, and the scheduler together.

        Args:
            max_model_len: Longest prompt plus output the model's positions cover.
        """
        self.model, self.pool, self.blocks, self.eos_id = model, pool, blocks, eos_id
        self.max_model_len = max_model_len
        self.scheduler = Scheduler(config=config, blocks=blocks)
        self.token_bytes = token_bytes
        self.tracked: dict[str, tuple[Request, _Tracked]] = {}
        self.ids = count()

    def add_request(
        self, prompt: list[int], params: SamplingParams, request_id: Optional[str] = None
    ) -> str:
        """Queue a prompt; return its id.

        Raises:
            ValueError: If prompt plus `max_tokens` is longer than the model's positions,
                or needs more blocks than the whole pool holds: such a request could never
                be admitted and would wait forever.
        """
        if len(prompt) + params.max_tokens > self.max_model_len:  # noqa: NAR001
            raise ValueError(f"request longer than max_model_len={self.max_model_len}")  # noqa: NAR001
        total = len(self.blocks.refcount)  # noqa: NAR001
        if self.blocks.blocks_needed(num_tokens=len(prompt) + params.max_tokens) > total:  # noqa: NAR001
            raise ValueError(f"request needs more than the pool's {total} blocks")  # noqa: NAR001
        request_id = request_id or f"req-{next(self.ids)}"  # noqa: NAR001
        request = Request(request_id=request_id, prompt=prompt, max_new_tokens=params.max_tokens)
        generator = None if params.seed is None else torch.Generator().manual_seed(params.seed)  # noqa: NAR001
        state = _Tracked(
            params=params,
            generator=generator,
            detokenizer=IncrementalDetokenizer(token_bytes=self.token_bytes),
            stops=StopChecker(stops=params.stop),
        )
        self.tracked[request_id] = (request, state)
        self.scheduler.add(request=request)
        return request_id

    def has_work(self) -> bool:
        """Whether anything is waiting or running."""
        return self.scheduler.has_work()

    @torch.inference_mode()
    def step(self) -> list[Event]:
        """Run one engine step; return the tokens it produced."""
        batch = self.scheduler.schedule()
        if not batch:
            return []
        items = [(r.tokens, r.computed, n, self.blocks.tables[r.request_id]) for r, n in batch]
        logits = execute(
            model=self.model,
            pool=self.pool,
            batch=build_batch(items=items, block_size=self.blocks.block_size),
        )
        events, now = [], time.perf_counter()
        for row, (request, num_new) in enumerate(batch):  # noqa: NAR001
            request.computed += num_new
            self.blocks.commit_full_blocks(
                seq_id=request.request_id, tokens=request.tokens[: request.computed]
            )
            if request.remaining == 0:
                events.append(self._emit(request=request, logits=logits[row], now=now))  # noqa: NAR001
        return events

    def _emit(self, request: Request, logits: torch.Tensor, now: float) -> Event:
        """Sample, detokenize, check stops, and retire `request` if it is done."""
        _, state = self.tracked[request.request_id]
        token = sample(logits=logits, params=state.params, generator=state.generator)
        request.output.append(token)  # noqa: NAR001
        text = state.stops.feed(text=state.detokenizer.push(token=token))
        eos = token == self.eos_id and not state.params.ignore_eos
        done = eos or state.stops.stopped or len(request.output) >= request.max_new_tokens  # noqa: NAR001
        if done:
            if not state.stops.stopped:
                text += state.stops.feed(text=state.detokenizer.flush()) + state.stops.flush()
            self.scheduler.finish(request=request)
            del self.tracked[request.request_id]
        return Event(request_id=request.request_id, token=token, text=text, finished=done, time=now)

    def generate(self, prompts: list[list[int]], params: SamplingParams) -> list[list[int]]:
        """Run every prompt to completion offline; return each one's output ids."""
        requests = [self.add_request(prompt=p, params=params) for p in prompts]
        outputs: dict[str, list[int]] = {r: [] for r in requests}
        while self.has_work():
            for event in self.step():
                outputs[event.request_id].append(event.token)  # noqa: NAR001
        return [outputs[r] for r in requests]
