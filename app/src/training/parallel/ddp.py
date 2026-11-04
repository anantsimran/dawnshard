"""Data parallelism by hand: naive, then bucketed with communication under backward.

Both wrappers broadcast rank 0's weights at construction, so every replica starts
equal, and make `loss.backward()` return with every gradient already averaged across
ranks, so `train_step` needs no change. They differ in when the communication runs:

```
naive:     backward ──────────────────────────► all_reduce p1, p2, ... pN ──► step
                                                (N small collectives, after backward)

bucketed:  backward ──┬─────────┬─────────┬───► wait ──► step
                      ▼         ▼         ▼
                 all_reduce  all_reduce  all_reduce     one per ~25 MB bucket,
                 bucket 0    bucket 1    bucket 2       async, while backward runs
```

Buckets hold parameters in reverse registration order, roughly the order backward
produces their gradients, so the first bucket fills while later layers are still in
backward. Each gradient's post-accumulate hook counts it into its bucket; a full
bucket is flattened and all-reduced asynchronously. A callback queued on the
autograd engine runs once backward finishes: it waits for every handle, launches
any bucket that never filled (a parameter with no gradient), and copies the averaged
values back into `.grad`.

`no_sync()` turns the hooks off, so gradients accumulate locally over micro-batches
and only the last backward of a group communicates. `train_step` enters it on every
micro-batch that does not step.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import chain

import torch
import torch.distributed as dist
from torch import nn

BUCKET_MB = 25


def broadcast_parameters(module: nn.Module, src: int = 0) -> None:
    """Copy rank `src`'s parameters and buffers to every rank, in place.

    Side effects:
        One broadcast per tensor.
    """
    for tensor in chain(module.parameters(), module.buffers()):  # noqa: NAR001
        dist.broadcast(tensor=tensor.data, src=src)


def _queue_after_backward(callback) -> None:  # noqa: NAR002
    """Run `callback` once the current backward pass has finished."""
    torch.autograd.Variable._execution_engine.queue_callback(callback)  # noqa: NAR001


class NaiveDDP(nn.Module):
    """Data parallel by one all-reduce per gradient, after backward.

    Wraps `module`; in and out are whatever `module` takes and returns. Correct and
    slow: N parameters cost N collectives, each latency-bound, and none overlaps
    compute. It exists to be measured against `BucketedDDP`.

    ```
    forward ──► module(*args) ──► loss.backward() ──► _all_reduce_grads ──► .grad = mean
                                       │                  (queued callback)
                                       └─ post-accumulate hook on each parameter
    ```

    The first gradient hook of a backward queues the callback; the callback averages
    every `.grad`. `require_sync` is the flag `no_sync` clears.

    `forward` holds no state; a backward queues one callback.
    """

    def __init__(self, module: nn.Module) -> None:
        """Broadcast rank 0's weights and hook every trainable parameter.

        Side effects:
            Overwrites `module`'s parameters on ranks other than 0.
        """
        super().__init__()
        self.module = module
        self.require_sync = True
        self.queued = False
        broadcast_parameters(module=module)
        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook=self._on_grad)

    def _on_grad(self, param: torch.Tensor) -> None:
        """Queue the all-reduce once per backward, when syncing."""
        if self.require_sync and not self.queued:
            self.queued = True
            _queue_after_backward(callback=self._all_reduce_grads)

    def _all_reduce_grads(self) -> None:
        """Average every gradient across ranks, one collective each."""
        self.queued = False
        world = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(tensor=param.grad)
                param.grad.div_(world)  # noqa: NAR001

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Skip communication for backwards inside the block."""
        previous, self.require_sync = self.require_sync, False
        try:
            yield
        finally:
            self.require_sync = previous

    def forward(self, *args, **kwargs):  # noqa: NAR002
        return self.module(*args, **kwargs)


def make_buckets(params: list[nn.Parameter], bucket_bytes: int) -> list[list[nn.Parameter]]:
    """Group `params`, in order, into consecutive buckets of at most `bucket_bytes`.

    A parameter larger than a bucket gets a bucket of its own.
    """
    buckets: list[list[nn.Parameter]] = [[]]
    size = 0
    for param in params:
        nbytes = param.numel() * param.element_size()
        if buckets[-1] and size + nbytes > bucket_bytes:
            buckets.append([])  # noqa: NAR001
            size = 0
        buckets[-1].append(param)  # noqa: NAR001
        size += nbytes
    return buckets if buckets[0] else []


class BucketedDDP(nn.Module):
    """Data parallel by bucketed, asynchronous all-reduces overlapped with backward.

    Wraps `module`; in and out are whatever `module` takes and returns. See the
    module docstring for the timeline.

    ```
    param.grad ready ──► ready[bucket] += 1 ──(bucket full)──► flatten ──► all_reduce(async)
                                                                              │ handle
    end of backward (queued callback) ──► wait every handle ──► / world ──► copy into .grad
    ```

    A gradient that arrives late (a tied weight used twice, a parameter used only by
    an early layer) only delays its bucket; correctness never depends on the order,
    because the callback finishes whatever is left.

    `forward` holds no state; a backward fills `ready` and `handles`, and the
    callback resets them.
    """

    def __init__(self, module: nn.Module, bucket_mb: float = BUCKET_MB) -> None:
        """Broadcast rank 0's weights, bucket the parameters, and hook each one.

        Side effects:
            Overwrites `module`'s parameters on ranks other than 0.
        """
        super().__init__()
        self.module = module
        self.require_sync = True
        self.queued = False
        broadcast_parameters(module=module)
        trainable = [p for p in module.parameters() if p.requires_grad]
        self.buckets = make_buckets(params=trainable[::-1], bucket_bytes=int(bucket_mb * 2**20))  # noqa: NAR001
        self.bucket_of = {id(p): i for i, bucket in enumerate(self.buckets) for p in bucket}  # noqa: NAR001
        self.ready = [0] * len(self.buckets)  # noqa: NAR001
        self.handles: dict[int, tuple[dist.Work, torch.Tensor]] = {}
        for param in trainable:
            param.register_post_accumulate_grad_hook(hook=self._on_grad)

    def _launch(self, index: int) -> None:
        """Flatten bucket `index`'s gradients (zeros where none) and all-reduce them async."""
        flat = torch.cat(  # noqa: NAR001
            [
                (p.grad if p.grad is not None else torch.zeros_like(input=p)).flatten()
                for p in self.buckets[index]
            ]
        )
        work = dist.all_reduce(tensor=flat, async_op=True)
        assert work is not None
        self.handles[index] = (work, flat)

    def _on_grad(self, param: torch.Tensor) -> None:
        """Count `param` into its bucket and launch the bucket when it is full."""
        if not self.require_sync:
            return
        if not self.queued:
            self.queued = True
            _queue_after_backward(callback=self._finish)
        index = self.bucket_of[id(param)]  # noqa: NAR001
        self.ready[index] += 1
        if self.ready[index] == len(self.buckets[index]):  # noqa: NAR001
            self._launch(index=index)

    def _finish(self) -> None:
        """Launch unfilled buckets, wait for all, and write the averages into `.grad`."""
        for index in range(len(self.buckets)):  # noqa: NAR001
            if index not in self.handles:
                self._launch(index=index)
        world = dist.get_world_size()
        for index, (handle, flat) in self.handles.items():
            handle.wait()
            flat.div_(world)  # noqa: NAR001
            offset = 0
            for param in self.buckets[index]:
                values = flat[offset : offset + param.numel()].view_as(other=param)
                param.grad = values.clone() if param.grad is None else param.grad.copy_(values)  # noqa: NAR001
                offset += param.numel()
        self.handles, self.ready, self.queued = {}, [0] * len(self.buckets), False  # noqa: NAR001

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Skip communication for backwards inside the block."""
        previous, self.require_sync = self.require_sync, False
        try:
            yield
        finally:
            self.require_sync = previous

    def forward(self, *args, **kwargs):  # noqa: NAR002
        return self.module(*args, **kwargs)
