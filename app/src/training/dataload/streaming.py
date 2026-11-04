"""
streaming.py
Stream packed token windows from shards so that every rank and worker sees its own
slice of one global order, and a run can stop at any step and resume exactly.

Three rules make resume exact, even on a different number of GPUs:

1. **The order is a pure function.** Sample `i` is token window `i` of the shard
   stream. Epoch `e`'s order is a permutation of the sample indices drawn from a
   generator seeded with `(seed, e)` alone; there is no RNG state anywhere else, in
   the main process or in any worker.
1. **Global batch k is `order[k*B : (k+1)*B]`,** whatever the world size. Rank `r`
   of `R` takes the `r`-th contiguous `B / R` of it; DataLoader worker `w` of `W`
   builds the batches for steps `w, w + W, w + 2W, ...`, which the DataLoader's
   round-robin hands back in step order.
1. **The loader state is two integers:** the epoch and the samples consumed globally.
   Resuming sets the next step to `samples_consumed // B`. A new world size changes
   only the per-rank split, so the global batches are identical, as long as `B`
   stays fixed.

Shards are read from a local directory through memmaps, or from an `http(s)://`
prefix (a public or presigned bucket URL) with byte-range GETs, fetched in parallel
and cached locally per shard when a cache directory is given.

`shuffle_buffer` is the other way to shuffle a stream: a bounded buffer that emits a
random element as each new one arrives. It needs no index, but an item moves at most
about `size` positions, and its state is the buffer itself, which is why the
deterministic order above uses a global permutation instead.
"""

import json
import random
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from training.dataload.shards import MANIFEST
from training.dataload.tokens import TOKEN_DTYPE

type Batch = tuple[torch.Tensor, torch.Tensor]


def global_order(num_samples: int, seed: int, epoch: int) -> np.ndarray:
    """Return epoch `epoch`'s permutation of `range(num_samples)`; depends on nothing else."""
    return np.random.default_rng(seed=[seed, epoch]).permutation(num_samples)  # noqa: NAR001


def rank_slice(global_batch: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    """Return rank `rank`'s contiguous share of one global batch.

    Raises:
        ValueError: If the global batch does not divide by the world size.
    """
    if len(global_batch) % world_size:  # noqa: NAR001
        raise ValueError(  # noqa: NAR001
            f"global batch {len(global_batch)} not divisible by world size {world_size}"  # noqa: NAR001
        )  # noqa: NAR001
    per_rank = len(global_batch) // world_size  # noqa: NAR001
    return global_batch[rank * per_rank : (rank + 1) * per_rank]


def worker_steps(start_step: int, num_steps: int, worker: int, num_workers: int) -> range:
    """Steps worker `worker` of `num_workers` builds, from `start_step` on."""
    return range(start_step + worker, num_steps, max(num_workers, 1))  # noqa: NAR001


def shuffle_buffer(items: Iterable, size: int, seed: int) -> Iterator:
    """Yield every item of `items` exactly once, shuffled through a buffer of `size`."""
    rng = random.Random(x=seed)  # noqa: NAR001
    buffer: list = []
    for item in items:
        buffer.append(item)  # noqa: NAR001
        if len(buffer) > size:  # noqa: NAR001
            yield buffer.pop(rng.randrange(len(buffer)))  # noqa: NAR001
    rng.shuffle(buffer)  # noqa: NAR001
    yield from buffer


class TokenSource:
    """The shard stream as one flat token array, read from disk or over HTTP."""

    def __init__(self, root: str, cache_dir: Optional[Path] = None, max_parallel: int = 8) -> None:
        """Read the manifest under `root`, a directory or an `http(s)://` prefix.

        Args:
            cache_dir: For HTTP roots, where whole shards are cached after first use;
                None reads every window with a byte-range GET instead.
            max_parallel: Concurrent GETs when prefetching.
        """
        self.root, self.cache_dir, self.max_parallel = root.rstrip("/"), cache_dir, max_parallel  # noqa: NAR001
        manifest = json.loads(s=self._read_bytes(name=MANIFEST))
        self.names = [entry["name"] for entry in manifest["shards"]]
        self.offsets = np.cumsum([0] + [entry["num_tokens"] for entry in manifest["shards"]])  # noqa: NAR001
        self.remote = self.root.startswith(("http://", "https://"))  # noqa: NAR001

    @property
    def num_tokens(self) -> int:
        """Tokens in the whole stream."""
        return int(self.offsets[-1])  # noqa: NAR001

    def _read_bytes(
        self, name: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> bytes:
        """Return bytes `[start, end)` of file `name` under the root (whole file when None)."""
        if not self.root.startswith(("http://", "https://")):  # noqa: NAR001
            with open(file=Path(self.root) / name, mode="rb") as handle:  # noqa: NAR001
                handle.seek(start or 0)  # noqa: NAR001
                return handle.read() if end is None else handle.read(end - (start or 0))  # noqa: NAR001
        headers = {} if start is None else {"Range": f"bytes={start}-{end - 1}"}  # type: ignore[operator]
        request = urllib.request.Request(url=f"{self.root}/{name}", headers=headers)
        with urllib.request.urlopen(url=request, timeout=60) as response:
            return response.read()

    def _shard(self, index: int) -> np.ndarray:
        """Shard `index` as a memmap: local, or downloaded once into the cache."""
        name = self.names[index]
        if self.remote:
            assert self.cache_dir is not None
            path = self.cache_dir / name
            if not path.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(suffix=".part")
                partial.write_bytes(data=self._read_bytes(name=name))
                partial.rename(target=path)
        else:
            path = Path(self.root) / name  # noqa: NAR001
        return np.memmap(filename=path, dtype=TOKEN_DTYPE, mode="r")

    def prefetch(self) -> None:
        """Download every shard into the cache in parallel. A no-op for local roots."""
        if self.remote and self.cache_dir is not None:
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                list(pool.map(lambda index: self._shard(index=index), range(len(self.names))))  # noqa: NAR001

    def read(self, start: int, length: int) -> np.ndarray:
        """Return tokens `[start, start + length)` of the stream as int64, across shards."""
        pieces, position, end = [], start, start + length
        while position < end:
            index = int(np.searchsorted(a=self.offsets, v=position, side="right")) - 1  # noqa: NAR001
            local, stop = (
                position - int(self.offsets[index]),  # noqa: NAR001
                min(end, int(self.offsets[index + 1])),  # noqa: NAR001
            )  # noqa: NAR001
            if self.remote and self.cache_dir is None:
                raw = self._read_bytes(
                    name=self.names[index], start=4 * local, end=4 * (local + stop - position)
                )
                pieces.append(np.frombuffer(buffer=raw, dtype=TOKEN_DTYPE))  # noqa: NAR001
            else:
                pieces.append(self._shard(index=index)[local : local + stop - position])  # noqa: NAR001
            position = stop
        return np.concatenate(pieces).astype(np.int64)  # noqa: NAR001


class StreamingTokens(IterableDataset[Batch]):
    """Per-rank batches of `(X (b, L), Y (b*L,))` in the deterministic global order.

    Yields whole batches, so its DataLoader uses `batch_size=None`. `epoch` and
    `start_step` say where to begin; `StatefulLoader` sets them.
    """

    def __init__(
        self,
        source: TokenSource,
        seq_len: int,
        global_batch: int,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        """Describe the slice of the stream this rank reads; nothing is read yet."""
        self.source, self.seq_len, self.global_batch, self.seed = (
            source,
            seq_len,
            global_batch,
            seed,
        )
        self.rank, self.world_size = rank, world_size
        self.num_samples = (source.num_tokens - 1) // seq_len
        self.epoch, self.start_step = 0, 0
        self.cached_order: tuple[int, np.ndarray] = (-1, np.empty(shape=0))

    @property
    def steps_per_epoch(self) -> int:
        """Whole global batches per epoch; the tail that does not fill one is dropped."""
        return self.num_samples // self.global_batch

    def batch(self, epoch: int, step: int) -> Batch:
        """Return this rank's share of global batch `step` of `epoch`."""
        if self.cached_order[0] != epoch:
            self.cached_order = (
                epoch,
                global_order(num_samples=self.num_samples, seed=self.seed, epoch=epoch),
            )
        order = self.cached_order[1]
        indices = rank_slice(
            global_batch=order[step * self.global_batch : (step + 1) * self.global_batch],
            rank=self.rank,
            world_size=self.world_size,
        )
        rows = torch.from_numpy(  # noqa: NAR001
            np.stack(  # noqa: NAR001
                [
                    self.source.read(start=int(i) * self.seq_len, length=self.seq_len + 1)  # noqa: NAR001
                    for i in indices
                ]
            )
        )  # noqa: NAR001
        return rows[:, :-1], rows[:, 1:].reshape(-1)  # noqa: NAR001

    def __iter__(self) -> Iterator[Batch]:  # noqa: NAR002
        info = get_worker_info()
        worker, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        for step in worker_steps(
            start_step=self.start_step,
            num_steps=self.steps_per_epoch,
            worker=worker,
            num_workers=num_workers,
        ):
            yield self.batch(epoch=self.epoch, step=step)


class StatefulLoader:
    """A DataLoader over `StreamingTokens` whose position is two integers.

    Counts a batch as consumed when it is handed out, so a checkpoint taken after
    training on batch k resumes at batch k + 1. `state_dict` and `load_state_dict`
    make it checkpointable through `TrainState.loader`.
    """

    def __init__(
        self, dataset: StreamingTokens, num_workers: int = 0, pin_memory: bool = False
    ) -> None:
        """Wrap `dataset`; start at epoch 0, nothing consumed."""
        self.dataset, self.num_workers, self.pin_memory = dataset, num_workers, pin_memory
        self.epoch, self.samples_consumed = 0, 0

    def __len__(self) -> int:
        """Batches left in the current epoch."""
        return self.dataset.steps_per_epoch - self.samples_consumed // self.dataset.global_batch

    def __iter__(self) -> Iterator[Batch]:  # noqa: NAR002
        self.dataset.epoch = self.epoch
        self.dataset.start_step = self.samples_consumed // self.dataset.global_batch
        loader = DataLoader(
            dataset=self.dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        for batch in loader:
            self.samples_consumed += self.dataset.global_batch
            yield batch
        self.epoch, self.samples_consumed = self.epoch + 1, 0

    def state_dict(self) -> dict[str, int]:
        """The loader's whole position."""
        return {"epoch": self.epoch, "samples_consumed": self.samples_consumed}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Resume at `state_dict`'s position; the world size may differ from the saver's."""
        self.epoch, self.samples_consumed = state_dict["epoch"], state_dict["samples_consumed"]
