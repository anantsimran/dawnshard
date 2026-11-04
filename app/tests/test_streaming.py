import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import torch
from training.dataload.shards import write_manifest, write_shards
from training.dataload.streaming import (
    StatefulLoader,
    StreamingTokens,
    TokenSource,
    global_order,
    rank_slice,
    shuffle_buffer,
    worker_steps,
)


@pytest.mark.parametrize("world", range(1, 9))
@pytest.mark.parametrize("workers", range(0, 5))
def test_ranks_and_workers_partition_every_index_exactly_once(world, workers):
    global_batch, num_steps = 8 * 7 * 5 * 3, 6  # divisible by every world size 1-8
    order = global_order(num_samples=global_batch * num_steps, seed=1, epoch=2)
    seen = []
    for rank in range(world):
        for worker in range(max(workers, 1)):
            for step in worker_steps(start_step=0, num_steps=num_steps, worker=worker, num_workers=workers):
                seen += rank_slice(global_batch=order[step * global_batch : (step + 1) * global_batch], rank=rank, world_size=world).tolist()
    assert sorted(seen) == list(range(global_batch * num_steps))


def test_global_order_is_a_deterministic_permutation_per_epoch():
    first = global_order(num_samples=100, seed=3, epoch=0)
    assert sorted(first.tolist()) == list(range(100))
    assert (first == global_order(num_samples=100, seed=3, epoch=0)).all()
    assert not (first == global_order(num_samples=100, seed=3, epoch=1)).all()


def test_shuffle_buffer_emits_every_item_once():
    out = list(shuffle_buffer(items=range(1000), size=37, seed=0))
    assert sorted(out) == list(range(1000))
    assert out != list(range(1000))


def _shards(tmp_path, num_tokens=4000):
    rng = np.random.default_rng(seed=0)
    documents = [rng.integers(low=1, high=70000, size=n).tolist() for n in rng.integers(low=5, high=40, size=num_tokens // 20)]
    write_shards(documents=documents, eos_id=0, out_dir=tmp_path, prefix="t", tokens_per_shard=500)
    write_manifest(out_dir=tmp_path)
    return TokenSource(root=str(tmp_path))


def test_reads_cross_shard_boundaries_and_keep_large_ids(tmp_path):
    source = _shards(tmp_path=tmp_path)
    stream = np.concatenate([np.fromfile(tmp_path / name, dtype=np.uint32) for name in source.names]).astype(np.int64)
    assert (source.read(start=490, length=30) == stream[490:520]).all()
    assert stream.max() > 65535


def _global_batches(source, world, state, steps):
    loaders = []
    for rank in range(world):
        loader = StatefulLoader(dataset=StreamingTokens(source=source, seq_len=16, global_batch=8, seed=5, rank=rank, world_size=world))
        if state is not None:
            loader.load_state_dict(state_dict=state)
        loaders.append(iter(loader))
    return [torch.cat([next(it)[0] for it in loaders]) for _ in range(steps)]


def test_loader_state_round_trips_and_resumes_on_another_world_size(tmp_path):
    source = _shards(tmp_path=tmp_path)
    reference = _global_batches(source=source, world=4, state=None, steps=12)
    loader = StatefulLoader(dataset=StreamingTokens(source=source, seq_len=16, global_batch=8, seed=5, rank=0, world_size=4), num_workers=2)
    for _ in zip(range(7), loader):
        pass
    state = loader.state_dict()
    assert state == {"epoch": 0, "samples_consumed": 7 * 8}
    for world in (4, 2, 1):
        resumed = _global_batches(source=source, world=world, state=state, steps=5)
        for got, expected in zip(resumed, reference[7:]):
            assert torch.equal(got, expected)


def test_y_is_x_shifted_by_one(tmp_path):
    source = _shards(tmp_path=tmp_path)
    x, y = StreamingTokens(source=source, seq_len=16, global_batch=4, seed=0, rank=0, world_size=1).batch(epoch=0, step=0)
    assert x.shape == (4, 16) and y.shape == (64,)
    assert torch.equal(x[:, 1:], y.view(4, 16)[:, :-1])


class _RangeHandler(SimpleHTTPRequestHandler):
    """Serve files with byte-range support, as an object store does."""

    def do_GET(self):  # noqa: N802
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        header = self.headers.get("Range")
        status = 200
        if header:
            start, end = (int(v) for v in header.removeprefix("bytes=").split("-"))
            data, status = data[start : end + 1], 206
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("cache", [False, True])
def test_http_source_reads_the_same_tokens_with_ranges_or_a_cache(tmp_path, cache):
    local = _shards(tmp_path=tmp_path / "shards")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_RangeHandler, directory=str(tmp_path / "shards")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        remote = TokenSource(root=f"http://127.0.0.1:{server.server_port}", cache_dir=tmp_path / "cache" if cache else None)
        remote.prefetch()
        assert (remote.read(start=480, length=60) == local.read(start=480, length=60)).all()
    finally:
        server.shutdown()
