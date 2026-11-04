import torch
import torch.distributed as dist
from dist_helpers import run_distributed
from torch import nn
from training.parallel import ddp
from training.parallel.ddp import BucketedDDP, NaiveDDP, make_buckets


def _model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))


def _batch():
    generator = torch.Generator().manual_seed(123)
    return torch.randn(8, 4, generator=generator), torch.randn(8, 3, generator=generator)


def _reference_grads():
    model = _model(seed=0)
    x, y = _batch()
    nn.functional.mse_loss(model(x), y).backward()
    return [p.grad for p in model.parameters()]


def _check_parity(rank, world, wrapper_name):
    # Rank 1 starts from different weights: the wrapper must broadcast rank 0's.
    wrapper = {"naive": NaiveDDP, "bucketed": lambda module: BucketedDDP(module=module, bucket_mb=1e-4)}[wrapper_name]
    model = wrapper(_model(seed=rank))
    x, y = _batch()
    shard = slice(rank * 4, (rank + 1) * 4)
    nn.functional.mse_loss(model(x[shard]), y[shard]).backward()
    for grad, expected in zip([p.grad for p in model.module.parameters()], _reference_grads()):
        torch.testing.assert_close(actual=grad, expected=expected)


def test_naive_ddp_gradients_equal_single_process_on_the_combined_batch():
    run_distributed(fn=_check_parity, args=("naive",))


def test_bucketed_ddp_gradients_equal_single_process_on_the_combined_batch():
    run_distributed(fn=_check_parity, args=("bucketed",))


def _check_buckets_and_no_sync(rank, world):
    module = _model(seed=0)
    model = BucketedDDP(module=module, bucket_mb=1e-4)
    flat = [p for bucket in model.buckets for p in bucket]
    assert [id(p) for p in flat] == [id(p) for p in list(module.parameters())[::-1]]
    calls = []
    original = ddp.dist.all_reduce
    ddp.dist.all_reduce = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        x, y = _batch()
        with model.no_sync():
            nn.functional.mse_loss(model(x), y).backward()
        assert calls == []
        nn.functional.mse_loss(model(x), y).backward()
        assert len(calls) == len(model.buckets)
    finally:
        ddp.dist.all_reduce = original


def test_buckets_cover_every_parameter_in_reverse_order_and_no_sync_skips_collectives():
    run_distributed(fn=_check_buckets_and_no_sync)


def test_make_buckets_respects_the_size_cap():
    params = [nn.Parameter(torch.zeros(n)) for n in (10, 10, 30, 5)]
    buckets = make_buckets(params=params, bucket_bytes=80)
    assert [len(b) for b in buckets] == [2, 1, 1]
