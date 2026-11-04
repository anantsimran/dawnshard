import pytest
import torch
import torch.nn.functional as F
from dist_helpers import run_distributed
from torch.distributed.tensor import DTensor
from training.model.qwen3 import QWEN3_TINY, Qwen3, Qwen3Block
from training.parallel.fsdp import ParallelismSpec, apply_fsdp


def _loss(model, ids):
    return F.cross_entropy(model(ids[:, :-1]), ids[:, 1:].reshape(-1))


def _check_fsdp(rank, world):
    torch.manual_seed(0)
    reference = Qwen3(config=QWEN3_TINY)
    torch.manual_seed(0)
    model = apply_fsdp(
        model=Qwen3(config=QWEN3_TINY),
        block_type=Qwen3Block,
        spec=ParallelismSpec(param_dtype=None, reduce_dtype=None),
        device_type="cpu",
    )
    assert all(isinstance(p, DTensor) for p in model.parameters())
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(params=reference.parameters(), lr=0.1)
    ids = torch.randint(low=0, high=QWEN3_TINY.vocab_size, size=(4, 9), generator=torch.Generator().manual_seed(3))
    _loss(model, ids[rank * 2 : (rank + 1) * 2]).backward()
    optimizer.step()
    _loss(reference, ids).backward()
    reference_optimizer.step()
    for param, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual=param.full_tensor(), expected=expected, atol=1e-5, rtol=1e-4)


def test_fsdp_shards_every_parameter_and_matches_single_process_sgd():
    try:
        run_distributed(fn=_check_fsdp)
    except Exception as error:  # FSDP2 on CPU/gloo depends on the torch build
        if "NotImplementedError" in str(error) or "not supported" in str(error):
            pytest.skip(f"FSDP2 unavailable on CPU here: {error}")
        raise
