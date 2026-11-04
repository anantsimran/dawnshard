import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from dist_helpers import run_distributed
from torch import nn
from training.model.qwen3 import QWEN3_TINY, Qwen3, Qwen3Attention, Qwen3MLP, parallelize_dtensor, parallelize_tensor_parallel
from training.parallel.tensor import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    gather_sequence,
    scatter_sequence,
    vocab_parallel_cross_entropy,
)


def _x(shape, seed=1):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _check_mlp(rank, world):
    torch.manual_seed(0)
    dense = Qwen3MLP(config=QWEN3_TINY)
    sharded = copy.deepcopy(dense)
    sharded.gate_proj = ColumnParallelLinear(dense=dense.gate_proj, group=None)
    sharded.up_proj = ColumnParallelLinear(dense=dense.up_proj, group=None)
    sharded.down_proj = RowParallelLinear(dense=dense.down_proj, group=None)
    x = _x((2, 5, QWEN3_TINY.hidden_size)).requires_grad_()
    x_dense = x.detach().clone().requires_grad_()
    out, expected = sharded(x), dense(x_dense)
    torch.testing.assert_close(actual=out, expected=expected, atol=1e-5, rtol=1e-5)
    out.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual=x.grad, expected=x_dense.grad, atol=1e-5, rtol=1e-5)
    rows = dense.gate_proj.weight.grad.chunk(world, dim=0)[rank]
    torch.testing.assert_close(actual=sharded.gate_proj.weight.grad, expected=rows, atol=1e-5, rtol=1e-5)


def test_column_then_row_mlp_equals_the_dense_mlp_in_output_and_gradients():
    run_distributed(fn=_check_mlp)


def _check_attention(rank, world):
    torch.manual_seed(0)
    dense = Qwen3Attention(config=QWEN3_TINY)
    sharded = copy.deepcopy(dense)
    for name in ("q_proj", "k_proj", "v_proj"):
        setattr(sharded, name, ColumnParallelLinear(dense=getattr(dense, name), group=None))
    sharded.o_proj = RowParallelLinear(dense=dense.o_proj, group=None)
    sharded.num_heads //= world
    sharded.num_kv_heads //= world
    model = Qwen3(config=QWEN3_TINY)
    cos, sin = model.rope_cos[:7], model.rope_sin[:7]
    x = _x((2, 7, QWEN3_TINY.hidden_size))
    torch.testing.assert_close(actual=sharded(x=x, cos=cos, sin=sin), expected=dense(x=x, cos=cos, sin=sin), atol=1e-5, rtol=1e-5)


def test_head_sharded_gqa_attention_equals_dense_attention():
    run_distributed(fn=_check_attention)


def _check_vocab_ce(rank, world):
    logits = _x((9, 12)).double().requires_grad_()
    target = torch.tensor([0, 3, 11, 6, 5, -100, 7, 1, 10])
    expected = F.cross_entropy(logits, target)
    expected.backward()
    local = logits.detach().chunk(world, dim=1)[rank].clone().requires_grad_()
    loss = vocab_parallel_cross_entropy(logits=local, target=target, vocab_start=rank * 6, group=None)
    loss.backward()
    assert abs(loss.item() - expected.item()) < 1e-10
    torch.testing.assert_close(actual=local.grad, expected=logits.grad.chunk(world, dim=1)[rank], atol=1e-10, rtol=0)


def test_vocab_parallel_cross_entropy_equals_full_cross_entropy_in_loss_and_gradients():
    run_distributed(fn=_check_vocab_ce)


def _check_embedding_and_sequence(rank, world):
    torch.manual_seed(0)
    dense = nn.Embedding(8, 4)
    sharded = VocabParallelEmbedding(dense=dense, group=None)
    ids = torch.tensor([[0, 7, 3, 4]])
    torch.testing.assert_close(actual=sharded(ids), expected=dense(ids))
    x = _x((2, 6, 4), seed=rank)
    shards = scatter_sequence(x=x, group=None)
    assert shards.shape == (2, 3, 4)
    full = sum(_x((2, 6, 4), seed=r) for r in range(world))
    torch.testing.assert_close(actual=gather_sequence(x=shards, group=None), expected=full)


def test_vocab_parallel_embedding_and_sequence_parallel_round_trip():
    run_distributed(fn=_check_embedding_and_sequence)


def _model_loss(model, criterion, ids):
    return criterion(model(ids[:, :-1]), ids[:, 1:].reshape(-1))


def _check_full_model(rank, world, use_dtensor):
    torch.manual_seed(0)
    dense = Qwen3(config=QWEN3_TINY)
    sharded = copy.deepcopy(dense)
    if use_dtensor:
        from torch.distributed.device_mesh import init_device_mesh

        parallelize_dtensor(model=sharded, tp_mesh=init_device_mesh(device_type="cpu", mesh_shape=(world,)))
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = parallelize_tensor_parallel(model=sharded, group=None)
    ids = torch.randint(0, QWEN3_TINY.vocab_size, (2, 9), generator=torch.Generator().manual_seed(4))
    loss = _model_loss(sharded, criterion, ids)
    expected = _model_loss(dense, nn.CrossEntropyLoss(), ids)
    assert abs(loss.item() - expected.item()) < 1e-5
    loss.backward()
    expected.backward()
    norm_grad = sharded.layers[0].input_layernorm.weight.grad
    torch.testing.assert_close(actual=norm_grad, expected=dense.layers[0].input_layernorm.weight.grad, atol=1e-5, rtol=1e-4)


def test_hand_tensor_parallel_qwen3_matches_dense_loss_and_replicated_gradients():
    run_distributed(fn=_check_full_model, args=(False,))


def test_dtensor_plan_matches_dense_loss():
    run_distributed(fn=_check_full_model, args=(True,))
