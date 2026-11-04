import pytest
from dist_helpers import run_distributed
from training.parallel.mesh import MESH_DIMS, ParallelismSpec, build_mesh


def test_spec_rejects_a_tp_degree_that_does_not_divide_the_model():
    with pytest.raises(ValueError):
        ParallelismSpec(tp=3).validate(world_size=3, num_heads=16, num_kv_heads=8, ffn=6144, vocab=151936)
    with pytest.raises(ValueError):
        ParallelismSpec(tp=2).validate(world_size=4, num_heads=16, num_kv_heads=8, ffn=6144, vocab=151936)
    ParallelismSpec(dp_shard=2, tp=2).validate(world_size=4, num_heads=16, num_kv_heads=8, ffn=6144, vocab=151936)


def _check_mesh(rank, world):
    mesh = build_mesh(spec=ParallelismSpec(dp_shard=1, tp=2), device_type="cpu")
    assert mesh.mesh_dim_names == MESH_DIMS
    assert mesh["tp"].mesh.tolist() == [0, 1]  # TP holds consecutive ranks


def test_tp_is_the_innermost_dimension():
    run_distributed(fn=_check_mesh)
