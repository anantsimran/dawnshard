import torch
from dist_helpers import run_distributed
from training.parallel.parity import drift_test, flatten_shards, tensor_dots


def test_flatten_counts_replicated_tensors_once_across_ranks():
    sharded, whole = torch.tensor([1.0, 2.0]), torch.tensor([3.0])
    flat = flatten_shards(tensors=[sharded, whole], replicated=[False, True], world_size=4)
    # Four ranks each contribute 9 / 4 for the replicated entry: 9 in total.
    assert torch.isclose(input=4 * flat[2:] @ flat[2:], other=torch.tensor(9.0))
    assert torch.equal(flat[:2], sharded)


def test_zero_mean_errors_give_ratio_near_one_and_rare_alarms():
    generator = torch.Generator().manual_seed(0)
    alarms = 0
    for trial in range(200):
        errors = [torch.randn(500, generator=generator) for _ in range(30)]
        ratio, p_value = drift_test(errors=errors, num_flips=200, seed=trial)
        assert 0.5 < ratio < 1.5
        alarms += p_value <= 0.05
    # 10 expected at level 0.05; 25 is about 5 standard deviations above.
    assert alarms <= 25


def test_bias_far_below_per_step_noise_is_detected():
    generator = torch.Generator().manual_seed(1)
    direction = torch.randn(10_000, generator=generator)
    bias = 0.03 * direction / direction.norm()
    # Each step's error is ~100 in norm; the bias is 0.03, under 0.1% of the energy.
    errors = [bias + torch.randn(10_000, generator=generator) for _ in range(50)]
    noise_only = [torch.randn(10_000, generator=generator) for _ in range(50)]
    assert drift_test(errors=errors, num_flips=2000, seed=0)[1] > 0.01
    errors = [200 * bias + e for e in noise_only]  # a bias of 6 against noise of 100
    assert drift_test(errors=errors, num_flips=2000, seed=0)[1] <= 0.01


def _check_sharded_matches_dense(rank, world):
    generator = torch.Generator().manual_seed(2)
    dense = [(torch.randn(6, generator=generator), torch.randn(3, generator=generator)) for _ in range(8)]
    errors = [
        flatten_shards(tensors=[d[0].chunk(world)[rank], d[1]], replicated=[False, True], world_size=world)
        for d in dense
    ]
    full = [torch.cat([d[0], d[1]]) for d in dense]
    per_tensor = tensor_dots(a=errors[0], b=errors[1], sizes=[6 // world, 3])
    expected_dots = [float(dense[0][0] @ dense[1][0]), float(dense[0][1] @ dense[1][1])]
    assert torch.allclose(per_tensor, torch.tensor(expected_dots, dtype=torch.float64), atol=1e-5)
    sharded = drift_test(errors=errors, num_flips=500, seed=3)
    expected = drift_test(errors=full, num_flips=500, seed=3)
    assert abs(sharded[0] - expected[0]) < 1e-6 and sharded[1] == expected[1]


def test_sharded_drift_test_equals_the_unsharded_one():
    run_distributed(fn=_check_sharded_matches_dense)
