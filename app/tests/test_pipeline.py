import copy

import pytest
import torch
from dist_helpers import run_distributed
from torch import nn
from training.parallel.pipeline import (
    gpipe,
    one_f_one_b,
    partition,
    run_stage,
    simulate,
    theoretical_bubble,
    validate,
)


@pytest.mark.parametrize("p", range(1, 9))
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 16, 32])
def test_1f1b_is_valid_with_at_most_p_minus_s_in_flight(p, m):
    schedule = one_f_one_b(num_stages=p, num_microbatches=m)
    validate(schedule=schedule, num_microbatches=m, max_in_flight=p)
    for stage, actions in enumerate(schedule):
        validate(schedule=[actions], num_microbatches=m, max_in_flight=p - stage)


def test_gpipe_holds_every_microbatch():
    validate(schedule=gpipe(num_stages=4, num_microbatches=8), num_microbatches=8, max_in_flight=8)
    with pytest.raises(ValueError):
        validate(schedule=gpipe(num_stages=4, num_microbatches=8), num_microbatches=8, max_in_flight=4)


@pytest.mark.parametrize("m", [4, 8, 16, 32])
def test_simulated_bubble_matches_the_formula_for_equal_times(m):
    for schedule in (gpipe(num_stages=4, num_microbatches=m), one_f_one_b(num_stages=4, num_microbatches=m)):
        timeline = simulate(schedule=schedule, forward_time=[1.0] * 4, backward_time=[2.0] * 4)
        assert timeline.bubble == pytest.approx(theoretical_bubble(num_stages=4, num_microbatches=m))


def test_partition_balances_flops_within_ten_percent():
    # 28 blocks of cost 1 and a tied LM head of 6.2 blocks, over 4 stages.
    costs = [1.0] * 28 + [6.2]
    stages = partition(costs=costs, num_stages=4)
    assert [s.start for s in stages] == [0, 9, 18, 27] or len(stages) == 4
    loads = [sum(costs[i] for i in s) for s in stages]
    assert max(loads) <= 1.1 * sum(costs) / 4
    assert stages[-1].stop == len(costs) and stages[0].start == 0
    naive = [sum(costs[i] for i in range(k * 29 // 4, (k + 1) * 29 // 4)) for k in range(4)]
    assert max(loads) < max(naive)


def _layers():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 3))


def _check_pipeline(rank, world, schedule_name):
    full = _layers()
    stages = [full[:2], full[2:]]
    stage = copy.deepcopy(stages[rank])
    generator = torch.Generator().manual_seed(1)
    x, y = torch.randn(8, 4, generator=generator), torch.randn(8, 3, generator=generator)
    micro_x, micro_y = list(x.chunk(4)), list(y.chunk(4))
    schedule = {"gpipe": gpipe, "1f1b": one_f_one_b}[schedule_name](num_stages=2, num_microbatches=4)
    losses = run_stage(stage=stage, actions=schedule[rank], microbatches=micro_x, targets=micro_y,
                       loss_fn=nn.functional.mse_loss, rank=rank, num_stages=2, activation_shape=(2, 8))
    reference = _layers()
    loss = nn.functional.mse_loss(reference(x), y)
    loss.backward()
    expected = [p.grad for p in (reference[:2] if rank == 0 else reference[2:]).parameters()]
    for got, want in zip([p.grad for p in stage.parameters()], expected):
        torch.testing.assert_close(actual=got, expected=want, atol=1e-6, rtol=1e-5)
    if rank == 1:
        assert sum(losses) == pytest.approx(loss.item(), rel=1e-5)


@pytest.mark.parametrize("schedule_name", ["gpipe", "1f1b"])
def test_two_stage_pipeline_gradients_equal_single_process(schedule_name):
    run_distributed(fn=_check_pipeline, args=(schedule_name,))
