import pytest
from training.experiments.fault_tolerance import account, failure_schedule, simulate_goodput


def test_the_failure_schedule_is_a_function_of_its_seed():
    assert failure_schedule(seed=1, mtbf_s=600, duration_s=7200) == failure_schedule(seed=1, mtbf_s=600, duration_s=7200)
    assert failure_schedule(seed=1, mtbf_s=600, duration_s=7200) != failure_schedule(seed=2, mtbf_s=600, duration_s=7200)
    assert all(0 < t < 7200 for t in failure_schedule(seed=1, mtbf_s=600, duration_s=7200))


def test_goodput_without_failures_is_the_checkpoint_overhead_only():
    assert simulate_goodput(interval_s=100, checkpoint_s=10, restart_s=60, failures=[], duration_s=1100) == pytest.approx(100 / 110)


def test_a_failure_loses_the_uncommitted_work_and_the_restart():
    # Saves commit at work 100 (t=110); the failure at t=160 loses 50 s of work and costs 60.
    with_failure = simulate_goodput(interval_s=100, checkpoint_s=10, restart_s=60, failures=[160.0], duration_s=500)
    assert with_failure < simulate_goodput(interval_s=100, checkpoint_s=10, restart_s=60, failures=[], duration_s=500)


def test_goodput_accounting_from_a_synthetic_event_log():
    events = [
        {"kind": "start", "time": 0.0, "step": 0},
        *[{"kind": "step", "time": float(s), "step": s} for s in range(1, 8)],
        {"kind": "saved", "time": 5.1, "step": 5},
        {"kind": "start", "time": 10.0, "step": 5},  # killed at 8.5, resumed from the step-5 commit
        *[{"kind": "step", "time": 10.0 + s, "step": 5 + s} for s in range(1, 6)],
    ]
    rows = account(events=events, kills=[8.5])
    assert rows[0]["step_reached"] == 7 and rows[0]["resumed_from_step"] == 5
    assert rows[0]["steps_lost"] == 2
    assert rows[0]["restart_and_reload_s"] == 1.5 and rows[0]["to_first_step_s"] == 2.5
    assert rows[-1]["steps_lost"] == 2
