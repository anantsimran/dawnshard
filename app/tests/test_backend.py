import pytest
from training.launch.backend import run_cost, torchrun_command
from training.launch.spec import JobSpec


def test_cost_matches_the_two_node_gke_example():
    assert run_cost(hourly_rates=[0.7068, 0.7068], hours=0.5) == pytest.approx(0.7068)


def test_spot_rework_raises_the_effective_price():
    on_demand = 1.0
    spot = run_cost(hourly_rates=[0.57 * on_demand], hours=1.0, rework_fraction=0.1)
    assert spot == pytest.approx(0.627)


def test_torchrun_command_multi_node_uses_c10d_rendezvous():
    spec = JobSpec(name="job", image="i", module="training.model.qwen3", args=("--steps", "5"), nodes=2)
    command = torchrun_command(spec=spec, rdzv_endpoint="host:29500")
    assert command[:2] == ["torchrun", "--nproc-per-node=1"]
    assert "--nnodes=2" in command and "--rdzv-endpoint=host:29500" in command
    assert command[-4:] == ["-m", "training.model.qwen3", "--steps", "5"]
