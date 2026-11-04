import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributed.checkpoint.filesystem import FileSystemReader
from training.train.checkpoint import (
    COMMIT_MARKER,
    Checkpointer,
    expected_goodput,
    latest_committed,
    rng_state,
    set_rng_state,
    young_daly_interval,
)
from training.train.model import TrainState


class Position:
    def __init__(self, consumed=0):
        self.consumed = consumed

    def state_dict(self):
        return {"samples_consumed": self.consumed}

    def load_state_dict(self, state_dict):
        self.consumed = state_dict["samples_consumed"]


def _state(seed=0):
    torch.manual_seed(seed)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scheduler.step()
    return TrainState(model=model, optimizer=optimizer, criterion=nn.MSELoss(), scheduler=scheduler, loader=Position(consumed=96))


def test_every_required_key_is_written(tmp_path):
    Checkpointer(root=tmp_path, config={"lr": 0.1}).save(state=_state(), step=7)
    keys = set(FileSystemReader(path=tmp_path / "step_00000007").read_metadata().state_dict_metadata)
    for prefix in ("model.", "optimizer.", "scheduler", "loader", "step", "rank0/rng", "meta"):
        assert any(k.startswith(prefix) for k in keys), prefix


def test_round_trip_restores_weights_optimizer_scheduler_loader_rng_and_step(tmp_path):
    saved = _state(seed=0)
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    Checkpointer(root=tmp_path).save(state=saved, step=7)
    expected_draws = (torch.rand(3), random.random(), np.random.rand())
    restored = _state(seed=1)
    restored.loader.consumed = 0
    torch.manual_seed(99)
    assert Checkpointer(root=tmp_path).load_latest(state=restored) == 7
    assert (torch.rand(3), random.random(), np.random.rand())[1:] == expected_draws[1:]
    for a, b in zip(saved.model.parameters(), restored.model.parameters()):
        assert torch.equal(a, b)
    assert restored.loader.consumed == 96
    assert restored.scheduler.last_epoch == saved.scheduler.last_epoch
    torch.testing.assert_close(actual=restored.optimizer.state_dict()["state"][0]["exp_avg"], expected=saved.optimizer.state_dict()["state"][0]["exp_avg"])


def test_rng_state_round_trips():
    blob = rng_state()
    first = (torch.rand(2), random.random(), np.random.rand())
    set_rng_state(blob=blob)
    second = (torch.rand(2), random.random(), np.random.rand())
    assert torch.equal(first[0], second[0]) and first[1:] == second[1:]


def test_a_crash_mid_write_leaves_the_previous_checkpoint_loadable(tmp_path):
    checkpointer = Checkpointer(root=tmp_path)
    checkpointer.save(state=_state(), step=10)
    (tmp_path / "step_00000020").mkdir()  # a write that died before its commit marker
    (tmp_path / "step_00000020" / "__0_0.distcp").write_bytes(b"partial")
    assert latest_committed(root=tmp_path).name == "step_00000010"
    assert Checkpointer(root=tmp_path).load_latest(state=_state(seed=3)) == 10


def test_async_save_commits_on_wait_and_old_checkpoints_are_pruned(tmp_path):
    checkpointer = Checkpointer(root=tmp_path, async_save=True, keep=2)
    for step in (1, 2, 3):
        checkpointer.save(state=_state(), step=step)
    checkpointer.wait()
    committed = sorted(p.name for p in tmp_path.glob("step_*") if (p / COMMIT_MARKER).exists())
    assert committed == ["step_00000002", "step_00000003"]


def test_no_checkpoint_means_a_fresh_start(tmp_path):
    assert Checkpointer(root=tmp_path / "none").load_latest(state=_state()) is None


def test_young_daly_and_goodput_match_the_worked_examples():
    assert young_daly_interval(checkpoint_s=30, mtbf_s=10_800) == pytest.approx(805, abs=1)
    assert young_daly_interval(checkpoint_s=3, mtbf_s=10_800) == pytest.approx(255, abs=1)
    sync = expected_goodput(interval_s=805, checkpoint_s=30, mtbf_s=10_800, restart_s=300)
    async_ = expected_goodput(interval_s=255, checkpoint_s=3, mtbf_s=10_800, restart_s=300)
    assert sync == pytest.approx(0.898, abs=0.002)
    assert async_ == pytest.approx(0.949, abs=0.002)
