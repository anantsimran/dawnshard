import json

import pytest
import torch
import torch.nn as nn

from training.metrics.loss_only_metrics import (
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.train.model import EpochRecord, EpochSpec, Metrics, SystemMetrics, TrainState
from training.train.train_loop import eval_step, load, run_epoch, train_step
from training.train.utils import save_history, save_state
from training.utils.serialization import serialize_epoch_record


def make_state_and_config():
    model = nn.Linear(in_features=2, out_features=1)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    state = TrainState(model=model, optimizer=optimizer, criterion=criterion)
    config = EpochSpec(
        device=torch.device("cpu"),  # noqa: NAR001
        metrics_type=Metrics,
        calculate_metrics=calculate_metrics,
        accumulate_metrics=accumulate_metrics,
        reduce_metrics=reduce_metrics,
    )
    return state, config


def test_train_state_create_moves_model_to_device():
    device = torch.device("cpu")  # noqa: NAR001
    model = nn.Linear(in_features=2, out_features=1)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    state = TrainState.create(
        model=model, optimizer=optimizer, criterion=criterion, device=device
    )
    assert next(state.model.parameters()).device.type == "cpu"  # noqa: NAR001


def test_train_step_returns_loss_and_batch_size():
    state, config = make_state_and_config()
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001
    metrics = train_step(train_state=state, epoch_spec=config, batch=batch, epoch=1, batch_index=0)
    assert isinstance(metrics.loss, float)  # noqa: NAR001
    assert metrics.count == 4


def test_train_step_updates_weights():
    state, config = make_state_and_config()
    weights_before = state.model.weight.clone()
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001
    train_step(train_state=state, epoch_spec=config, batch=batch, epoch=1, batch_index=0)
    assert not torch.equal(weights_before, state.model.weight)  # noqa: NAR001


def test_eval_step_does_not_update_weights():
    state, config = make_state_and_config()
    weights_before = state.model.weight.clone()
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001
    eval_step(train_state=state, epoch_spec=config, batch=batch, epoch=1, batch_index=0)
    assert torch.equal(weights_before, state.model.weight)  # noqa: NAR001


def test_eval_step_returns_loss_and_batch_size():
    state, config = make_state_and_config()
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001
    metrics = eval_step(train_state=state, epoch_spec=config, batch=batch, epoch=1, batch_index=0)
    assert isinstance(metrics.loss, float)  # noqa: NAR001
    assert metrics.count == 4


def test_run_epoch_returns_nonnegative_loss():
    state, config = make_state_and_config()
    batches = [
        (torch.randn(4, 2), torch.randn(4, 1)),  # noqa: NAR001
        (torch.randn(4, 2), torch.randn(4, 1)),  # noqa: NAR001
    ]
    metrics = run_epoch(
        state=state,
        epoch_spec=config,
        loader=batches,
        step_fn=train_step,
        epoch=1,
        train=True,
    )
    assert isinstance(metrics.loss, float)  # noqa: NAR001
    assert metrics.loss >= 0.0
    assert metrics.count == 8


def test_run_epoch_starts_each_epoch_with_a_fresh_accumulator():
    state, config = make_state_and_config()
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    for _ in range(2):  # noqa: NAR001
        metrics = run_epoch(
            state=state,
            epoch_spec=config,
            loader=batches,
            step_fn=eval_step,
            epoch=1,
            train=False,
        )
        assert metrics.count == 4


def test_eval_step_passes_batch_prediction_and_position_to_eval_probe():
    state, config = make_state_and_config()
    calls = []
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001

    def probe(model, input_tensor, target_tensor, predicted, epoch, batch_index):
        calls.append((model, input_tensor, target_tensor, predicted, epoch, batch_index))  # noqa: NAR001

    config.eval_probe = probe
    eval_step(train_state=state, epoch_spec=config, batch=batch, epoch=3, batch_index=2)
    [(model, input_tensor, target_tensor, predicted, epoch, batch_index)] = calls
    assert model is state.model
    assert torch.equal(input_tensor, batch[0])  # noqa: NAR001
    assert torch.equal(target_tensor, batch[1])  # noqa: NAR001
    assert torch.equal(predicted, state.model(batch[0]).detach())  # noqa: NAR001
    assert (epoch, batch_index) == (3, 2)


def test_run_epoch_calls_only_the_matching_probe_with_epoch_and_batch_index():
    state, config = make_state_and_config()
    calls = []
    config.train_probe = lambda *args: calls.append(("train", args[4], args[5]))  # noqa: NAR001
    config.eval_probe = lambda *args: calls.append(("eval", args[4], args[5]))  # noqa: NAR001
    batches = [(torch.randn(4, 2), torch.randn(4, 1))] * 3  # noqa: NAR001
    run_epoch(
        state=state, epoch_spec=config, loader=batches, step_fn=train_step, epoch=2, train=True
    )
    run_epoch(
        state=state, epoch_spec=config, loader=batches, step_fn=eval_step, epoch=2, train=False
    )
    assert calls == [
        ("train", 2, 0),
        ("train", 2, 1),
        ("train", 2, 2),
        ("eval", 2, 0),
        ("eval", 2, 1),
        ("eval", 2, 2),
    ]


def test_save_and_load_roundtrip(tmp_path):
    state, config = make_state_and_config()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_state(state=state, checkpoint_path=checkpoint_path)

    original_weight = state.model.weight.clone()
    state.model.weight.data.fill_(99.0)  # noqa: NAR001

    load(state=state, epoch_spec=config, checkpoint_path=checkpoint_path)
    assert torch.allclose(state.model.weight, original_weight)  # noqa: NAR001


def test_save_history_writes_valid_json(tmp_path):
    history = [{"epoch": 1, "train_loss": 0.5}]
    history_path = tmp_path / "history.json"
    save_history(history=history, history_path=history_path)

    with open(file=history_path) as history_file:
        data = json.load(fp=history_file)

    assert data["history"] == history
    assert "git_commit" in data["meta"]


def test_save_history_includes_train_state_meta(tmp_path):
    state, config = make_state_and_config()
    history = [{"epoch": 1, "train_loss": 0.3}]
    history_path = tmp_path / "history.json"
    save_history(
        history=history,
        history_path=history_path,
        train_state=state,
        epoch_spec=config,
    )

    with open(file=history_path) as history_file:
        data = json.load(fp=history_file)

    assert "train_state" in data["meta"]
    assert "train_config" in data["meta"]
    assert data["meta"]["train_state"]["model"] == "Linear"


def test_serialize_epoch_record_flattens_and_drops_unset_fields():
    record = EpochRecord(
        epoch=2,
        train_loss=0.4,
        train_metrics=Metrics(loss=0.4, count=8),
        epoch_duration_seconds=1.5,
        system_metrics=SystemMetrics(cpu_usage_percent=10.0, ram_used_gb=3.0),
    )
    assert serialize_epoch_record(record=record) == {
        "epoch": 2,
        "train_loss": 0.4,
        "train_metrics": {"loss": 0.4, "count": 8},
        "epoch_duration_seconds": 1.5,
        "cpu_usage_percent": 10.0,
        "ram_used_gb": 3.0,
    }
