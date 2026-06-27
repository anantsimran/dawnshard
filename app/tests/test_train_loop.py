import json

import pytest
import torch
import torch.nn as nn

from train.mean_loss import accumulate_loss, compute_reduced_loss
from train.train_loop import (
    RuntimeConfig,
    TrainState,
    eval_step,
    load,
    run_epoch,
    save_history,
    save_state,
    train_step,
)


def make_state_and_config():
    model = nn.Linear(in_features=2, out_features=1)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    state = TrainState(model=model, optimizer=optimizer, criterion=criterion)
    config = RuntimeConfig(
        device=torch.device("cpu"),
        accumulate_loss=accumulate_loss,
        compute_reduced_loss=compute_reduced_loss,
    )
    return state, config


def test_train_state_create_moves_model_to_device():
    device = torch.device("cpu")
    model = nn.Linear(in_features=2, out_features=1)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    state = TrainState.create(
        model=model, optimizer=optimizer, criterion=criterion, device=device
    )
    assert next(state.model.parameters()).device.type == "cpu"


def test_train_step_returns_loss_and_batch_size():
    state, config = make_state_and_config()
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    batch_loss, batch_size = train_step(
        train_state=state, config=config, batch=batch
    )
    assert isinstance(batch_loss, float)
    assert batch_size == 4


def test_train_step_updates_weights():
    state, config = make_state_and_config()
    weights_before = state.model.weight.clone()
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    train_step(train_state=state, config=config, batch=batch)
    assert not torch.equal(weights_before, state.model.weight)


def test_eval_step_does_not_update_weights():
    state, config = make_state_and_config()
    weights_before = state.model.weight.clone()
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    eval_step(train_state=state, config=config, batch=batch)
    assert torch.equal(weights_before, state.model.weight)


def test_eval_step_returns_loss_and_batch_size():
    state, config = make_state_and_config()
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    batch_loss, batch_size = eval_step(
        train_state=state, config=config, batch=batch
    )
    assert isinstance(batch_loss, float)
    assert batch_size == 4


def test_run_epoch_returns_nonnegative_loss():
    state, config = make_state_and_config()
    batches = [
        (torch.randn(4, 2), torch.randn(4, 1)),
        (torch.randn(4, 2), torch.randn(4, 1)),
    ]
    loss = run_epoch(
        state=state,
        config=config,
        loader=batches,
        step_fn=train_step,
        train=True,
    )
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_save_and_load_roundtrip(tmp_path):
    state, config = make_state_and_config()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_state(state=state, checkpoint_path=checkpoint_path)

    original_weight = state.model.weight.clone()
    state.model.weight.data.fill_(99.0)

    load(state=state, config=config, checkpoint_path=checkpoint_path)
    assert torch.allclose(state.model.weight, original_weight)


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
        runtime_config=config,
    )

    with open(file=history_path) as history_file:
        data = json.load(fp=history_file)

    assert "train_state" in data["meta"]
    assert "train_config" in data["meta"]
    assert data["meta"]["train_state"]["model"] == "Linear"
