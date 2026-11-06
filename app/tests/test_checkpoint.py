import json

import torch
import torch.nn as nn

from training.constants import TEST_SEED, VAL_SEED
from training.dataclass.config import (
    DataloadConfig,
    DataloaderConfig,
    EarlyStoppingConfig,
    FitConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
)
from training.dataclass.model import (
    EpochRecord,
    EpochSpec,
    Metrics,
    SystemMetrics,
    TrainState,
)
from training.metrics.loss_only_metrics import (
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.train.checkpoint import (
    checkpoint,
    load,
    save_history,
    serialize_epoch_record,
)
from training.train.train_loop import fit


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


def make_record(epoch):
    return EpochRecord(
        epoch=epoch,
        train_loss=0.4,
        train_metrics=Metrics(loss=0.4, count=8),
        epoch_duration_seconds=1.5,
        system_metrics=SystemMetrics(cpu_usage_percent=10.0, ram_used_gb=3.0),
        train_seed=1,
        val_seed=2,
        test_seed=3,
    )


def read_history(history_path):
    with open(file=history_path) as history_file:
        return json.load(fp=history_file)


def test_checkpoint_and_load_roundtrip_into_a_new_folder(tmp_path):
    state, config = make_state_and_config()
    checkpoint_path = tmp_path / "checkpoints" / "run.pt"
    checkpoint(state=state, checkpoint_path=checkpoint_path)

    original_weight = state.model.weight.clone()
    state.model.weight.data.fill_(99.0)  # noqa: NAR001

    load(state=state, epoch_spec=config, checkpoint_path=checkpoint_path)
    assert torch.allclose(state.model.weight, original_weight)  # noqa: NAR001


def test_save_history_serializes_each_record(tmp_path):
    history_path = tmp_path / "history.json"
    save_history(history=[make_record(epoch=1)], history_path=history_path)

    data = read_history(history_path=history_path)
    assert data["history"] == [serialize_epoch_record(record=make_record(epoch=1))]
    assert "git_commit" in data["meta"]


def test_save_history_includes_train_state_meta(tmp_path):
    state, config = make_state_and_config()
    history_path = tmp_path / "history.json"
    save_history(
        history=[make_record(epoch=1)],
        history_path=history_path,
        train_state=state,
        epoch_spec=config,
    )

    meta = read_history(history_path=history_path)["meta"]
    assert "epoch_spec" in meta
    assert meta["train_state"]["model"] == "Linear"


def test_save_history_records_model_shape(tmp_path):
    state, config = make_state_and_config()
    history_path = tmp_path / "history.json"
    save_history(
        history=[],
        history_path=history_path,
        train_state=state,
        epoch_spec=config,
    )

    train_state_meta = read_history(history_path=history_path)["meta"]["train_state"]
    assert train_state_meta["parameter_shapes"] == {"weight": [1, 2], "bias": [1]}
    assert train_state_meta["num_parameters"] == 3


def test_save_history_records_the_whole_train_config_and_last_epoch(tmp_path):
    fit_config = FitConfig(
        num_epochs=5,
        val_epoch_list=[2, 4],
        train_seed=1,
        val_seed=2,
        test_seed=3,
        epoch_start=2,
        run_uuid="abc",
        early_stopping=EarlyStoppingConfig(patience=3, min_delta=0.01),
    )
    train_config = TrainConfig(
        model=ModelConfig(),
        fit=fit_config,
        optimizer=OptimizerConfig(name="sgd", lr=0.01),
        dataloader=DataloaderConfig(batch_size=4),
        data=DataloadConfig(),
    )
    history_path = tmp_path / "history.json"
    save_history(
        history=[make_record(epoch=3), make_record(epoch=4)],
        history_path=history_path,
        train_config=train_config,
    )

    meta = read_history(history_path=history_path)["meta"]
    assert meta["epoch"] == 4
    assert meta["train_config"] == {
        "model": {},
        "fit": {
            "num_epochs": 5,
            "val_epoch_list": [2, 4],
            "train_seed": 1,
            "val_seed": 2,
            "test_seed": 3,
            "epoch_start": 2,
            "run_uuid": "abc",
            "early_stopping": {"patience": 3, "min_delta": 0.01},
        },
        "optimizer": {"name": "sgd", "lr": 0.01, "weight_decay": 0.0},
        "dataloader": {"batch_size": 4},
        "data": {},
        "scheduler": None,
    }


def test_save_history_leaves_out_the_epoch_when_history_is_empty(tmp_path):
    history_path = tmp_path / "history.json"
    save_history(history=[], history_path=history_path)

    assert "epoch" not in read_history(history_path=history_path)["meta"]


def test_save_history_merges_profiling_metrics_into_their_epochs_only(tmp_path):
    history_path = tmp_path / "history.json"
    save_history(
        history=[make_record(epoch=1), make_record(epoch=2)],
        history_path=history_path,
        profiling_metrics={1: {"trace_path": "epoch_1.json"}},
    )

    rows = read_history(history_path=history_path)["history"]
    assert rows[0]["trace_path"] == "epoch_1.json"
    assert "trace_path" not in rows[1]


def test_fit_then_save_history_records_the_seeds_fit_ran_with(tmp_path):
    state, config = make_state_and_config()
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    fit_config = FitConfig(
        num_epochs=1,
        val_epoch_list=[1],
        train_seed=5,
        val_seed=VAL_SEED,
        test_seed=TEST_SEED,
    )
    history = fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=fit_config,
    )
    history_path = tmp_path / "history.json"
    save_history(history=history, history_path=history_path, train_config=fit_config)

    data = read_history(history_path=history_path)
    recorded = data["meta"]["train_config"]
    assert (recorded["train_seed"], recorded["val_seed"], recorded["test_seed"]) == (
        5,
        VAL_SEED,
        TEST_SEED,
    )
    row = data["history"][0]
    assert (row["train_seed"], row["val_seed"], row["test_seed"]) == (5, VAL_SEED, TEST_SEED)


def test_serialize_epoch_record_flattens_and_drops_unset_fields():
    assert serialize_epoch_record(record=make_record(epoch=2)) == {
        "epoch": 2,
        "train_loss": 0.4,
        "train_metrics": {"loss": 0.4, "count": 8},
        "epoch_duration_seconds": 1.5,
        "cpu_usage_percent": 10.0,
        "ram_used_gb": 3.0,
        "train_seed": 1,
        "val_seed": 2,
        "test_seed": 3,
    }
