import pytest
import torch
import torch.nn as nn

from training.metrics.loss_only_metrics import (
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.dataclass.config import EarlyStoppingConfig, FitConfig
from training.dataclass.model import EpochSpec, Metrics, TrainState
from training.train.checkpoint import serialize_epoch_record
from training.constants import TEST_SEED, VAL_SEED
from training.train.train_loop import (
    eval_step,
    fit,
    profiled_fit,
    run_epoch,
    train_step,
)


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
    state = TrainState.create(model=model, optimizer=optimizer, criterion=criterion, device=device)
    assert next(state.model.parameters()).device.type == "cpu"  # noqa: NAR001


def test_train_step_returns_loss_and_batch_size():
    state, config = make_state_and_config()
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001
    metrics = train_step(
        train_state=state, epoch_spec=config, batch=batch, epoch=1, batch_index=0, seed=0
    )
    assert isinstance(metrics.loss, float)  # noqa: NAR001
    assert metrics.count == 4


def test_train_step_updates_weights():
    state, config = make_state_and_config()
    weights_before = state.model.weight.clone()
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001
    train_step(train_state=state, epoch_spec=config, batch=batch, epoch=1, batch_index=0, seed=0)
    assert not torch.equal(weights_before, state.model.weight)  # noqa: NAR001


def test_eval_step_does_not_update_weights():
    state, config = make_state_and_config()
    weights_before = state.model.weight.clone()
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001
    eval_step(train_state=state, epoch_spec=config, batch=batch, epoch=1, batch_index=0, seed=0)
    assert torch.equal(weights_before, state.model.weight)  # noqa: NAR001


def test_eval_step_returns_loss_and_batch_size():
    state, config = make_state_and_config()
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001
    metrics = eval_step(
        train_state=state, epoch_spec=config, batch=batch, epoch=1, batch_index=0, seed=0
    )
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
        seed=0,
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
            seed=0,
        )
        assert metrics.count == 4


def test_eval_step_passes_batch_prediction_and_position_to_eval_probe():
    state, config = make_state_and_config()
    calls = []
    batch = (torch.randn(4, 2), torch.randn(4, 1))  # noqa: NAR001

    def probe(model, input_tensor, target_tensor, predicted, epoch, batch_index, seed):
        calls.append(  # noqa: NAR001
            (model, input_tensor, target_tensor, predicted, epoch, batch_index, seed)
        )

    config.eval_probe = probe
    eval_step(train_state=state, epoch_spec=config, batch=batch, epoch=3, batch_index=2, seed=7)
    [(model, input_tensor, target_tensor, predicted, epoch, batch_index, seed)] = calls
    assert model is state.model
    assert torch.equal(input_tensor, batch[0])  # noqa: NAR001
    assert torch.equal(target_tensor, batch[1])  # noqa: NAR001
    assert torch.equal(predicted, state.model(batch[0]).detach())  # noqa: NAR001
    assert (epoch, batch_index, seed) == (3, 2, 7)


def test_run_epoch_calls_only_the_matching_probe_with_epoch_and_batch_index():
    state, config = make_state_and_config()
    calls = []
    config.train_probe = lambda *args: calls.append(("train", args[4], args[5]))  # noqa: NAR001
    config.eval_probe = lambda *args: calls.append(("eval", args[4], args[5]))  # noqa: NAR001
    batches = [(torch.randn(4, 2), torch.randn(4, 1))] * 3  # noqa: NAR001
    run_epoch(
        state=state,
        epoch_spec=config,
        loader=batches,
        step_fn=train_step,
        epoch=2,
        train=True,
        seed=0,
    )
    run_epoch(
        state=state,
        epoch_spec=config,
        loader=batches,
        step_fn=eval_step,
        epoch=2,
        train=False,
        seed=0,
    )
    assert calls == [
        ("train", 2, 0),
        ("train", 2, 1),
        ("train", 2, 2),
        ("eval", 2, 0),
        ("eval", 2, 1),
        ("eval", 2, 2),
    ]


def test_fit_seeds_the_global_rng_with_train_seed():
    state, config = make_state_and_config()
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    torch.manual_seed(seed=999)
    fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=FitConfig(
            num_epochs=1,
            train_seed=1234,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
        ),
    )
    assert torch.initial_seed() == 1234


def test_fit_repeats_dropout_draws_from_train_seed_whatever_the_rng_state_was():
    # The two runs enter fit with different global RNG states, so identical losses
    # can only come from fit reseeding it with train_seed.
    losses = []
    for extra_draws in (0, 100):  # noqa: NAR001
        torch.manual_seed(seed=0)
        state, config = make_state_and_config()
        state.model = nn.Sequential(  # noqa: NAR001
            nn.Linear(in_features=2, out_features=2),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2, out_features=1),
        )
        state.optimizer = torch.optim.SGD(params=state.model.parameters(), lr=0.01)
        torch.randn(extra_draws)  # noqa: NAR001
        batches = [(torch.ones(4, 2), torch.ones(4, 1))]  # noqa: NAR001
        history = fit(
            state=state,
            epoch_spec=config,
            train_loader=batches,
            val_loader=batches,
            fit_config=FitConfig(
                num_epochs=3,
                train_seed=42,
                val_seed=VAL_SEED,
                test_seed=TEST_SEED,
            ),
        )
        losses.append([record.train_loss for record in history])  # noqa: NAR001
    assert losses[0] == losses[1]


def test_fit_hands_train_probes_no_seed_and_eval_probes_the_val_seed():
    state, config = make_state_and_config()
    seeds = []
    config.train_probe = lambda *args: seeds.append(("train", args[6]))  # noqa: NAR001
    config.eval_probe = lambda *args: seeds.append(("eval", args[6]))  # noqa: NAR001
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=FitConfig(
            num_epochs=1,
            val_epoch_list=[1],
            train_seed=11,
            val_seed=22,
            test_seed=33,
        ),
    )
    assert seeds == [("train", None), ("eval", 22)]


def test_fit_runs_from_epoch_start_to_num_epochs():
    state, config = make_state_and_config()
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    history = fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=FitConfig(
            num_epochs=4,
            train_seed=0,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
            epoch_start=3,
        ),
    )

    assert [record.epoch for record in history] == [3, 4]


def test_fit_stops_after_patience_validation_passes_without_a_gain():
    state, config = make_state_and_config()
    state.optimizer.param_groups[0]["lr"] = 0.0
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    history = fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=FitConfig(
            num_epochs=10,
            val_epoch_list=list(range(1, 11)),  # noqa: NAR001
            train_seed=0,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
            early_stopping=EarlyStoppingConfig(patience=2),
        ),
    )

    # Epoch 1 sets the best loss; epochs 2 and 3 repeat it, so patience=2 runs out.
    assert [record.epoch for record in history] == [1, 2, 3]


def test_fit_counts_patience_in_validation_passes_not_epochs():
    state, config = make_state_and_config()
    state.optimizer.param_groups[0]["lr"] = 0.0
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    history = fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=FitConfig(
            num_epochs=10,
            val_epoch_list=[2, 4, 6, 8, 10],
            train_seed=0,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
            early_stopping=EarlyStoppingConfig(patience=1),
        ),
    )

    # The pass at epoch 2 sets the best loss; the pass at epoch 4 repeats it.
    assert [record.epoch for record in history] == [1, 2, 3, 4]


def test_fit_min_delta_makes_a_small_gain_not_count():
    state, config = make_state_and_config()
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    history = fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=FitConfig(
            num_epochs=10,
            val_epoch_list=list(range(1, 11)),  # noqa: NAR001
            train_seed=0,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
            early_stopping=EarlyStoppingConfig(patience=1, min_delta=1e9),
        ),
    )

    # No later loss beats the best by 1e9, so the second pass ends the run.
    assert [record.epoch for record in history] == [1, 2]


def test_profiled_fit_returns_profiler_metrics_for_profiled_epochs(tmp_path, monkeypatch):
    monkeypatch.setattr("training.train.train_loop.TRACES_PATH", tmp_path / "traces")
    state, config = make_state_and_config()
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    history, profiling_metrics = profiled_fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=FitConfig(
            num_epochs=2,
            val_epoch_list=[2],
            train_seed=7,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
            run_uuid="run-a",
        ),
        profile_epoch_list=[1],
    )

    assert [record.epoch for record in history] == [1, 2]
    assert torch.initial_seed() == 7
    assert list(profiling_metrics) == [1]  # noqa: NAR001
    trace_path = tmp_path / "traces" / "run-a" / "epoch_1.json"
    assert profiling_metrics[1]["trace_path"] == str(trace_path)  # noqa: NAR001
    assert trace_path.exists()


def test_fit_and_profiled_fit_stamp_the_seeds_on_every_epoch_record(tmp_path, monkeypatch):
    monkeypatch.setattr("training.train.train_loop.TRACES_PATH", tmp_path / "traces")
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    fit_config = FitConfig(
        num_epochs=2,
        train_seed=11,
        val_seed=22,
        test_seed=33,
    )
    state, config = make_state_and_config()
    fit_history = fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=fit_config,
    )
    state, config = make_state_and_config()
    profiled_history, _ = profiled_fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=fit_config,
        profile_epoch_list=[],
    )

    for record in fit_history + profiled_history:  # noqa: NAR001
        assert (record.train_seed, record.val_seed, record.test_seed) == (11, 22, 33)
    row = serialize_epoch_record(record=fit_history[0])
    assert (row["train_seed"], row["val_seed"], row["test_seed"]) == (11, 22, 33)


def test_fit_rejects_a_missing_train_seed():
    state, config = make_state_and_config()
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    with pytest.raises(expected_exception=ValueError, match="train_seed"):
        fit(
            state=state,
            epoch_spec=config,
            train_loader=batches,
            val_loader=batches,
            fit_config=FitConfig(num_epochs=1, val_seed=VAL_SEED, test_seed=TEST_SEED),
        )


def test_fit_validates_every_epoch_when_val_epoch_list_is_none():
    state, config = make_state_and_config()
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    history = fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=FitConfig(
            num_epochs=3, train_seed=0, val_seed=VAL_SEED, test_seed=TEST_SEED
        ),
    )
    assert all(record.val_loss is not None for record in history)  # noqa: NAR001
    assert len(history) == 3  # noqa: NAR001


def test_fit_checkpoints_after_every_validation_pass_only(tmp_path):
    state, config = make_state_and_config()
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]  # noqa: NAR001
    fit(
        state=state,
        epoch_spec=config,
        train_loader=batches,
        val_loader=batches,
        fit_config=FitConfig(
            num_epochs=3,
            val_epoch_list=[2],
            train_seed=0,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
            run_uuid="run-a",
        ),
    )

    run_dir = tmp_path / "checkpoints" / "run-a"
    assert sorted(path.name for path in run_dir.iterdir()) == ["epoch_2.pt"]  # noqa: NAR001
    saved = torch.load(f=run_dir / "epoch_2.pt")
    assert sorted(saved) == ["model", "optimizer"]  # noqa: NAR001
