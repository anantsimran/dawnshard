"""train_loop.py -- functional PyTorch training loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from training.constants import CHECKPOINTS_PATH, TRACES_PATH
from training.dataclass.config import FitConfig
from training.dataclass.model import EpochRecord, EpochSpec, Metrics, TrainState
from training.train.checkpoint import checkpoint, serialize_epoch_record
from training.train.early_stopper import EarlyStopper
from training.train.environment import collect_system_metrics
from training.utils.expect import expect

Batch = tuple[Tensor, Tensor]
StepFn = Callable[["TrainState", "EpochSpec", Batch, int, int, Optional[int]], "Metrics"]


def train_step(
    train_state: TrainState,
    epoch_spec: EpochSpec,
    batch: Batch,
    epoch: int,
    batch_index: int,
    seed: Optional[int],
) -> Metrics:
    """One optimization step. Returns batch metrics with mean batch loss and batch size.

    If `epoch_spec.train_probe` is set, it is called after the optimizer step with
    this batch's input, target, detached predictions, `epoch`, `batch_index`, and
    `seed`.

    Args:
        epoch: 1-based epoch number, passed through to the probe.
        batch_index: 0-based position of `batch` in the epoch, passed through to the probe.
        seed: Seed of this pass, passed through to the probe. `fit` passes None on a
            training pass. The step draws no randomness of its own.
    """
    input_tensor, target_tensor = (tensor.to(device=epoch_spec.device) for tensor in batch)
    train_state.optimizer.zero_grad(set_to_none=True)
    predicted = train_state.model(input_tensor)  # noqa: NAR001
    loss = train_state.criterion(predicted, target_tensor)  # noqa: NAR001
    loss.backward()
    train_state.optimizer.step()
    predicted = predicted.detach()
    metrics = epoch_spec.calculate_metrics(predicted, target_tensor)  # noqa: NAR001
    metrics.loss = loss.item()
    metrics.count = target_tensor.size(dim=0)
    if epoch_spec.train_probe is not None:
        epoch_spec.train_probe(  # noqa: NAR001
            train_state.model,
            input_tensor,
            target_tensor,
            predicted,
            epoch,
            batch_index,
            seed,
        )
    return metrics


@torch.no_grad()
def eval_step(
    train_state: TrainState,
    epoch_spec: EpochSpec,
    batch: Batch,
    epoch: int,
    batch_index: int,
    seed: Optional[int],
) -> Metrics:
    """One forward-only step. Returns batch metrics with mean batch loss and batch size.

    If `epoch_spec.eval_probe` is set, it is called after the forward pass with this
    batch's input, target, predictions, `epoch`, `batch_index`, and `seed`.

    Args:
        epoch: 1-based epoch number, passed through to the probe.
        batch_index: 0-based position of `batch` in the epoch, passed through to the probe.
        seed: Seed of this pass, passed through to the probe. The step draws no
            randomness of its own.
    """
    input_tensor, target_tensor = (tensor.to(device=epoch_spec.device) for tensor in batch)
    predicted = train_state.model(input_tensor)  # noqa: NAR001
    loss = train_state.criterion(predicted, target_tensor)  # noqa: NAR001
    metrics = epoch_spec.calculate_metrics(predicted, target_tensor)  # noqa: NAR001
    metrics.loss = loss.item()
    metrics.count = target_tensor.size(dim=0)
    if epoch_spec.eval_probe is not None:
        epoch_spec.eval_probe(  # noqa: NAR001
            train_state.model,
            input_tensor,
            target_tensor,
            predicted,
            epoch,
            batch_index,
            seed,
        )
    return metrics


def run_epoch(
    state: TrainState,
    epoch_spec: EpochSpec,
    loader: DataLoader,
    step_fn: StepFn,
    *,
    epoch: int,
    train: bool,
    seed: Optional[int],
) -> Metrics:
    """Drive one pass over loader with step_fn; return the reduced epoch metrics.

    Builds a fresh metrics accumulator for the epoch and passes `epoch`, each batch's
    index, and `seed` to the step.

    Args:
        seed: Seed of this pass, threaded to the step and on to its probe, or None
            when the pass has none. `run_epoch` does not reseed anything with it.
    """
    state.model.train(mode=train)
    accumulator = epoch_spec.metrics_type()
    for batch_index, batch in enumerate(loader):  # noqa: NAR001
        batch_metrics = step_fn(  # noqa: NAR001
            state, epoch_spec, batch, epoch, batch_index, seed
        )
        epoch_spec.accumulate_metrics(accumulator, batch_metrics)  # noqa: NAR001
    return epoch_spec.reduce_metrics(accumulator)  # noqa: NAR001


def fit(
    state: TrainState,
    epoch_spec: EpochSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fit_config: FitConfig,
    wandb_run: Optional[Any] = None,
) -> list[EpochRecord]:
    """Run epochs `epoch_start` to `num_epochs` of `fit_config`; validate on its `val_epoch_list`.

    Args:
        fit_config: The run's epochs and seeds; `fit` reads it and never writes it.
            `val_epoch_list=None` validates after every epoch.
            If `early_stopping` is set, `fit` stops after the first validation pass
            that leaves `patience` consecutive passes without a `val_loss` that beats
            the best by more than `min_delta`; the history then ends at that epoch.

            - `train_seed`: Required, though the field defaults to None so a YAML can
              leave it out. Seeds the global torch RNG once, before the first epoch,
              so shuffling, dropout, and any randomness a collate draws repeat across
              runs. The caller builds the model before `fit` sees it, so weight init
              is not covered: seed before constructing the model if you want that
              too. It is not handed to training steps or their probe; they get None,
              and any randomness they need comes from the global RNG this seeds.
            - `val_seed`: Handed to every validation step and on to its probe, so a
              probe that draws randomness draws the same thing on each validation
              pass. Pass the repo-wide `VAL_SEED` unless a run has a reason not to;
              `fit` does not reseed the global RNG with it.
            - `test_seed`: Stamped on every record. `fit` runs no test pass, so
              nothing reads it yet; it is here so a run's three seeds are written
              down together.
            - `run_uuid`: Names the checkpoint folder, `CHECKPOINTS_PATH / run_uuid`;
              a fresh uuid4 stands in when it is None. Name the history file with
              the same uuid so the two pair up.

    Returns:
        Per-epoch history. `fit` writes no history file; pass it to
        `checkpoint.save_history` to keep it.

    Raises:
        ValueError: If `fit_config.train_seed` is None.

    Side effects:
        Seeds the global torch RNG from `fit_config.train_seed`. After every
        validation pass, writes a `checkpoint` of `state` to
        `CHECKPOINTS_PATH / run_uuid / f"epoch_{epoch}.pt"`, after the scheduler has
        stepped, so `load` resumes the run at the next epoch.
    """
    run_uuid = fit_config.run_uuid
    if run_uuid is None:
        run_uuid = str(uuid4())  # noqa: NAR001
    train_seed = expect(value=fit_config.train_seed, message="draw train_seed before calling fit")
    torch.manual_seed(seed=train_seed)
    logger.info(  # noqa: NAR001
        "fit | epochs {}-{} | val_epochs {} | seeds {}/{}/{} | device {} | wandb {} | run {}",
        fit_config.epoch_start,
        fit_config.num_epochs,
        fit_config.val_epoch_list,
        train_seed,
        fit_config.val_seed,
        fit_config.test_seed,
        epoch_spec.device,
        wandb_run is not None,
        run_uuid,
    )
    val_epoch_set = set(  # noqa: NAR001
        fit_config.val_epoch_list or range(1, fit_config.num_epochs + 1)  # noqa: NAR001
    )
    early_stopper = EarlyStopper(early_stopping=fit_config.early_stopping)
    history: list[EpochRecord] = []
    for epoch_number in range(fit_config.epoch_start, fit_config.num_epochs + 1):  # noqa: NAR001
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start_time = time.perf_counter()
        train_metrics = run_epoch(
            state=state,
            epoch_spec=epoch_spec,
            loader=train_loader,
            step_fn=train_step,
            epoch=epoch_number,
            train=True,
            seed=None,
        )
        if state.scheduler is not None:
            state.scheduler.step()
        train_loss = train_metrics.loss
        epoch_duration_seconds = time.perf_counter() - epoch_start_time
        val_loss = None
        val_metrics = None
        if epoch_number in val_epoch_set:
            val_metrics = run_epoch(
                state=state,
                epoch_spec=epoch_spec,
                loader=val_loader,
                step_fn=eval_step,
                epoch=epoch_number,
                train=False,
                seed=fit_config.val_seed,
            )
            val_loss = val_metrics.loss
            logger.info(  # noqa: NAR001
                "epoch {:>3} | train {:.4f} | val {:.4f}",
                epoch_number,
                train_loss,
                val_loss,
            )
            checkpoint(
                state=state,
                checkpoint_path=CHECKPOINTS_PATH / run_uuid / f"epoch_{epoch_number}.pt",
            )
        else:
            logger.info("epoch {:>3} | train {:.4f}", epoch_number, train_loss)  # noqa: NAR001
        epoch_record = EpochRecord(
            epoch=epoch_number,
            train_loss=train_loss,
            train_metrics=train_metrics,
            epoch_duration_seconds=epoch_duration_seconds,
            system_metrics=collect_system_metrics(),
            train_seed=train_seed,
            val_seed=fit_config.val_seed,
            test_seed=fit_config.test_seed,
            val_loss=val_loss,
            val_metrics=val_metrics,
        )
        record_dict = serialize_epoch_record(record=epoch_record)
        if wandb_run is not None:
            wandb_run.log(data=record_dict)
        history.append(epoch_record)  # noqa: NAR001
        if val_loss is not None and early_stopper.update(val_loss=val_loss):
            logger.info("early stopping after epoch {}", epoch_number)  # noqa: NAR001
            break
    return history


def _profiled_run_epoch(
    state: TrainState,
    epoch_spec: EpochSpec,
    loader: DataLoader,
    step_fn: StepFn,
    *,
    epoch: int,
    train: bool,
    seed: Optional[int],
) -> Metrics:
    """Like run_epoch but with record_function labels so the profiler can attribute time."""
    from torch.profiler import record_function

    state.model.train(mode=train)
    accumulator = epoch_spec.metrics_type()
    loader_iter = iter(loader)  # noqa: NAR001
    batch_index = 0
    while True:
        with record_function(name="dataloader"):
            try:
                batch = next(loader_iter)  # noqa: NAR001
            except StopIteration:
                break
        with record_function(name="step"):
            batch_metrics = step_fn(  # noqa: NAR001
                state, epoch_spec, batch, epoch, batch_index, seed
            )
            epoch_spec.accumulate_metrics(accumulator, batch_metrics)  # noqa: NAR001
        batch_index += 1
    return epoch_spec.reduce_metrics(accumulator)  # noqa: NAR001


def _extract_profiler_metrics(prof: Any, *, epoch_number: int, traces_dir: Path) -> dict:
    """Extract profiler timing metrics and export a Chrome trace file to `traces_dir`."""
    key_averages = prof.key_averages()
    dataloader_event = next(  # noqa: NAR001
        (event for event in key_averages if event.key == "dataloader"), None
    )
    step_event = next(  # noqa: NAR001
        (event for event in key_averages if event.key == "step"), None
    )
    metrics: dict = {}
    if dataloader_event is not None:
        metrics["dataloader_cpu_time_ms"] = dataloader_event.cpu_time_total / 1000
    if step_event is not None:
        metrics["step_cpu_time_ms"] = step_event.cpu_time_total / 1000
        if torch.cuda.is_available():
            metrics["step_cuda_time_ms"] = step_event.cuda_time_total / 1000
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"epoch_{epoch_number}.json"
    prof.export_chrome_trace(str(trace_path))  # noqa: NAR001
    metrics["trace_path"] = str(trace_path)  # noqa: NAR001
    return metrics


def profiled_fit(
    state: TrainState,
    epoch_spec: EpochSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fit_config: FitConfig,
    profile_epoch_list: list[int],
    wandb_run: Optional[Any] = None,
) -> tuple[list[EpochRecord], dict[int, dict]]:
    """Like fit, but wraps profiling epochs with torch.profiler.profile.

    Stops early under the same `fit_config.early_stopping` rule as `fit`. Like `fit`,
    it writes no history file; pass both return values to `checkpoint.save_history`
    for a history whose profiled epochs carry the profiler's timings and trace path.

    Args:
        fit_config: As in `fit`, with the same fields and seeds. Its `run_uuid`
            also names the trace directory, `TRACES_PATH / run_uuid`.
        profile_epoch_list: Epochs whose training pass runs under the profiler.

    Returns:
        The per-epoch history, as `fit` returns it, and the profiler metrics of each
        profiled epoch keyed by epoch number, `trace_path` included.

    Raises:
        ValueError: If `fit_config.train_seed` is None.

    Side effects:
        Seeds the global torch RNG from `fit_config.train_seed`, writes one Chrome
        trace per profiled epoch, and checkpoints after every validation pass as
        `fit` does.
    """
    from torch.profiler import ProfilerActivity, profile

    run_uuid = fit_config.run_uuid
    if run_uuid is None:
        run_uuid = str(uuid4())  # noqa: NAR001
    traces_dir = TRACES_PATH / run_uuid
    train_seed = expect(value=fit_config.train_seed, message="draw train_seed before calling fit")
    torch.manual_seed(seed=train_seed)
    logger.info(  # noqa: NAR001
        "profiled_fit epochs={} val={} profile={} seeds={}/{}/{} device={} traces={}",
        fit_config.num_epochs,
        fit_config.val_epoch_list,
        profile_epoch_list,
        train_seed,
        fit_config.val_seed,
        fit_config.test_seed,
        epoch_spec.device,
        traces_dir,
    )
    val_epoch_set = set(  # noqa: NAR001
        fit_config.val_epoch_list or range(1, fit_config.num_epochs + 1)  # noqa: NAR001
    )
    profile_epoch_set = set(profile_epoch_list)  # noqa: NAR001
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)  # noqa: NAR001
    early_stopper = EarlyStopper(early_stopping=fit_config.early_stopping)
    history: list[EpochRecord] = []
    profiling_metrics: dict[int, dict] = {}
    for epoch_number in range(fit_config.epoch_start, fit_config.num_epochs + 1):  # noqa: NAR001
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start_time = time.perf_counter()
        if epoch_number in profile_epoch_set:
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                train_metrics = _profiled_run_epoch(
                    state=state,
                    epoch_spec=epoch_spec,
                    loader=train_loader,
                    step_fn=train_step,
                    epoch=epoch_number,
                    train=True,
                    seed=None,
                )
            profiling_metrics[epoch_number] = _extract_profiler_metrics(
                prof=prof,
                epoch_number=epoch_number,
                traces_dir=traces_dir,
            )
        else:
            train_metrics = run_epoch(
                state=state,
                epoch_spec=epoch_spec,
                loader=train_loader,
                step_fn=train_step,
                epoch=epoch_number,
                train=True,
                seed=None,
            )
        if state.scheduler is not None:
            state.scheduler.step()
        train_loss = train_metrics.loss
        epoch_duration_seconds = time.perf_counter() - epoch_start_time
        val_loss = None
        val_metrics = None
        if epoch_number in val_epoch_set:
            val_metrics = run_epoch(
                state=state,
                epoch_spec=epoch_spec,
                loader=val_loader,
                step_fn=eval_step,
                epoch=epoch_number,
                train=False,
                seed=fit_config.val_seed,
            )
            val_loss = val_metrics.loss
            logger.info(  # noqa: NAR001
                "epoch {:>3} | train {:.4f} | val {:.4f}",
                epoch_number,
                train_loss,
                val_loss,
            )
            checkpoint(
                state=state,
                checkpoint_path=CHECKPOINTS_PATH / run_uuid / f"epoch_{epoch_number}.pt",
            )
        else:
            logger.info(  # noqa: NAR001
                "epoch {:>3} | train {:.4f}", epoch_number, train_loss
            )
        epoch_record = EpochRecord(
            epoch=epoch_number,
            train_loss=train_loss,
            train_metrics=train_metrics,
            epoch_duration_seconds=epoch_duration_seconds,
            system_metrics=collect_system_metrics(),
            train_seed=train_seed,
            val_seed=fit_config.val_seed,
            test_seed=fit_config.test_seed,
            val_loss=val_loss,
            val_metrics=val_metrics,
        )
        if wandb_run is not None:
            wandb_run.log(data=serialize_epoch_record(record=epoch_record))
        history.append(epoch_record)  # noqa: NAR001
        if val_loss is not None and early_stopper.update(val_loss=val_loss):
            logger.info("early stopping after epoch {}", epoch_number)  # noqa: NAR001
            break
    return history, profiling_metrics
