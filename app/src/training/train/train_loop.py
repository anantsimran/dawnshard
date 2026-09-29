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
from training.constants import HISTORY_PATH, TRACES_PATH
from training.train.environment import collect_system_metrics
from training.train.model import EpochRecord, EpochSpec, Metrics, TrainState
from training.train.serialization import serialize_epoch_record
from training.train.utils import save_history

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


def load(state: TrainState, epoch_spec: EpochSpec, checkpoint_path: Path) -> None:
    """Restore model + optimizer (+ scheduler) from a checkpoint."""
    checkpoint = torch.load(f=checkpoint_path, map_location=epoch_spec.device)
    state.model.load_state_dict(state_dict=checkpoint["model"])
    state.optimizer.load_state_dict(state_dict=checkpoint["optimizer"])
    if state.scheduler is not None and "scheduler" in checkpoint:
        state.scheduler.load_state_dict(state_dict=checkpoint["scheduler"])


def fit(
    state: TrainState,
    epoch_spec: EpochSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    val_epoch_list: list[int],
    train_seed: int,
    val_seed: int,
    test_seed: int,
    history_path: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
    run_uuid: Optional[str] = None,
) -> list[EpochRecord]:
    """Run num_epochs of training; validate only on epochs in val_epoch_list.

    Args:
        train_seed: Seeds the global torch RNG once, before the first epoch, so
            shuffling, dropout, and any randomness a collate draws repeat across runs.
            The caller builds the model before `fit` sees it, so weight init is not
            covered: seed before constructing the model if you want that too. It is
            not handed to training steps or their probe; they get None, and any
            randomness they need comes from the global RNG this seeds.
        val_seed: Handed to every validation step and on to its probe, so a probe that
            draws randomness draws the same thing on each validation pass. Pass the
            repo-wide `VAL_SEED` unless a run has a reason not to; `fit` does not
            reseed the global RNG with it.
        test_seed: Recorded in the history file. `fit` runs no test pass, so nothing
            reads it yet; it is here so a run's three seeds are written down together.
        run_uuid: Identifies this run when `history_path` is not given, so the
            history file lands at `HISTORY_PATH / f"{run_uuid}.json"`. Defaults to a
            fresh uuid4.

    Returns:
        Per-epoch history.

    Side effects:
        Seeds the global torch RNG from `train_seed`, and writes the history file.
    """
    if run_uuid is None:
        run_uuid = str(uuid4())  # noqa: NAR001
    if history_path is None:
        history_path = HISTORY_PATH / f"{run_uuid}.json"
    torch.manual_seed(seed=train_seed)
    logger.info(  # noqa: NAR001
        "fit | epochs {} | val_epochs {} | seeds {}/{}/{} | device {} | wandb {} | history {}",
        num_epochs,
        val_epoch_list,
        train_seed,
        val_seed,
        test_seed,
        epoch_spec.device,
        wandb_run is not None,
        history_path,
    )
    val_epoch_set = set(val_epoch_list)  # noqa: NAR001
    history: list[EpochRecord] = []
    for epoch_number in range(1, num_epochs + 1):  # noqa: NAR001
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
                seed=val_seed,
            )
            val_loss = val_metrics.loss
            logger.info(  # noqa: NAR001
                "epoch {:>3} | train {:.4f} | val {:.4f}",
                epoch_number,
                train_loss,
                val_loss,
            )
        else:
            logger.info("epoch {:>3} | train {:.4f}", epoch_number, train_loss)  # noqa: NAR001
        epoch_record = EpochRecord(
            epoch=epoch_number,
            train_loss=train_loss,
            train_metrics=train_metrics,
            epoch_duration_seconds=epoch_duration_seconds,
            system_metrics=collect_system_metrics(),
            val_loss=val_loss,
            val_metrics=val_metrics,
        )
        record_dict = serialize_epoch_record(record=epoch_record)
        if wandb_run is not None:
            wandb_run.log(data=record_dict)
        history.append(epoch_record)  # noqa: NAR001
    save_history(
        history=[serialize_epoch_record(record=record) for record in history],
        history_path=history_path,
        train_state=state,
        epoch_spec=epoch_spec,
        seeds={"train": train_seed, "val": val_seed, "test": test_seed},
    )
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
    num_epochs: int,
    val_epoch_list: list[int],
    profile_epoch_list: list[int],
    train_seed: int,
    val_seed: int,
    test_seed: int,
    history_path: Optional[Path] = None,
    history_detailed_path: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
    run_uuid: Optional[str] = None,
) -> list[EpochRecord]:
    """Like fit, but wraps profiling epochs with torch.profiler.profile.

    Writes history.json (base metrics, same as fit) and history_detailed.json
    (same records, with profiling metrics and trace_path merged in on
    profiling epochs). Returns the base per-epoch history list.

    Args:
        train_seed: As in `fit`; seeds the global torch RNG for the run.
        val_seed: As in `fit`; handed to every validation step and its probe.
        test_seed: As in `fit`; recorded in both history files, read by nothing.
        run_uuid: As in `fit`; also names the trace directory,
            `TRACES_PATH / run_uuid`. Defaults to a fresh uuid4.

    Side effects:
        Seeds the global torch RNG from `train_seed`, and writes both history files
        and one Chrome trace per profiled epoch.
    """
    from torch.profiler import ProfilerActivity, profile

    if run_uuid is None:
        run_uuid = str(uuid4())  # noqa: NAR001
    if history_path is None:
        history_path = HISTORY_PATH / f"{run_uuid}.json"
    if history_detailed_path is None:
        history_detailed_path = HISTORY_PATH / f"{run_uuid}_detailed.json"
    traces_dir = TRACES_PATH / run_uuid
    torch.manual_seed(seed=train_seed)
    logger.info(  # noqa: NAR001
        "profiled_fit epochs={} val={} profile={} seeds={}/{}/{} device={} history={} detailed={}",
        num_epochs,
        val_epoch_list,
        profile_epoch_list,
        train_seed,
        val_seed,
        test_seed,
        epoch_spec.device,
        history_path,
        history_detailed_path,
    )
    val_epoch_set = set(val_epoch_list)  # noqa: NAR001
    profile_epoch_set = set(profile_epoch_list)  # noqa: NAR001
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)  # noqa: NAR001
    history: list[EpochRecord] = []
    history_detailed: list[dict] = []
    for epoch_number in range(1, num_epochs + 1):  # noqa: NAR001
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
            profiling_metrics = _extract_profiler_metrics(
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
            profiling_metrics = None
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
                seed=val_seed,
            )
            val_loss = val_metrics.loss
            logger.info(  # noqa: NAR001
                "epoch {:>3} | train {:.4f} | val {:.4f}",
                epoch_number,
                train_loss,
                val_loss,
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
            val_loss=val_loss,
            val_metrics=val_metrics,
        )
        record_dict = serialize_epoch_record(record=epoch_record)
        if wandb_run is not None:
            wandb_run.log(data=record_dict)
        history.append(epoch_record)  # noqa: NAR001
        detailed_record = (
            {**record_dict, **profiling_metrics} if profiling_metrics is not None else record_dict
        )
        history_detailed.append(detailed_record)  # noqa: NAR001
    seeds = {"train": train_seed, "val": val_seed, "test": test_seed}
    save_history(
        history=[serialize_epoch_record(record=record) for record in history],
        history_path=history_path,
        train_state=state,
        epoch_spec=epoch_spec,
        seeds=seeds,
    )
    save_history(
        history=history_detailed,
        history_path=history_detailed_path,
        train_state=state,
        epoch_spec=epoch_spec,
        seeds=seeds,
    )
    return history
