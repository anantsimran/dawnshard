"""train_loop.py -- functional PyTorch training loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import psutil
import torch
from constants import HISTORY_PATH
from loguru import logger
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from utils.git import get_git_commit
from utils.serialization import serialize_runtime_config, serialize_train_state

Batch = tuple[Tensor, Tensor]
StepFn = Callable[["TrainState", "RuntimeConfig", Batch], tuple[float, int]]
AccumulateFn = Callable[[dict, float, int], None]
ReduceFn = Callable[[dict], float]


@dataclass
class RuntimeConfig:
    device: torch.device
    accumulate_loss: AccumulateFn
    compute_reduced_loss: ReduceFn


@dataclass
class TrainState:
    model: nn.Module
    optimizer: Optimizer
    criterion: nn.Module
    scheduler: Optional[LRScheduler] = None

    @classmethod
    def create(
        cls,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[LRScheduler] = None,
    ) -> "TrainState":
        # Build the optimizer from model.parameters() before calling create();
        # we move the model here so optimizer buffers land on device on first step.
        model.to(device=device)
        return cls(
            model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler
        )


def train_step(
    train_state: TrainState, config: RuntimeConfig, batch: Batch
) -> tuple[float, int]:
    """One optimization step. Returns (mean batch loss, batch size)."""
    input_tensor, target_tensor = (tensor.to(device=config.device) for tensor in batch)
    train_state.optimizer.zero_grad(set_to_none=True)
    predicted = train_state.model(input_tensor)  # noqa: NAR001
    loss = train_state.criterion(predicted, target_tensor)  # noqa: NAR001
    loss.backward()
    train_state.optimizer.step()
    return loss.item(), target_tensor.size(dim=0)


@torch.no_grad()
def eval_step(
    train_state: TrainState, config: RuntimeConfig, batch: Batch
) -> tuple[float, int]:
    """One forward-only step. Returns (mean batch loss, batch size)."""
    input_tensor, target_tensor = (tensor.to(device=config.device) for tensor in batch)
    predicted = train_state.model(input_tensor)  # noqa: NAR001
    loss = train_state.criterion(predicted, target_tensor)  # noqa: NAR001
    return loss.item(), target_tensor.size(dim=0)


def run_epoch(
    state: TrainState,
    config: RuntimeConfig,
    loader: DataLoader,
    step_fn: StepFn,
    *,
    train: bool,
) -> float:
    """Drive one pass over loader with step_fn; return the epoch mean loss."""
    state.model.train(mode=train)
    accumulator = {"accumulated_loss": 0.0, "count": 0}
    for batch in loader:
        batch_loss, batch_size = step_fn(state, config, batch)  # noqa: NAR001
        config.accumulate_loss(accumulator, batch_loss, batch_size)  # noqa: NAR001
    return config.compute_reduced_loss(accumulator)  # noqa: NAR001


def _collect_system_metrics() -> dict:
    cpu_usage_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    metrics: dict = {
        "cpu_usage_percent": cpu_usage_percent,
        "ram_used_gb": ram.used / 1e9,
    }
    if torch.cuda.is_available():
        metrics["gpu_peak_memory_allocated_gb"] = (
            torch.cuda.max_memory_allocated() / 1e9
        )
        metrics["gpu_peak_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    elif torch.backends.mps.is_available():
        metrics["gpu_memory_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
    return metrics


def save_state(state: TrainState, checkpoint_path: Path) -> None:
    """Checkpoint model + optimizer (+ scheduler) so training can resume."""
    payload: dict = {
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
    }
    if state.scheduler is not None:
        payload["scheduler"] = state.scheduler.state_dict()
    torch.save(obj=payload, f=checkpoint_path)


def save_history(
    history: list[dict],
    history_path: Path,
    train_state: Optional[TrainState] = None,
    runtime_config: Optional[RuntimeConfig] = None,
) -> None:
    """Write per-epoch history and optional run metadata to a JSON file."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {"git_commit": get_git_commit()}
    if train_state is not None:
        meta["train_state"] = serialize_train_state(state=train_state)
    if runtime_config is not None:
        meta["train_config"] = serialize_runtime_config(config=runtime_config)
    payload = {"meta": meta, "history": history}
    with open(file=history_path, mode="w") as history_file:
        json.dump(obj=payload, fp=history_file, indent=2)
    logger.info("saved history to {}", history_path)  # noqa: NAR001


def load(state: TrainState, config: RuntimeConfig, checkpoint_path: Path) -> None:
    """Restore model + optimizer (+ scheduler) from a checkpoint."""
    checkpoint = torch.load(f=checkpoint_path, map_location=config.device)
    state.model.load_state_dict(state_dict=checkpoint["model"])
    state.optimizer.load_state_dict(state_dict=checkpoint["optimizer"])
    if state.scheduler is not None and "scheduler" in checkpoint:
        state.scheduler.load_state_dict(state_dict=checkpoint["scheduler"])


def fit(
    state: TrainState,
    config: RuntimeConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    val_epoch_list: list[int],
    history_path: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
) -> list[dict]:
    """Run num_epochs of training; validate only on epochs in val_epoch_list.

    Returns per-epoch history.
    """
    if history_path is None:
        history_path = HISTORY_PATH / f"{uuid4()}.json"
    val_epoch_set = set(val_epoch_list)  # noqa: NAR001
    history: list[dict] = []
    for epoch_number in range(1, num_epochs + 1):  # noqa: NAR001
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start_time = time.perf_counter()
        train_loss = run_epoch(
            state=state,
            config=config,
            loader=train_loader,
            step_fn=train_step,
            train=True,
        )
        if state.scheduler is not None:
            state.scheduler.step()
        epoch_duration_seconds = time.perf_counter() - epoch_start_time
        epoch_record: dict = {
            "epoch": epoch_number,
            "train_loss": train_loss,
            "epoch_duration_seconds": epoch_duration_seconds,
        }
        if epoch_number in val_epoch_set:
            val_loss = run_epoch(
                state=state,
                config=config,
                loader=val_loader,
                step_fn=eval_step,
                train=False,
            )
            epoch_record["val_loss"] = val_loss
            logger.info(  # noqa: NAR001
                "epoch {:>3} | train {:.4f} | val {:.4f}",
                epoch_number,
                train_loss,
                val_loss,
            )
        else:
            logger.info("epoch {:>3} | train {:.4f}", epoch_number, train_loss)  # noqa: NAR001
        epoch_record.update(_collect_system_metrics())  # noqa: NAR001
        if wandb_run is not None:
            wandb_run.log(data=epoch_record)
        history.append(epoch_record)  # noqa: NAR001
    save_history(
        history=history,
        history_path=history_path,
        train_state=state,
        runtime_config=config,
    )
    return history
