"""checkpoint.py -- persist a run to disk: weights to a checkpoint, metrics to history.

`checkpoint` and `load` round-trip a `TrainState`'s weights and optimizer state.
`save_history` writes a run's per-epoch records and the config it was built from to
JSON. The `serialize_*` functions turn the loop's dataclasses into the plain dicts
that the history file and wandb hold. `fit` calls none of the writers; the caller
decides what a run leaves behind.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from training.dataclass.config import Config
from training.dataclass.model import EpochRecord, EpochSpec, TrainState
from training.utils.git import get_git_commit


def checkpoint(state: TrainState, checkpoint_path: Path) -> None:
    """Save the model weights and optimizer (+ scheduler) state, for `load` to restore.

    The file holds `state_dict`s only, so loading it needs a `TrainState` built the same
    way: same model class and constructor arguments, same optimizer and scheduler.

    Side effects:
        Creates `checkpoint_path`'s parent folder, and overwrites `checkpoint_path`.
    """
    payload: dict = {
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
    }
    if state.scheduler is not None:
        payload["scheduler"] = state.scheduler.state_dict()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj=payload, f=checkpoint_path)
    logger.info("saved checkpoint to {}", checkpoint_path)  # noqa: NAR001


def load(state: TrainState, epoch_spec: EpochSpec, checkpoint_path: Path) -> None:
    """Restore model + optimizer (+ scheduler) from a `checkpoint` file.

    Side effects:
        Overwrites the parameters and state of `state`'s model, optimizer, and
        scheduler in place.
    """
    payload = torch.load(f=checkpoint_path, map_location=epoch_spec.device)
    state.model.load_state_dict(state_dict=payload["model"])
    state.optimizer.load_state_dict(state_dict=payload["optimizer"])
    if state.scheduler is not None and "scheduler" in payload:
        state.scheduler.load_state_dict(state_dict=payload["scheduler"])


def save_history(
    history: list[EpochRecord],
    history_path: Path,
    train_state: Optional[TrainState] = None,
    epoch_spec: Optional[EpochSpec] = None,
    train_config: Optional[Config] = None,
    profiling_metrics: Optional[dict[int, dict]] = None,
) -> None:
    """Write per-epoch history and optional run metadata to a JSON file.

    `meta.epoch` is the last record's epoch, and is left out when `history` is empty.

    Args:
        history: As `fit` returns it; each record goes through
            `serialize_epoch_record`.
        train_config: The whole config the run was built from, with the seeds and
            `run_uuid` it actually ran with, so the run can be repeated. Written under
            `meta.train_config`.
        profiling_metrics: Profiler metrics keyed by epoch, as `profiled_fit` returns
            them, merged into those epochs' rows.

    Side effects:
        Creates `history_path`'s parent folder, and overwrites `history_path`.
    """
    rows = [serialize_epoch_record(record=record) for record in history]
    if profiling_metrics is not None:
        rows = [
            {**row, **profiling_metrics.get(row["epoch"], {})}  # noqa: NAR001
            for row in rows
        ]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {"git_commit": get_git_commit()}
    if train_state is not None:
        meta["train_state"] = serialize_train_state(state=train_state)
    if epoch_spec is not None:
        meta["epoch_spec"] = serialize_epoch_spec(spec=epoch_spec)
    if train_config is not None:
        meta["train_config"] = train_config.to_dict()
    if history:
        meta["epoch"] = history[-1].epoch
    payload = {"meta": meta, "history": rows}
    with open(file=history_path, mode="w") as history_file:
        json.dump(obj=payload, fp=history_file, indent=2)
    logger.info("saved history to {}", history_path)  # noqa: NAR001


def serialize_train_state(state: TrainState) -> dict:
    """Summarize a TrainState as JSON-serializable metadata, never weights.

    The model is recorded as its class name, its parameter count, and the shape of
    every named parameter, so two runs can be told apart by architecture even when
    they share a class.
    """
    optimizer_param_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in state.optimizer.state_dict()["param_groups"]
    ]
    return {
        "model": type(state.model).__name__,  # noqa: NAR001
        "num_parameters": sum(  # noqa: NAR001
            parameter.numel() for parameter in state.model.parameters()
        ),
        "parameter_shapes": {
            name: list(parameter.shape)  # noqa: NAR001
            for name, parameter in state.model.named_parameters()
        },
        "optimizer": {
            "type": type(state.optimizer).__name__,  # noqa: NAR001
            "param_groups": optimizer_param_groups,
        },
        "criterion": type(state.criterion).__name__,  # noqa: NAR001
        "scheduler": (
            type(state.scheduler).__name__  # noqa: NAR001
            if state.scheduler is not None
            else None
        ),
    }


def serialize_epoch_spec(spec: EpochSpec) -> dict:
    """Summarize an EpochSpec as JSON-serializable metadata."""
    return {"device": str(spec.device)}  # noqa: NAR001


def serialize_epoch_record(record: EpochRecord) -> dict:
    """Flatten an EpochRecord into the history-JSON / wandb dict.

    System metrics are hoisted to the top level and unset fields are dropped, so
    history files keep the flat shape that the viz scripts and older runs use.
    """
    flat = asdict(obj=record)
    flat.update(flat.pop("system_metrics"))  # noqa: NAR001
    return {key: value for key, value in flat.items() if value is not None}
