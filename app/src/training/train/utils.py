"""utils.py -- persist checkpoints and run history to disk."""

import json
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from training.train.model import EpochSpec, TrainState
from training.utils.git import get_git_commit
from training.utils.serialization import serialize_epoch_spec, serialize_train_state


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
    epoch_spec: Optional[EpochSpec] = None,
) -> None:
    """Write per-epoch history and optional run metadata to a JSON file."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {"git_commit": get_git_commit()}
    if train_state is not None:
        meta["train_state"] = serialize_train_state(state=train_state)
    if epoch_spec is not None:
        meta["train_config"] = serialize_epoch_spec(spec=epoch_spec)
    payload = {"meta": meta, "history": history}
    with open(file=history_path, mode="w") as history_file:
        json.dump(obj=payload, fp=history_file, indent=2)
    logger.info("saved history to {}", history_path)  # noqa: NAR001
