from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.train.model import EpochRecord, EpochSpec, TrainState


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
