from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from train.train_loop import RuntimeConfig, TrainState


def serialize_train_state(state: TrainState) -> dict:
    optimizer_param_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in state.optimizer.state_dict()["param_groups"]
    ]
    return {
        "model": type(state.model).__name__,  # noqa: NAR001
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


def serialize_runtime_config(config: RuntimeConfig) -> dict:
    return {"device": str(config.device)}  # noqa: NAR001
