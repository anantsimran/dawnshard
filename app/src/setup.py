import os

import torch

import wandb


def get_device() -> torch.device:
    if os.environ.get(key="DISABLE_GPU") == "1":
        return torch.device("cpu")  # noqa: NAR001
    if torch.cuda.is_available():
        return torch.device("cuda")  # noqa: NAR001
    if torch.backends.mps.is_available():
        return torch.device("mps")  # noqa: NAR001
    return torch.device("cpu")  # noqa: NAR001


DEVICE = get_device()


def init_wandb(
    project: str,
    run_name: str | None = None,
    config: dict | None = None,
):
    wandb.login(key=os.environ["WANDB_API_KEY"])
    return wandb.init(
        entity="anantsimran-self",
        project=project,
        name=run_name,
        config=config,
    )
