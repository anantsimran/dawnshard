"""optim.py -- build the optimizer and scheduler a `TrainConfig` names."""

from typing import Callable, Iterable

from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    OneCycleLR,
    SequentialLR,
)
from training.dataclass.config import OptimizerConfig, OptimizerName, SchedulerConfig

OPTIMIZERS: dict[OptimizerName, Callable[..., Optimizer]] = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
}

WARMUP_EPOCHS = 5
# OneCycleLR's default div_factor, so both warmups start from the same LR.
WARMUP_START_FACTOR = 1 / 25


def build_optimizer(config: OptimizerConfig, params: Iterable[nn.Parameter]) -> Optimizer:
    """Return the optimizer `config.name` names, over `params`."""
    return OPTIMIZERS[config.name](params=params, lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(config: SchedulerConfig, optimizer: Optimizer, num_epochs: int) -> LRScheduler:
    """Return the scheduler `config.name` names, over `optimizer`, sized in epochs.

    Every schedule peaks at the optimizer's LR. The warmups ramp up in steps of one
    epoch, because `fit` steps the scheduler once per epoch, after the training pass.
    Each decay reaches its floor only after the last step, so the final epoch trains
    at a small but nonzero LR.

    Args:
        num_epochs: Total epochs `fit` will run. A schedule sized shorter would end
            early, and one_cycle raises if stepped past its length.
    """
    match config.name:
        case "cosine":
            return CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
        case "warmup_cosine":
            return SequentialLR(
                optimizer=optimizer,
                schedulers=[
                    LinearLR(
                        optimizer=optimizer,
                        start_factor=WARMUP_START_FACTOR,
                        total_iters=WARMUP_EPOCHS,
                    ),
                    CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs - WARMUP_EPOCHS),
                ],
                milestones=[WARMUP_EPOCHS],
            )
        case "one_cycle":
            # Momentum cycling off, so this differs from warmup_cosine in the LR's shape
            # alone: a cosine-shaped warmup over the first 30% of the run (pct_start=0.3)
            # instead of a linear 5 epochs.
            return OneCycleLR(
                optimizer=optimizer,
                max_lr=optimizer.param_groups[0]["lr"],
                total_steps=num_epochs,
                cycle_momentum=False,
            )
