"""
common.py
What every config study shares: LR schedules, norm names, seed repeats, and the
`grid_search` loop that trains a list of configs and ranks them by best val loss.

A study module defines its own frozen `Config` dataclass and the three functions
`grid_search` takes: one that returns loaders for a config, one that builds a fresh
`TrainState`, and one that builds the `EpochSpec`.
"""

import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Optional, cast
from uuid import uuid4

import torch
from loguru import logger
from torch import nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    OneCycleLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from training.constants import ATTENTION_PROBES_PATH, TEST_SEED, VAL_SEED
from training.metrics.masked_token_metrics import MaskedTokenMetrics
from training.train.model import EpochSpec, TrainState
from training.train.train_loop import fit

Schedule = Literal["constant", "cosine", "warmup_cosine", "one_cycle"]

# A config names its norm as a string, not the class, so it reads the same in a log
# line and in a history file; this maps it.
NORMS: dict[str, type[nn.LayerNorm] | type[nn.RMSNorm]] = {
    "layernorm": nn.LayerNorm,
    "rmsnorm": nn.RMSNorm,
}

WARMUP_EPOCHS = 5
# OneCycleLR's default div_factor, so both warmups start from the same LR.
WARMUP_START_FACTOR = 1 / 25


def repeat_for_seeds[C](configs: list[C], seeds: int) -> list[C]:
    """Return `configs` with every entry repeated `seeds` times, adjacent.

    `grid_search` draws a fresh train seed per run, so repeating a config is how one
    arm gets several seeds. The spread inside an arm is the noise floor that a gap
    between arms has to clear to mean anything.
    """
    return [config for config in configs for _ in range(seeds)]  # noqa: NAR001


def build_scheduler(
    schedule: Schedule, optimizer: torch.optim.Optimizer, num_epochs: int
) -> Optional[LRScheduler]:
    """Return the scheduler `schedule` names, sized in epochs, or None for constant.

    Every schedule peaks at the optimizer's LR. The warmups ramp up in steps of one
    epoch, because `fit` steps the scheduler once per epoch, after the training pass.
    Each decay reaches its floor only after the last step, so the final epoch trains
    at a small but nonzero LR.

    Args:
        num_epochs: Total epochs `fit` will run. A schedule sized shorter would end
            early, and one_cycle raises if stepped past its length.
    """
    if schedule == "constant":
        return None
    if schedule == "cosine":
        return CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
    if schedule == "warmup_cosine":
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
    # Momentum cycling off, so this arm differs from warmup_cosine in the LR's shape
    # alone: a cosine-shaped warmup over the first 30% of the run (pct_start=0.3)
    # instead of a linear 5 epochs.
    return OneCycleLR(
        optimizer=optimizer,
        max_lr=optimizer.param_groups[0]["lr"],
        total_steps=num_epochs,
        cycle_momentum=False,
    )


def grid_search[C, D](
    configs: list[C],
    num_epochs: int,
    get_loaders: Callable[[C], tuple[DataLoader, DataLoader, D]],
    build_train_state: Callable[[C, D, int], TrainState],
    build_epoch_spec: Callable[[Path], EpochSpec],
) -> None:
    """Train every config in `configs` for `num_epochs` and log them by best val loss.

    Every run draws its own train seed, so two configs differ in weight init and batch
    order as well as in the config; each run's seed is in its history file.

    Args:
        get_loaders: Returns `(train_loader, val_loader, data)` for a config, where
            `data` is whatever `build_train_state` needs from the loaders, such as
            vocab sizes. Called once per config, before the run is seeded, so cache
            inside it to share loaders between configs.
        build_train_state: Called as `(config, data, num_epochs)`; returns a fresh
            model, optimizer, criterion, and scheduler.
        build_epoch_spec: Called with the run's probe directory under
            ATTENTION_PROBES_PATH. The val metrics must be `MaskedTokenMetrics`,
            because the ranking logs their perplexity and accuracy.

    Side effects:
        Per config: seeds the global torch RNG with a fresh train seed before building
        the model, so weight init repeats too; `fit` writes the history file, seed
        included, and whatever probe the `EpochSpec` holds writes under
        ATTENTION_PROBES_PATH.
    """
    results: list[tuple[float, int, MaskedTokenMetrics, C, str]] = []
    for index, config in enumerate(configs):  # noqa: NAR001
        logger.info("grid {}/{}", index + 1, len(configs))  # noqa: NAR001
        train_loader, val_loader, data = get_loaders(config)  # noqa: NAR001
        # Fresh per run, so no single lucky draw is trained on forever. fit records it;
        # the same data and config are also needed to replay a past setup.
        train_seed = random.getrandbits(32)  # noqa: NAR001
        run_uuid = str(uuid4())  # noqa: NAR001
        # fit reseeds from it, but only after the model exists; seed here so weight
        # init uses the recorded seed too. See the README on what the rewind implies.
        torch.manual_seed(seed=train_seed)
        state = build_train_state(config, data, num_epochs)  # noqa: NAR001
        logger.info("config {} | run {}", config, run_uuid)  # noqa: NAR001
        history = fit(
            run_uuid=run_uuid,
            state=state,
            epoch_spec=build_epoch_spec(ATTENTION_PROBES_PATH / run_uuid),  # noqa: NAR001
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            val_epoch_list=list(range(1, num_epochs + 1)),  # noqa: NAR001
            train_seed=train_seed,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
        )
        best_loss, best = min(  # noqa: NAR001
            ((record.val_loss, record) for record in history if record.val_loss is not None),
            key=lambda pair: pair[0],
        )
        results.append(  # noqa: NAR001
            (
                best_loss,
                best.epoch,
                cast(MaskedTokenMetrics, best.val_metrics),  # noqa: NAR001
                config,
                run_uuid,
            )
        )

    results.sort(key=lambda result: result[0])
    logger.info("grid done | ranked by best val loss")  # noqa: NAR001
    for best_loss, epoch, metrics, config, run_uuid in results:
        logger.info(  # noqa: NAR001
            "val {:.4f} | ppl {:.2f} | acc {:.4f} | epoch {:>2} | {} | {}",
            best_loss,
            metrics.perplexity,
            metrics.accuracy,
            epoch,
            config,
            run_uuid,
        )
