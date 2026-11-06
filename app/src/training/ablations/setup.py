"""
setup.py
What every config study shares: seed repeats, and the `grid_search` loop that trains a
list of configs and ranks them by best val loss.

A study is a list of `TrainConfig`s, each its pinned YAML under
`CONFIGS_PATH / "ablations"` with a few fields overridden (`Config.with_overrides`),
and the three functions `grid_search` takes: one that returns loaders for a config,
one that builds a fresh `TrainState`, and one that builds the `EpochSpec`. The model's
`run/` module provides all three.
"""

import random
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast
from uuid import uuid4

import torch
from loguru import logger
from torch.utils.data import DataLoader
from training.constants import ATTENTION_PROBES_PATH, HISTORY_PATH
from training.dataclass.config import TrainConfig
from training.dataclass.model import EpochSpec, TrainState
from training.metrics.masked_token_metrics import MaskedTokenMetrics
from training.train.checkpoint import save_history
from training.train.train_loop import fit


def repeat_for_seeds[C](configs: list[C], seeds: int) -> list[C]:
    """Return `configs` with every entry repeated `seeds` times, adjacent.

    `grid_search` draws a fresh train seed per run, so repeating a config is how one
    arm gets several seeds. The spread inside an arm is the noise floor that a gap
    between arms has to clear to mean anything.
    """
    return [config for config in configs for _ in range(seeds)]  # noqa: NAR001


def grid_search[C: TrainConfig, D](
    configs: list[C],
    get_loaders: Callable[[C], tuple[DataLoader, DataLoader, D]],
    build_train_state: Callable[[C, D], TrainState],
    build_epoch_spec: Callable[[Path], EpochSpec],
) -> None:
    """Train every config in `configs` under its own `fit` section; log them by best val loss.

    Every run draws its own train seed, so two configs differ in weight init and batch
    order as well as in the config; each run's seed is in its history file. A
    `train_seed` set in a config is ignored for the same reason.

    Args:
        get_loaders: Returns `(train_loader, val_loader, data)` for a config, where
            `data` is whatever `build_train_state` needs from the loaders, such as
            tokenizers. Called once per config, before the run is seeded, so cache
            inside it to share loaders between configs.
        build_train_state: Called as `(config, data)`; returns a fresh model,
            optimizer, criterion, and scheduler.
        build_epoch_spec: Called with the run's probe directory under
            ATTENTION_PROBES_PATH. The val metrics must be `MaskedTokenMetrics`,
            because the ranking logs their perplexity and accuracy.

    Side effects:
        Per config: seeds the global torch RNG with a fresh train seed before building
        the model, so weight init repeats too; writes the run's history file under
        HISTORY_PATH, its config and seed included; whatever probe the `EpochSpec`
        holds writes under ATTENTION_PROBES_PATH.
    """
    results: list[tuple[float, int, MaskedTokenMetrics, C, str]] = []
    for index, config in enumerate(configs):  # noqa: NAR001
        logger.info("grid {}/{}", index + 1, len(configs))  # noqa: NAR001
        train_loader, val_loader, data = get_loaders(config)  # noqa: NAR001
        # Fresh per run, so no single lucky draw is trained on forever. fit records it;
        # the same data and config are also needed to replay a past setup.
        train_seed = random.getrandbits(32)  # noqa: NAR001
        run_uuid = str(uuid4())  # noqa: NAR001
        run_config = replace(  # noqa: NAR001
            config,
            fit=replace(config.fit, train_seed=train_seed, run_uuid=run_uuid),  # noqa: NAR001
        )
        # fit reseeds from it, but only after the model exists; seed here so weight
        # init uses the recorded seed too. See the README on what the rewind implies.
        torch.manual_seed(seed=train_seed)
        state = build_train_state(run_config, data)  # noqa: NAR001
        epoch_spec = build_epoch_spec(ATTENTION_PROBES_PATH / run_uuid)  # noqa: NAR001
        logger.info("config {} | run {}", config, run_uuid)  # noqa: NAR001
        history = fit(
            state=state,
            epoch_spec=epoch_spec,
            train_loader=train_loader,
            val_loader=val_loader,
            fit_config=run_config.fit,
        )
        save_history(
            history=history,
            history_path=HISTORY_PATH / f"{run_uuid}.json",
            train_state=state,
            epoch_spec=epoch_spec,
            train_config=run_config,
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
