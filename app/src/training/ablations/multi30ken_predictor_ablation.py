"""
multi30ken_predictor_ablation.py
Config studies of `Multi30kEnPredictor` (see model.multi30ken_predictor for the model),
each a `grid_search` (from ablations.setup) over a list of
`Multi30kEnPredictorTrainConfig`s.

BASELINE is `ablations/multi30ken_predictor.yaml` under CONFIGS_PATH, pinned to the
values the studies ran with, and every study config is that file with a few fields
overridden. Editing the model's own YAML for a one-off run changes no study.

`main` runs `grid_search` over ABLATION_ARMS repeated ABLATION_SEEDS times for
ABLATION_EPOCHS: ABLATION_REFERENCE, one arm per component removed from it
(COMPONENT_ARMS), and one arm per LR schedule in place of its constant LR
(SCHEDULE_ARMS); its results are in
`app/src/training/ablations/multi30ken_predictor/2026-10-15/`.
Two earlier studies are kept here as the configs that produced
them — `grid_search` over
`one_at_a_time_configs(baseline=BASELINE, sweep=SWEEP)` for BASELINE's 20 epochs
(BASELINE plus each of SWEEP's values changed one at a time, each with a constant and
a cosine LR), and `grid_search` over FOLLOW_UP for FOLLOW_UP_EPOCHS. Their results are
written up in `app/src/training/ablations/multi30ken_predictor/2026-09-29/`. All of
them use the `multi30k-en-{bpe_num_merges}.json` tokenizer under TOKENIZERS_PATH,
trained the first time; changing `bpe_num_merges` names another file, which the next
run trains.

Run from the repo root; losses are logged to the terminal after every epoch, and the
ranked results once every config has finished:

    uv run python app/src/training/ablations/multi30ken_predictor_ablation.py
"""

from functools import cache, partial
from typing import Any

from torch.utils.data import DataLoader
from training.ablations.setup import grid_search, repeat_for_seeds
from training.constants import CONFIGS_PATH
from training.dal.dao.dao import Dao
from training.dal.dao.sqlite_dao import SqliteDao
from training.dataclass.model import TrainState
from training.dataload.constants import CACHE_PATH
from training.preprocess.tokenizer import Tokenizer
from training.run.multi30ken_predictor import (
    Multi30kEnPredictorTrainConfig,
    build_epoch_spec,
    build_loaders,
    build_train_state,
)

PROBE_SENTENCES = 10

type Overrides = dict[str, Any]


def _dropout(rate: float) -> Overrides:
    """Return overrides setting one dropout rate on the embedding and every block."""
    return {"model": {"embedding": {"dropout": rate}, "encoder": {"dropout": rate}}}


def _schedule(name: str) -> Overrides:
    """Return overrides swapping the constant LR for the schedule `name`."""
    return {"scheduler": {"name": name}}


CONSTANT_LR: Overrides = {"scheduler": None}

BASELINE = Multi30kEnPredictorTrainConfig.from_yaml(
    path=CONFIGS_PATH / "ablations" / "multi30ken_predictor.yaml"
)
# Values other than BASELINE's per field, each tried with every other field at
# BASELINE. Every d_model stays divisible by every number of heads.
SWEEP: list[Overrides] = [
    *({"model": {"d_model": value}} for value in (32, 128)),
    *({"model": {"embedding": {"seq_length": value}}} for value in (16, 64)),
    *(_dropout(rate=value) for value in (0.0, 0.2, 0.3)),
    *({"model": {"num_encoders": value}} for value in (3, 9)),
    *({"model": {"encoder": {"attention": {"h": value}}}} for value in (2, 8)),
    *({"model": {"encoder": {"ffn_multiplier": value}}} for value in (4, 8)),
]
# Follow-ups at d_model=128 for 50 epochs. The control runs twice, so the gap between
# its two seeds says how big a difference has to be before it is real.
FOLLOW_UP_EPOCHS = 50
FOLLOW_UP_CONTROL = BASELINE.with_overrides(
    overrides={"model": {"d_model": 128}, "fit": {"num_epochs": FOLLOW_UP_EPOCHS}}
)
FOLLOW_UP = [
    FOLLOW_UP_CONTROL,
    FOLLOW_UP_CONTROL,
    FOLLOW_UP_CONTROL.with_overrides(
        overrides={"model": {"encoder": {"attention": {"qk_norm": False}}}}
    ),
    FOLLOW_UP_CONTROL.with_overrides(overrides=_dropout(rate=0.0)),
]

# The ablation. Every arm is ABLATION_REFERENCE with exactly one thing changed, so a
# gap against the reference is that change's doing. The reference is the best setting
# the 2026-09-29 study found, trained past where that study stopped: every one of its
# 34 runs was still improving at its last epoch, so ablating at 50 would measure
# partial training as much as the component.
ABLATION_EPOCHS = 100
ABLATION_REFERENCE = BASELINE.with_overrides(
    overrides={"model": {"d_model": 128}, "fit": {"num_epochs": ABLATION_EPOCHS}}
)
# One component removed each. Cut to fit a 5-hour budget at 100 epochs and 3 seeds,
# keeping dropout, the verdict most likely to flip with length. qk_norm=False has a
# clear 50-epoch result (+0.095), and final_norm=False, one head, norm="rmsnorm",
# num_encoders=3, and position_encoding=False wait for a longer run. The model keeps
# every flag, so each is one line to restore.
# Post-norm is missing on purpose: EncoderBlock hardcodes the pre-norm order, and
# switching it is a change to shared code rather than a flag on this model.
COMPONENT_ARMS = [
    ABLATION_REFERENCE.with_overrides(overrides=_dropout(rate=0.0)),
]
# The reference's constant LR swapped for a schedule with the same peak LR: cosine
# against the reference is what decay buys. warmup_cosine and one_cycle, which would
# price the warmup, were cut for the same budget.
SCHEDULE_ARMS = [
    ABLATION_REFERENCE.with_overrides(overrides=_schedule(name="cosine")),
]
# The reference is the control for both lists, so it runs once.
ABLATION_ARMS = [ABLATION_REFERENCE] + COMPONENT_ARMS + SCHEDULE_ARMS
ABLATION_SEEDS = 3

# Configs that share the data settings, seq_length, batch size, and seeds share one
# pair of loaders.
get_loaders = cache(build_loaders)  # noqa: NAR001


def one_at_a_time_configs(
    baseline: Multi30kEnPredictorTrainConfig, sweep: list[Overrides]
) -> list[Multi30kEnPredictorTrainConfig]:
    """Return `baseline`, then one config per sweep entry, each without and with cosine.

    Each non-baseline config differs from `baseline` in exactly one sweep entry, so a
    difference in its result is that entry's doing. Cosine annealing is crossed with
    all of them rather than swept alone, doubling the count.
    """
    architectures = [baseline] + [baseline.with_overrides(overrides=entry) for entry in sweep]
    return [
        config.with_overrides(overrides=schedule)
        for config in architectures
        for schedule in (CONSTANT_LR, _schedule(name="cosine"))
    ]


def _loaders_for(
    config: Multi30kEnPredictorTrainConfig, cache: Dao
) -> tuple[DataLoader, DataLoader, Tokenizer]:
    """Return `config`'s loaders and tokenizer, shared with every config that matches it."""
    return get_loaders(
        dataload_config=config.data,
        seq_len=config.model.embedding.seq_length,
        batch_size=config.dataloader.batch_size,
        val_seed=config.fit.val_seed,
        test_seed=config.fit.test_seed,
        cache=cache,
    )


def _state_for(config: Multi30kEnPredictorTrainConfig, tokenizer: Tokenizer) -> TrainState:
    """Return a fresh `build_train_state` for `config`, sized to `tokenizer`'s vocab."""
    return build_train_state(config=config, vocab_size=len(tokenizer.vocab))  # noqa: NAR001


def main():
    """Run the component ablation: every arm, ABLATION_SEEDS times each."""
    with SqliteDao(path=CACHE_PATH) as cache:
        grid_search(
            configs=repeat_for_seeds(configs=ABLATION_ARMS, seeds=ABLATION_SEEDS),
            get_loaders=lambda config: _loaders_for(config=config, cache=cache),
            build_train_state=_state_for,
            build_epoch_spec=partial(build_epoch_spec, num_sentences=PROBE_SENTENCES),  # noqa: NAR001
        )


if __name__ == "__main__":
    main()
