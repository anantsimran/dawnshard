"""
multi30k_translator_ablation.py
Component, LR-schedule, and source-language ablation of `Multi30kTranslator` (see
model.multi30k_translator for the model), a `grid_search` (from ablations.setup)
over a list of `Multi30kTranslatorTrainConfig`s.

ABLATION_REFERENCE is `ablations/multi30k_translator.yaml` under CONFIGS_PATH, pinned
to the values the study runs with, and every arm is that file with a few fields
overridden. Editing the model's own YAML for a one-off run changes no study. The
ablation's results are written up in
`app/src/training/ablations/multi30k_translator/2026-10-13/README.md`, and the scale
study's in `app/src/training/ablations/multi30k_translator/2026-10-15/README.md`.

`main` runs one study from STUDIES, by name, as `grid_search` over its arms repeated
ABLATION_SEEDS times; each arm runs for its own `fit.num_epochs`. The default,
"ablation", is ABLATION_ARMS: ABLATION_REFERENCE, one arm per component changed from
it (COMPONENT_ARMS), one arm per LR schedule in place of its constant LR
(SCHEDULE_ARMS), and one arm per source language in place of English
(LANGUAGE_ARMS). "scale" is SCALE_ARMS for SCALE_EPOCHS: the same model at 256 wide,
deeper, at 512 wide, and with other schedules and dropout. Both use the
`multi30k-{lang}-{bpe_num_merges}.json` tokenizers under TOKENIZERS_PATH, trained the
first time; changing `bpe_num_merges` names other files, which the next run trains.
"best" is not a study: it retrains the scale study's best arm once (`train_best`);
`fit` checkpoints it after every epoch under CHECKPOINTS_PATH / run_uuid.

Run from the repo root, naming the study (default "ablation"); losses are logged to the
terminal after every epoch, and the ranked results once every config has finished:

    uv run python app/src/training/ablations/multi30k_translator_ablation.py scale
"""

import argparse
from dataclasses import replace
from functools import cache, partial
from typing import Any
from uuid import uuid4

import torch
from loguru import logger
from torch.utils.data import DataLoader
from training.ablations.setup import grid_search, repeat_for_seeds
from training.constants import ATTENTION_PROBES_PATH, CONFIGS_PATH, HISTORY_PATH
from training.dal.dao.dao import Dao
from training.dal.dao.sqlite_dao import SqliteDao
from training.dataclass.model import TrainState
from training.dataload.constants import CACHE_PATH
from training.preprocess.tokenizer import Tokenizer
from training.run.multi30k_translator import (
    Multi30kTranslatorTrainConfig,
    build_epoch_spec,
    build_loaders,
    build_train_state,
    loader_seq_len,
)
from training.train.checkpoint import save_history
from training.train.train_loop import fit

PROBE_SENTENCES = 10

type Overrides = dict[str, Any]


def _blocks(**fields: Any) -> Overrides:
    """Return overrides setting `fields` on both stacks' blocks, which every arm shares."""
    return {"model": {"encoder": fields, "decoder": fields}}


def _dropout(rate: float) -> Overrides:
    """Return overrides setting one dropout rate on both embeddings and both stacks."""
    side = {"dropout": rate}
    return {"model": {"src": side, "tgt": side, "encoder": side, "decoder": side}}


def _schedule(name: str) -> Overrides:
    """Return overrides swapping the constant LR for the schedule `name`."""
    return {"scheduler": {"name": name}}


# Every arm is ABLATION_REFERENCE with exactly one thing changed, so a gap against the
# reference is that change's doing. The translator has no sweep of its own yet, so the
# reference borrows the best setting of the Multi30k predictor's 2026-09-29 study:
# d_model=128, dropout 0.1, QK norm, 4 heads, FFN 6x, and 6 blocks a side.
ABLATION_REFERENCE = Multi30kTranslatorTrainConfig.from_yaml(
    path=CONFIGS_PATH / "ablations" / "multi30k_translator.yaml"
)
# One component changed each. The two depth arms each cut one side to 2 blocks, so
# both have 8 in total: against each other they ask which side the capacity is worth
# more on, and against the reference what the 4 dropped blocks cost.
# Post-norm is missing on purpose: EncoderBlock and DecoderBlock hardcode the pre-norm
# order, and switching it is a change to shared code rather than a flag on this model.
COMPONENT_ARMS = [
    ABLATION_REFERENCE.with_overrides(overrides=_blocks(attention={"qk_norm": False})),
    ABLATION_REFERENCE.with_overrides(overrides=_dropout(rate=0.0)),
    ABLATION_REFERENCE.with_overrides(overrides=_blocks(norm="rmsnorm")),
    ABLATION_REFERENCE.with_overrides(overrides={"model": {"num_decoders": 2}}),
    ABLATION_REFERENCE.with_overrides(overrides={"model": {"num_encoders": 2}}),
]
# The reference's constant LR swapped for a schedule, each with the same peak LR.
# Read as a chain: cosine against the reference is what decay buys, warmup_cosine
# against cosine is what warmup buys, one_cycle against warmup_cosine is what a 9-
# rather than 5-epoch warmup costs.
SCHEDULE_ARMS = [
    ABLATION_REFERENCE.with_overrides(overrides=_schedule(name="cosine")),
    ABLATION_REFERENCE.with_overrides(overrides=_schedule(name="warmup_cosine")),
    ABLATION_REFERENCE.with_overrides(overrides=_schedule(name="one_cycle")),
]
# The source swapped, the German target kept, so the arm shares the reference's target
# tokenizer and val sentences and its val loss compares directly. Czech is Slavic and
# its captions are shorter than the English ones, so the arm asks what a distant
# source costs without losing more of it to truncation at seq_length.
LANGUAGE_ARMS = [
    ABLATION_REFERENCE.with_overrides(overrides={"data": {"src_lang": "cs"}}),
]
# The reference is the control for every list, so it runs once.
ABLATION_ARMS = [ABLATION_REFERENCE] + COMPONENT_ARMS + SCHEDULE_ARMS + LANGUAGE_ARMS
ABLATION_SEEDS = 3

# Every arm in the 128 study peaked by epoch 13, and a wider model on the same data
# should peak no later.
SCALE_EPOCHS = 25
# The scale study: the ablation's model at 256 wide is its control, and every arm
# changes one thing from it except the two 512 schedule arms, which change the
# schedule of WIDE_ARM and compare against it. The question is whether more capacity
# helps on 29,000 pairs, and whether the schedules and dropout matter more at this
# size than they did at 128, where all three schedules landed inside noise.
SCALE_REFERENCE = ABLATION_REFERENCE.with_overrides(
    overrides={"model": {"d_model": 256}, "fit": {"num_epochs": SCALE_EPOCHS}}
)
WIDE_ARM = SCALE_REFERENCE.with_overrides(overrides={"model": {"d_model": 512}})
SCALE_ARMS = [
    SCALE_REFERENCE,
    SCALE_REFERENCE.with_overrides(overrides={"model": {"num_encoders": 12, "num_decoders": 12}}),
    WIDE_ARM,
    SCALE_REFERENCE.with_overrides(overrides=_schedule(name="one_cycle")),
    SCALE_REFERENCE.with_overrides(overrides=_schedule(name="warmup_cosine")),
    WIDE_ARM.with_overrides(overrides=_schedule(name="one_cycle")),
    WIDE_ARM.with_overrides(overrides=_schedule(name="warmup_cosine")),
    SCALE_REFERENCE.with_overrides(overrides=_dropout(rate=0.0)),
    SCALE_REFERENCE.with_overrides(overrides=_dropout(rate=0.3)),
]

# Each study is its configs in run order. The ablation keeps a config's seeds
# adjacent, as it ran. The scale study runs every arm once before any arm twice, so a
# third of the way in every arm has one seed, and a clear loser can be cut.
STUDIES: dict[str, list[Multi30kTranslatorTrainConfig]] = {
    "ablation": repeat_for_seeds(configs=ABLATION_ARMS, seeds=ABLATION_SEEDS),
    "scale": SCALE_ARMS * ABLATION_SEEDS,
}

# The scale study's best arm by mean best val loss (2.077), retrained by `train_best`
# from the seed of its best run (96ff2746, 2.068) and stopped at that run's best epoch,
# the median of the arm's three (7, 8, 9). fit keeps no best-epoch weights, so the
# epoch is fixed in advance. The replay is not exact, since the loader shuffles its
# rows before the seed is set, so expect a val loss inside the arm's 2.064-2.099.
BEST_CONFIG = SCALE_REFERENCE.with_overrides(overrides=_schedule(name="warmup_cosine"))
BEST_TRAIN_SEED = 1091505212
BEST_EPOCH = 8

# Arms that share the data settings, seq_length, and batch size share one pair of
# loaders. Val loss is a mean over the target's non-pad positions, so every arm with
# the same `tgt_lang` is scored on the same set of positions.
get_loaders = cache(build_loaders)  # noqa: NAR001


def _loaders_for(
    config: Multi30kTranslatorTrainConfig, cache: Dao
) -> tuple[DataLoader, DataLoader, tuple[Tokenizer, Tokenizer]]:
    """Return `config`'s loaders and tokenizers, shared with every arm that matches it."""
    return get_loaders(
        dataload_config=config.data,
        seq_len=loader_seq_len(model_config=config.model),
        batch_size=config.dataloader.batch_size,
        cache=cache,
    )


def _state_for(
    config: Multi30kTranslatorTrainConfig, tokenizers: tuple[Tokenizer, Tokenizer]
) -> TrainState:
    """Return a fresh `build_train_state` for `config`, sized to `tokenizers`' vocabs."""
    src_tokenizer, tgt_tokenizer = tokenizers
    return build_train_state(
        config=config,
        src_vocab_size=len(src_tokenizer.vocab),  # noqa: NAR001
        tgt_vocab_size=len(tgt_tokenizer.vocab),  # noqa: NAR001
    )


def train_best(cache: Dao) -> None:
    """Retrain BEST_CONFIG for BEST_EPOCH epochs, which `fit` checkpoints.

    The scheduler is sized for BEST_CONFIG's `fit.num_epochs`, SCALE_EPOCHS, as in the
    study, so the epochs that run see the same learning rates the study's run did.

    Args:
        cache: Where the texts and encodings live; `main` opens it.

    Side effects:
        Seeds the global torch RNG, and writes the run's history file, attention
        probes, and, through `fit`, one checkpoint per validation epoch under
        CHECKPOINTS_PATH / run_uuid; the last is `epoch_{BEST_EPOCH}.pt`.
    """
    train_loader, val_loader, tokenizers = _loaders_for(config=BEST_CONFIG, cache=cache)
    # fit reseeds from it, but only after the model exists; seed here so weight init
    # uses the recorded seed too.
    torch.manual_seed(seed=BEST_TRAIN_SEED)
    state = _state_for(config=BEST_CONFIG, tokenizers=tokenizers)
    run_uuid = str(uuid4())  # noqa: NAR001
    # What fit ran. The scheduler above was sized from BEST_CONFIG, not this.
    run_config = replace(  # noqa: NAR001
        BEST_CONFIG,
        fit=replace(  # noqa: NAR001
            BEST_CONFIG.fit,
            num_epochs=BEST_EPOCH,
            train_seed=BEST_TRAIN_SEED,
            run_uuid=run_uuid,
        ),
    )
    epoch_spec = build_epoch_spec(
        probe_dir=ATTENTION_PROBES_PATH / run_uuid, num_sentences=PROBE_SENTENCES
    )
    logger.info("config {} | run {}", BEST_CONFIG, run_uuid)  # noqa: NAR001
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


def main():
    """Run the study named on the command line: every arm, ABLATION_SEEDS times each,
    in the order STUDIES lists them. "best" runs `train_best` instead."""
    parser = argparse.ArgumentParser(description="Run one Multi30kTranslator study.")
    parser.add_argument(  # noqa: NAR001
        "study",
        nargs="?",
        default="ablation",
        choices=list(STUDIES) + ["best"],  # noqa: NAR001
    )
    args = parser.parse_args()
    with SqliteDao(path=CACHE_PATH) as cache:
        if args.study == "best":
            train_best(cache=cache)
            return
        grid_search(
            configs=STUDIES[args.study],
            get_loaders=lambda config: _loaders_for(config=config, cache=cache),
            build_train_state=_state_for,
            build_epoch_spec=partial(build_epoch_spec, num_sentences=PROBE_SENTENCES),  # noqa: NAR001
        )


if __name__ == "__main__":
    main()
