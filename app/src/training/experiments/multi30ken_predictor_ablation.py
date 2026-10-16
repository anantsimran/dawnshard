"""
multi30ken_predictor_ablation.py
Config studies of `Multi30kEnPredictor` (see model.multi30ken_predictor for the model),
each a `grid_search` (from experiments.common) over a list of `Config`s.

The constants and BASELINE here are pinned to the values the studies ran with, apart
from the model module's constants, so editing those for a one-off run changes no study.

`main` runs `grid_search` over ABLATION_ARMS repeated ABLATION_SEEDS times for
ABLATION_EPOCHS: ABLATION_REFERENCE, one arm per component removed from it
(COMPONENT_ARMS), and one arm per LR schedule in place of its constant LR
(SCHEDULE_ARMS); its results are in `experiments/multi30ken_predictor/2026-10-15/`.
Two earlier studies are kept here as the configs that produced
them — `grid_search` over
`one_at_a_time_configs(baseline=BASELINE, sweep=SWEEP)` for NUM_EPOCHS (BASELINE plus
each of SWEEP's values changed one at a time, each with a constant and a cosine LR),
and `grid_search` over FOLLOW_UP for FOLLOW_UP_EPOCHS. Their results are written up in
`experiments/multi30ken_predictor/2026-09-29/`. All of them load the default merges file
rather than retraining BPE; to change it, call the loader once with
`retrain_tokenizer=True, set_default_tokenizer=True`.

Run from the repo root; losses are logged to the terminal after every epoch, and the
ranked results once every config has finished:

    uv run python app/src/training/experiments/multi30ken_predictor_ablation.py
"""

from dataclasses import dataclass, replace
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from training.constants import TEST_SEED, VAL_SEED
from training.dataload.multi30k import get_multi30ken_mlm_dataloader
from training.experiments.common import (
    NORMS,
    Schedule,
    build_scheduler,
    grid_search,
    repeat_for_seeds,
)
from training.metrics.masked_token_metrics import (
    MaskedTokenMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.model.multi30ken_predictor import Multi30kEnPredictor
from training.setup import DEVICE
from training.train.model import EpochSpec, TrainState
from training.transformer.probes import save_attention_maps

PROBE_SENTENCES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class Config:
    d_model: int
    seq_len: int
    dropout_rate: float
    num_encoders: int
    num_heads: int
    ffn_multiplier: int
    # Every schedule is sized in epochs, because fit steps the scheduler once per
    # epoch; build_scheduler maps the name to one.
    schedule: Schedule = "constant"
    qk_norm: bool = True
    # The last three name a component the model can drop, for the ablation below.
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    position_encoding: bool = True
    final_norm: bool = True


BASELINE = Config(
    d_model=64,
    seq_len=32,
    dropout_rate=0.1,
    num_encoders=6,
    num_heads=4,
    ffn_multiplier=6,
)
# Values other than BASELINE's per field, each tried with every other field at
# BASELINE. Every d_model stays divisible by every num_heads.
SWEEP: dict[str, tuple[Any, ...]] = {
    "d_model": (32, 128),
    "seq_len": (16, 64),
    "dropout_rate": (0.0, 0.2, 0.3),
    "num_encoders": (3, 9),
    "num_heads": (2, 8),
    "ffn_multiplier": (4, 8),
}
# Follow-ups at d_model=128 for 50 epochs. The control runs twice, so the gap between
# its two seeds says how big a difference has to be before it is real.
FOLLOW_UP_CONTROL = replace(BASELINE, d_model=128)  # noqa: NAR001
FOLLOW_UP = [
    FOLLOW_UP_CONTROL,
    FOLLOW_UP_CONTROL,
    replace(FOLLOW_UP_CONTROL, qk_norm=False),  # noqa: NAR001
    replace(FOLLOW_UP_CONTROL, dropout_rate=0.0),  # noqa: NAR001
]
FOLLOW_UP_EPOCHS = 50

# The ablation. Every arm is ABLATION_REFERENCE with exactly one thing changed, so a
# gap against the reference is that change's doing. The reference is the best setting
# the 2026-09-29 study found, trained past where that study stopped: every one of its
# 34 runs was still improving at its last epoch, so ablating at 50 would measure
# partial training as much as the component.
ABLATION_REFERENCE = replace(BASELINE, d_model=128)  # noqa: NAR001
# One component removed each. Cut to fit a 5-hour budget at 100 epochs and 3 seeds,
# keeping dropout, the verdict most likely to flip with length. qk_norm=False has a
# clear 50-epoch result (+0.095), and final_norm=False, num_heads=1, norm="rmsnorm",
# num_encoders=3, and position_encoding=False wait for a longer run. The model keeps
# every flag, so each is one line to restore.
# Post-norm is missing on purpose: EncoderBlock hardcodes the pre-norm order, and
# switching it is a change to shared code rather than a flag on this model.
COMPONENT_ARMS = [
    replace(ABLATION_REFERENCE, dropout_rate=0.0),  # noqa: NAR001
]
# The reference's constant LR swapped for a schedule with the same peak LR: cosine
# against the reference is what decay buys. warmup_cosine and one_cycle, which would
# price the warmup, were cut for the same budget.
SCHEDULE_ARMS = [
    replace(ABLATION_REFERENCE, schedule="cosine"),  # noqa: NAR001
]
# The reference is the control for both lists, so it runs once.
ABLATION_ARMS = [ABLATION_REFERENCE] + COMPONENT_ARMS + SCHEDULE_ARMS
ABLATION_SEEDS = 3
ABLATION_EPOCHS = 100


def one_at_a_time_configs(baseline: Config, sweep: dict[str, tuple[Any, ...]]) -> list[Config]:
    """Return `baseline`, then one config per sweep value, each without and with cosine.

    Each non-baseline config differs from `baseline` in exactly one sweep field, so a
    difference in its result is that field's doing. Cosine annealing is crossed with
    all of them rather than swept alone, doubling the count.
    """
    architectures = [baseline] + [
        replace(baseline, **{name: value})  # noqa: NAR001
        for name, values in sweep.items()
        for value in values
    ]
    return [
        replace(config, schedule=schedule)  # noqa: NAR001
        for config in architectures
        for schedule in ("constant", "cosine")
    ]


@cache
def get_loaders(seq_len: int) -> tuple[DataLoader, DataLoader, int]:
    """Return the train and val loaders for `seq_len`, and the vocab size.

    Cached, so configs that share a `seq_len` share loaders. They share a tokenizer
    only if a default merges file has been configured; on a fresh clone without one,
    each `seq_len` retrains BPE and the sweep does not share one tokenizer. Val loss
    is a mean over selected positions, and a shorter `seq_len` truncates long
    sentences and drops their tail positions from val, so its val loss is over a
    slightly different set of positions than the others'.
    """
    train_loader, val_loader, _, _, vocab = get_multi30ken_mlm_dataloader(
        seq_len=seq_len,
        val_seed=VAL_SEED,
        test_seed=TEST_SEED,
        batch_size=BATCH_SIZE,
    )
    return train_loader, val_loader, len(vocab)  # noqa: NAR001


def build_train_state(config: Config, vocab_size: int, num_epochs: int) -> TrainState:
    """Return a fresh model, optimizer, criterion, and scheduler for `config`.

    Args:
        vocab_size: `len(vocab)` of the tokenizer the loaders were built with.
        num_epochs: The schedule's length, when `config` has one; unused for a
            constant LR.

    Side effects:
        Initializing the weights draws from the global torch RNG, so seed it first
        for the init to repeat.
    """
    model = Multi30kEnPredictor(
        dropout_rate=config.dropout_rate,
        seq_len=config.seq_len,
        d_model=config.d_model,
        vocab_size=vocab_size,
        num_encoders=config.num_encoders,
        h=config.num_heads,
        ffn_multiplier=config.ffn_multiplier,
        qk_norm=config.qk_norm,
        norm=NORMS[config.norm],
        position_encoding=config.position_encoding,
        final_norm=config.final_norm,
    ).to(device=DEVICE)  # noqa: NAR001
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    return TrainState(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=build_scheduler(
            schedule=config.schedule, optimizer=optimizer, num_epochs=num_epochs
        ),
    )


def build_epoch_spec(probe_dir: Path) -> EpochSpec:
    """Return the masked-token policy, saving attention maps under `probe_dir`."""
    return EpochSpec(
        device=DEVICE,
        metrics_type=MaskedTokenMetrics,
        calculate_metrics=calculate_metrics,
        accumulate_metrics=accumulate_metrics,
        reduce_metrics=reduce_metrics,
        eval_probe=partial(  # noqa: NAR001
            save_attention_maps,
            out_dir=probe_dir,
            num_sentences=PROBE_SENTENCES,
        ),
    )


def main():
    """Run the component ablation: every arm, ABLATION_SEEDS times each."""
    grid_search(
        configs=repeat_for_seeds(configs=ABLATION_ARMS, seeds=ABLATION_SEEDS),
        num_epochs=ABLATION_EPOCHS,
        get_loaders=lambda config: get_loaders(seq_len=config.seq_len),
        build_train_state=build_train_state,
        build_epoch_spec=build_epoch_spec,
    )


if __name__ == "__main__":
    main()
