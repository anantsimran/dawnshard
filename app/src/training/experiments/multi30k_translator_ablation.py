"""
multi30k_translator_ablation.py
Component, LR-schedule, and source-language ablation of `Multi30kTranslator` (see
model.multi30k_translator for the model), a `grid_search` (from experiments.common)
over a list of `Config`s.

The constants and ABLATION_REFERENCE here are pinned to the values the study runs
with, apart from the model module's constants, so editing those for a one-off run
changes no study. The ablation's results are written up in
`experiments/multi30k_translator/2026-10-13/README.md`, and the scale study's in
`experiments/multi30k_translator/2026-10-15/README.md`.

`main` runs one study from STUDIES, by name, as `grid_search` over its arms repeated
ABLATION_SEEDS times. The default, "ablation", is ABLATION_ARMS for ABLATION_EPOCHS:
ABLATION_REFERENCE, one arm per component changed from it (COMPONENT_ARMS), one arm
per LR schedule in place of its constant LR (SCHEDULE_ARMS), and one arm per source
language in place of English (LANGUAGE_ARMS). "scale" is SCALE_ARMS for SCALE_EPOCHS:
the same model at 256 wide, deeper, at 512 wide, and with other schedules and
dropout. Both load the default merges files rather than retraining BPE; to set them,
call the loader once per language pair with `retrain_tokenizer=True,
set_default_tokenizer=True`.

Run from the repo root, naming the study (default "ablation"); losses are logged to the
terminal after every epoch, and the ranked results once every config has finished:

    uv run python app/src/training/experiments/multi30k_translator_ablation.py scale
"""

import argparse
from dataclasses import dataclass, replace
from functools import cache, partial
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from training.dataload.multi30k import get_multi30k_translation_dataloader
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
from training.model.multi30k_translator import Multi30kTranslator
from training.setup import DEVICE
from training.train.model import EpochSpec, TrainState
from training.transformer.probes import save_attention_maps

PROBE_SENTENCES = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class Config:
    d_model: int
    seq_len: int
    dropout_rate: float
    num_encoders: int
    num_decoders: int
    # Shared by the encoder and decoder stacks, as are ffn_multiplier and norm.
    num_heads: int
    ffn_multiplier: int
    # Every schedule is sized in epochs, because fit steps the scheduler once per
    # epoch; build_scheduler maps the name to one.
    schedule: Schedule = "constant"
    qk_norm: bool = True
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    # Multi30k language codes. Val loss is per target BPE token, so only arms that share
    # tgt_lang share a tokenizer and are comparable by it.
    src_lang: str = "en"
    tgt_lang: str = "de"


# Every arm is ABLATION_REFERENCE with exactly one thing changed, so a gap against the
# reference is that change's doing. The translator has no sweep of its own yet, so the
# reference borrows the best setting of the Multi30k predictor's 2026-09-29 study:
# d_model=128, dropout 0.1, QK norm, 4 heads, FFN 6x, and 6 blocks a side.
ABLATION_REFERENCE = Config(
    d_model=128,
    seq_len=32,
    dropout_rate=0.1,
    num_encoders=6,
    num_decoders=6,
    num_heads=4,
    ffn_multiplier=6,
)
# One component changed each. The two depth arms each cut one side to 2 blocks, so
# both have 8 in total: against each other they ask which side the capacity is worth
# more on, and against the reference what the 4 dropped blocks cost.
# Post-norm is missing on purpose: EncoderBlock and DecoderBlock hardcode the pre-norm
# order, and switching it is a change to shared code rather than a flag on this model.
COMPONENT_ARMS = [
    replace(ABLATION_REFERENCE, qk_norm=False),  # noqa: NAR001
    replace(ABLATION_REFERENCE, dropout_rate=0.0),  # noqa: NAR001
    replace(ABLATION_REFERENCE, norm="rmsnorm"),  # noqa: NAR001
    replace(ABLATION_REFERENCE, num_decoders=2),  # noqa: NAR001
    replace(ABLATION_REFERENCE, num_encoders=2),  # noqa: NAR001
]
# The reference's constant LR swapped for a schedule, each with the same peak LR.
# Read as a chain: cosine against the reference is what decay buys, warmup_cosine
# against cosine is what warmup buys, one_cycle against warmup_cosine is what a 9-
# rather than 5-epoch warmup costs.
SCHEDULE_ARMS = [
    replace(ABLATION_REFERENCE, schedule="cosine"),  # noqa: NAR001
    replace(ABLATION_REFERENCE, schedule="warmup_cosine"),  # noqa: NAR001
    replace(ABLATION_REFERENCE, schedule="one_cycle"),  # noqa: NAR001
]
# The source swapped, the German target kept, so the arm shares the reference's target
# tokenizer and val sentences and its val loss compares directly. Czech is Slavic and
# its captions are shorter than the English ones, so the arm asks what a distant
# source costs without losing more of it to truncation at seq_len.
LANGUAGE_ARMS = [
    replace(ABLATION_REFERENCE, src_lang="cs"),  # noqa: NAR001
]
# The reference is the control for every list, so it runs once.
ABLATION_ARMS = [ABLATION_REFERENCE] + COMPONENT_ARMS + SCHEDULE_ARMS + LANGUAGE_ARMS
ABLATION_SEEDS = 3
# The first control run (100 epochs) bottomed out in val loss at epoch 9 and climbed
# steadily after, so 30 epochs covers every arm's minimum at under a third of the cost.
ABLATION_EPOCHS = 30

# The scale study: the ablation's model at 256 wide is its control, and every arm
# changes one thing from it except the two 512 schedule arms, which change the
# schedule of WIDE_ARM and compare against it. The question is whether more capacity
# helps on 29,000 pairs, and whether the schedules and dropout matter more at this
# size than they did at 128, where all three schedules landed inside noise.
SCALE_REFERENCE = replace(ABLATION_REFERENCE, d_model=256)  # noqa: NAR001
WIDE_ARM = replace(SCALE_REFERENCE, d_model=512)  # noqa: NAR001
SCALE_ARMS = [
    SCALE_REFERENCE,
    replace(SCALE_REFERENCE, num_encoders=12, num_decoders=12),  # noqa: NAR001
    WIDE_ARM,
    replace(SCALE_REFERENCE, schedule="one_cycle"),  # noqa: NAR001
    replace(SCALE_REFERENCE, schedule="warmup_cosine"),  # noqa: NAR001
    replace(WIDE_ARM, schedule="one_cycle"),  # noqa: NAR001
    replace(WIDE_ARM, schedule="warmup_cosine"),  # noqa: NAR001
    replace(SCALE_REFERENCE, dropout_rate=0.0),  # noqa: NAR001
    replace(SCALE_REFERENCE, dropout_rate=0.3),  # noqa: NAR001
]
# Every arm in the 128 study peaked by epoch 13, and a wider model on the same data
# should peak no later.
SCALE_EPOCHS = 25

# Each study is its configs in run order and its epochs. The ablation keeps a config's
# seeds adjacent, as it ran. The scale study runs every arm once before any arm twice,
# so a third of the way in every arm has one seed, and a clear loser can be cut.
STUDIES: dict[str, tuple[list[Config], int]] = {
    "ablation": (
        repeat_for_seeds(configs=ABLATION_ARMS, seeds=ABLATION_SEEDS),
        ABLATION_EPOCHS,
    ),
    "scale": (SCALE_ARMS * ABLATION_SEEDS, SCALE_EPOCHS),
}


@cache
def get_loaders(
    seq_len: int, src_lang: str, tgt_lang: str
) -> tuple[DataLoader, DataLoader, tuple[int, int]]:
    """Return the train and val loaders, and the source and target vocab sizes.

    Cached, so every arm with the same arguments shares one pair of loaders and one
    pair of tokenizers. Val loss is a mean over the target's non-pad positions, so
    every arm with the same `tgt_lang` is scored on the same set of positions.
    """
    train_loader, val_loader, _, _, src_vocab, _, tgt_vocab = get_multi30k_translation_dataloader(
        seq_len=seq_len,
        batch_size=BATCH_SIZE,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    return train_loader, val_loader, (len(src_vocab), len(tgt_vocab))  # noqa: NAR001


def build_train_state(config: Config, vocab_sizes: tuple[int, int], num_epochs: int) -> TrainState:
    """Return a fresh model, optimizer, criterion, and scheduler for `config`.

    Args:
        vocab_sizes: `(len(src_vocab), len(tgt_vocab))` of the tokenizers the loaders
            were built with.
        num_epochs: The schedule's length, when `config` has one; unused for a
            constant LR.

    Side effects:
        Initializing the weights draws from the global torch RNG, so seed it first
        for the init to repeat.
    """
    src_vocab_size, tgt_vocab_size = vocab_sizes
    model = Multi30kTranslator(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.d_model,
        src_seq_length=config.seq_len,
        tgt_seq_length=config.seq_len,
        dropout=config.dropout_rate,
        num_encoder_heads=config.num_heads,
        num_decoder_heads=config.num_heads,
        encoder_ffn_multiplier=config.ffn_multiplier,
        decoder_ffn_multiplier=config.ffn_multiplier,
        num_encoders=config.num_encoders,
        num_decoders=config.num_decoders,
        qk_norm=config.qk_norm,
        norm=NORMS[config.norm],
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
    """Return the next-token policy, saving attention maps under `probe_dir`.

    `MaskedTokenMetrics` scores every position whose target isn't
    `CROSS_ENTROPY_IGNORE_INDEX`, which the translation loader puts on the pads, so
    its accuracy and perplexity are over every real target token.
    """
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
    """Run the study named on the command line: every arm, ABLATION_SEEDS times each,
    in the order STUDIES lists them."""
    parser = argparse.ArgumentParser(description="Run one Multi30kTranslator study.")
    parser.add_argument(  # noqa: NAR001
        "study",
        nargs="?",
        default="ablation",
        choices=list(STUDIES),  # noqa: NAR001
    )
    args = parser.parse_args()
    configs, num_epochs = STUDIES[args.study]
    grid_search(
        configs=configs,
        num_epochs=num_epochs,
        get_loaders=lambda config: get_loaders(
            seq_len=config.seq_len, src_lang=config.src_lang, tgt_lang=config.tgt_lang
        ),
        build_train_state=build_train_state,
        build_epoch_spec=build_epoch_spec,
    )


if __name__ == "__main__":
    main()
