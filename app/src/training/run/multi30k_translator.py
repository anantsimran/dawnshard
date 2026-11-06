"""
multi30k_translator.py
Train one `Multi30kTranslator` (see model.multi30k_translator) from a YAML config.

`Multi30kTranslatorTrainConfig` is the YAML's shape, and `multi30k_translator.yaml`
under CONFIGS_PATH is the default. Each side's tokenizer is
`multi30k-{lang}-{bpe_num_merges}.json` under TOKENIZERS_PATH: the first run trains and
writes it, and later runs load it. `build_loaders`, `build_train_state`, and
`build_epoch_spec` are shared with `ablations/multi30k_translator_ablation.py`, so a
study arm is built exactly as a single run is.

Train from the repo root, optionally naming another YAML; losses are logged to the
terminal after every epoch:

    uv run python app/src/training/run/multi30k_translator.py
    uv run python app/src/training/run/multi30k_translator.py --config my_run.yaml
"""

import argparse
import random
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import DataLoader
from training.constants import ATTENTION_PROBES_PATH, CONFIGS_PATH, HISTORY_PATH
from training.dal.dao.dao import Dao
from training.dal.dao.sqlite_dao import SqliteDao
from training.dataclass.config import DataloadConfig, TrainConfig, TransformerConfig
from training.dataclass.model import EpochSpec, TrainState
from training.dataload.constants import CACHE_PATH, TOKENIZERS_PATH
from training.dataload.multi30k import get_multi30k_texts, get_multi30k_translation_dataloader
from training.metrics.masked_token_metrics import (
    MaskedTokenMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.model.multi30k_translator import Multi30kTranslator
from training.preprocess.tokenizer import Tokenizer, get_tokenizer
from training.setup import DEVICE, init_wandb
from training.train.checkpoint import save_history
from training.train.optim import build_optimizer, build_scheduler
from training.train.train_loop import fit
from training.transformer.probes import save_attention_maps
from training.viz.attention_heatmap import plot_attention_probes

DEFAULT_CONFIG_PATH = CONFIGS_PATH / "multi30k_translator.yaml"
PROBE_SENTENCES = 20


@dataclass(frozen=True)
class Multi30kTranslationDataConfig(DataloadConfig):
    # Multi30k language codes. Val loss is per target BPE token, so only runs that
    # share tgt_lang share a tokenizer and are comparable by it.
    src_lang: str
    tgt_lang: str
    # Names the tokenizer files, so changing it trains new ones on the next run.
    bpe_num_merges: int


@dataclass(frozen=True)
class Multi30kTranslatorTrainConfig(TrainConfig):
    model: TransformerConfig
    data: Multi30kTranslationDataConfig


def loader_seq_len(model_config: TransformerConfig) -> int:
    """Return the one length the loader pads both sides to.

    Raises:
        ValueError: If `src.seq_length` and `tgt.seq_length` differ. The loader stacks
            source and target in one `(B, 2, L)` tensor, so they share `L`.
    """
    if model_config.src.seq_length != model_config.tgt.seq_length:
        raise ValueError("src.seq_length and tgt.seq_length must match")  # noqa: NAR001
    return model_config.src.seq_length


def build_loaders(
    dataload_config: Multi30kTranslationDataConfig, seq_len: int, batch_size: int, cache: Dao
) -> tuple[DataLoader, DataLoader, tuple[Tokenizer, Tokenizer]]:
    """Return the train and val loaders, and the source and target tokenizers.

    Every argument is hashable, so a study can wrap this in `functools.cache` and
    share loaders between arms.

    Args:
        cache: Where the texts and encodings live; the caller opens it.
    """
    src_tokenizer, tgt_tokenizer = (
        get_tokenizer(
            merges_path=TOKENIZERS_PATH / f"multi30k-{lang}-{dataload_config.bpe_num_merges}.json",
            train_texts=get_multi30k_texts(lang=lang, cache=cache).train,  # noqa: NAR001
            num_merges=dataload_config.bpe_num_merges,
        )
        for lang in (dataload_config.src_lang, dataload_config.tgt_lang)
    )
    train_loader, val_loader, _ = get_multi30k_translation_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_lang=dataload_config.src_lang,
        tgt_lang=dataload_config.tgt_lang,
        cache=cache,
    )
    return train_loader, val_loader, (src_tokenizer, tgt_tokenizer)


def build_train_state(
    config: Multi30kTranslatorTrainConfig, src_vocab_size: int, tgt_vocab_size: int
) -> TrainState:
    """Return a fresh model, optimizer, criterion, and scheduler for `config`.

    The scheduler is sized for `config.fit.num_epochs`.

    Args:
        src_vocab_size: Size of the source tokenizer's vocab the loaders encode with.
        tgt_vocab_size: Size of the target tokenizer's vocab the loaders encode with.

    Side effects:
        Initializing the weights draws from the global torch RNG, so seed it first
        for the init to repeat.
    """
    model = Multi30kTranslator(
        model_config=config.model,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
    ).to(device=DEVICE)  # noqa: NAR001
    optimizer = build_optimizer(config=config.optimizer, params=model.parameters())
    return TrainState(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=build_scheduler(
            config=config.scheduler, optimizer=optimizer, num_epochs=config.fit.num_epochs
        )
        if config.scheduler is not None
        else None,
    )


def build_epoch_spec(probe_dir: Path, num_sentences: int) -> EpochSpec:
    """Return the next-token policy, saving attention maps under `probe_dir`.

    `MaskedTokenMetrics` scores every position whose target isn't
    `CROSS_ENTROPY_IGNORE_INDEX`, which the translation loader puts on the pads, so
    its accuracy and perplexity are over every real target token.

    Args:
        num_sentences: How many val sentences the probe saves maps for.
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
            num_sentences=num_sentences,
        ),
    )


def main():
    """Train one Multi30kTranslator from the `--config` YAML, optionally logging to W&B."""
    parser = argparse.ArgumentParser(description="Train one Multi30kTranslator.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)  # noqa: NAR001
    args = parser.parse_args()
    config = Multi30kTranslatorTrainConfig.from_yaml(path=args.config)
    SHOULD_LOG_WANDB: bool = False
    wandb_run = None
    if SHOULD_LOG_WANDB:
        wandb_run = init_wandb(project="dawnshard", run_name="multi30k_translator")
    with SqliteDao(path=CACHE_PATH) as cache:
        train_loader, val_loader, tokenizers = build_loaders(
            dataload_config=config.data,
            seq_len=loader_seq_len(model_config=config.model),
            batch_size=config.dataloader.batch_size,
            cache=cache,
        )
    # Fresh per run unless the YAML pins one, so no single lucky draw is trained on
    # forever; the history file records it, so a run can be repeated by putting its
    # recorded seed in the YAML.
    train_seed = config.fit.train_seed
    if train_seed is None:
        train_seed = random.getrandbits(32)  # noqa: NAR001
    run_uuid = str(uuid4())  # noqa: NAR001
    config = replace(  # noqa: NAR001
        config,
        fit=replace(config.fit, train_seed=train_seed, run_uuid=run_uuid),  # noqa: NAR001
    )
    # fit reseeds from it, but only after the model exists; seed here so weight init
    # uses the recorded seed too. See the README on what the rewind implies.
    torch.manual_seed(seed=train_seed)
    src_tokenizer, tgt_tokenizer = tokenizers
    state = build_train_state(
        config=config,
        src_vocab_size=len(src_tokenizer.vocab),  # noqa: NAR001
        tgt_vocab_size=len(tgt_tokenizer.vocab),  # noqa: NAR001
    )
    epoch_spec = build_epoch_spec(
        probe_dir=ATTENTION_PROBES_PATH / run_uuid, num_sentences=PROBE_SENTENCES
    )
    history = fit(
        state=state,
        epoch_spec=epoch_spec,
        train_loader=train_loader,
        val_loader=val_loader,
        fit_config=config.fit,
        wandb_run=wandb_run,
    )
    save_history(
        history=history,
        history_path=HISTORY_PATH / f"{run_uuid}.json",
        train_state=state,
        epoch_spec=epoch_spec,
        train_config=config,
    )
    plot_attention_probes(
        probe_dir=ATTENTION_PROBES_PATH / run_uuid,
        vocab=src_tokenizer.vocab,
        tgt_vocab=tgt_tokenizer.vocab,
    )


if __name__ == "__main__":
    main()
