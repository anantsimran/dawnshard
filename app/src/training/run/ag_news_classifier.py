"""
ag_news_classifier.py
Train one `AGNewsClassifier` (see model.ag_news_classifier) from a YAML config.

`AGNewsClassifierTrainConfig` is the YAML's shape, and `ag_news_classifier.yaml` under
CONFIGS_PATH is the default. Its `data` section sizes the sample and names the
tokenizer; the model module's docstring says how the sample and the tokenizer are
cached.

Train from the repo root, optionally naming another YAML; losses are logged to the
terminal after every epoch:

    uv run python app/src/training/run/ag_news_classifier.py
    uv run python app/src/training/run/ag_news_classifier.py --config my_run.yaml
"""

import argparse
import random
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from training.constants import ATTENTION_PROBES_PATH, CONFIGS_PATH, HISTORY_PATH
from training.dal.dao.sqlite_dao import SqliteDao
from training.dataclass.config import DataloadConfig, EncoderConfig, TrainConfig
from training.dataclass.model import EpochSpec, TrainState
from training.dataload.ag_news import get_ag_news_dataloader, get_ag_news_texts
from training.dataload.constants import CACHE_PATH, TOKENIZERS_PATH
from training.metrics.classification_metrics import (
    ClassificationMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.model.ag_news_classifier import AGNewsClassifier
from training.preprocess.tokenizer import get_tokenizer
from training.setup import DEVICE, init_wandb
from training.train.checkpoint import save_history
from training.train.optim import build_optimizer, build_scheduler
from training.train.train_loop import fit
from training.transformer.probes import save_attention_maps
from training.viz.attention_heatmap import plot_attention_probes

DEFAULT_CONFIG_PATH = CONFIGS_PATH / "ag_news_classifier.yaml"
PROBE_SENTENCES = 4


@dataclass(frozen=True)
class AGNewsDataConfig(DataloadConfig):
    # Rows sampled per split; the sample is cached by these and sample_seed.
    n_train: int
    n_eval: int
    n_test: int
    sample_seed: int
    # Names the tokenizer file, so changing it trains a new one on the next run.
    bpe_num_merges: int


@dataclass(frozen=True)
class AGNewsClassifierTrainConfig(TrainConfig):
    model: EncoderConfig
    data: AGNewsDataConfig


def main():
    """Train one AGNewsClassifier from the `--config` YAML, optionally logging to W&B."""
    parser = argparse.ArgumentParser(description="Train one AGNewsClassifier.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)  # noqa: NAR001
    args = parser.parse_args()
    config = AGNewsClassifierTrainConfig.from_yaml(path=args.config)
    data = config.data
    SHOULD_LOG_WANDB: bool = False
    wandb_run = None
    if SHOULD_LOG_WANDB:
        wandb_run = init_wandb(project="dawnshard", run_name="ag_news_classifier")
    run_uuid = str(uuid4())  # noqa: NAR001
    epoch_spec = EpochSpec(
        device=DEVICE,
        metrics_type=ClassificationMetrics,
        calculate_metrics=calculate_metrics,
        accumulate_metrics=accumulate_metrics,
        reduce_metrics=reduce_metrics,
        eval_probe=partial(  # noqa: NAR001
            save_attention_maps,
            out_dir=ATTENTION_PROBES_PATH / run_uuid,
            num_sentences=PROBE_SENTENCES,
        ),
    )
    with SqliteDao(path=CACHE_PATH) as cache:
        texts = get_ag_news_texts(  # noqa: NAR001
            n_train=data.n_train,
            n_eval=data.n_eval,
            n_test=data.n_test,
            seed=data.sample_seed,
            cache=cache,
        )
        tokenizer = get_tokenizer(
            merges_path=TOKENIZERS_PATH / f"ag_news-{data.bpe_num_merges}.json",
            train_texts=[text for text, _ in texts["train"]],
            num_merges=data.bpe_num_merges,
        )
        train_loader, val_loader, _ = get_ag_news_dataloader(
            tokenizer=tokenizer,
            batch_size=config.dataloader.batch_size,
            n_train=data.n_train,
            n_eval=data.n_eval,
            n_test=data.n_test,
            max_len=config.model.embedding.seq_length,
            seed=data.sample_seed,
            cache=cache,
        )
    # Fresh per run unless the YAML pins one, so no single lucky draw is trained on
    # forever; the history file records it, so a run can be repeated by putting its
    # recorded seed in the YAML.
    train_seed = config.fit.train_seed
    if train_seed is None:
        train_seed = random.getrandbits(32)  # noqa: NAR001
    config = replace(  # noqa: NAR001
        config,
        fit=replace(config.fit, train_seed=train_seed, run_uuid=run_uuid),  # noqa: NAR001
    )
    # fit reseeds from it, but only after the model exists; seed here so weight init
    # is reproducible too. See the README on what the rewind implies.
    torch.manual_seed(seed=train_seed)
    model = AGNewsClassifier(
        model_config=config.model,
        vocab_size=len(tokenizer.vocab),  # noqa: NAR001
    ).to(device=DEVICE)  # noqa: NAR001
    optimizer = build_optimizer(config=config.optimizer, params=model.parameters())
    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=build_scheduler(
            config=config.scheduler, optimizer=optimizer, num_epochs=config.fit.num_epochs
        )
        if config.scheduler is not None
        else None,
    )
    history = fit(
        state=train_state,
        epoch_spec=epoch_spec,
        train_loader=train_loader,
        val_loader=val_loader,
        fit_config=config.fit,
        wandb_run=wandb_run,
    )
    save_history(
        history=history,
        history_path=HISTORY_PATH / f"{run_uuid}.json",
        train_state=train_state,
        epoch_spec=epoch_spec,
        train_config=config,
    )
    plot_attention_probes(probe_dir=ATTENTION_PROBES_PATH / run_uuid, vocab=tokenizer.vocab)


if __name__ == "__main__":
    main()
