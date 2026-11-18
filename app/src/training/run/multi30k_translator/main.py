"""
main.py
Train one `TransformerTranslator` (see model.transformer_translator) from a YAML config.

`multi30k_translator.yaml` under CONFIGS_PATH is the default. `setup.py` beside this
file holds the config types and the builders for the loaders, the `TrainState`, and
the `EpochSpec`, and `load_transformer` for restoring a trained one.

Train from the repo root, optionally naming another YAML; losses are logged to the
terminal after every epoch. `infer` translates one English sentence to German with
the best run so far and prints it:

    uv run python app/src/training/run/multi30k_translator/main.py train
    uv run python app/src/training/run/multi30k_translator/main.py train --config-path my_run.yaml
    uv run python app/src/training/run/multi30k_translator/main.py infer "A man is riding a bike."
"""

import random
from dataclasses import replace
from pathlib import Path
from typing import cast
from uuid import uuid4

import cyclopts
import torch
from training.constants import ATTENTION_PROBES_PATH, CHECKPOINTS_PATH, CONFIGS_PATH, HISTORY_PATH
from training.dal.dao.sqlite_dao import SqliteDao
from training.dataload.constants import CACHE_PATH
from training.model.transformer_translator import TransformerTranslator
from training.preprocess.constants import BOS_ID, EOS_ID
from training.preprocess.tokenizer import Tokenizer
from training.run.multi30k_translator.setup import (
    PROBE_SENTENCES,
    Multi30kTranslatorTrainConfig,
    build_epoch_spec,
    build_loaders,
    build_train_state,
    load_transformer,
    loader_seq_len,
)
from training.setup import init_wandb
from training.train.checkpoint import save_history
from training.train.train_loop import fit
from training.viz.attention_heatmap import plot_attention_probes

DEFAULT_CONFIG_PATH = CONFIGS_PATH / "multi30k_translator.yaml"

# Lowest val loss so far (2.083 at epoch 8). The run's config is copied beside its
# checkpoints, from its history file, without the run's uuid and train seed.
BEST_TRANSLATOR_RUN_PATH = CHECKPOINTS_PATH / "1bf10f5a-e622-4349-947f-790af297971e"
BEST_TRANSLATOR_CHECKPOINT_PATH = BEST_TRANSLATOR_RUN_PATH / "epoch_8.pt"
BEST_TRANSLATOR_CONFIG_PATH = BEST_TRANSLATOR_RUN_PATH / "config.yaml"


def train(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Train one TransformerTranslator from the YAML at `config_path`, optionally logging to W&B.

    Side effects:
        Writes checkpoints, the history file, and attention probes under the run's uuid.
    """
    config = Multi30kTranslatorTrainConfig.from_yaml(path=config_path)
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


@torch.no_grad()
def greedy_translate(
    src_sentence: str,
    transformer_model: TransformerTranslator,
    device: torch.device,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
) -> str:
    """Translate one sentence by appending the argmax token until EOS or `tgt_seq_length`.

    The source is framed as the loader frames it, `BOS_ID` then the ids, truncated to
    `src_seq_length`. Each step re-runs the whole target prefix, since
    `decode_next_token` keeps no cache.

    Args:
        transformer_model: Must already be in eval mode, so dropout is off.
        src_tokenizer: The tokenizer the model's source side was trained with.
        tgt_tokenizer: The tokenizer the model's target side was trained with.
    """
    ids = src_tokenizer.encode(texts=[src_sentence])[0]
    src = torch.tensor(
        data=[[BOS_ID] + ids[: transformer_model.src_seq_length - 1]],
        dtype=torch.int64,
        device=device,
    )  # (1, S)
    encoder_mask = transformer_model.build_encoder_mask(src=src)
    memory, _ = transformer_model.encode(src=src, encoder_mask=encoder_mask)
    tgt_in = torch.tensor(data=[[BOS_ID]], dtype=torch.int64, device=device)  # (1, 1)
    # The position table has tgt_seq_length rows, so the prefix never grows past it.
    for _ in range(transformer_model.tgt_seq_length):  # noqa: NAR001
        next_token = transformer_model.decode_next_token(
            tgt_in=tgt_in,
            memory=memory,
            decoder_mask=transformer_model.build_decoder_later_positions_mask(tgt_in=tgt_in),
            memory_mask=encoder_mask,
        ).argmax(dim=-1)  # (1,)
        if next_token.item() == EOS_ID:
            break
        tgt_in = torch.cat(tensors=[tgt_in, next_token.unsqueeze(dim=-1)], dim=1)  # (1, L+1)
    # decode drops protected ids, so the leading BOS does not reach the text.
    return tgt_tokenizer.decode(ids=tgt_in.tolist())[0]


def infer(src_sentence: str) -> str:
    """Translate `src_sentence` with the best run so far and return the translation.

    Loads the checkpoint at BEST_TRANSLATOR_CHECKPOINT_PATH with the config beside it.
    """
    train_state, epoch_spec, _, (src_tokenizer, tgt_tokenizer) = load_transformer(
        config_path=BEST_TRANSLATOR_CONFIG_PATH,
        checkpoint_path=BEST_TRANSLATOR_CHECKPOINT_PATH,
    )
    model = cast(typ=TransformerTranslator, val=train_state.model)
    model.eval()
    return greedy_translate(
        src_sentence=src_sentence,
        transformer_model=model,
        device=epoch_spec.device,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )


def main() -> None:
    """Dispatch the `train` or `infer` subcommand named on the command line."""
    app = cyclopts.App(help="Train or run one TransformerTranslator.")
    app.command(obj=train)
    app.command(obj=infer)
    app()


if __name__ == "__main__":
    main()
