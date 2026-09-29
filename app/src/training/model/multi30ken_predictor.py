# TODO: Write one from scratch
# TODO: Implement loss function
# TODO: Implement beam/grid search

"""
multi30ken_predictor.py
Masked-token predictor for Multi30k English sentences (see dataload.multi30ken for
loading and masking).

The loader yields `(X, Y)` int64 batches. X is the sentence with BERT-style masking
applied, `(B, SEQ_LEN)`; Y holds the original id at each selected position and
`CROSS_ENTROPY_IGNORE_INDEX` everywhere else, flattened to `(B*SEQ_LEN,)`, so only the
selected positions are scored. Rows longer than SEQ_LEN are truncated and shorter ones
right-padded with PAD_ID, so a model reading them needs an embedding with `len(vocab)`
rows and `padding_idx=PAD_ID`.

`main` either trains BASELINE once or, with SHOULD_GRID_SEARCH, runs `grid_search`:
BASELINE plus each of SWEEP's values changed one at a time, every one of those with
and without cosine annealing. Both load the default merges file rather than retraining
BPE; to change it, call the loader once with `retrain_tokenizer=True,
set_default_tokenizer=True`.

Train from the repo root; losses are logged to the terminal after every epoch:

    uv run python app/src/training/model/multi30ken_predictor.py
"""

import random
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Optional, cast
from uuid import uuid4

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from training.common.constants import PAD_ID
from training.constants import ATTENTION_PROBES_PATH, TEST_SEED, VAL_SEED
from training.dataload.multi30ken import get_multi30ken_mlm_dataloader
from training.metrics.masked_token_metrics import (
    MaskedTokenMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.setup import DEVICE, init_wandb
from training.train.model import EpochRecord, EpochSpec, TrainState
from training.train.train_loop import fit
from training.transformer.constants import B, L, V
from training.transformer.mask import padding_keep_mask
from training.transformer.modules import EncoderBlock, SinusoidalEmbedding
from training.transformer.probes import save_attention_maps
from training.viz.attention_heatmap import plot_attention_probes

D_MODEL = 64
SEQ_LEN = 32
DROPOUT_RATE = 0.1
NUM_ENCODERS = 6
NUM_HEADS = 4
FFN_MULTIPLIER = 6
PROBE_SENTENCES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3


class Multi30kEnPredictor(nn.Module):
    """Embedding → sin/cos positions → pre-norm encoder stack → vocab logits per token.

    Predicts the original token at every position of a masked sentence, for
    masked-language-model pretraining. In are `(B, L)` token ids; out are
    `(B*L, vocab)` logits, one row per position. The loss and the metrics score only
    the positions the loader selected; every other row is computed and ignored.

    ```
                       ids (B, L)                             right-padded with PAD_ID,
                            │                                 MASK_ID where masked
              ┌─────────────┴─────────────┐
              ▼                           ▼
      padding_keep_mask             nn_embedding              padding_idx=PAD_ID
        (B, 1, 1, L)                      │  (B, L, d_model)
              │                           ▼
              │                 nn_embedding_dropout          on token content only;
              │                           │                   positions are added
              │                           ▼                   after, never dropped
              │                  position_embedding           + fixed sin/cos table,
              │                           │  (B, L, d_model)  L ≤ seq_len
              │                           ▼
              └────────────► EncoderBlock × num_encoders ──► weights (B, h, L, L) each
                                          │  (B, L, d_model)
                                          ▼
                                     final_norm               the stack is pre-norm,
                                          │                   so its output is not
                                          ▼                   normed otherwise
                                        head                  Linear to vocab
                                          │  (B, L, vocab)
                                          ▼
                                       reshape                one row per position,
                                          │                   row b*L + l
                                          ▼  logits (B*L, vocab)
    ```

    The padding mask is derived inside `forward`, because the training loop calls
    `model(ids)` with one tensor and has nowhere to pass a mask. It hides `PAD_ID` keys
    only. `MASK_ID` is an ordinary token to attention, as in BERT: a masked position
    still attends to its context, and that context is all it has to recover the
    original token from.

    `nn_embedding_dropout` runs before `position_embedding`, so dropout zeroes features
    of the token embedding only and every position keeps its full sin/cos signal.
    `AGNewsClassifier` applies dropout after adding positions instead, as in the
    original Transformer.

    `forward` flattens the logits to `(B*L, vocab)` because `CrossEntropyLoss` reads
    dim 1 as the class axis and would take `L` for the classes of a `(B, L, vocab)`
    tensor. The loader flattens its target the same way, to `(B*L,)`, so row
    `b*L + l` of both is the same position and a plain `CrossEntropyLoss` and
    `MaskedTokenMetrics` work on them unchanged.

    Every `forward` overwrites `attention_map` with each block's detached attention
    weights, (B, h, L, L) per block in block order, so it describes the most recent
    call only. `save_attention_maps` reads it as an eval probe; nothing else may rely
    on it surviving the next call.

    Output: raw logits (no softmax), as CrossEntropyLoss expects.
    """

    def __init__(
        self,
        dropout_rate: float,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        num_encoders: int,
        h: int,
        ffn_multiplier: int,
        qk_norm: bool,
    ) -> None:
        """Build the embedding, position table, encoder stack, final norm, and head.

        Args:
            dropout_rate: Used on the token embeddings and inside every encoder block.
            seq_len: Length of the position table, so the longest `L` that `forward`
                accepts. Must be at least the loader's `seq_len`.
            d_model: Embedding width. Even only, because the sin/cos table splits it
                in half.
            vocab_size: `len(vocab)` from `get_multi30ken_mlm_dataloader`; the ids
                are contiguous, so this covers the protected ids and every merge.
            h: Attention heads per block.
            ffn_multiplier: Feed-forward width of each block, as a multiple of
                `d_model`.
            qk_norm: RMSNorm queries and keys per head before the dot product; see
                `MultiHeadAttentionLayer`.

        Raises:
            ValueError: If `d_model` is not divisible by `h`, from `EncoderBlock`.
        """
        super().__init__()
        self.nn_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=PAD_ID
        )
        self.nn_embedding_dropout = nn.Dropout(p=dropout_rate)
        self.position_embedding = SinusoidalEmbedding(seq_length=seq_len, d_model=d_model)

        self.encoder_blocks = nn.ModuleList(
            modules=[
                EncoderBlock(
                    d_model=d_model,
                    h=h,
                    qk_norm=qk_norm,
                    d_ff_multiplier=ffn_multiplier,
                    dropout=dropout_rate,
                    norm=nn.LayerNorm,
                )
                for _ in range(num_encoders)  # noqa: NAR001
            ]
        )
        self.final_norm = nn.LayerNorm(normalized_shape=d_model)
        self.head = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.attention_map: list[torch.Tensor] = []

    @jaxtyped(typechecker=beartype)
    def forward(self, ids: Int[torch.Tensor, f"{B} {L}"]) -> Float[torch.Tensor, f"{B}*{L} {V}"]:
        """Return logits for every position of a batch of masked, right-padded ids.

        Args:
            ids: `(B, L)` with `L ≤ seq_len`, right-padded with `PAD_ID`. The mask is
                derived from `PAD_ID` alone, so `MASK_ID` is attended to like any
                other id.

        Returns:
            Raw logits, `(B*L, vocab)`. Row `b*L + l` is position `l` of sentence
            `b`, the order the loader flattens its target in.

        Raises:
            ValueError: If `L` exceeds `seq_len`, the length of the position table.

        Side effects:
            Replaces `self.attention_map` with this call's attention weights, one
            detached `(B, h, L, L)` tensor per block in block order.
        """
        pad_mask = padding_keep_mask(token_ids=ids, pad_id=PAD_ID)
        embedding = self.nn_embedding_dropout(self.nn_embedding(ids))  # noqa: NAR001
        embedding = self.position_embedding(batch=embedding)  # {B L d_model}
        attention_map = []
        for encoder_block in self.encoder_blocks:
            embedding, attention_vector = encoder_block(batch=embedding, pad_mask=pad_mask)
            attention_map.append(attention_vector.detach())  # noqa: NAR001
        self.attention_map = attention_map
        encoded = self.final_norm(embedding)  # noqa: NAR001
        logits = self.head(encoded)  # noqa: NAR001
        return logits.reshape(-1, logits.size()[-1])  # noqa: NAR001


@dataclass(frozen=True)
class Config:
    d_model: int
    seq_len: int
    dropout_rate: float
    num_encoders: int
    num_heads: int
    ffn_multiplier: int
    cosine_annealing: bool


BASELINE = Config(
    d_model=D_MODEL,
    seq_len=SEQ_LEN,
    dropout_rate=DROPOUT_RATE,
    num_encoders=NUM_ENCODERS,
    num_heads=NUM_HEADS,
    ffn_multiplier=FFN_MULTIPLIER,
    cosine_annealing=False,
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
        replace(config, cosine_annealing=cosine)  # noqa: NAR001
        for config in architectures
        for cosine in (False, True)
    ]


def train_config(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    wandb_run: Optional[Any] = None,
) -> tuple[str, list[EpochRecord]]:
    """Train one model on `config` for NUM_EPOCHS, validating every epoch.

    Args:
        train_loader: Built with `seq_len=config.seq_len`.
        vocab_size: `len(vocab)` of the tokenizer the loaders were built with.

    Returns:
        `(run_uuid, history)`. `run_uuid` names the history file under HISTORY_PATH
        and the attention probe directory under ATTENTION_PROBES_PATH.

    Side effects:
        Draws a fresh train seed and seeds the global torch RNG from it before
        building the model, so weight init repeats too; `fit` writes the history
        file, seed included, and the probe writes one attention capture per epoch.
    """
    # Fresh per run, so no single lucky draw is trained on forever. fit records it;
    # the same data and config are also needed to replay a past setup.
    train_seed = random.getrandbits(32)  # noqa: NAR001
    run_uuid = str(uuid4())  # noqa: NAR001
    epoch_spec = EpochSpec(
        device=DEVICE,
        metrics_type=MaskedTokenMetrics,
        calculate_metrics=calculate_metrics,
        accumulate_metrics=accumulate_metrics,
        reduce_metrics=reduce_metrics,
        eval_probe=partial(  # noqa: NAR001
            save_attention_maps,
            out_dir=ATTENTION_PROBES_PATH / run_uuid,
            num_sentences=PROBE_SENTENCES,
        ),
    )
    # fit reseeds from it, but only after the model exists; seed here so weight init
    # uses the recorded seed too. See the README on what the rewind implies.
    torch.manual_seed(seed=train_seed)
    model = Multi30kEnPredictor(
        dropout_rate=config.dropout_rate,
        seq_len=config.seq_len,
        d_model=config.d_model,
        vocab_size=vocab_size,
        num_encoders=config.num_encoders,
        h=config.num_heads,
        ffn_multiplier=config.ffn_multiplier,
        qk_norm=True,
    ).to(device=DEVICE)  # noqa: NAR001
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        # fit steps the scheduler once per epoch, so this decays to 0 by the last one.
        scheduler=CosineAnnealingLR(optimizer=optimizer, T_max=NUM_EPOCHS)
        if config.cosine_annealing
        else None,
    )
    logger.info("config {} | run {}", config, run_uuid)  # noqa: NAR001
    history = fit(
        run_uuid=run_uuid,
        state=train_state,
        epoch_spec=epoch_spec,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        val_epoch_list=list(range(1, NUM_EPOCHS + 1)),  # noqa: NAR001
        train_seed=train_seed,
        val_seed=VAL_SEED,
        test_seed=TEST_SEED,
        wandb_run=wandb_run,
    )
    return run_uuid, history


def grid_search() -> None:
    """Train every config from `one_at_a_time_configs` and log them by best val loss.

    Every run draws its own train seed, so two configs differ in weight init and batch
    order as well as in the config; each run's seed is in its history file. Loaders are
    built once per `seq_len`. They share a tokenizer only if a default merges file
    has been configured.

    On a fresh clone without a configured default merges file, loader construction
    retrains BPE for each `seq_len`; the sweep then does not share one tokenizer.
    Val loss is a mean over selected positions. A shorter `seq_len` truncates long
    sentences and drops their tail positions from val, so its val loss is over a
    slightly different set of positions than the others'.

    Side effects:
        Everything `train_config` does, once per config.
    """
    loaders: dict[int, tuple[DataLoader, DataLoader, int]] = {}
    results: list[tuple[float, int, MaskedTokenMetrics, Config, str]] = []
    configs = one_at_a_time_configs(baseline=BASELINE, sweep=SWEEP)
    for index, config in enumerate(configs):  # noqa: NAR001
        logger.info("grid {}/{}", index + 1, len(configs))  # noqa: NAR001
        if config.seq_len not in loaders:
            train_loader, val_loader, _, _, vocab = get_multi30ken_mlm_dataloader(
                seq_len=config.seq_len,
                val_seed=VAL_SEED,
                test_seed=TEST_SEED,
                batch_size=BATCH_SIZE,
            )
            loaders[config.seq_len] = (train_loader, val_loader, len(vocab))  # noqa: NAR001
        train_loader, val_loader, vocab_size = loaders[config.seq_len]
        run_uuid, history = train_config(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_size=vocab_size,
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


def main():
    """Run `grid_search`, or train BASELINE once, optionally logging to W&B."""
    SHOULD_GRID_SEARCH: bool = True
    SHOULD_LOG_WANDB: bool = False
    if SHOULD_GRID_SEARCH:
        grid_search()
        return
    wandb_run = None
    if SHOULD_LOG_WANDB:
        wandb_run = init_wandb(project="dawnshard", run_name="multi30ken_predictor")
    train_loader, val_loader, _, _, vocab = get_multi30ken_mlm_dataloader(
        seq_len=BASELINE.seq_len,
        val_seed=VAL_SEED,
        test_seed=TEST_SEED,
        batch_size=BATCH_SIZE,
    )
    run_uuid, _ = train_config(
        config=BASELINE,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=len(vocab),  # noqa: NAR001
        wandb_run=wandb_run,
    )
    plot_attention_probes(probe_dir=ATTENTION_PROBES_PATH / run_uuid, vocab=vocab)


if __name__ == "__main__":
    main()
