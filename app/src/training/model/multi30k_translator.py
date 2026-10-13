# TODO: KV cache
# TODO: main + beam search
# TODO: complete all
"""
multi30k_translator.py
Encoder-decoder translator for Multi30k sentence pairs (see dataload.multi30k for
loading).

The loader yields `(X, Y)` int64 batches. X stacks the source ids and the target input
ids, `(B, 2, SEQ_LEN)`: the target input starts with `BOS_ID`, and Y is the same target
shifted one to the left and ended with `EOS_ID`, flattened to `(B*SEQ_LEN,)`, with
`CROSS_ENTROPY_IGNORE_INDEX` on its padding. So every real target position is scored
on the next token, which teacher forcing needs.

`main` trains one model built from the constants below. The ablation is in
`experiments/multi30k_translator_ablation.py`.

Train from the repo root; losses are logged to the terminal after every epoch:

    uv run python app/src/training/model/multi30k_translator.py
"""

import random
from functools import partial
from uuid import uuid4

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from training.common.constants import PAD_ID
from training.constants import ATTENTION_PROBES_PATH, TEST_SEED, VAL_SEED
from training.dataload.multi30k import get_multi30k_translation_dataloader
from training.metrics.masked_token_metrics import (
    MaskedTokenMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.setup import DEVICE, init_wandb
from training.train.model import EpochSpec, TrainState
from training.train.train_loop import fit
from training.transformer.constants import B, L, V
from training.transformer.mask import causal_keep_mask, padding_keep_mask
from training.transformer.modules import (
    DecoderBlock,
    EncoderBlock,
    SinusoidalEmbedding,
)
from training.transformer.probes import save_attention_maps
from training.viz.attention_heatmap import plot_attention_probes

SOURCE_LANG = "en"
TARGET_LANG = "de"
D_MODEL_SIZE = 512
SEQ_LEN = 32
NUM_SRC_MERGES = 10000
NUM_TGT_MERGES = 100000
DROPOUT_RATE = 0.3
NUM_ENCODERS = 6
NUM_DECODERS = 6
NUM_HEADS = 4
FFN_MULTIPLIER = 6
QK_NORM = False
NORM: type[nn.LayerNorm] | type[nn.RMSNorm] = nn.LayerNorm
COSINE_ANNEALING = True
PROBE_SENTENCES = 20
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3


class Multi30kTranslator(nn.Module):
    """Encoder over the source → causal decoder over the target → next-token logits.

    Translates by teacher forcing: given the source sentence and the target sentence so
    far, predicts the next target token at every target position at once. In is one
    `(B, 2, L)` tensor, the source ids at index 0 and the target input ids at index 1;
    out are `(B*L, vocab)` logits over the target vocab, one row per target position.

    ```
                batch (B, 2, L)                     src at index 0, tgt_in at 1
         ┌─────────────┴─────────────┐
         ▼  src (B, L)               ▼  tgt_in (B, L)
     src_embedding             tgt_embedding        padding_idx=PAD_ID, one per
         │                           │              tokenizer
         ▼                           ▼
      dropout                     dropout           on token content only
         │                           │
         ▼                           ▼
    positional_embedding     positional_embedding   one fixed sin/cos table
         │  (B, S, d_model)          │  (B, L, d_model)
         ▼                           │
    EncoderBlock × num_encoders      │              encoder_mask: source padding,
         │                           │              weights (B, h, S, S) each
         ▼                           │
     decoder_norm                    │              norms the memory, once
         │  memory (B, S, d_model)   ▼
         └──────────► DecoderBlock × num_decoders   decoder_mask: causal & target
                                     │              padding; memory_mask=encoder_mask
                                     ▼
                                 pred_norm
                                     │
                                     ▼
                                   pred             Linear to the target vocab
                                     │  (B, L, vocab)
                                     ▼
                                  reshape           one row per position, b*L + l
                                     │
                                     ▼  logits (B*L, vocab)
    ```

    Source and target arrive stacked in one tensor because the training loop calls
    `model(X)` with a single input and has nowhere to pass a second one. Both masks are
    derived inside `forward` from `PAD_ID` for the same reason.

    The two embeddings are separate, not tied: source and target have their own
    tokenizers, so the same id names a different token on each side.

    `dropout` runs before `positional_embedding`, as in `Multi30kEnPredictor`, so it
    zeroes features of the token embedding only and every position keeps its full
    sin/cos signal. One `nn.Dropout` serves both sides, because a Dropout holds no
    state. One position table serves both sides too, sized to the longer of the two
    lengths: it is a fixed function of position, so sharing it shares no weights.

    `decoder_norm` is the encoder stack's final norm. The stack is pre-norm, so its
    output is an unnormed residual, and `DecoderBlock` does not norm `memory` itself;
    norming it here, once, hands every decoder block the same normed memory.
    `pred_norm` is the decoder stack's final norm, for the same reason.

    `forward` flattens the logits to `(B*L, vocab)` because `CrossEntropyLoss` reads
    dim 1 as the class axis. The loader flattens its target the same way, so row
    `b*L + l` of both is the same position.

    Output: raw logits (no softmax). Every `forward` overwrites `attention_map`.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        src_seq_length: int,
        tgt_seq_length: int,
        dropout: float,
        num_encoder_heads: int,
        num_decoder_heads: int,
        encoder_ffn_multiplier: int,
        decoder_ffn_multiplier: int,
        num_encoders: int,
        num_decoders: int,
        qk_norm: bool,
        norm: type[nn.LayerNorm] | type[nn.RMSNorm] = nn.LayerNorm,
    ) -> None:
        """Build both embeddings, the position table, both stacks, and the head.

        Args:
            src_vocab_size: `len(src_vocab)` from `get_multi30k_translation_dataloader`.
            tgt_vocab_size: `len(tgt_vocab)` from the same call; the logits' width.
            src_seq_length: Longest source `L` that `forward` accepts.
            tgt_seq_length: Longest target `L` that `forward` accepts.
            dropout: Used on both embeddings and inside every block.
            qk_norm: Forwarded to every block; RMSNorms queries and keys per head
                before the dot product.
            norm: Class behind every norm in the model, each block's and both final
                ones, so one setting switches them together.

        Raises:
            ValueError: If `d_model` is not divisible by `num_encoder_heads` or by
                `num_decoder_heads`, from `EncoderBlock` and `DecoderBlock`.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.src_embedding = nn.Embedding(
            num_embeddings=src_vocab_size, embedding_dim=d_model, padding_idx=PAD_ID
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size, embedding_dim=d_model, padding_idx=PAD_ID
        )
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.positional_embedding = SinusoidalEmbedding(
            seq_length=max(src_seq_length, tgt_seq_length),  # noqa: NAR001
            d_model=d_model,
        )

        self.encoder_blocks = nn.ModuleList(
            modules=[
                EncoderBlock(
                    d_model=d_model,
                    dropout=dropout,
                    h=num_encoder_heads,
                    d_ff_multiplier=encoder_ffn_multiplier,
                    norm=norm,
                    qk_norm=qk_norm,
                )
                for _ in range(num_encoders)  # noqa: NAR001
            ]
        )

        self.decoder_norm = norm(normalized_shape=d_model)

        self.decoder_blocks = nn.ModuleList(
            modules=[
                DecoderBlock(
                    d_model=d_model,
                    dropout=dropout,
                    h=num_decoder_heads,
                    d_ff_multiplier=decoder_ffn_multiplier,
                    norm=norm,
                    qk_norm=qk_norm,
                )
                for _ in range(num_decoders)  # noqa: NAR001
            ]
        )

        self.pred_norm = norm(normalized_shape=d_model)
        self.pred = nn.Linear(in_features=d_model, out_features=tgt_vocab_size)
        # Overwritten by every forward: one detached (B, h, L, S) tensor per block, keyed
        # by kind ("encoder", "decoder_self", "decoder_cross").
        self.attention_map: dict[str, list[torch.Tensor]] = {}

    @jaxtyped(typechecker=beartype)
    def forward(
        self, batch: Int[torch.Tensor, f"{B} 2 {L}"]
    ) -> Float[torch.Tensor, f"{B}*{L} {V}"]:
        """Return next-token logits for every target position of a batch of pairs.

        Args:
            batch: Source ids at index 0 and target input ids at index 1, each
                right-padded with `PAD_ID` and no longer than its `seq_length`. Both
                masks come from `PAD_ID` alone.

        Returns:
            Raw logits, `(B*L, vocab)`. Row `b*L + l` is target position `l` of pair
            `b`, the order the loader flattens its target in.

        Raises:
            ValueError: If either side is longer than the position table.

        Side effects:
            Replaces `self.attention_map` with this call's detached attention weights
            under `"encoder"`, `"decoder_self"`, and `"decoder_cross"`, one tensor per
            block in block order.
        """
        src, tgt_in = batch[:, 0], batch[:, 1]
        src_vector = self.positional_embedding(
            batch=self.dropout(self.src_embedding(src))  # noqa: NAR001
        )
        tgt_vector = self.positional_embedding(
            batch=self.dropout(self.tgt_embedding(tgt_in))  # noqa: NAR001
        )

        # Self attention
        encoder_mask = padding_keep_mask(token_ids=src, pad_id=PAD_ID)
        encoder_attn = []
        for encoder in self.encoder_blocks:
            src_vector, src_attn = encoder(batch=src_vector, pad_mask=encoder_mask)
            encoder_attn.append(src_attn.detach())  # noqa: NAR001
        decoder_pad_mask = padding_keep_mask(token_ids=tgt_in, pad_id=PAD_ID)
        decoder_causal_mask = causal_keep_mask(seq_len=tgt_in.shape[1], device=tgt_in.device)
        decoder_mask = decoder_pad_mask & decoder_causal_mask

        src_vector = self.decoder_norm(src_vector)  # noqa: NAR001
        decoder_self_attn, decoder_cross_attn = [], []
        for decoder in self.decoder_blocks:
            tgt_vector, tgt_self_attn, tgt_cross_attn = decoder(
                x=tgt_vector, memory=src_vector, x_mask=decoder_mask, memory_mask=encoder_mask
            )
            decoder_self_attn.append(tgt_self_attn.detach())  # noqa: NAR001
            decoder_cross_attn.append(tgt_cross_attn.detach())  # noqa: NAR001

        self.attention_map = {
            "encoder": encoder_attn,
            "decoder_self": decoder_self_attn,
            "decoder_cross": decoder_cross_attn,
        }
        logits = self.pred(self.pred_norm(tgt_vector))  # noqa: NAR001
        return logits.reshape(-1, logits.shape[-1])  # noqa: NAR001


def main():
    """Train one Multi30kTranslator from the module constants, optionally to W&B."""
    # Fresh per run, so no single lucky draw is trained on forever; fit writes it to
    # the history file, so a run can be repeated by passing its recorded seed back.
    train_seed = random.getrandbits(32)  # noqa: NAR001
    # fit reseeds from it, but only after the model exists; seed here so weight init
    # uses the recorded seed too. See the README on what the rewind implies.
    torch.manual_seed(seed=train_seed)
    SHOULD_LOG_WANDB: bool = False
    wandb_run = None
    if SHOULD_LOG_WANDB:
        wandb_run = init_wandb(project="dawnshard", run_name="multi30k_translator")
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
    train_loader, val_loader, _, _, src_vocab, _, tgt_vocab = get_multi30k_translation_dataloader(
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        cache=False,
        src_lang=SOURCE_LANG,
        tgt_lang=TARGET_LANG,
        retrain_tokenizer=True,
        set_default_tokenizer=True,
    )

    model = Multi30kTranslator(
        src_vocab_size=len(src_vocab),  # noqa: NAR001
        tgt_vocab_size=len(tgt_vocab),  # noqa: NAR001
        d_model=D_MODEL_SIZE,
        src_seq_length=SEQ_LEN,
        tgt_seq_length=SEQ_LEN,
        dropout=DROPOUT_RATE,
        num_encoder_heads=NUM_HEADS,
        num_decoder_heads=NUM_HEADS,
        encoder_ffn_multiplier=FFN_MULTIPLIER,
        decoder_ffn_multiplier=FFN_MULTIPLIER,
        num_encoders=NUM_ENCODERS,
        num_decoders=NUM_DECODERS,
        qk_norm=QK_NORM,
        norm=NORM,
    ).to(device=DEVICE)  # noqa: NAR001
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        # fit steps the scheduler once per epoch, so this decays to 0 by the last one.
        scheduler=CosineAnnealingLR(optimizer=optimizer, T_max=NUM_EPOCHS)
        if COSINE_ANNEALING
        else None,
    )
    fit(
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
    plot_attention_probes(
        probe_dir=ATTENTION_PROBES_PATH / run_uuid, vocab=src_vocab, tgt_vocab=tgt_vocab
    )


if __name__ == "__main__":
    main()
