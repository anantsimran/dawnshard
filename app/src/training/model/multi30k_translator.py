# TODO: KV cache
# TODO: main + beam search
# TODO: complete all
"""
multi30k_translator.py
Encoder-decoder translator for Multi30k sentence pairs (see dataload.multi30k for
loading).

The loader yields `(X, Y)` int64 batches. X stacks the source ids and the target input
ids, `(B, 2, seq_len)`: the target input starts with `BOS_ID`, and Y is the same target
shifted one to the left and ended with `EOS_ID`, flattened to `(B*seq_len,)`, with
`CROSS_ENTROPY_IGNORE_INDEX` on its padding. So every real target position is scored
on the next token, which teacher forcing needs.

The architecture comes from a `TransformerConfig`. `run/multi30k_translator.py` trains
one from its YAML, and `ablations/multi30k_translator_ablation.py` holds the studies.
"""

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import nn
from training.dataclass.config import TransformerConfig
from training.preprocess.constants import PAD_ID
from training.transformer.constants import B, L, V
from training.transformer.mask import causal_keep_mask, padding_keep_mask
from training.transformer.modules import (
    NORMS,
    DecoderBlock,
    EncoderBlock,
    SinusoidalEmbedding,
)


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
    src_dropout               tgt_dropout           on token content only
         │                           │
         ▼                           ▼
    positional_embedding     positional_embedding   one fixed sin/cos table
         │  (B, S, d_model)          │  (B, L, d_model)
         ▼                           │
    encoder_blocks                   │              EncoderBlock × num_encoders,
         │                           │              encoder_mask: source padding,
         │                           │              weights (B, h, S, S) each
         ▼                           │
     decoder_norm                    │              norms the memory, once
         │  memory (B, S, d_model)   ▼
         └──────────────────► decoder_blocks        DecoderBlock × num_decoders,
                                     │              decoder_mask: causal & target
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

    `src_dropout` and `tgt_dropout` run before `positional_embedding`, as in
    `Multi30kEnPredictor`, so they zero features of the token embedding only and every
    position keeps its full sin/cos signal. Each side takes its rate from its own
    `EmbeddingConfig`. One position table serves both sides, sized to the longer of the
    two lengths: it is a fixed function of position, so sharing it shares no weights.

    `decoder_norm` is the encoder stack's final norm. The stack is pre-norm, so its
    output is an unnormed residual, and `DecoderBlock` does not norm `memory` itself;
    norming it here, once, hands every decoder block the same normed memory.
    `pred_norm` is the decoder stack's final norm, for the same reason. Each final norm
    is the class its own stack's blocks use, so `model_config.encoder` and
    `model_config.decoder` can pick different norms without a third setting.

    `forward` flattens the logits to `(B*L, vocab)` because `CrossEntropyLoss` reads
    dim 1 as the class axis. The loader flattens its target the same way, so row
    `b*L + l` of both is the same position.

    Output: raw logits (no softmax). Every `forward` overwrites `attention_map`.
    """

    def __init__(
        self, model_config: TransformerConfig, src_vocab_size: int, tgt_vocab_size: int
    ) -> None:
        """Build both embeddings, the position table, both stacks, and the head.

        Args:
            model_config: The architecture. Each side's block setting also picks its
                final norm: `encoder.norm` for `decoder_norm`, `decoder.norm` for
                `pred_norm`.
            src_vocab_size: `len(src_tokenizer.vocab)` of the source tokenizer.
            tgt_vocab_size: `len(tgt_tokenizer.vocab)` of the target tokenizer; the
                logits' width.

        Raises:
            ValueError: If `model_config.d_model` is not divisible by either stack's
                `attention.h`, from `EncoderBlock` and `DecoderBlock`.
            ValueError: If either side sets `position_encoding=False`. One position
                table serves both sides, so it cannot be dropped for one of them.
        """
        super().__init__()
        if not (model_config.src.position_encoding and model_config.tgt.position_encoding):
            raise ValueError("Multi30kTranslator always adds position encoding")  # noqa: NAR001
        d_model = model_config.d_model
        encoder, decoder = model_config.encoder, model_config.decoder
        self.src_dropout = nn.Dropout(p=model_config.src.dropout)
        self.tgt_dropout = nn.Dropout(p=model_config.tgt.dropout)

        self.src_embedding = nn.Embedding(
            num_embeddings=src_vocab_size, embedding_dim=d_model, padding_idx=PAD_ID
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size, embedding_dim=d_model, padding_idx=PAD_ID
        )
        self.src_seq_length = model_config.src.seq_length
        self.tgt_seq_length = model_config.tgt.seq_length
        self.positional_embedding = SinusoidalEmbedding(
            seq_length=max(self.src_seq_length, self.tgt_seq_length),  # noqa: NAR001
            d_model=d_model,
        )

        self.encoder_blocks = nn.ModuleList(
            modules=[
                EncoderBlock(
                    d_model=d_model,
                    dropout=encoder.dropout,
                    h=encoder.attention.h,
                    d_ff_multiplier=encoder.ffn_multiplier,
                    norm=NORMS[encoder.norm],
                    qk_norm=encoder.attention.qk_norm,
                )
                for _ in range(model_config.num_encoders)  # noqa: NAR001
            ]
        )

        self.decoder_norm = NORMS[encoder.norm](normalized_shape=d_model)

        self.decoder_blocks = nn.ModuleList(
            modules=[
                DecoderBlock(
                    d_model=d_model,
                    dropout=decoder.dropout,
                    h=decoder.attention.h,
                    d_ff_multiplier=decoder.ffn_multiplier,
                    norm=NORMS[decoder.norm],
                    qk_norm=decoder.attention.qk_norm,
                )
                for _ in range(model_config.num_decoders)  # noqa: NAR001
            ]
        )

        self.pred_norm = NORMS[decoder.norm](normalized_shape=d_model)
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
            batch=self.src_dropout(self.src_embedding(src))  # noqa: NAR001
        )
        tgt_vector = self.positional_embedding(
            batch=self.tgt_dropout(self.tgt_embedding(tgt_in))  # noqa: NAR001
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
