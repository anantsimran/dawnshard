# ── Notes: how the decoder learns, and decoding one token at a time ──────────────
#
# How the decoder learns (teacher forcing)
# - Self-attention mixes in the target tokens up to and including each position.
#   `mask_later_positions` is lower-triangular WITH the diagonal. That is safe
#   because tgt_in is shifted: position i holds token i but is scored on token i+1,
#   so a token sees its own input, never its own answer. A zero diagonal would leave
#   the BOS row with no visible key, and softmax over nothing is NaN.
#   So the model learns to predict the next token from the tokens up to its own.
# - Cross-attention takes its query from the target and its keys AND values from
#   memory. Values from the source are fine: the output is added to the residual
#   stream, which still carries the target side, and `pred` maps that mix to
#   target words.
#
# Masks hide keys, never queries
# - `memory_mask` is the source padding mask because cross-attention's keys are
#   source positions. It hides key columns only; a fully masked query row would
#   be NaN after softmax.
#
# Decoding one token at a time: reuse `decode`, shrink the query to one row
# - No new attention math. Feed only the newest token, x (B, 1, d_model). Then q is
#   (B, h, 1, d_k), the cached k and v are (B, h, t, d_k), and the scores are
#   (B, h, 1, t). Matmul does not care whether the query has 1 row or L.
# - The only change inside attention is where k and v come from: the cache plus
#   the new token, instead of the whole sequence. Cache them after projection
#   (and after `k_norm`), per block.
# - The self-attention mask becomes None, since the newest token may see every
#   cached key. `memory_mask` stays (B, 1, 1, S).
# - The newest token is at index 0 of what `decode` sees, so the caller passes its
#   position row (`positional_embedding.at_pos(pos=t)`) as `pos_encoding`.
#
# `run/multi30k_translator/main.py` builds the caches (`init_decoder_cache`) and the
# cached decode loop (`translate_with_cache`); the transformer package README's
# "KV cache" section explains the layer side. Decode under `model.eval()` so
# dropout is off.
# ─────────────────────────────────────────────────────────────────────────────────

"""
transformer_translator.py
Encoder-decoder translator for Multi30k sentence pairs (see dataload.multi30k for
loading).

The loader yields `(X, Y)` int64 batches. X stacks the source ids and the target input
ids, `(B, 2, seq_len)`: the target input starts with `BOS_ID`, and Y is the same target
shifted one to the left and ended with `EOS_ID`, flattened to `(B*seq_len,)`, with
`CROSS_ENTROPY_IGNORE_INDEX` on its padding. So every real target position is scored
on the next token, which teacher forcing needs.

The architecture comes from a `TransformerConfig`. `run/multi30k_translator/main.py`
trains one from its YAML, and `ablations/multi30k_translator_ablation.py` holds the
studies.
"""

from typing import Optional

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import nn
from training.dataclass.config import TransformerConfig
from training.preprocess.constants import PAD_ID
from training.transformer.constants import D_MODEL, B, H, L, S, T, V
from training.transformer.masks import mask_later_positions, mask_tokenizer_pad_ids
from training.transformer.modules import (
    NORMS,
    DecoderBlock,
    DecoderBlockCache,
    EncoderBlock,
    SinusoidalEmbedding,
)
from training.transformer.probes import AttentionProbePerEpoch


class TransformerTranslator(nn.Module):
    """Encoder over the source → causal decoder over the target → next-token logits.

    Translates by teacher forcing: given the source sentence and the target sentence so
    far, predicts the next target token at every target position at once. In is one
    `(B, 2, L)` tensor, the source ids at index 0 and the target input ids at index 1;
    out are `(B*L, vocab)` logits over the target vocab, one row per target position.

    ```
                batch (B, 2, L)                     src at index 0, tgt_in at 1
         ┌─────────────┴─────────────┐
         ▼  src (B, L)               ▼  tgt_in (B, L)
     src_embedding             tgt_embedding        .weight (vocab, d_model) each,
         │                           │              vocab from that side's own
         │                           │              tokenizer; padding_idx=PAD_ID
         ▼                           ▼
    src_dropout               tgt_dropout           on token content only
         │                           │
         ▼                           ▼
    positional_embedding     positional_embedding   one shared fixed sin/cos table,
         │                           │              .pe (seq_length, d_model)
         │  (B, S, d_model)          │  (B, L, d_model)
         ▼                           │
    encoder_blocks                   │              EncoderBlock × num_encoders,
         │                           │              encoder_mask: source padding,
         │                           │              weights (B, h, S, S) each
         ▼                           │
     encoder_norm                    │              norms the memory, once
         │  memory (B, S, d_model)   ▼
         └──────────────────► decoder_blocks        DecoderBlock × num_decoders,
                                     │              decoder_mask: causal & target
                                     │              padding; memory_mask=encoder_mask
                                     ▼
                                 pred_norm
                                     │
                                     ▼
                                   pred             Linear to the target vocab,
                                     │              .weight (vocab, d_model)
                                     │  (B, L, vocab)
                                     ▼
                                  reshape           one row per position, b*L + l
                                     │
                                     ▼  logits (B*L, vocab)
    ```

    Source and target arrive stacked in one tensor because the training loop calls
    `model(X)` with a single input and has nowhere to pass a second one. All three masks
    are derived inside `forward` from `PAD_ID` for the same reason.

    The two embeddings are separate, not tied: source and target have their own
    tokenizers, so the same id names a different token on each side.

    `src_dropout` and `tgt_dropout` run before `positional_embedding`, as in
    `Multi30kEnPredictor`, so they zero features of the token embedding only and every
    position keeps its full sin/cos signal. Each side takes its rate from its own
    `EmbeddingConfig`. One position table serves both sides, sized to the longer of the
    two lengths: it is a fixed function of position, so sharing it shares no weights.

    `decode` takes the target's position rows as `pos_encoding` rather than reading
    the table itself. `forward` passes `positional_embedding.pe[:L]`; a cached decode
    step feeds one token that sits at index 0 of its input but at position t of the
    translation, so it passes `positional_embedding.at_pos(pos=t)`, and the model
    keeps no position counter between steps.

    `encoder_norm` is the encoder stack's final norm. The stack is pre-norm, so its
    output is an unnormed residual, and `DecoderBlock` does not norm `memory` itself;
    norming it here, once, hands every decoder block the same normed memory.
    `pred_norm` is the decoder stack's final norm, for the same reason. Each final norm
    is the class its own stack's blocks use, so `model_config.encoder` and
    `model_config.decoder` can pick different norms without a third setting.

    `forward` flattens the logits to `(B*L, vocab)` because `CrossEntropyLoss` reads
    dim 1 as the class axis. The loader flattens its target the same way, so row
    `b*L + l` of both is the same position.

    Output: raw logits (no softmax). Every `forward` overwrites `attention_maps`.
    """

    def __init__(
        self, model_config: TransformerConfig, src_vocab_size: int, tgt_vocab_size: int
    ) -> None:
        """Build both embeddings, the position table, both stacks, and the head.

        Args:
            model_config: The architecture. Each side's block setting also picks its
                final norm: `encoder.norm` for `encoder_norm`, `decoder.norm` for
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
            raise ValueError("TransformerTranslator always adds position encoding")  # noqa: NAR001
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

        self.encoder_norm = NORMS[encoder.norm](normalized_shape=d_model)

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
        self.attention_maps = AttentionProbePerEpoch()

    @jaxtyped(typechecker=beartype)
    def build_encoder_mask(
        self, src: Int[torch.Tensor, f"{B} {S}"]
    ) -> Bool[torch.Tensor, f"{B} 1 1 {S}"]:
        """Return the encoder self-attention mask, which hides source padding."""
        return mask_tokenizer_pad_ids(token_ids=src, pad_id=PAD_ID)

    @jaxtyped(typechecker=beartype)
    def build_decoder_pad_mask(
        self, tgt_in: Int[torch.Tensor, f"{B} {L}"]
    ) -> Bool[torch.Tensor, f"{B} 1 1 {L}"]:
        """Return the mask that hides target padding from decoder self-attention."""
        return mask_tokenizer_pad_ids(token_ids=tgt_in, pad_id=PAD_ID)

    @jaxtyped(typechecker=beartype)
    def build_decoder_later_positions_mask(
        self, tgt_in: Int[torch.Tensor, f"{B} {L}"]
    ) -> Bool[torch.Tensor, f"1 1 {L} {L}"]:
        """Return the mask that hides later target positions from decoder self-attention."""
        return mask_later_positions(seq_len=tgt_in.shape[1], device=tgt_in.device)

    @jaxtyped(typechecker=beartype)
    def encode(
        self,
        src: Int[torch.Tensor, f"{B} {S}"],
        encoder_mask: Bool[torch.Tensor, f"{B} 1 1 {S}"],
    ) -> tuple[
        Float[torch.Tensor, f"{B} {S} {D_MODEL}"],
        list[Float[torch.Tensor, f"{B} {H} {S} {S}"]],
    ]:
        """Run the source through the encoder stack and its final norm.

        Returns:
            `(memory, encoder_attn)`: the normed memory, and each encoder block's
            detached attention weights in block order.
        """
        memory = self.positional_embedding(
            batch=self.src_dropout(self.src_embedding(src))  # noqa: NAR001
        )
        encoder_attn = []
        for encoder in self.encoder_blocks:
            memory, attn = encoder(batch=memory, pad_mask=encoder_mask)
            encoder_attn.append(attn.detach())  # noqa: NAR001
        return self.encoder_norm(memory), encoder_attn  # noqa: NAR001

    @jaxtyped(typechecker=beartype)
    def decode(
        self,
        tgt_in: Int[torch.Tensor, f"{B} {L}"],
        pos_encoding: Float[torch.Tensor, f"{L} {D_MODEL}"],
        memory: Float[torch.Tensor, f"{B} {S} {D_MODEL}"],
        decoder_mask: Optional[Bool[torch.Tensor, f"#{B} 1 {L} {T}"]],
        memory_mask: Bool[torch.Tensor, f"{B} 1 1 {S}"],
        decoder_cache: Optional[list[DecoderBlockCache]] = None,
    ) -> tuple[
        Float[torch.Tensor, f"{B} {L} {V}"],
        list[Float[torch.Tensor, f"{B} {H} {L} {T}"]],
        list[Float[torch.Tensor, f"{B} {H} {L} {S}"]],
    ]:
        """Run the target through the decoder stack against `memory` to logits.

        Training passes the whole target and no cache. A decode loop either passes
        the whole prefix so far with no cache, and reads the last row of the logits,
        or passes only the newest token with caches it allocated and owns.

        Args:
            tgt_in: Target ids. With a cache, only the ids not yet cached.
            pos_encoding: The position table's rows for those ids, added after
                `tgt_dropout`: `positional_embedding.pe[:L]` for a prefix starting
                at 0, or `positional_embedding.at_pos(pos=t)` as one row for the
                single cached token at position t.
            decoder_mask: Causal mask, `& target padding` in training, sliced to
                (L, T). It is needed even when only the last row is read: without
                it, earlier positions see later tokens and feed every block after
                the first different keys and values. None only for a single cached
                position.
            decoder_cache: One `DecoderBlockCache` per decoder block, in block order,
                or None to cache nothing. The caller allocates it for one decode
                and keeps it between calls; the model only writes into it.

        Returns:
            `(logits, decoder_self_attn, decoder_cross_attn)`: unflattened raw
            logits, then each decoder block's detached self- and cross-attention
            weights in block order.

        Raises:
            ValueError: If `decoder_cache` does not have one entry per block, or a
                block rejects its mask or cache.

        Side effects:
            Writes into every cache in `decoder_cache`.
        """
        tgt_vector = self.tgt_dropout(self.tgt_embedding(tgt_in)) + pos_encoding  # noqa: NAR001
        block_caches = (
            [None] * len(self.decoder_blocks)  # noqa: NAR001
            if decoder_cache is None
            else decoder_cache
        )
        decoder_self_attn, decoder_cross_attn = [], []
        for decoder, block_cache in zip(  # noqa: NAR001
            self.decoder_blocks, block_caches, strict=True
        ):
            tgt_vector, self_attn, cross_attn = decoder(
                x=tgt_vector,
                memory=memory,
                x_mask=decoder_mask,
                memory_mask=memory_mask,
                decoder_block_cache=block_cache,
            )
            decoder_self_attn.append(self_attn.detach())  # noqa: NAR001
            decoder_cross_attn.append(cross_attn.detach())  # noqa: NAR001

        logits = self.pred(self.pred_norm(tgt_vector))  # noqa: NAR001
        return logits, decoder_self_attn, decoder_cross_attn

    @jaxtyped(typechecker=beartype)
    def forward(
        self, batch: Int[torch.Tensor, f"{B} 2 {L}"]
    ) -> Float[torch.Tensor, f"{B}*{L} {V}"]:
        """Return next-token logits for every target position of a batch of pairs.

        Args:
            batch: Source ids at index 0 and target input ids at index 1, each
                right-padded with `PAD_ID` and no longer than its `seq_length`. All
                three masks come from `PAD_ID` alone.

        Returns:
            Raw logits, `(B*L, vocab)`. Row `b*L + l` is target position `l` of pair
            `b`, the order the loader flattens its target in.

        Raises:
            ValueError: If the source is longer than the position table. A target
                longer than it fails `decode`'s shape check instead.

        Side effects:
            Replaces `self.attention_maps` with three kinds, `"encoder"`,
            `"decoder_self"`, and `"decoder_cross"`, each a list of detached weights in
            block order.
        """
        src, tgt_in = batch[:, 0], batch[:, 1]
        encoder_mask = self.build_encoder_mask(src=src)
        memory, encoder_attn = self.encode(src=src, encoder_mask=encoder_mask)
        decoder_mask = self.build_decoder_pad_mask(
            tgt_in=tgt_in
        ) & self.build_decoder_later_positions_mask(tgt_in=tgt_in)
        logits, decoder_self_attn, decoder_cross_attn = self.decode(
            tgt_in=tgt_in,
            pos_encoding=self.positional_embedding.pe[: tgt_in.shape[1]],
            memory=memory,
            decoder_mask=decoder_mask,
            memory_mask=encoder_mask,
        )
        self.attention_maps = AttentionProbePerEpoch(
            map={
                "encoder": encoder_attn,
                "decoder_self": decoder_self_attn,
                "decoder_cross": decoder_cross_attn,
            }
        )
        return logits.reshape(-1, logits.shape[-1])  # noqa: NAR001
