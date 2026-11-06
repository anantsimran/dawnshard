"""
multi30ken_predictor.py
Masked-token predictor for Multi30k English sentences (see dataload.multi30k for
loading and masking).

The loader yields `(X, Y)` int64 batches. X is the sentence with BERT-style masking
applied, `(B, seq_len)`; Y holds the original id at each selected position and
`CROSS_ENTROPY_IGNORE_INDEX` everywhere else, flattened to `(B*seq_len,)`, so only the
selected positions are scored. Rows longer than seq_len are truncated and shorter ones
right-padded with PAD_ID, so a model reading them needs an embedding with `len(tokenizer.vocab)`
rows and `padding_idx=PAD_ID`.

The architecture comes from an `EncoderConfig`. `run/multi30ken_predictor.py` trains
one from its YAML, and `ablations/multi30ken_predictor_ablation.py` holds the studies.
"""

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import nn
from training.dataclass.config import EncoderConfig
from training.preprocess.constants import PAD_ID
from training.transformer.constants import B, L, V
from training.transformer.mask import padding_keep_mask
from training.transformer.modules import NORMS, EncoderBlock, SinusoidalEmbedding


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
              │                           │  (B, L, d_model)  L ≤ seq_length; Identity
              │                           ▼                   if not position_encoding
              └────────────► EncoderBlock × num_encoders ──► weights (B, h, L, L) each
                                          │  (B, L, d_model)  norm inside each
                                          ▼
                                     final_norm               `encoder.norm`, or
                                          │                   Identity if not
                                          │                   final_norm; the
                                          │                   stack is pre-norm, so
                                          ▼                   nothing else norms here
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

    `encoder.norm`, `embedding.position_encoding`, and `final_norm` exist so one
    component at a time can be switched off and the cost measured; `COMPONENT_ARMS` in
    `ablations/multi30ken_predictor_ablation.py` is where an arm switches one off.
    Each off state is `nn.Identity`, not a skipped call, so the forward pass keeps one
    shape and one code path whichever way the flags are set. Dropping the position
    encoding leaves attention as the only thing that mixes positions, and attention is
    permutation-equivariant, so that model cannot tell `a dog bit` from `bit a dog`;
    it is the one ablation expected to be catastrophic rather than merely worse.

    `forward` flattens the logits to `(B*L, vocab)` because `CrossEntropyLoss` reads
    dim 1 as the class axis and would take `L` for the classes of a `(B, L, vocab)`
    tensor. The loader flattens its target the same way, to `(B*L,)`, so row
    `b*L + l` of both is the same position and a plain `CrossEntropyLoss` and
    `MaskedTokenMetrics` work on them unchanged.

    Every `forward` overwrites `attention_map` with each block's detached attention
    weights, `{"encoder": [(B, h, L, L) per block]}`, so it describes the most recent
    call only. `save_attention_maps` reads it as an eval probe; nothing else may rely
    on it surviving the next call.

    Output: raw logits (no softmax), as CrossEntropyLoss expects.
    """

    def __init__(self, model_config: EncoderConfig, vocab_size: int) -> None:
        """Build the embedding, position table, encoder stack, final norm, and head.

        Args:
            model_config: The architecture. `embedding.seq_length` sizes the position
                table, so it is the longest `L` that `forward` accepts and must be at
                least the loader's `seq_len`. `d_model` must be even, because the
                sin/cos table splits it in half. `encoder.norm` is the class behind
                every norm in the model, both of each block's and the final one, so
                one setting switches them together. `embedding.position_encoding=False`
                swaps the sin/cos table for `nn.Identity`, which makes the whole model
                permutation-equivariant: permuting the input permutes the output rows
                the same way and changes nothing else, so word order carries no
                information. `final_norm=False` swaps the norm after the stack for
                `nn.Identity`, which sends the last block's unnormed residual straight
                to `head`.
            vocab_size: `len(tokenizer.vocab)` of the tokenizer the loader used; the
                ids are contiguous, so this covers the protected ids and every merge.

        Raises:
            ValueError: If `d_model` is not divisible by `encoder.attention.h`, from
                `EncoderBlock`.
        """
        super().__init__()
        d_model = model_config.d_model
        embedding, encoder = model_config.embedding, model_config.encoder
        norm = NORMS[encoder.norm]
        self.nn_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=PAD_ID
        )
        self.nn_embedding_dropout = nn.Dropout(p=embedding.dropout)
        self.position_embedding = (
            SinusoidalEmbedding(seq_length=embedding.seq_length, d_model=d_model)
            if embedding.position_encoding
            else nn.Identity()
        )

        self.encoder_blocks = nn.ModuleList(
            modules=[
                EncoderBlock(
                    d_model=d_model,
                    h=encoder.attention.h,
                    qk_norm=encoder.attention.qk_norm,
                    d_ff_multiplier=encoder.ffn_multiplier,
                    dropout=encoder.dropout,
                    norm=norm,
                )
                for _ in range(model_config.num_encoders)  # noqa: NAR001
            ]
        )
        self.final_norm = (
            norm(normalized_shape=d_model) if model_config.final_norm else nn.Identity()
        )
        self.head = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.attention_map: dict[str, list[torch.Tensor]] = {}

    @jaxtyped(typechecker=beartype)
    def forward(self, ids: Int[torch.Tensor, f"{B} {L}"]) -> Float[torch.Tensor, f"{B}*{L} {V}"]:
        """Return logits for every position of a batch of masked, right-padded ids.

        Args:
            ids: `(B, L)` with `L ≤ embedding.seq_length`, right-padded with `PAD_ID`. The mask is
                derived from `PAD_ID` alone, so `MASK_ID` is attended to like any
                other id.

        Returns:
            Raw logits, `(B*L, vocab)`. Row `b*L + l` is position `l` of sentence
            `b`, the order the loader flattens its target in.

        Raises:
            ValueError: If `L` exceeds `embedding.seq_length`, the length of the
                position table.

        Side effects:
            Replaces `self.attention_map` with this call's attention weights under
            `"encoder"`, one detached `(B, h, L, L)` tensor per block in block order.
        """
        pad_mask = padding_keep_mask(token_ids=ids, pad_id=PAD_ID)
        embedding = self.nn_embedding_dropout(self.nn_embedding(ids))  # noqa: NAR001
        # Positional: nn.Identity, the no-position-encoding case, has no `batch` kwarg.
        embedding = self.position_embedding(embedding)  # noqa: NAR001
        attention_map = []
        for encoder_block in self.encoder_blocks:
            embedding, attention_vector = encoder_block(batch=embedding, pad_mask=pad_mask)
            attention_map.append(attention_vector.detach())  # noqa: NAR001
        self.attention_map = {"encoder": attention_map}
        encoded = self.final_norm(embedding)  # noqa: NAR001
        logits = self.head(encoded)  # noqa: NAR001
        return logits.reshape(-1, logits.size()[-1])  # noqa: NAR001
