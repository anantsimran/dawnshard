"""
multi30ken_predictor.py
Masked-token predictor for Multi30k English sentences (see dataload.multi30k for
loading and masking).

The loader yields `(X, Y)` int64 batches. X is the sentence with BERT-style masking
applied, `(B, SEQ_LEN)`; Y holds the original id at each selected position and
`CROSS_ENTROPY_IGNORE_INDEX` everywhere else, flattened to `(B*SEQ_LEN,)`, so only the
selected positions are scored. Rows longer than SEQ_LEN are truncated and shorter ones
right-padded with PAD_ID, so a model reading them needs an embedding with `len(vocab)`
rows and `padding_idx=PAD_ID`.

`main` trains one model built from the constants below; change them to train a
different one. The config sweep, the follow-ups, and the component ablation are in
`experiments/multi30ken_predictor_ablation.py`, which pins its own configs, so editing
these constants does not change a past study. Both load the default merges file rather
than retraining BPE; to change it, call the loader once with
`retrain_tokenizer=True, set_default_tokenizer=True`.

Train from the repo root; losses are logged to the terminal after every epoch:

    uv run python app/src/training/model/multi30ken_predictor.py
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
from training.dataload.multi30k import get_multi30ken_mlm_dataloader
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
from training.transformer.mask import padding_keep_mask
from training.transformer.modules import EncoderBlock, SinusoidalEmbedding
from training.transformer.probes import save_attention_maps
from training.viz.attention_heatmap import plot_attention_probes

D_MODEL = 512
SEQ_LEN = 32
DROPOUT_RATE = 0.3
NUM_ENCODERS = 6
NUM_HEADS = 4
FFN_MULTIPLIER = 6
QK_NORM = True
NORM: type[nn.LayerNorm] | type[nn.RMSNorm] = nn.LayerNorm
POSITION_ENCODING = True
FINAL_NORM = True
COSINE_ANNEALING = True
PROBE_SENTENCES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 50
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
              │                           │  (B, L, d_model)  L ≤ seq_len; Identity
              │                           ▼                   if not position_encoding
              └────────────► EncoderBlock × num_encoders ──► weights (B, h, L, L) each
                                          │  (B, L, d_model)  norm inside each
                                          ▼
                                     final_norm               `norm`, or Identity if
                                          │                   not final_norm; the
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

    `norm`, `position_encoding`, and `final_norm` exist so one component at a time can
    be switched off and the cost measured; `COMPONENT_ARMS` in
    `experiments/multi30ken_predictor_ablation.py` is where an arm switches one off.
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
        norm: type[nn.LayerNorm] | type[nn.RMSNorm] = nn.LayerNorm,
        position_encoding: bool = True,
        final_norm: bool = True,
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
            norm: Class behind every norm in the model — both of each block's and the
                final one — so one setting switches them together.
            position_encoding: False swaps the sin/cos table for `nn.Identity`, which
                makes the whole model permutation-equivariant: permuting the input
                permutes the output rows the same way and changes nothing else, so
                word order carries no information.
            final_norm: False swaps the norm after the stack for `nn.Identity`, which
                sends the last block's unnormed residual straight to `head`.

        Raises:
            ValueError: If `d_model` is not divisible by `h`, from `EncoderBlock`.
        """
        super().__init__()
        self.nn_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=PAD_ID
        )
        self.nn_embedding_dropout = nn.Dropout(p=dropout_rate)
        self.position_embedding = (
            SinusoidalEmbedding(seq_length=seq_len, d_model=d_model)
            if position_encoding
            else nn.Identity()
        )

        self.encoder_blocks = nn.ModuleList(
            modules=[
                EncoderBlock(
                    d_model=d_model,
                    h=h,
                    qk_norm=qk_norm,
                    d_ff_multiplier=ffn_multiplier,
                    dropout=dropout_rate,
                    norm=norm,
                )
                for _ in range(num_encoders)  # noqa: NAR001
            ]
        )
        self.final_norm = norm(normalized_shape=d_model) if final_norm else nn.Identity()
        self.head = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.attention_map: dict[str, list[torch.Tensor]] = {}

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


def main():
    """Train one Multi30kEnPredictor from the module constants, optionally to W&B."""
    SHOULD_LOG_WANDB: bool = False
    wandb_run = None
    if SHOULD_LOG_WANDB:
        wandb_run = init_wandb(project="dawnshard", run_name="multi30ken_predictor")
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
    train_loader, val_loader, _, _, vocab = get_multi30ken_mlm_dataloader(
        seq_len=SEQ_LEN,
        val_seed=VAL_SEED,
        test_seed=TEST_SEED,
        batch_size=BATCH_SIZE,
    )
    # Fresh per run, so no single lucky draw is trained on forever; fit writes it to
    # the history file, so a run can be repeated by passing its recorded seed back.
    train_seed = random.getrandbits(32)  # noqa: NAR001
    # fit reseeds from it, but only after the model exists; seed here so weight init
    # uses the recorded seed too. See the README on what the rewind implies.
    torch.manual_seed(seed=train_seed)
    model = Multi30kEnPredictor(
        dropout_rate=DROPOUT_RATE,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        vocab_size=len(vocab),  # noqa: NAR001
        num_encoders=NUM_ENCODERS,
        h=NUM_HEADS,
        ffn_multiplier=FFN_MULTIPLIER,
        qk_norm=QK_NORM,
        norm=NORM,
        position_encoding=POSITION_ENCODING,
        final_norm=FINAL_NORM,
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
    plot_attention_probes(probe_dir=ATTENTION_PROBES_PATH / run_uuid, vocab=vocab)


if __name__ == "__main__":
    main()
