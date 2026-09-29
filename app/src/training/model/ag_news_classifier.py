"""
ag_news_classifier.py
Text classifier for the AG News dataset (see dataload.ag_news for loading).

AG News is a set of news article titles and descriptions, labelled World, Sports,
Business, or Sci/Tech. The loader turns it into padded token ids:

    from training.dataload.ag_news import (
        get_ag_news_dataloader,
        inspect_ag_news_dataset,
    )

    train_loader, eval_loader, test_loader, merges, vocab = get_ag_news_dataloader(
        max_len=128
    )
    inspect_ag_news_dataset(loader=train_loader, vocab=vocab)

BPE is trained on the train rows only, which come from `train.csv`. The eval and test
rows are two non-overlapping samples from `test.csv`, so the tokenizer never sees
them.

By default the loader samples 20,000 train rows and 2,000 each for eval and test, with
a fixed seed, and caches them as CSV files. The cache only checks that those files
exist, so pass `refresh_cache=True` after changing the sample sizes or the seed. The
trained merges are cached next to them and retrained whenever the rows, `num_merges`,
or BPE's pre-tokenization pattern change.

Each split is tokenized once, into a single `(N, max_len)` tensor padded on the right
with `PAD_ID`. Because the data is already in memory, the `DataLoader`s don't use
worker processes, which would only add startup and pickling time. A batch is
`(ids, labels)` with shapes `(B, max_len)` and `(B,)`, and a model reading it needs an
embedding with `len(vocab)` rows and `padding_idx=PAD_ID`.

Train from the repo root; losses are logged to the terminal after every epoch:

    uv run python app/src/training/model/ag_news_classifier.py
"""

import random
from functools import partial
from uuid import uuid4

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import nn
from training.common.constants import PAD_ID
from training.constants import ATTENTION_PROBES_PATH, TEST_SEED, VAL_SEED
from training.dataload.ag_news import NUM_CLASSES, get_ag_news_dataloader
from training.metrics.classification_metrics import (
    ClassificationMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.setup import DEVICE, init_wandb
from training.train.model import EpochSpec, TrainState
from training.train.train_loop import fit
from training.transformer.constants import B, L
from training.transformer.mask import pad_masked_mean, padding_keep_mask
from training.transformer.modules import EncoderBlock, SinusoidalEmbedding
from training.transformer.probes import save_attention_maps
from training.viz.attention_heatmap import plot_attention_probes

MAX_LEN = 128
D_MODEL = 64
H = 4
NUM_BLOCKS = 2
DROPOUT = 0.1
PROBE_SENTENCES = 4


class AGNewsClassifier(nn.Module):
    """Embedding + sin/cos positions → 2 pre-norm encoder blocks → mean pool → 4 logits.

    ```
                    token_ids (B, L)  right-padded with PAD_ID
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        padding_keep_mask             embedding          padding_idx=PAD_ID
          (B, 1, 1, L)                    │  (B, L, d_model)
              │                           ▼
              │                   position_encoding      + fixed sin/cos table,
              │                           │  (B, L, d_model)   L ≤ MAX_LEN
              │                           ▼
              │                   embedding_dropout
              │                           │
              ├──────────────► EncoderBlock × 2 ──► weights (B, h, L, L) each
              │                           │  (B, L, d_model)
              │                           ▼
              │                      final_norm                the stack is
              │                           │                    pre-norm, so its
              │                           │                    output is not
              │                           │                    normed otherwise
              └─► [:, 0, 0, :] ─► pad_masked_mean
                     (B, L)              │  (B, d_model)
                                         ▼
                                       head                    Linear to 4
                                         │
                                         ▼  logits (B, 4)
    ```

    The padding mask is derived inside `forward`, because the training loop calls
    `model(ids)` with one tensor and has nowhere to pass a mask. One mask does both
    jobs: hiding pad keys inside attention, and keeping pad positions out of the mean.
    It is built once and shared by every block, since padding doesn't change with
    depth.

    The pool is a mean over real tokens rather than a `[CLS]` position, because
    nothing here trains such a position: every token is a news word, so averaging what
    attention produced is the cheaper answer. The pool itself is order-blind, but by
    then each vector already carries its position, so order survives it.

    `position_encoding` is what makes this a sequence model rather than a bag of
    tokens: without it, attention and the mean pool together are permutation
    invariant, and "Apple sues Samsung" reads the same as "Samsung sues Apple". The
    table is fixed, so it adds no parameters, and it is added before
    `embedding_dropout` so the dropout acts on the sum, as in the original
    Transformer. Pad positions get encodings too; the padding mask keeps them out.

    Every `forward` overwrites `attention_map` with each block's detached attention
    weights, (B, h, L, L) per block in block order, so it describes the most recent
    call only. `save_attention_maps` reads it as an eval probe; nothing else may rely
    on it surviving the next call.

    Output: raw logits (no softmax), as CrossEntropyLoss expects.
    """

    def __init__(self, vocab_size: int) -> None:
        """Build the embedding, encoder stack, and classifier head.

        Args:
            vocab_size: `len(vocab)` from `get_ag_news_dataloader`; the ids are
                contiguous, so this covers `PAD_ID` and every merge.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=D_MODEL,
            padding_idx=PAD_ID,
        )
        self.position_encoding = SinusoidalEmbedding(seq_length=MAX_LEN, d_model=D_MODEL)
        self.embedding_dropout = nn.Dropout(p=DROPOUT)
        self.blocks = nn.ModuleList(
            modules=[
                EncoderBlock(d_model=D_MODEL, h=H, qk_norm=True, dropout=DROPOUT)
                for _ in range(NUM_BLOCKS)  # noqa: NAR001
            ]
        )
        self.final_norm = nn.LayerNorm(normalized_shape=D_MODEL)
        self.head = nn.Linear(in_features=D_MODEL, out_features=NUM_CLASSES)
        self.attention_map: list[torch.Tensor] = []

    @jaxtyped(typechecker=beartype)
    def forward(
        self, token_ids: Int[torch.Tensor, f"{B} {L}"]
    ) -> Float[torch.Tensor, f"{B} {NUM_CLASSES}"]:
        """Return logits for a batch of right-padded token ids.

        Args:
            token_ids: Padded ids from `get_ag_news_dataloader`, `(B, L)` with
                `L ≤ MAX_LEN`; the mask is derived from `PAD_ID`, not from a length
                argument.

        Returns:
            Raw class logits, `(B, NUM_CLASSES)`.

        Raises:
            ValueError: If `L` exceeds `MAX_LEN`, the length of the position table.

        Side effects:
            Replaces `self.attention_map` with this call's attention weights, one
            detached `(B, h, L, L)` tensor per block in block order.
        """
        pad_mask = padding_keep_mask(token_ids=token_ids, pad_id=PAD_ID)
        hidden = self.embedding(token_ids)  # noqa: NAR001
        hidden = self.position_encoding(batch=hidden)
        hidden = self.embedding_dropout(hidden)  # noqa: NAR001
        attention_map = []
        for block in self.blocks:
            hidden, weights = block(batch=hidden, pad_mask=pad_mask)
            attention_map.append(weights.detach())  # noqa: NAR001
        self.attention_map = attention_map
        normalized = self.final_norm(hidden)  # noqa: NAR001
        # (B, 1, 1, L) is the attention view; the pool wants (B, L).
        pooled = pad_masked_mean(attn=normalized, mask=pad_mask[:, 0, 0, :])
        return self.head(pooled)  # noqa: NAR001


def main():
    """Train AGNewsClassifier on AG News, optionally logging to Weights & Biases."""
    SHOULD_LOG_WANDB: bool = False
    wandb_run = None
    if SHOULD_LOG_WANDB:
        wandb_run = init_wandb(project="dawnshard", run_name="ag_news_classifier")
    probe_dir = ATTENTION_PROBES_PATH / str(uuid4())  # noqa: NAR001
    epoch_spec = EpochSpec(
        device=DEVICE,
        metrics_type=ClassificationMetrics,
        calculate_metrics=calculate_metrics,
        accumulate_metrics=accumulate_metrics,
        reduce_metrics=reduce_metrics,
        eval_probe=partial(  # noqa: NAR001
            save_attention_maps,
            out_dir=probe_dir,
            num_sentences=PROBE_SENTENCES,
        ),
    )
    train_loader, val_loader, _, _, vocab = get_ag_news_dataloader(max_len=MAX_LEN)
    # Fresh per run, so no single lucky draw is trained on forever; fit writes it to
    # the history file, so a run can be repeated by passing its recorded seed back.
    train_seed = random.getrandbits(32)  # noqa: NAR001
    # fit reseeds from it, but only after the model exists; seed here so weight init
    # is reproducible too. See the README on what the rewind implies.
    torch.manual_seed(seed=train_seed)
    model = AGNewsClassifier(vocab_size=len(vocab)).to(device=DEVICE)  # noqa: NAR001
    train_state = TrainState(
        model=model,
        optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
    )
    fit(
        state=train_state,
        epoch_spec=epoch_spec,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        val_epoch_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        train_seed=train_seed,
        val_seed=VAL_SEED,
        test_seed=TEST_SEED,
        wandb_run=wandb_run,
    )
    plot_attention_probes(probe_dir=probe_dir, vocab=vocab)


if __name__ == "__main__":
    main()
