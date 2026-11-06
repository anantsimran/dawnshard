"""
ag_news_classifier.py
Text classifier for the AG News dataset (see dataload.ag_news for loading).

AG News is a set of news article titles and descriptions, labelled World, Sports,
Business, or Sci/Tech. The loader turns it into padded token ids:

    from training.dal.dao.sqlite_dao import SqliteDao
    from training.dataload.ag_news import (
        get_ag_news_dataloader,
        get_ag_news_texts,
        inspect_ag_news_dataset,
    )
    from training.dataload.constants import CACHE_PATH, TOKENIZERS_PATH
    from training.preprocess.tokenizer import get_tokenizer

    with SqliteDao(path=CACHE_PATH) as cache:
        texts = get_ag_news_texts(n_train=100_000, n_eval=2_000, n_test=2_000, seed=0, cache=cache)
        tokenizer = get_tokenizer(
            merges_path=TOKENIZERS_PATH / "ag_news-1000.json",
            train_texts=[text for text, _ in texts["train"]],
            num_merges=1000,
        )
        train_loader, eval_loader, test_loader = get_ag_news_dataloader(
            tokenizer=tokenizer, max_len=128, cache=cache
        )
    inspect_ag_news_dataset(loader=train_loader, tokenizer=tokenizer)

BPE is trained on the train rows only, which come from `train.csv`. The eval and test
rows are two non-overlapping samples from `test.csv`, so the tokenizer never sees
them.

`run/ag_news_classifier.py` trains one model from its YAML, which samples 100,000 train
rows and 2,000 each for eval and test, with a fixed seed. The sample is stored in the
cache database keyed by those sizes and the seed, so changing any of them samples
afresh. The tokenizer's merges live in `ag_news-{bpe_num_merges}.json` under
TOKENIZERS_PATH: the first run trains and writes it, and later runs load it. The path
alone identifies it, so rename the file, or delete it, after changing the rows it
trains on.

Each split is tokenized once, into a single `(N, max_len)` tensor padded on the right
with `PAD_ID`. Because the data is already in memory, the `DataLoader`s don't use
worker processes, which would only add startup and pickling time. A batch is
`(ids, labels)` with shapes `(B, max_len)` and `(B,)`, and a model reading it needs an
embedding with `len(tokenizer.vocab)` rows and `padding_idx=PAD_ID`.
"""

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import nn
from training.dataclass.config import EncoderConfig
from training.dataload.ag_news import NUM_CLASSES
from training.preprocess.constants import PAD_ID
from training.transformer.constants import B, L
from training.transformer.mask import pad_masked_mean, padding_keep_mask
from training.transformer.modules import NORMS, EncoderBlock, SinusoidalEmbedding


class AGNewsClassifier(nn.Module):
    """Embedding + sin/cos positions → pre-norm encoder stack → mean pool → 4 logits.

    ```
                    token_ids (B, L)  right-padded with PAD_ID
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        padding_keep_mask             embedding          padding_idx=PAD_ID
          (B, 1, 1, L)                    │  (B, L, d_model)
              │                           ▼
              │                   position_encoding      + fixed sin/cos table,
              │                           │  (B, L, d_model)   L ≤ seq_length
              │                           ▼
              │                   embedding_dropout
              │                           │
              ├──────► EncoderBlock × num_encoders ──► weights (B, h, L, L) each
              │                           │  (B, L, d_model)
              │                           ▼
              │                      final_norm                `encoder.norm`; the
              │                           │                    stack is pre-norm,
              │                           │                    so its output is
              │                           │                    not normed otherwise
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
    weights, `{"encoder": [(B, h, L, L) per block]}`, so it describes the most recent
    call only. `save_attention_maps` reads it as an eval probe; nothing else may rely
    on it surviving the next call.

    Output: raw logits (no softmax), as CrossEntropyLoss expects.
    """

    def __init__(self, model_config: EncoderConfig, vocab_size: int) -> None:
        """Build the embedding, encoder stack, and classifier head.

        Args:
            model_config: The architecture. `embedding.seq_length` sizes the position
                table, so it must be at least the loader's `max_len`. `encoder.norm`
                is the class of every norm, the final one included.
                `embedding.position_encoding=False` and `final_norm=False` swap those
                layers for `nn.Identity`; without positions the model is a bag of
                tokens.
            vocab_size: `len(tokenizer.vocab)` of the tokenizer the loader used; the
                ids are contiguous, so this covers `PAD_ID` and every merge.

        Raises:
            ValueError: If `d_model` is not divisible by `encoder.attention.h`, from
                `EncoderBlock`.
        """
        super().__init__()
        d_model = model_config.d_model
        embedding, encoder = model_config.embedding, model_config.encoder
        norm = NORMS[encoder.norm]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=PAD_ID,
        )
        self.position_encoding = (
            SinusoidalEmbedding(seq_length=embedding.seq_length, d_model=d_model)
            if embedding.position_encoding
            else nn.Identity()
        )
        self.embedding_dropout = nn.Dropout(p=embedding.dropout)
        self.blocks = nn.ModuleList(
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
        self.head = nn.Linear(in_features=d_model, out_features=NUM_CLASSES)
        self.attention_map: dict[str, list[torch.Tensor]] = {}

    @jaxtyped(typechecker=beartype)
    def forward(
        self, token_ids: Int[torch.Tensor, f"{B} {L}"]
    ) -> Float[torch.Tensor, f"{B} {NUM_CLASSES}"]:
        """Return logits for a batch of right-padded token ids.

        Args:
            token_ids: Padded ids from `get_ag_news_dataloader`, `(B, L)` with
                `L ≤ embedding.seq_length`; the mask is derived from `PAD_ID`, not
                from a length argument.

        Returns:
            Raw class logits, `(B, NUM_CLASSES)`.

        Raises:
            ValueError: If `L` exceeds `embedding.seq_length`, the length of the
                position table.

        Side effects:
            Replaces `self.attention_map` with this call's attention weights under
            `"encoder"`, one detached `(B, h, L, L)` tensor per block in block order.
        """
        pad_mask = padding_keep_mask(token_ids=token_ids, pad_id=PAD_ID)
        hidden = self.embedding(token_ids)  # noqa: NAR001
        # Positional: nn.Identity, the no-position-encoding case, has no `batch` kwarg.
        hidden = self.position_encoding(hidden)  # noqa: NAR001
        hidden = self.embedding_dropout(hidden)  # noqa: NAR001
        attention_map = []
        for block in self.blocks:
            hidden, weights = block(batch=hidden, pad_mask=pad_mask)
            attention_map.append(weights.detach())  # noqa: NAR001
        self.attention_map = {"encoder": attention_map}
        normalized = self.final_norm(hidden)  # noqa: NAR001
        # (B, 1, 1, L) is the attention view; the pool wants (B, L).
        pooled = pad_masked_mean(attn=normalized, mask=pad_mask[:, 0, 0, :])
        return self.head(pooled)  # noqa: NAR001
