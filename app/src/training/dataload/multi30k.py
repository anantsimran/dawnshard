"""
multi30k.py
Multi30k sentences as translation pairs or as English masked-language-modelling
batches, tokenized with our from-scratch BPE.

Splits are the official train, val, and test_2016_flickr files for one language.
`get_multi30k_texts` downloads all three and, given a `Dao`, stores them, so a caller
can train a tokenizer from `.train` without a second download. The loaders take their
tokenizers instead of training them: the caller decides which texts a tokenizer trains
on and what to call it (see `get_tokenizer`), and the loader decides how to reach the
data with it. Given a `Dao`, the encodings are stored there too.

Translation pairs draw one set of train indices and apply it to both languages, so row
i of the source and the target are the same sentence; the two languages must have the
same number of rows in every split. Each side has its own tokenizer. The source and
the target input start with BOS_ID; the target output is the target shifted one left
and ended with EOS_ID, so position l of the target input predicts position l of the
target output.

Masking follows BERT: MASK_PROB of the non-protected tokens are selected, and of
those MASK_TOKEN_PROB become MASK_ID, RANDOM_TOKEN_PROB a random non-protected id,
and the rest stay unchanged. Train masks are drawn fresh every batch; val and test
are seeded per sample, so their masks are the same on every pass. The DataLoaders
use no worker processes, because the ids are already in memory.
"""

import gzip
import random
import urllib.request
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader
from training.constants import CROSS_ENTROPY_IGNORE_INDEX
from training.dal.cache import cached
from training.dal.dao.dao import Dao
from training.dataload.datasets import InMemoryDataset
from training.preprocess.constants import BOS_ID, EOS_ID, MASK_ID, PAD_ID, PROTECTED_IDS
from training.preprocess.tokenizer import Tokenizer
from training.utils.timing import log_elapsed

# (index, ids: (seq_len,) int64); the index lets _mlm_collate seed masking per sample.
type Multi30kENDatasetChunk = tuple[int, torch.Tensor]
type Multi30kENBatch = tuple[torch.Tensor, torch.Tensor]  # X (B, seq_len), Y (B*seq_len,)

MULTI30K_URL = (
    "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/{name}.{lang}.gz"
)
MULTI30K_RAW_NAMES = {"train": "train", "val": "val", "test": "test_2016_flickr"}
MASK_PROB = 0.15
MASK_TOKEN_PROB = 0.8  # of the selected 15%
RANDOM_TOKEN_PROB = 0.1  # the next 10%; the rest stay unchanged


@dataclass
class Multi30kTexts:
    """The train, val, and test sentences of one language, one string per row."""

    train: list[str]
    val: list[str]
    test: list[str]


@cached(version="1")
def get_multi30k_texts(lang: str) -> Multi30kTexts:
    """
    Return the train, val, and test sentences for `lang`, downloading them.

    The test split is test_2016_flickr. Blank lines are dropped. With a `cache`
    (the keyword `@cached` adds), the result is stored only after all three splits
    downloaded, so a failure partway leaves nothing behind and the error propagates.

    Args:
        lang: Multi30k language code, such as "en" or "de". Not every language has a
            test_2016_flickr file upstream, so the download fails for those.
    """
    splits = {}
    with log_elapsed(label=f"Downloaded Multi30k {lang}"):
        for split, name in MULTI30K_RAW_NAMES.items():
            with urllib.request.urlopen(url=MULTI30K_URL.format(name=name, lang=lang)) as response:
                text = gzip.decompress(data=response.read()).decode(encoding="utf-8")
            splits[split] = [line for line in text.splitlines() if line.strip()]
    return Multi30kTexts(**splits)


def _pad_and_tensorize(
    rows: list[list[int]], seq_len: int, append_bos: bool = False, append_eos: bool = False
) -> torch.Tensor:
    """Right-pad (and truncate) token id rows to seq_len.

    Args:
        rows: Token id sequences, possibly of different lengths.
        seq_len: Target length, BOS_ID and EOS_ID included. Longer rows are
            truncated, and a truncated row still ends with EOS_ID if append_eos.
        append_bos: Put BOS_ID before each row.
        append_eos: Put EOS_ID after each row, before its padding.

    Returns:
        int64 tensor of shape (len(rows), seq_len), padded with PAD_ID.
    """
    bos = [BOS_ID] if append_bos else []
    eos = [EOS_ID] if append_eos else []
    keep_len = seq_len - len(eos) - len(bos)  # noqa: NAR001

    return torch.tensor(
        data=[
            bos + row[:keep_len] + eos + [PAD_ID] * (keep_len - len(row))  # noqa: NAR001
            for row in rows
        ],
        dtype=torch.int64,
    )


def _sample_train_indices(num_rows: int, train_num: Optional[int]) -> list[int]:
    """
    Return the train row indices to keep, in sampled order.

    Args:
        num_rows: Rows in the train split.
        train_num: Rows to sample without replacement; None keeps all of them in
            file order.

    Raises:
        ValueError: If train_num is larger than num_rows.

    Side effects:
        Draws one float from the global torch RNG to seed the sample, so seeding torch
        before the call makes the sample repeat. The draw happens even when train_num
        is None.
    """
    # Drawn even when train_num is None, so the global RNG advances the same way
    # whether or not the train split is sampled.
    sample_seed = float(torch.rand(size=()))  # noqa: NAR001
    if train_num is None:
        return list(range(num_rows))  # noqa: NAR001
    return random.Random(x=sample_seed).sample(population=range(num_rows), k=train_num)  # noqa: NAR001


def get_multi30k_translation_dataloader(
    seq_len: int,
    batch_size: int,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    src_lang: str = "en",
    tgt_lang: str = "de",
    train_num: Optional[int] = 29000,
    val_num: Optional[int] = 1014,
    test_num: Optional[int] = 1000,
    cache: Optional[Dao] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader).

    Each loader yields (X, Y), both int64. X is (B, 2, seq_len): the source ids at
    index 0 and the target input ids at index 1, each starting with BOS_ID. Y is the
    target ids ended with EOS_ID, so Y at position l is the token after X[:, 1] at
    position l, with CROSS_ENTROPY_IGNORE_INDEX on its padding. Y is flattened
    row-major to (B*seq_len,), as in get_multi30ken_mlm_dataloader.

    Args:
        src_tokenizer: Tokenizer for the source side, trained on `src_lang` rows.
        tgt_tokenizer: Tokenizer for the target side, trained on `tgt_lang` rows.
        src_lang: Multi30k language code of the source side.
        tgt_lang: Multi30k language code of the target side.
        train_num: Train pairs to sample; None keeps all of them. One set of indices
            serves both languages, so the pairs stay aligned.
        val_num: Leading val pairs to keep; None keeps all of them.
        test_num: Leading test pairs to keep; None keeps all of them.
        cache: Stores the downloaded texts and the encodings; None stores nothing.

    Raises:
        ValueError: If a split has a different number of rows in the two languages,
            or train_num exceeds the train split.

    Side effects:
        May download both languages' splits and store them and the encodings in
        `cache`. Draws once from the global torch RNG.
    """
    src_texts = get_multi30k_texts(lang=src_lang, cache=cache)  # noqa: NAR001
    tgt_texts = get_multi30k_texts(lang=tgt_lang, cache=cache)  # noqa: NAR001
    for split in MULTI30K_RAW_NAMES:
        src_len, tgt_len = len(getattr(src_texts, split)), len(getattr(tgt_texts, split))  # noqa: NAR001
        if src_len != tgt_len:
            raise ValueError(  # noqa: NAR001
                f"Multi30k {split} has {src_len} {src_lang} rows but {tgt_len} {tgt_lang} "
                "rows, so the pairs cannot be aligned"
            )

    train_indices = _sample_train_indices(num_rows=len(src_texts.train), train_num=train_num)  # noqa: NAR001
    ds = {}
    for split, src_rows, tgt_rows in (
        (
            "train",
            [src_texts.train[i] for i in train_indices],
            [tgt_texts.train[i] for i in train_indices],
        ),
        ("val", src_texts.val[:val_num], tgt_texts.val[:val_num]),
        ("test", src_texts.test[:test_num], tgt_texts.test[:test_num]),
    ):
        src = _pad_and_tensorize(
            rows=src_tokenizer.encode(texts=src_rows, cache=cache),
            seq_len=seq_len,
            append_bos=True,
        )
        encoded_tgt_rows = tgt_tokenizer.encode(texts=tgt_rows, cache=cache)
        tgt_in = _pad_and_tensorize(rows=encoded_tgt_rows, seq_len=seq_len, append_bos=True)
        tgt_out = _pad_and_tensorize(rows=encoded_tgt_rows, seq_len=seq_len, append_eos=True)
        tgt_out = tgt_out.masked_fill(mask=tgt_out == PAD_ID, value=CROSS_ENTROPY_IGNORE_INDEX)

        ds[split] = InMemoryDataset(
            items=list(  # noqa: NAR001
                zip(torch.stack(tensors=(src, tgt_in), dim=1), tgt_out)  # noqa: NAR001
            )
        )

    # Data is already tokenized in memory, so worker processes would only add
    # startup and pickling cost.
    return (
        DataLoader(
            dataset=ds["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_translation_collate,
        ),
        DataLoader(
            dataset=ds["val"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_translation_collate,
        ),
        DataLoader(
            dataset=ds["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_translation_collate,
        ),
    )


def get_multi30ken_mlm_dataloader(
    seq_len: int,
    val_seed: int,
    test_seed: int,
    batch_size: int,
    tokenizer: Tokenizer,
    train_num: Optional[int] = 29000,
    val_num: Optional[int] = 1014,
    test_num: Optional[int] = 1000,
    cache: Optional[Dao] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) over the English sentences.

    Each loader yields (X, Y), both int64: X is the masked input, (B, seq_len), and Y
    the original id at each selected position, CROSS_ENTROPY_IGNORE_INDEX elsewhere.
    Y is flattened row-major to (B*seq_len,), so position l of row b is Y[b*seq_len +
    l]: CrossEntropyLoss takes (N, C) logits against (N,) targets, so a model that
    flattens its logits the same way needs no reshaping in the loss or the metrics.

    Args:
        val_seed: Val sample i is masked with seed val_seed + i. Keep val_seed and
            test_seed further apart than the larger split, or val sample j and test
            sample i share masks whenever val_seed + j == test_seed + i. VAL_SEED
            and TEST_SEED in training.constants are.
        test_seed: As val_seed, for the test split.
        tokenizer: Tokenizer trained on English rows.
        train_num: Train rows to sample; None keeps all of them.
        val_num: Leading val rows to keep; None keeps all of them.
        test_num: Leading test rows to keep; None keeps all of them.
        cache: Stores the downloaded texts and the encodings; None stores nothing.

    Raises:
        ValueError: If train_num exceeds the train split.

    Side effects:
        May download the splits and store them and the encodings in `cache`. Draws
        once from the global torch RNG when train_num is set.
    """
    texts = get_multi30k_texts(lang="en", cache=cache)  # noqa: NAR001
    train_indices = _sample_train_indices(num_rows=len(texts.train), train_num=train_num)  # noqa: NAR001
    train_ds, val_ds, test_ds = (
        InMemoryDataset(
            items=list(  # noqa: NAR001
                enumerate(  # noqa: NAR001
                    _pad_and_tensorize(
                        rows=tokenizer.encode(texts=rows, cache=cache),
                        seq_len=seq_len,
                    ).unbind(dim=0)
                )
            )
        )
        for rows in (
            [texts.train[i] for i in train_indices],
            texts.val[:val_num],
            texts.test[:test_num],
        )
    )

    # Data is already tokenized in memory, so worker processes would only add
    # startup and pickling cost.
    return (
        DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(_mlm_collate, vocab_length=len(tokenizer.vocab)),  # noqa: NAR001
        ),
        DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(  # noqa: NAR001
                _mlm_collate,
                vocab_length=len(tokenizer.vocab),  # noqa: NAR001
                seed=val_seed,
            ),
        ),
        DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(  # noqa: NAR001
                _mlm_collate,
                vocab_length=len(tokenizer.vocab),  # noqa: NAR001
                seed=test_seed,
            ),
        ),
    )


def _translation_collate(
    rows: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a translation batch, flattening the target row-major to `(B*seq_len,)`.

    The model returns its logits flattened to `(B*seq_len, vocab)`, because
    `CrossEntropyLoss` reads dim 1 as the class axis. Flattening the target here, as
    `_mlm_collate` does, keeps row `b*seq_len + l` of both on the same position.

    Args:
        rows: `(src and tgt_in stacked (2, seq_len), tgt_out (seq_len,))` items, as
            `get_multi30k_translation_dataloader` stores them.

    Returns:
        `(X, Y)`: X is `(B, 2, seq_len)`, Y is `(B*seq_len,)`.
    """
    inputs, targets = zip(*rows)  # noqa: NAR001
    return torch.stack(tensors=inputs), torch.cat(tensors=targets)


def _mlm_collate(
    rows: list[Multi30kENDatasetChunk],
    vocab_length: int,
    seed: Optional[int] = None,
) -> Multi30kENBatch:
    """
    Mask a batch for MLM with BERT's 80/10/10 rule, never on a protected id.

    Args:
        rows: (index, ids) items as get_multi30ken_mlm_dataloader stores them, ids
            (seq_len,) int64. index must be the sample's position in its split.
        vocab_length: len(vocab); random replacements are drawn from its
            non-protected ids.
        seed: None draws from the global RNG, so masks change on every call. An int
            seeds sample i with seed + i, so its masks are the same on every pass
            and the global RNG is left untouched.

    Returns:
        (X, Y), int64. X is (B, seq_len); Y is the original id at each selected
        position and CROSS_ENTROPY_IGNORE_INDEX elsewhere, flattened row-major to
        (B*seq_len,).

    Raises:
        ValueError: If rows is empty.
    """
    if not rows:
        raise ValueError("_mlm_collate got an empty batch")  # noqa: NAR001
    protected = torch.tensor(data=PROTECTED_IDS)
    x_rows = []
    y_rows = []
    for index, ids in rows:
        gen = None
        if seed is not None:
            # Generator.manual_seed is a C method and takes no keyword arguments.
            gen = torch.Generator().manual_seed(seed + index)  # noqa: NAR001
        seq_len = ids.shape[0]
        x = ids.clone()
        chance = torch.rand(size=(seq_len,), generator=gen)
        selected = (chance < MASK_PROB) & ~torch.isin(elements=x, test_elements=protected)
        y_rows.append(  # noqa: NAR001
            torch.where(condition=selected, input=x, other=CROSS_ENTROPY_IGNORE_INDEX)
        )

        action = torch.rand(size=(seq_len,), generator=gen)
        to_mask = selected & (action < MASK_TOKEN_PROB)
        to_random = (
            selected & (action >= MASK_TOKEN_PROB) & (action < MASK_TOKEN_PROB + RANDOM_TOKEN_PROB)
        )
        random_ids = torch.randint(
            low=0,
            high=vocab_length - len(PROTECTED_IDS),  # noqa: NAR001
            size=(int(to_random.sum()),),  # noqa: NAR001
            generator=gen,
        )
        # PROTECTED_IDS are contiguous from PAD_ID, so shifting every draw at or
        # above PAD_ID past them leaves only byte and merge ids.
        random_ids += len(PROTECTED_IDS) * (random_ids >= PAD_ID)  # noqa: NAR001
        x[to_mask] = MASK_ID
        x[to_random] = random_ids
        x_rows.append(x)  # noqa: NAR001
    return torch.stack(tensors=x_rows), torch.cat(tensors=y_rows)
