# 1. Rename to multi30k.py
# 2. Change change, encoding to both de and en
# 3. ADD BOS and EOS (Depending on source and target language)
# 4. The output batch should be stacked, not cat
#
#
"""
multi30k.py
Multi30k sentences as translation pairs or as English masked-language-modelling
batches, tokenized with our from-scratch BPE.

Splits are the official train, val, and test_2016_flickr files for one language,
downloaded once to data_dir/multi30k/<lang>/. The download cache only checks that the
files exist.

Each tokenizer trains on train rows only. A trained tokenizer is reused only once it
is the default: the merges file that a tokenizer_path.txt pointer file names, which
only set_default_tokenizer=True writes. Until then every call retrains and logs a
warning that the new merges file will not be reused. This is intentional: nothing
becomes the default by accident. The default is loaded as is, so pass
retrain_tokenizer=True after changing train_num or BPE_NUM_MERGES.

Translation pairs read both languages with one train sample seed, so row i of the
source and the target are the same sentence. Each side gets its own tokenizer. The
source and the target input start with BOS_ID; the target output is the target
shifted one left and ended with EOS_ID, so position l of the target input predicts
position l of the target output.

Masking follows BERT: MASK_PROB of the non-protected tokens are selected, and of
those MASK_TOKEN_PROB become MASK_ID, RANDOM_TOKEN_PROB a random non-protected id,
and the rest stay unchanged. Train masks are drawn fresh every batch; val and test
are seeded per sample, so their masks are the same on every pass. The DataLoaders
use no worker processes, because the ids are already in memory.
"""

import gzip
import random
import time
import urllib.request
from dataclasses import astuple, dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader
from training.common import bpe
from training.common.constants import BOS_ID, EOS_ID, MASK_ID, PAD_ID, PROTECTED_IDS
from training.constants import CROSS_ENTROPY_IGNORE_INDEX
from training.dataload.constants import DATASETS_CACHE_DIR
from training.dataload.datasets import InMemoryDataset
from training.utils.timing import log_elapsed

# (index, ids: (seq_len,) int64); the index lets _mlm_collate seed masking per sample.
type Multi30kENDatasetChunk = tuple[int, torch.Tensor]
type Multi30kENBatch = tuple[torch.Tensor, torch.Tensor]  # X (B, seq_len), Y (B*seq_len,)

TRAIN_TXT_NAME = "train.txt"
TEST_TXT_NAME = "test.txt"
VAL_TXT_NAME = "val.txt"
MULTI30K_URL = (
    "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/{name}.{lang}.gz"
)
MULTI30K_RAW_NAMES = {"train": "train", "val": "val", "test": "test_2016_flickr"}
DEFAULT_TOKENIZER_PATHS_FILE_NAME = "tokenizer_path.txt"
DATA_SUBDIR_NAME = "multi30k"
BPE_NUM_MERGES = 10000
MASK_PROB = 0.15
MASK_TOKEN_PROB = 0.8  # of the selected 15%
RANDOM_TOKEN_PROB = 0.1  # the next 10%; the rest stay unchanged


@dataclass(frozen=True)
class Multi30kPaths:
    """Paths of the three split text files, one sentence per line."""

    train: Path
    val: Path
    test: Path


def _get_cached_or_downloaded_txts(cache: bool, data_dir: Path, lang: str) -> Multi30kPaths:
    """
    Return the paths of the train, val, and test text files for `lang`.

    On a cache miss, downloads the official Multi30k splits for `lang` (test is
    test_2016_flickr). The cache only checks that the files exist.

    Args:
        lang: Multi30k language code, such as "en" or "de". Not every language has a
            test_2016_flickr file upstream, so the download fails for those.

    Side effects:
        On a cache miss, writes the three files under data_dir/multi30k/<lang>/.
    """
    txt_dir = data_dir / DATA_SUBDIR_NAME
    paths = Multi30kPaths(
        train=txt_dir / lang / TRAIN_TXT_NAME,
        val=txt_dir / lang / VAL_TXT_NAME,
        test=txt_dir / lang / TEST_TXT_NAME,
    )
    if cache and all(path.exists() for path in astuple(obj=paths)):  # noqa: NAR001
        logger.info("Loading Multi30k {} splits from {}", lang, txt_dir / lang)  # noqa: NAR001
        return paths

    (txt_dir / lang).mkdir(parents=True, exist_ok=True)
    with log_elapsed(label=f"Downloaded Multi30k {lang}"):
        for split, name in MULTI30K_RAW_NAMES.items():
            with urllib.request.urlopen(url=MULTI30K_URL.format(name=name, lang=lang)) as response:
                text = gzip.decompress(data=response.read()).decode(encoding="utf-8")
            rows = [line for line in text.splitlines() if line.strip()]
            path = getattr(paths, split)  # noqa: NAR001
            path.write_text(  # noqa: NAR001
                data="\n".join(rows) + "\n",  # noqa: NAR001
                encoding="utf-8",
            )
            logger.info(  # noqa: NAR001
                "Wrote {} {} sentences to {}",
                len(rows),  # noqa: NAR001
                split,
                path,
            )
    return paths


def _save_merges(merges: list[bpe.Phrase], path: Path) -> None:
    """Write each merge as a "first_id second_id" line, in rank order."""
    ids = {bpe.Phrase(encoding=b): b for b in range(256)}  # noqa: NAR001
    ids.update(  # noqa: NAR001
        {m: bpe.FIRST_MERGE_ID + k for k, m in enumerate(merges)}  # noqa: NAR001
    )
    lines = []
    for m in merges:
        assert m.first is not None and m.second is not None
        lines.append(f"{ids[m.first]} {ids[m.second]}")  # noqa: NAR001
    path.write_text(data="\n".join(lines), encoding="utf-8")  # noqa: NAR001


def _load_merges(path: Path) -> tuple[list[bpe.Phrase], dict[int, str]]:
    """Rebuild (merges, vocab) as bpe.train returns them from a _save_merges file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    pairs = [(int(a), int(b)) for a, b in (line.split() for line in lines)]  # noqa: NAR001
    return bpe.tokenizer_from_pairs(pairs=pairs)


def _train_tokenizer_or_cached(
    retrain_tokenizer: bool,
    set_default_tokenizer: bool,
    train_rows: list[str],
    data_dir: Path,
) -> tuple[list[bpe.Phrase], dict[int, str]]:
    """
    Return (merges, vocab), trained on train_rows or loaded from the default tokenizer.

    The default tokenizer is the merges file that the pointer file
    data_dir/multi30k/tokenizer_path.txt names. It is loaded as is: nothing
    checks that it was trained on these rows or with BPE_NUM_MERGES, so pass
    retrain_tokenizer=True after changing either.

    Args:
        retrain_tokenizer: Train a new tokenizer even if a default exists. With no
            default, it trains regardless.
        set_default_tokenizer: After training, point the pointer file at the new
            merges. Ignored when the tokenizer is loaded.
        train_rows: Text the tokenizer trains on, one sentence per row.
        data_dir: Root of this tokenizer's pointer and merges files. Pass a
            different one per tokenizer that must not share a default, such as one
            per language.

    Side effects:
        On training, writes merges_{len(train_rows)}_{timestamp}.txt under
        data_dir/multi30k/, and overwrites the pointer file if
        set_default_tokenizer. Without it, nothing references the new merges file,
        a later call will not load it, and a warning says so. With no default and
        set_default_tokenizer=False, every call retrains; this is intentional.
    """
    tokenization_paths_file = data_dir / DATA_SUBDIR_NAME / DEFAULT_TOKENIZER_PATHS_FILE_NAME
    merges_path = None
    if tokenization_paths_file.exists() and tokenization_paths_file.stat().st_size > 0:
        merges_path = Path(  # noqa: NAR001
            tokenization_paths_file.read_text(encoding="utf-8").strip()
        )
    if merges_path is not None and merges_path.stat().st_size > 0 and not retrain_tokenizer:
        return _load_merges(path=merges_path)

    merges_path = (
        data_dir
        / DATA_SUBDIR_NAME
        / (
            f"merges_{len(train_rows)}_"  # noqa: NAR001
            f"{time.strftime('%Y%m%d-%H%M%S')}.txt"  # noqa: NAR001
        )
    )
    merges_path.parent.mkdir(parents=True, exist_ok=True)
    with log_elapsed(label=f"BPE trained {BPE_NUM_MERGES} merges"):
        merges, vocab = bpe.train(
            text="\n".join(train_rows),  # noqa: NAR001
            num_merges=BPE_NUM_MERGES,
        )
        _save_merges(merges=merges, path=merges_path)
    if set_default_tokenizer:
        tokenization_paths_file.write_text(
            data=str(merges_path),  # noqa: NAR001
            encoding="utf-8",
        )
    else:
        logger.warning(  # noqa: NAR001
            "Trained a tokenizer at {} but it is not the default, so no later call "
            "will load it; pass retrain_tokenizer=True, set_default_tokenizer=True "
            "to keep the next one",
            merges_path,
        )
    return merges, vocab


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


def _read_rows(
    datapaths: Multi30kPaths,
    train_sample_seed: float,
    train_num: Optional[int],
    val_num: Optional[int],
    test_num: Optional[int],
) -> tuple[list[str], list[str], list[str]]:
    """
    Return (train_rows, val_rows, test_rows) read from datapaths.

    Args:
        train_sample_seed: Seeds the sample of train_num train rows. Two languages
            read with the same seed keep the same sentence indices, so their rows
            stay aligned, because the files have the same number of lines.
    """
    train_rows = datapaths.train.read_text(encoding="utf-8").splitlines()
    if train_num is not None:
        train_rows = random.Random(x=train_sample_seed).sample(  # noqa: NAR001
            population=train_rows, k=train_num
        )
    val_rows = datapaths.val.read_text(encoding="utf-8").splitlines()[:val_num]
    test_rows = datapaths.test.read_text(encoding="utf-8").splitlines()[:test_num]
    return train_rows, val_rows, test_rows


def _validate_flags(cache: bool, retrain_tokenizer: bool, set_default_tokenizer: bool) -> None:
    """
    Reject cache and tokenizer flag combinations that cannot work together.

    Raises:
        ValueError: If cache=False without retrain_tokenizer, or
            set_default_tokenizer without retrain_tokenizer.
    """
    if not cache and not retrain_tokenizer:
        raise ValueError(  # noqa: NAR001
            "cache=False re-downloads the data, so retrain_tokenizer must be True"
        )
    if set_default_tokenizer and not retrain_tokenizer:
        raise ValueError(  # noqa: NAR001
            "set_default_tokenizer=True requires retrain_tokenizer=True"
        )


def get_multi30k_translation_dataloader(
    seq_len: int,
    batch_size: int,
    cache: bool = True,
    src_lang: str = "en",
    tgt_lang: str = "de",
    train_num: Optional[int] = 29000,
    val_num: Optional[int] = 1014,
    test_num: Optional[int] = 1000,
    retrain_tokenizer: bool = False,
    set_default_tokenizer: bool = False,
    data_dir: Path = DATASETS_CACHE_DIR,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    list[bpe.Phrase],
    dict[int, str],
    list[bpe.Phrase],
    dict[int, str],
]:
    """
    Return (train_loader, val_loader, test_loader, src_merges, src_vocab, tgt_merges,
    tgt_vocab).

    Each loader yields (X, Y), both int64. X is (B, 2, seq_len): the source ids at
    index 0 and the target input ids at index 1, each starting with BOS_ID. Y is the
    target ids ended with EOS_ID, so Y at position l is the token after X[:, 1] at
    position l, with CROSS_ENTROPY_IGNORE_INDEX on its padding. Y is flattened
    row-major to (B*seq_len,), as in get_multi30ken_mlm_dataloader.

    Args:
        src_lang: Multi30k language code of the source side.
        tgt_lang: Multi30k language code of the target side.
        train_num: Train pairs to sample; None keeps all of them. Both languages
            sample with one seed, so the pairs stay aligned.
        val_num: Leading val pairs to keep; None keeps all of them.
        test_num: Leading test pairs to keep; None keeps all of them.
        retrain_tokenizer: Forwarded to both tokenizers; see
            _train_tokenizer_or_cached.
        set_default_tokenizer: Forwarded to both tokenizers.

    Raises:
        ValueError: On an invalid flag combination; see _validate_flags.

    Side effects:
        May download both languages' splits, and writes each side's merges file and
        pointer under data_dir/<lang>/multi30k/; see _get_cached_or_downloaded_txts
        and _train_tokenizer_or_cached.
    """
    _validate_flags(
        cache=cache,
        retrain_tokenizer=retrain_tokenizer,
        set_default_tokenizer=set_default_tokenizer,
    )

    src_datapaths = _get_cached_or_downloaded_txts(cache=cache, data_dir=data_dir, lang=src_lang)
    tgt_datapaths = _get_cached_or_downloaded_txts(cache=cache, data_dir=data_dir, lang=tgt_lang)

    # One seed for both languages, so source and target sample the same sentences.
    train_sample_seed = float(torch.rand(size=()))  # noqa: NAR001
    src_train_rows, src_val_rows, src_test_rows = _read_rows(
        datapaths=src_datapaths,
        train_sample_seed=train_sample_seed,
        train_num=train_num,
        val_num=val_num,
        test_num=test_num,
    )
    tgt_train_rows, tgt_val_rows, tgt_test_rows = _read_rows(
        datapaths=tgt_datapaths,
        train_sample_seed=train_sample_seed,
        train_num=train_num,
        val_num=val_num,
        test_num=test_num,
    )

    src_merges, src_vocab = _train_tokenizer_or_cached(
        retrain_tokenizer=retrain_tokenizer,
        set_default_tokenizer=set_default_tokenizer,
        train_rows=src_train_rows,
        data_dir=data_dir / src_lang,
    )

    tgt_merges, tgt_vocab = _train_tokenizer_or_cached(
        retrain_tokenizer=retrain_tokenizer,
        set_default_tokenizer=set_default_tokenizer,
        train_rows=tgt_train_rows,
        data_dir=data_dir / tgt_lang,
    )
    ds = {}
    for split, src_rows, tgt_rows in zip(  # noqa: NAR001
        ("train", "val", "test"),
        (src_train_rows, src_val_rows, src_test_rows),
        (tgt_train_rows, tgt_val_rows, tgt_test_rows),
    ):
        src = _pad_and_tensorize(
            rows=[bpe.encode(text=row, merges=src_merges) for row in src_rows],
            seq_len=seq_len,
            append_bos=True,
        )
        encoded_tgt_rows = [bpe.encode(text=row, merges=tgt_merges) for row in tgt_rows]
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
        src_merges,
        src_vocab,
        tgt_merges,
        tgt_vocab,
    )


def get_multi30ken_mlm_dataloader(
    seq_len: int,
    val_seed: int,
    test_seed: int,
    batch_size: int,
    cache: bool = True,
    train_num: Optional[int] = 29000,
    val_num: Optional[int] = 1014,
    test_num: Optional[int] = 1000,
    retrain_tokenizer: bool = False,
    set_default_tokenizer: bool = False,
    data_dir: Path = DATASETS_CACHE_DIR,
) -> tuple[DataLoader, DataLoader, DataLoader, list[bpe.Phrase], dict[int, str]]:
    """
    Return (train_loader, val_loader, test_loader, merges, vocab).

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
        train_num: Train rows to sample; None keeps all of them.
        val_num: Leading val rows to keep; None keeps all of them.
        test_num: Leading test rows to keep; None keeps all of them.

    Raises:
        ValueError: On an invalid flag combination; see _validate_flags.

    Side effects:
        May download the splits and write a merges file under data_dir; see
        _get_cached_or_downloaded_txts and _train_tokenizer_or_cached.
    """
    _validate_flags(
        cache=cache,
        retrain_tokenizer=retrain_tokenizer,
        set_default_tokenizer=set_default_tokenizer,
    )

    datapaths = _get_cached_or_downloaded_txts(cache=cache, data_dir=data_dir, lang="en")

    train_rows, val_rows, test_rows = _read_rows(
        datapaths=datapaths,
        train_sample_seed=float(torch.rand(size=())),  # noqa: NAR001
        train_num=train_num,
        val_num=val_num,
        test_num=test_num,
    )

    merges, vocab = _train_tokenizer_or_cached(
        retrain_tokenizer=retrain_tokenizer,
        set_default_tokenizer=set_default_tokenizer,
        train_rows=train_rows,
        data_dir=data_dir,
    )
    train_ds, val_ds, test_ds = (
        InMemoryDataset(
            items=list(  # noqa: NAR001
                enumerate(  # noqa: NAR001
                    _pad_and_tensorize(
                        rows=[bpe.encode(text=row, merges=merges) for row in rows],
                        seq_len=seq_len,
                    ).unbind(dim=0)
                )
            )
        )
        for rows in (train_rows, val_rows, test_rows)
    )

    # Data is already tokenized in memory, so worker processes would only add
    # startup and pickling cost.
    return (
        DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(_mlm_collate, vocab_length=len(vocab)),  # noqa: NAR001
        ),
        DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(  # noqa: NAR001
                _mlm_collate,
                vocab_length=len(vocab),  # noqa: NAR001
                seed=val_seed,
            ),
        ),
        DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(  # noqa: NAR001
                _mlm_collate,
                vocab_length=len(vocab),  # noqa: NAR001
                seed=test_seed,
            ),
        ),
        merges,
        vocab,
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
