"""
multi30ken.py
Multi30k English sentences as masked-language-modelling batches, tokenized with our
from-scratch BPE.

Splits are the official train, val, and test_2016_flickr files, downloaded once to
data_dir/multi30k_en/. The download cache only checks that the files exist.

The tokenizer trains on train rows only. A trained tokenizer is reused only once it
is the default: the merges file that data_dir/multi30k_en/tokenizer_path.txt names,
which only set_default_tokenizer=True writes. Until then every call retrains and
logs a warning that the new merges file will not be reused. This is intentional:
nothing becomes the default by accident. The default is loaded as is, so pass
retrain_tokenizer=True after changing train_num or BPE_NUM_MERGES.

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
from torch.utils.data import DataLoader, Dataset
from training.common import bpe
from training.common.constants import MASK_ID, PAD_ID, PROTECTED_IDS
from training.constants import CROSS_ENTROPY_IGNORE_INDEX
from training.dataload.constants import DATASETS_CACHE_DIR
from training.utils.timing import log_elapsed

type Multi30kENDatasetChunk = tuple[int, torch.Tensor]  # (index, ids: (seq_len,) int64)
type Multi30kENBatch = tuple[torch.Tensor, torch.Tensor]  # X (B, seq_len), Y (B*seq_len,)

TRAIN_TXT_NAME = "train.txt"
TEST_TXT_NAME = "test.txt"
VAL_TXT_NAME = "val.txt"
MULTI30K_URL = (
    "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/{name}.en.gz"
)
MULTI30K_RAW_NAMES = {"train": "train", "val": "val", "test": "test_2016_flickr"}
DEFAULT_TOKENIZER_PATHS_FILE_NAME = "tokenizer_path.txt"
DATA_SUBDIR_NAME = "multi30k_en"
BPE_NUM_MERGES = 2000
MASK_PROB = 0.15
MASK_TOKEN_PROB = 0.8  # of the selected 15%
RANDOM_TOKEN_PROB = 0.1  # the next 10%; the rest stay unchanged


@dataclass(frozen=True)
class Multi30kENPaths:
    """Paths of the three split text files, one English sentence per line."""

    train: Path
    val: Path
    test: Path


def _get_cached_or_downloaded_txts(cache: bool, data_dir: Path) -> Multi30kENPaths:
    """
    Return the paths of the train, val, and test text files.

    On a cache miss, downloads the official Multi30k English splits (test is
    test_2016_flickr). The cache only checks that the files exist.

    Side effect: on a cache miss, writes the three files under data_dir/multi30k_en/.
    """
    txt_dir = data_dir / DATA_SUBDIR_NAME
    paths = Multi30kENPaths(
        train=txt_dir / TRAIN_TXT_NAME,
        val=txt_dir / VAL_TXT_NAME,
        test=txt_dir / TEST_TXT_NAME,
    )
    if cache and all(path.exists() for path in astuple(obj=paths)):  # noqa: NAR001
        logger.info("Loading Multi30k EN splits from {}", txt_dir)  # noqa: NAR001
        return paths

    txt_dir.mkdir(parents=True, exist_ok=True)
    with log_elapsed(label="Downloaded Multi30k EN"):
        for split, name in MULTI30K_RAW_NAMES.items():
            with urllib.request.urlopen(url=MULTI30K_URL.format(name=name)) as response:
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
    data_dir/multi30k_en/tokenizer_path.txt names. It is loaded as is: nothing
    checks that it was trained on these rows or with BPE_NUM_MERGES, so pass
    retrain_tokenizer=True after changing either.

    Args:
        retrain_tokenizer: Train a new tokenizer even if a default exists. With no
            default, it trains regardless.
        set_default_tokenizer: After training, point the pointer file at the new
            merges. Ignored when the tokenizer is loaded.
        train_rows: Text the tokenizer trains on, one sentence per row.

    Side effects:
        On training, writes merges_{len(train_rows)}_{timestamp}.txt under
        data_dir/multi30k_en/, and overwrites the pointer file if
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


def _pad_and_tensorize(rows: list[list[int]], seq_len: int) -> torch.Tensor:
    """Right-pad (and truncate) token id rows to seq_len.

    Args:
        rows: Token id sequences, possibly of different lengths.
        seq_len: Target length. Longer rows are truncated.

    Returns:
        int64 tensor of shape (len(rows), seq_len), padded with PAD_ID.
    """
    return torch.tensor(
        data=[row[:seq_len] + [PAD_ID] * (seq_len - len(row)) for row in rows],  # noqa: NAR001
        dtype=torch.int64,
    )


class Multi30kENDataset(Dataset):
    """One split of Multi30k EN, as a (N, seq_len) tensor of PAD_ID-padded token ids."""

    def __init__(self, ids: torch.Tensor) -> None:
        """Wrap already-encoded ids; get_multi30ken_mlm_dataloader builds them."""
        super().__init__()
        self._ids = ids

    def __getitem__(self, index: int) -> Multi30kENDatasetChunk:
        """Return (index, ids); the index lets collate seed masking per sample."""
        return index, self._ids[index]

    def __len__(self) -> int:
        """Return the number of sentences in this split."""
        return self._ids.shape[0]


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

    datapaths = _get_cached_or_downloaded_txts(cache=cache, data_dir=data_dir)

    train_rows = datapaths.train.read_text(encoding="utf-8").splitlines()
    if train_num is not None:
        train_rows = random.Random(x=float(torch.rand(size=()))).sample(  # noqa: NAR001
            population=train_rows, k=train_num
        )
    val_rows = datapaths.val.read_text(encoding="utf-8").splitlines()[:val_num]
    test_rows = datapaths.test.read_text(encoding="utf-8").splitlines()[:test_num]

    merges, vocab = _train_tokenizer_or_cached(
        retrain_tokenizer=retrain_tokenizer,
        set_default_tokenizer=set_default_tokenizer,
        train_rows=train_rows,
        data_dir=data_dir,
    )

    train_ds, val_ds, test_ds = (
        Multi30kENDataset(
            ids=_pad_and_tensorize(
                rows=[bpe.encode(text=row, merges=merges) for row in rows],
                seq_len=seq_len,
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


def _mlm_collate(
    rows: list[Multi30kENDatasetChunk],
    vocab_length: int,
    seed: Optional[int] = None,
) -> Multi30kENBatch:
    """
    Mask a batch for MLM with BERT's 80/10/10 rule, never on a protected id.

    Args:
        rows: (index, ids) pairs from Multi30kENDataset.
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
