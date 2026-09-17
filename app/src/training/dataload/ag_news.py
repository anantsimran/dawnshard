"""
ag_news.py
Load and inspect a sampled AG News dataset, tokenized with our from-scratch BPE.
"""

import csv
import json
import random
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from training.common import bpe
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from training.dataload.constants import DATASETS_CACHE_DIR

AG_NEWS_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/{split}.csv"
)
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = len(CLASSES)  # noqa: NAR001


class AGNewsDataset(Dataset):
    """Pre-tokenized (ids, label) pairs; ids is (N, max_len), truncated and right-padded."""

    def __init__(self, ids: torch.Tensor, labels: List[int]) -> None:
        """Store pre-tokenized ids and labels; see class docstring for shapes."""
        self.ids = ids
        self.labels = labels

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.labels)  # noqa: NAR001

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return the (ids, label) pair at `index`."""
        return self.ids[index], self.labels[index]


def _download_csv(split: str, data_dir: Path) -> Path:
    """Download and cache the AG News CSV for `split`, skipping if already present."""
    path = data_dir / "ag_news" / f"{split}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url=AG_NEWS_URL.format(split=split), filename=path)
    return path


def _read_raw_rows(path: Path) -> List[Tuple[str, int]]:
    """All (text, label) rows of a raw CSV; text is title + description, labels are 0-3."""
    with open(file=path, newline="", encoding="utf-8") as f:
        return [
            # The CSV encodes line breaks inside a description as a literal backslash.
            (f"{title} {description}".replace("\\", " "), int(label) - 1)  # noqa: NAR001
            for label, title, description in csv.reader(f)  # noqa: NAR001
        ]


def _write_split(rows: List[Tuple[str, int]], path: Path) -> None:
    """Write `rows` as a (label, text) CSV to `path`, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(file=path, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([(label, text) for text, label in rows])  # noqa: NAR001


def _read_split(path: Path) -> List[Tuple[str, int]]:
    """Read (text, label) rows from a CSV written by `_write_split`."""
    with open(file=path, newline="", encoding="utf-8") as f:
        return [(text, int(label)) for label, text in csv.reader(f)]  # noqa: NAR001


def _load_splits(
    n_train: int, n_eval: int, n_test: int, seed: int, data_dir: Path, refresh_cache: bool
) -> Tuple[Dict[str, List[Tuple[str, int]]], bool]:
    """
    Return ({"train", "eval", "test"} rows, whether they came from the cache),
    sampling from the raw CSVs on a cache miss.

    Train comes from train.csv; eval and test are disjoint samples of test.csv.
    The cache only checks that the files exist, so pass refresh_cache=True after
    changing any n_* or seed.
    """
    paths = {
        split: data_dir / "ag_news" / "sampled" / split / "rows.csv"
        for split in ("train", "eval", "test")
    }
    if not refresh_cache and all(path.exists() for path in paths.values()):  # noqa: NAR001
        logger.info("Loading sampled AG News splits from {}", paths["train"].parent.parent)  # noqa: NAR001
        return {split: _read_split(path=path) for split, path in paths.items()}, True

    rng = random.Random(x=seed)
    train_rows = _read_raw_rows(path=_download_csv(split="train", data_dir=data_dir))
    test_rows = _read_raw_rows(path=_download_csv(split="test", data_dir=data_dir))
    held_out = rng.sample(population=test_rows, k=n_eval + n_test)
    splits = {
        "train": rng.sample(population=train_rows, k=n_train),
        "eval": held_out[:n_eval],
        "test": held_out[n_eval:],
    }
    for split, rows in splits.items():
        _write_split(rows=rows, path=paths[split])
    return splits, False


def _save_merges(merges: List[bpe.Phrase], num_merges: int, path: Path) -> None:
    """Store each merge as its (first_id, second_id) pair, which is all encode needs."""
    ids = {bpe.Phrase(encoding=b): b for b in range(256)}  # noqa: NAR001
    pairs = []
    for rank, merge in enumerate(merges):  # noqa: NAR001
        assert merge.first is not None and merge.second is not None
        pairs.append([ids[merge.first], ids[merge.second]])  # noqa: NAR001
        ids[merge] = bpe.FIRST_MERGE_ID + rank
    # first_merge_id lets _load_merges reject caches written with a different id layout.
    cached = {
        "num_merges": num_merges,
        "first_merge_id": bpe.FIRST_MERGE_ID,
        "merges": pairs,
    }
    path.write_text(data=json.dumps(obj=cached), encoding="utf-8")


def _load_merges(path: Path, num_merges: int) -> Optional[Tuple[List[bpe.Phrase], Dict[int, str]]]:
    """Rebuild (merges, vocab) as bpe.train returns them; None if missing or stale."""
    if not path.exists():
        return None
    cached = json.loads(s=path.read_text(encoding="utf-8"))
    if (
        cached["num_merges"] != num_merges or cached.get("first_merge_id") != bpe.FIRST_MERGE_ID  # noqa: NAR001
    ):
        return None
    return bpe.tokenizer_from_pairs(pairs=cached["merges"])


def _encode_rows(
    rows: List[Tuple[str, int]], merges: List[bpe.Phrase], max_len: int
) -> AGNewsDataset:
    """Tokenize `rows` into a fixed-length AGNewsDataset, right-padded with bpe.PAD_ID."""
    ids = torch.full(size=(len(rows), max_len), fill_value=bpe.PAD_ID, dtype=torch.long)  # noqa: NAR001
    for i, (text, _) in enumerate(rows):  # noqa: NAR001
        row = bpe.encode(text=text, merges=merges)[:max_len]
        ids[i, : len(row)] = torch.tensor(data=row, dtype=torch.long)  # noqa: NAR001
    return AGNewsDataset(ids=ids, labels=[label for _, label in rows])


def get_ag_news_dataloader(
    batch_size: int = 64,
    n_train: int = 20_000,
    n_eval: int = 2_000,
    n_test: int = 2_000,
    num_merges: int = 2000,
    max_len: int = 128,
    seed: int = 0,
    data_dir: Path = DATASETS_CACHE_DIR,
    refresh_cache: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[bpe.Phrase], Dict[int, str]]:
    """
    Load the sampled AG News splits, train BPE on the train texts, and return
    (train_loader, eval_loader, test_loader, merges, vocab).

    Train rows come from train.csv and eval/test rows from test.csv, so neither is
    seen by the tokenizer. Sampled rows, and the BPE merges trained on them, are
    cached under data_dir; pass refresh_cache=True to resample (required after
    changing any n_* or seed). Merges are retrained whenever the rows are
    resampled or num_merges changes.
    Batches are (ids (B, max_len) long, labels (B,) long) with ids right-padded using
    bpe.PAD_ID; vocab includes it, so the embedding needs len(vocab) rows with
    padding_idx=bpe.PAD_ID.
    """
    splits, from_cache = _load_splits(
        n_train=n_train,
        n_eval=n_eval,
        n_test=n_test,
        seed=seed,
        data_dir=data_dir,
        refresh_cache=refresh_cache,
    )

    merges, vocab = get_bpe(
        num_merges=num_merges, data_dir=data_dir, splits=splits, from_cache=from_cache
    )

    start = time.perf_counter()
    datasets = {
        split: _encode_rows(rows=rows, merges=merges, max_len=max_len)
        for split, rows in splits.items()
    }
    n_texts = sum(len(ds) for ds in datasets.values())  # noqa: NAR001
    logger.info("encoded {} texts in {:.1f}s", n_texts, time.perf_counter() - start)  # noqa: NAR001

    # Data is already tokenized in memory, so worker processes would only add
    # startup and pickling cost.
    return (
        DataLoader(dataset=datasets["train"], batch_size=batch_size, shuffle=True),
        DataLoader(dataset=datasets["eval"], batch_size=batch_size, shuffle=False),
        DataLoader(dataset=datasets["test"], batch_size=batch_size, shuffle=False),
        merges,
        vocab,
    )


def get_bpe(
    num_merges: int,
    data_dir: Path,
    splits: Dict[str, List[Tuple[str, int]]],
    from_cache: bool,
) -> Tuple[List[bpe.Phrase], Dict[int, str]]:
    """
    Return (merges, vocab) trained on the train split, loading cached merges if valid.

    Side effect: on a cache miss, trains BPE and writes the merges to
    data_dir/ag_news/sampled/train/bpe.json.

    Args:
        from_cache: Whether `splits` came from the cache; if not, cached merges are
            ignored because they were trained on different rows.
    """
    # Merges are only valid for the rows they were trained on, so a resample
    # always retrains.
    bpe_path = data_dir / "ag_news" / "sampled" / "train" / "bpe.json"
    cached_bpe = _load_merges(path=bpe_path, num_merges=num_merges) if from_cache else None
    if cached_bpe is not None:
        merges, vocab = cached_bpe
        logger.info("Loaded {} BPE merges from {}", len(merges), bpe_path)  # noqa: NAR001
    else:
        start = time.perf_counter()
        merges, vocab = bpe.train(
            text="\n".join(text for text, _ in splits["train"]),  # noqa: NAR001
            num_merges=num_merges,
        )
        logger.info("BPE trained {} merges in {:.1f}s", len(merges), time.perf_counter() - start)  # noqa: NAR001
        _save_merges(merges=merges, num_merges=num_merges, path=bpe_path)
    return merges, vocab


def inspect_ag_news_dataset(loader: DataLoader, vocab: Dict[int, str]) -> None:
    """
    Print size, class balance, and one-batch tensor statistics.
    Call this before training to sanity-check the data pipeline.

    Args:
        loader: A DataLoader from get_ag_news_dataloader.
        vocab: The vocab returned alongside it.
    """
    ds = loader.dataset
    assert isinstance(ds, AGNewsDataset)  # noqa: NAR001
    counts = [ds.labels.count(c) for c in range(NUM_CLASSES)]  # noqa: NAR001
    logger.info("Total samples : {}", len(ds))  # noqa: NAR001
    logger.info("Classes       : {}", dict(zip(CLASSES, counts)))  # noqa: NAR001
    logger.info("Vocab size    : {} (incl. pad_id={})", len(vocab), bpe.PAD_ID)  # noqa: NAR001

    ids, labels = next(iter(loader))  # noqa: NAR001
    real = ids != bpe.PAD_ID
    logger.info("Batch shape   : {}  (B, L)", ids.shape)  # noqa: NAR001
    logger.info("Label sample  : {}", labels[:8].tolist())  # noqa: NAR001
    lengths = real.sum(dim=1)
    logger.info(  # noqa: NAR001
        "Tokens / row  : mean {:.1f}, max {}",
        lengths.float().mean(),
        lengths.max(),
    )
    logger.info(  # noqa: NAR001
        "Decoded row 0 : {}",
        bpe.decode(ids=ids[0][real[0]].tolist(), vocab=vocab),
    )
