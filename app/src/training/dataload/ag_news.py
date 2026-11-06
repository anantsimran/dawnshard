"""
ag_news.py
Load and inspect a sampled AG News dataset, tokenized with our from-scratch BPE.

The caller trains the tokenizer, from the train rows `get_ag_news_texts` returns, and
passes it in; the loader only samples, encodes, and pads. With a `Dao`, the raw CSVs,
the sample, and the encodings are stored. The sample is keyed
by the sizes and the seed, so changing either samples afresh.
"""

import csv
import io
import random
import urllib.request
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from training.dal.cache import cached
from training.dal.dao.dao import Dao
from training.preprocess.constants import PAD_ID
from training.preprocess.tokenizer import Tokenizer
from training.utils.timing import log_elapsed

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


@cached(version="1")
def _get_raw_rows(split: str) -> List[Tuple[str, int]]:
    """
    Download one AG News CSV and parse it in memory into (text, label) rows.

    Text is title + description and labels are 0-3. Bump this decorator's version
    when the URL, the parsing, or the backslash replacement changes.

    Args:
        split: "train" or "test", the upstream file name.
    """
    with urllib.request.urlopen(url=AG_NEWS_URL.format(split=split)) as response:
        text = response.read().decode(encoding="utf-8")
    return [
        # The CSV encodes line breaks inside a description as a literal backslash.
        (f"{title} {description}".replace("\\", " "), int(label) - 1)  # noqa: NAR001
        for label, title, description in csv.reader(io.StringIO(initial_value=text, newline=""))  # noqa: NAR001
    ]


@cached(version="1")
def get_ag_news_texts(
    n_train: int, n_eval: int, n_test: int, seed: int, cache: Optional[Dao] = None
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Return {"train", "eval", "test"} lists of (text, label) rows, sampled with `seed`.

    Train comes from train.csv; eval and test are disjoint samples of test.csv. The
    sizes and the seed are part of the cache key, so changing any of them samples
    afresh. Public because a caller trains its tokenizer from the train rows.

    Args:
        cache: Stores the raw CSVs and the sample; None downloads and samples every
            call.

    Raises:
        ValueError: If a sample is larger than its CSV.
    """
    rng = random.Random(x=seed)
    train_rows = _get_raw_rows(split="train", cache=cache)  # noqa: NAR001
    test_rows = _get_raw_rows(split="test", cache=cache)  # noqa: NAR001
    held_out = rng.sample(population=test_rows, k=n_eval + n_test)
    return {
        "train": rng.sample(population=train_rows, k=n_train),
        "eval": held_out[:n_eval],
        "test": held_out[n_eval:],
    }


def _encode_rows(encoded: List[List[int]], labels: List[int], max_len: int) -> AGNewsDataset:
    """Truncate each id list to `max_len` and right-pad it with PAD_ID into an AGNewsDataset."""
    ids = torch.full(size=(len(encoded), max_len), fill_value=PAD_ID, dtype=torch.long)  # noqa: NAR001
    for i, row in enumerate(encoded):  # noqa: NAR001
        kept = row[:max_len]
        ids[i, : len(kept)] = torch.tensor(data=kept, dtype=torch.long)  # noqa: NAR001
    return AGNewsDataset(ids=ids, labels=labels)


def get_ag_news_dataloader(
    tokenizer: Tokenizer,
    batch_size: int = 64,
    n_train: int = 100_000,
    n_eval: int = 2_000,
    n_test: int = 2_000,
    max_len: int = 128,
    seed: int = 0,
    cache: Optional[Dao] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, eval_loader, test_loader) over the sampled AG News splits.

    Train rows come from train.csv and eval/test rows from test.csv. The tokenizer
    comes from the caller, who should train it on the train rows from
    `get_ag_news_texts` with the same sizes and seed, so it never sees eval or test.
    Batches are (ids (B, max_len) long, labels (B,) long) with ids right-padded with
    PAD_ID; tokenizer.vocab includes it, so the embedding needs
    len(tokenizer.vocab) rows with padding_idx=PAD_ID.

    Args:
        tokenizer: A trained tokenizer, such as `get_tokenizer` returns.
        n_train: Train rows to sample.
        n_eval: Eval rows to sample.
        n_test: Test rows to sample.
        max_len: Rows are truncated or padded to this length.
        seed: Seeds the sample; with the sizes, it keys the cached rows.
        cache: Stores the raw CSVs, the sample, and the encodings; None stores
            nothing.

    Side effects:
        May download the CSVs and store them, the sample, and the encodings in
        `cache`.
    """
    splits = get_ag_news_texts(  # noqa: NAR001
        n_train=n_train, n_eval=n_eval, n_test=n_test, seed=seed, cache=cache
    )

    n_texts = sum(len(rows) for rows in splits.values())  # noqa: NAR001
    with log_elapsed(label=f"Encoded {n_texts} texts"):
        datasets = {
            split: _encode_rows(
                encoded=tokenizer.encode(texts=[text for text, _ in rows], cache=cache),
                labels=[label for _, label in rows],
                max_len=max_len,
            )
            for split, rows in splits.items()
        }

    # Data is already tokenized in memory, so worker processes would only add
    # startup and pickling cost.
    return (
        DataLoader(dataset=datasets["train"], batch_size=batch_size, shuffle=True),
        DataLoader(dataset=datasets["eval"], batch_size=batch_size, shuffle=False),
        DataLoader(dataset=datasets["test"], batch_size=batch_size, shuffle=False),
    )


def inspect_ag_news_dataset(loader: DataLoader, tokenizer: Tokenizer) -> None:
    """
    Print size, class balance, and one-batch tensor statistics.
    Call this before training to sanity-check the data pipeline.

    Args:
        loader: A DataLoader from get_ag_news_dataloader.
        tokenizer: The tokenizer returned alongside it.
    """
    ds = loader.dataset
    assert isinstance(ds, AGNewsDataset)  # noqa: NAR001
    counts = [ds.labels.count(c) for c in range(NUM_CLASSES)]  # noqa: NAR001
    logger.info("Total samples : {}", len(ds))  # noqa: NAR001
    logger.info("Classes       : {}", dict(zip(CLASSES, counts)))  # noqa: NAR001
    logger.info("Vocab size    : {} (incl. pad_id={})", len(tokenizer.vocab), PAD_ID)  # noqa: NAR001

    ids, labels = next(iter(loader))  # noqa: NAR001
    real = ids != PAD_ID
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
        tokenizer.decode(ids=[ids[0][real[0]].tolist()])[0],
    )
