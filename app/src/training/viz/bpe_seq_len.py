"""
bpe_seq_len.py
Plot mean BPE tokens per sample vs num_merges on the full AG News train split.

BPE is trained once with MAX_MERGES. Training is deterministic, so its first k
merges are exactly what build_tokenizer(num_merges=k) learns and one run covers every k.
The merges live in a file under TOKENIZERS_PATH and the downloaded split in the cache
database, so later runs skip training and the download.
"""

import json
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from training.constants import HISTORY_PATH
from training.dal.dao.sqlite_dao import SqliteDao
from training.dataload.ag_news import _get_raw_rows
from training.dataload.constants import CACHE_PATH, TOKENIZERS_PATH
from training.preprocess.bpe import gpt4_preprocessor
from training.preprocess.bpe.tokenizer import BpeTokenizer
from training.preprocess.constants import FIRST_MERGE_ID
from training.preprocess.tokenizer import get_tokenizer
from training.utils.timing import log_elapsed

MIN_MERGES = 50
MAX_MERGES = 20_000
TOKENIZER_NAME = "ag_news_full_train"
PLOT_PATH = HISTORY_PATH / "bpe_seq_len.png"
DATA_PATH = HISTORY_PATH / "bpe_seq_len.json"


def _load_or_train_pairs(texts: List[str]) -> List[Tuple[int, int]]:
    """Return MAX_MERGES merges as (first_id, second_id) pairs, training if none are saved.

    Raises:
        TypeError: If `get_tokenizer` returns a tokenizer that is not BPE.
    """
    tokenizer = get_tokenizer(
        merges_path=TOKENIZERS_PATH / f"{TOKENIZER_NAME}-{MAX_MERGES}.json",
        train_texts=texts,
        num_merges=MAX_MERGES,
    )
    if not isinstance(tokenizer, BpeTokenizer):  # noqa: NAR001
        raise TypeError(f"expected a BpeTokenizer, got {type(tokenizer).__name__}")  # noqa: NAR001
    return tokenizer.state()


def _byte_ids(text: str) -> List[int]:
    """Return the preprocessed text's bytes as ids, which is what no merges encode to."""
    return list(b"".join(gpt4_preprocessor.preprocess(text=text)))  # noqa: NAR001


def merge_counts(texts: List[str], pairs: Sequence[Sequence[int]]) -> np.ndarray:
    """
    counts[r] = how many times merge r fires when each text is encoded on its own,
    so after k merges the texts hold sum(bytes) - counts[:k].sum() tokens.

    Works chunk by chunk, as encode_chunks does, so no merge crosses a chunk.
    Within a chunk, repeatedly applies the lowest-rank merge present, left to
    right. A merge can only create pairs containing its own output token, and every
    merge using that token ranks higher, so this is encode_chunks's order minus
    merges that never occur, giving the same tokens in one pass instead of one per
    merge.
    """
    rank_of = {(first, second): rank for rank, (first, second) in enumerate(pairs)}  # noqa: NAR001
    no_merge = len(pairs)  # noqa: NAR001
    counts = np.zeros(shape=len(pairs), dtype=np.int64)  # noqa: NAR001
    chunks = (chunk for text in texts for chunk in gpt4_preprocessor.preprocess(text=text))
    for chunk in chunks:
        ids = list(chunk)  # noqa: NAR001
        while len(ids) >= 2:  # noqa: NAR001
            rank = min(rank_of.get(pair, no_merge) for pair in zip(ids, ids[1:]))  # noqa: NAR001
            if rank == no_merge:
                break
            first, second = pairs[rank]
            merged: List[int] = []
            i = 0
            while i < len(ids):  # noqa: NAR001
                if i + 1 < len(ids) and ids[i] == first and ids[i + 1] == second:  # noqa: NAR001
                    merged.append(FIRST_MERGE_ID + rank)  # noqa: NAR001
                    i += 2
                else:
                    merged.append(ids[i])  # noqa: NAR001
                    i += 1
            counts[rank] += len(ids) - len(merged)  # noqa: NAR001
            ids = merged
    return counts


def main() -> None:
    """Compute and plot mean BPE sequence length vs num_merges, writing data and plot files.

    Downloads the AG News train split, trains or loads the BPE merges, and writes
    results to `DATA_PATH` and `PLOT_PATH`. The split lives in the cache database
    and the merges in a file, so later runs skip the download and the training.
    """
    with SqliteDao(path=CACHE_PATH) as cache:
        rows = _get_raw_rows(split="train", cache=cache)  # noqa: NAR001
        texts = [text for text, _ in rows]
    pairs = _load_or_train_pairs(texts=texts)

    with log_elapsed(label=f"Encoded {len(texts)} texts"):  # noqa: NAR001
        counts = merge_counts(texts=texts, pairs=pairs)

    total_bytes = sum(len(_byte_ids(text=text)) for text in texts)  # noqa: NAR001
    # mean_len[k - 1] is the mean tokens per text after k merges.
    mean_len = (total_bytes - np.cumsum(a=counts)) / len(texts)  # noqa: NAR001
    num_merges = list(range(MIN_MERGES, len(pairs) + 1))  # noqa: NAR001
    seq_len = mean_len[MIN_MERGES - 1 :]
    for k in (MIN_MERGES, 100, 500, 1000, 2000, 5000, 10_000, len(pairs)):  # noqa: NAR001
        logger.info("num_merges {:>6}: mean seq len {:.1f}", k, mean_len[k - 1])  # noqa: NAR001

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(
        data=json.dumps(
            obj={
                "num_samples": len(texts),  # noqa: NAR001
                "total_bytes": total_bytes,
                "num_merges": num_merges,
                "mean_seq_len": seq_len.tolist(),
            }
        ),
        encoding="utf-8",
    )
    logger.info("Saved data to {}", DATA_PATH)  # noqa: NAR001

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(num_merges, seq_len)  # noqa: NAR001
    axis.set_xscale(value="log")
    axis.set_xlabel(xlabel="num_merges")
    axis.set_ylabel(ylabel="mean tokens per sample")
    axis.set_title(label=f"BPE sequence length, AG News train ({len(texts):,} samples)")  # noqa: NAR001
    axis.grid(visible=True, which="both")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname=PLOT_PATH, dpi=150)
    logger.info("Saved plot to {}", PLOT_PATH)  # noqa: NAR001


if __name__ == "__main__":
    main()
