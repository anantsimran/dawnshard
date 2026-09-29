"""
bpe_seq_len.py
Plot mean BPE tokens per sample vs num_merges on the full AG News train split.

BPE is trained once with MAX_MERGES. Training is deterministic, so its first k
merges are exactly what bpe.train(num_merges=k) learns and one run covers every k.
The merges are cached next to the dataset, so later runs skip training.
"""

import json
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from training.common import bpe
from training.constants import HISTORY_PATH
from training.dataload.ag_news import _download_csv, _load_merges, _read_raw_rows, _save_merges
from training.dataload.constants import DATASETS_CACHE_DIR

MIN_MERGES = 50
MAX_MERGES = 20_000
BPE_CACHE_PATH = DATASETS_CACHE_DIR / "ag_news" / "full_train" / "bpe.json"
PLOT_PATH = HISTORY_PATH / "bpe_seq_len.png"
DATA_PATH = HISTORY_PATH / "bpe_seq_len.json"


def _load_or_train_pairs(texts: List[str]) -> List[List[int]]:
    """Return MAX_MERGES merges as (first_id, second_id) pairs, training on a cache miss."""
    if _load_merges(path=BPE_CACHE_PATH, num_merges=MAX_MERGES) is None:
        start = time.perf_counter()
        merges, _ = bpe.train(text="\n".join(texts), num_merges=MAX_MERGES)  # noqa: NAR001
        logger.info(  # noqa: NAR001
            "BPE trained {} merges in {:.1f}s",
            len(merges),  # noqa: NAR001
            time.perf_counter() - start,
        )
        BPE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _save_merges(merges=merges, num_merges=MAX_MERGES, path=BPE_CACHE_PATH)
    return json.loads(s=BPE_CACHE_PATH.read_text(encoding="utf-8"))["merges"]


def merge_counts(texts: List[str], pairs: List[List[int]]) -> np.ndarray:
    """
    counts[r] = how many times merge r fires when each text is encoded on its own,
    so after k merges the texts hold sum(bytes) - counts[:k].sum() tokens.

    Repeatedly applies the lowest-rank merge present, left to right. A merge can
    only create pairs containing its own output token, and every merge using that
    token ranks higher, so this is bpe.encode's order minus merges that never
    occur, giving the same tokens in one pass instead of one per merge.
    """
    rank_of = {(first, second): rank for rank, (first, second) in enumerate(pairs)}  # noqa: NAR001
    no_merge = len(pairs)  # noqa: NAR001
    counts = np.zeros(shape=len(pairs), dtype=np.int64)  # noqa: NAR001
    for text in texts:
        ids = bpe.encode(text=text, merges=[])
        while len(ids) >= 2:  # noqa: NAR001
            rank = min(rank_of.get(pair, no_merge) for pair in zip(ids, ids[1:]))  # noqa: NAR001
            if rank == no_merge:
                break
            first, second = pairs[rank]
            merged: List[int] = []
            i = 0
            while i < len(ids):  # noqa: NAR001
                if i + 1 < len(ids) and ids[i] == first and ids[i + 1] == second:  # noqa: NAR001
                    merged.append(bpe.FIRST_MERGE_ID + rank)  # noqa: NAR001
                    i += 2
                else:
                    merged.append(ids[i])  # noqa: NAR001
                    i += 1
            counts[rank] += len(ids) - len(merged)  # noqa: NAR001
            ids = merged
    return counts


def main() -> None:
    """Compute and plot mean BPE sequence length vs num_merges, writing data and plot files.

    Downloads/caches the AG News train split, trains or loads the cached BPE
    merges, and writes results to `DATA_PATH` and `PLOT_PATH`.
    """
    rows = _read_raw_rows(path=_download_csv(split="train", data_dir=DATASETS_CACHE_DIR))
    texts = [text for text, _ in rows]
    pairs = _load_or_train_pairs(texts=texts)

    start = time.perf_counter()
    counts = merge_counts(texts=texts, pairs=pairs)
    logger.info("Encoded {} texts in {:.1f}s", len(texts), time.perf_counter() - start)  # noqa: NAR001

    total_bytes = sum(len(bpe.encode(text=text, merges=[])) for text in texts)  # noqa: NAR001
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
