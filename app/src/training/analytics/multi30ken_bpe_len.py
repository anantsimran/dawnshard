"""
multi30ken_bpe_len.py
Print BPE length stats (mean, min, max, percentiles) for each Multi30k EN split.

Lengths are counted from the datasets get_multi30ken_mlm_dataloader builds, so they
use the same tokenizer as training: the default one if set, otherwise a freshly
trained one. The datasets are padded to MAX_SEQ_LEN, and a length is the number of
non-PAD_ID ids in a row. Rows longer than MAX_SEQ_LEN would be truncated, so the
script refuses to report stats if any row reaches it.
"""

import numpy as np
from loguru import logger
from training.common.constants import PAD_ID
from training.constants import TEST_SEED, VAL_SEED
from training.dataload.multi30ken import Multi30kENDataset, get_multi30ken_mlm_dataloader

MAX_SEQ_LEN = 512
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9)


def bpe_lengths(dataset: Multi30kENDataset) -> np.ndarray:
    """Return the number of non-PAD_ID ids in each row of a Multi30kENDataset.

    Raises:
        ValueError: If a row fills all MAX_SEQ_LEN positions, since it may have been
            truncated and its length would be undercounted.
    """
    lengths = np.array(
        object=[int((dataset[i][1] != PAD_ID).sum()) for i in range(len(dataset))]  # noqa: NAR001
    )
    if lengths.max() >= MAX_SEQ_LEN:
        raise ValueError(  # noqa: NAR001
            f"a row has {lengths.max()} ids, reaching MAX_SEQ_LEN={MAX_SEQ_LEN}; "
            "raise MAX_SEQ_LEN so no row is truncated"
        )
    return lengths


def log_stats(split: str, lengths: np.ndarray) -> None:
    """Log count, mean, min, max, and PERCENTILES of one split's BPE lengths."""
    logger.info(  # noqa: NAR001
        "{:<5} n={:>6}  mean={:6.2f}  min={:>3}  max={:>3}",
        split,
        len(lengths),  # noqa: NAR001
        lengths.mean(),
        lengths.min(),
        lengths.max(),
    )
    values = np.percentile(a=lengths, q=PERCENTILES)
    logger.info(  # noqa: NAR001
        "{:<5} {}",
        split,
        "  ".join(f"p{q:g}={v:.1f}" for q, v in zip(PERCENTILES, values)),  # noqa: NAR001
    )


def main() -> None:
    """Load every row of each Multi30k EN split and log its BPE length stats."""
    train_loader, val_loader, test_loader, merges, _ = get_multi30ken_mlm_dataloader(
        seq_len=MAX_SEQ_LEN,
        val_seed=VAL_SEED,
        test_seed=TEST_SEED,
        batch_size=1,
        train_num=None,
        val_num=None,
        test_num=None,
    )
    logger.info("Tokenizer has {} merges", len(merges))  # noqa: NAR001
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        dataset = loader.dataset
        assert isinstance(dataset, Multi30kENDataset)  # noqa: NAR001
        log_stats(split=split, lengths=bpe_lengths(dataset=dataset))


if __name__ == "__main__":
    main()
