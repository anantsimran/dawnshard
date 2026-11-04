"""
data_loading.py
Checks and measurements of the streaming token pipeline, on Multi30k English captions
tokenized with the repo's BPE: who reads which sample, how well a shuffle buffer
mixes, whether resume is bitwise, and how much time training waits for data.

Each study writes a CSV under experiments/t6-data-loading/:

- `audit`: every sample index served by 4 ranks x 4 workers over one epoch, from the
  real `rank_slice` and `worker_steps`; duplicates and gaps against the epoch's
  global order. Then the same for an unsharded IterableDataset, where every rank and
  worker iterates the whole stream.
- `shuffle`: lag-1 same-shard rate of a shard-ordered stream after `shuffle_buffer`
  at several sizes, against a global permutation.
- `resume`: hashes of global batches from an uninterrupted run, from a run stopped at
  step 137 and resumed from `StatefulLoader.state_dict()`, and from the same resume
  on 2 ranks instead of 4.
- `stall`: a tiny Qwen3 trained through `fit` with `measure_data_stall`, fed either
  by a map-style dataset that runs BPE on raw text in `__getitem__` (v0) or by
  pre-tokenized shards (v1).

    uv run python app/src/training/experiments/data_loading.py all
"""

import argparse
import csv
import hashlib
from dataclasses import replace
from itertools import islice
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from training.common import bpe
from training.common.constants import EOS_ID
from training.constants import REPO_ROOT, TEST_SEED, VAL_SEED
from training.dataload.multi30k import read_multi30k_split
from training.dataload.shards import write_manifest, write_shards
from training.dataload.streaming import (
    StatefulLoader,
    StreamingTokens,
    TokenSource,
    global_order,
    rank_slice,
    shuffle_buffer,
    worker_steps,
)
from training.dataload.tokens import lm_collate
from training.metrics.loss_only_metrics import accumulate_metrics, calculate_metrics, reduce_metrics
from training.model.qwen3 import QWEN3_TINY, Qwen3
from training.train.model import EpochSpec, Metrics, TrainState
from training.train.train_loop import fit

RESULTS_DIR = REPO_ROOT / "experiments" / "t6-data-loading"
SHARDS_DIR = REPO_ROOT / "app" / "datasets" / "cache" / "multi30k_shards"
SEQ_LEN = 64
GLOBAL_BATCH = 32
SEED = 0
BPE_MERGES = 2000


def write_csv(name: str, rows: list[dict]) -> None:
    """Overwrite `RESULTS_DIR/<name>.csv` with `rows`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(file=RESULTS_DIR / f"{name}.csv", mode="w", newline="") as handle:
        writer = csv.DictWriter(f=handle, fieldnames=list(rows[0]))  # noqa: NAR001
        writer.writeheader()
        writer.writerows(rowdicts=rows)


def build_shards() -> list[bpe.Phrase]:
    """Tokenize the train captions into ~20K-token shards and return the merges.

    Side effects:
        Writes shards and a manifest under SHARDS_DIR.
    """
    rows = read_multi30k_split(split="train", lang="en")
    merges, _ = bpe.train(text="\n".join(rows[:5000]), num_merges=BPE_MERGES)  # noqa: NAR001
    write_shards(
        documents=bpe.encode_many(texts=rows, merges=merges),
        eos_id=EOS_ID,
        out_dir=SHARDS_DIR,
        prefix="train",
        tokens_per_shard=20_000,
    )
    write_manifest(out_dir=SHARDS_DIR)
    return merges


def served(world: int, workers: int, num_samples: int, sharded: bool) -> list[int]:
    """Every sample index the ranks and workers serve in one epoch."""
    order = global_order(num_samples=num_samples, seed=SEED, epoch=0)
    steps = num_samples // GLOBAL_BATCH
    indices: list[int] = []
    for rank in range(world):  # noqa: NAR001
        for worker in range(workers):  # noqa: NAR001
            if not sharded:
                indices.extend(order[: steps * GLOBAL_BATCH].tolist())  # noqa: NAR001
                continue
            for step in worker_steps(
                start_step=0, num_steps=steps, worker=worker, num_workers=workers
            ):
                batch = order[step * GLOBAL_BATCH : (step + 1) * GLOBAL_BATCH]
                indices.extend(rank_slice(global_batch=batch, rank=rank, world_size=world).tolist())  # noqa: NAR001
    return indices


def audit(num_samples: int) -> list[dict]:
    """Duplicates and gaps for the sharded loader and for an unsharded one, 4 x 4."""
    expected = set(  # noqa: NAR001
        global_order(num_samples=num_samples, seed=SEED, epoch=0)[
            : (num_samples // GLOBAL_BATCH) * GLOBAL_BATCH
        ].tolist()
    )  # noqa: NAR001
    rows = []
    for sharded in (True, False):
        seen = served(world=4, workers=4, num_samples=num_samples, sharded=sharded)
        counts = np.bincount(np.asarray(seen), minlength=num_samples)  # noqa: NAR001
        rows.append(  # noqa: NAR001
            {
                "loader": "rank x worker sharded" if sharded else "unsharded IterableDataset",
                "served": len(seen),  # noqa: NAR001
                "distinct": int((counts > 0).sum()),  # noqa: NAR001
                "duplicates": int((counts[counts > 1] - 1).sum()),  # noqa: NAR001
                "gaps": len(expected - set(seen)),  # noqa: NAR001
                "max_copies": int(counts.max()),  # noqa: NAR001
            }
        )  # noqa: NAR001
    return rows


def shuffle_quality(source: TokenSource) -> list[dict]:
    """Share of consecutive samples from the same shard, by shuffle method."""
    num_samples = (source.num_tokens - 1) // SEQ_LEN
    shard_of = (
        np.searchsorted(a=source.offsets, v=np.arange(num_samples) * SEQ_LEN, side="right") - 1  # noqa: NAR001
    )  # noqa: NAR001

    def same_shard_rate(order: list[int]) -> float:
        """Lag-1 rate of neighbours from one shard."""
        shards = shard_of[np.asarray(order)]  # noqa: NAR001
        return float((shards[1:] == shards[:-1]).mean())  # noqa: NAR001

    rows = [
        {
            "method": "no shuffle",
            "buffer": 0,
            "same_shard_rate": same_shard_rate(order=list(range(num_samples))),  # noqa: NAR001
        }
    ]  # noqa: NAR001
    for size in (10, 100, 1000):
        order = list(shuffle_buffer(items=range(num_samples), size=size, seed=SEED))  # noqa: NAR001
        assert sorted(order) == list(range(num_samples))  # noqa: NAR001
        rows.append(  # noqa: NAR001
            {
                "method": "shuffle buffer",
                "buffer": size,
                "same_shard_rate": same_shard_rate(order=order),
            }
        )  # noqa: NAR001
    rows.append(  # noqa: NAR001
        {
            "method": "global permutation",
            "buffer": num_samples,
            "same_shard_rate": same_shard_rate(
                order=global_order(num_samples=num_samples, seed=SEED, epoch=0).tolist()
            ),
        }
    )  # noqa: NAR001
    return rows


def global_batch_hashes(
    source: TokenSource, world: int, state: dict | None, steps: int
) -> list[str]:
    """Hash each global batch, rebuilt by concatenating every rank's loader output.

    Args:
        state: A `StatefulLoader.state_dict()` to resume from, or None to start fresh.
        steps: Global batches to hash.
    """
    loaders = []
    for rank in range(world):  # noqa: NAR001
        loader = StatefulLoader(
            dataset=StreamingTokens(
                source=source,
                seq_len=SEQ_LEN,
                global_batch=GLOBAL_BATCH,
                seed=SEED,
                rank=rank,
                world_size=world,
            ),
            num_workers=2,
        )
        if state is not None:
            loader.load_state_dict(state_dict=state)
        loaders.append(iter(loader))  # noqa: NAR001
    hashes = []
    for _ in range(steps):  # noqa: NAR001
        rows = torch.cat(tensors=[next(loader)[0] for loader in loaders])  # noqa: NAR001
        hashes.append(hashlib.sha256(rows.numpy().tobytes()).hexdigest()[:16])  # noqa: NAR001
    return hashes


def resume(source: TokenSource, kill_step: int = 137, after: int = 20) -> list[dict]:
    """Compare uninterrupted, resumed-on-4, and resumed-on-2 batch hashes after `kill_step`."""
    reference = global_batch_hashes(source=source, world=4, state=None, steps=kill_step + after)
    loader = StatefulLoader(
        dataset=StreamingTokens(
            source=source,
            seq_len=SEQ_LEN,
            global_batch=GLOBAL_BATCH,
            seed=SEED,
            rank=0,
            world_size=4,
        )
    )
    for _ in zip(  # noqa: NAR001
        range(kill_step),  # noqa: NAR001
        loader,  # noqa: NAR001
    ):  # consume, as training would, then "crash"  # noqa: NAR001
        pass
    state = loader.state_dict()
    on_4 = global_batch_hashes(source=source, world=4, state=state, steps=after)
    on_2 = global_batch_hashes(source=source, world=2, state=state, steps=after)
    return [
        {
            "step": kill_step + i,
            "uninterrupted": reference[kill_step + i],
            "resumed_on_4": on_4[i],
            "resumed_on_2": on_2[i],
            "match": reference[kill_step + i] == on_4[i] == on_2[i],
        }
        for i in range(after)  # noqa: NAR001
    ] + [
        {
            "step": "state",
            "uninterrupted": "",
            "resumed_on_4": str(state),  # noqa: NAR001
            "resumed_on_2": "",
            "match": "",
        }
    ]  # noqa: NAR001


class RawTextWindows(Dataset[torch.Tensor]):
    """v0: tokenize raw captions in `__getitem__`, a few per window, every time."""

    def __init__(self, rows: list[str], merges: list[bpe.Phrase]) -> None:
        """Keep the raw text; nothing is tokenized up front."""
        self.rows, self.merges = rows, merges

    def __len__(self) -> int:
        """One window per 4 captions."""
        return len(self.rows) // 4  # noqa: NAR001

    def __getitem__(self, index: int) -> torch.Tensor:
        """BPE-encode 4 captions with EOS between them and cut one window."""
        ids: list[int] = []
        for row in self.rows[4 * index : 4 * index + 4]:
            ids += bpe.encode(text=row, merges=self.merges) + [EOS_ID]
        ids = (ids * 2)[: SEQ_LEN + 1]
        return torch.tensor(data=ids, dtype=torch.int64)


def stall(source: TokenSource, merges: list[bpe.Phrase], steps: int = 60) -> list[dict]:
    """Train a tiny Qwen3 for `steps` steps per pipeline and record the data stall."""
    rows = []
    raw = DataLoader(
        dataset=RawTextWindows(rows=read_multi30k_split(split="train", lang="en"), merges=merges),
        batch_size=GLOBAL_BATCH,
        shuffle=True,
        collate_fn=lm_collate,
    )
    stream = StatefulLoader(
        dataset=StreamingTokens(
            source=source,
            seq_len=SEQ_LEN,
            global_batch=GLOBAL_BATCH,
            seed=SEED,
            rank=0,
            world_size=1,
        )
    )
    for name, loader in (("v0 BPE in __getitem__", raw), ("v1 pre-tokenized shards", stream)):
        torch.manual_seed(seed=SEED)
        model = Qwen3(config=replace(QWEN3_TINY, vocab_size=bpe.FIRST_MERGE_ID + BPE_MERGES))  # noqa: NAR001
        state = TrainState(
            model=model,
            optimizer=torch.optim.AdamW(params=model.parameters()),
            criterion=torch.nn.CrossEntropyLoss(),
        )
        spec = EpochSpec(
            device=torch.device("cpu"),  # noqa: NAR001
            metrics_type=Metrics,
            calculate_metrics=calculate_metrics,
            accumulate_metrics=accumulate_metrics,
            reduce_metrics=reduce_metrics,
            measure_data_stall=True,
            flops_per_target=1.0,
        )  # noqa: NAR001
        # flops_per_target only switches on targets_per_s; MFU is not computed (no peak).
        record = fit(
            state=state,
            epoch_spec=spec,
            train_loader=cast(DataLoader, islice(loader, steps)),  # noqa: NAR001
            val_loader=cast(DataLoader, []),  # noqa: NAR001
            num_epochs=1,
            val_epoch_list=[],
            train_seed=SEED,
            val_seed=VAL_SEED,
            test_seed=TEST_SEED,
            history_path=RESULTS_DIR / f"stall_{name.split()[0]}_history.json",
        )[0]
        rows.append(  # noqa: NAR001
            {
                "pipeline": name,
                "steps": steps,
                "data_stall_fraction": round(record.data_stall_fraction or 0.0, 4),  # noqa: NAR001
                "tokens_per_s": round(record.targets_per_s or 0.0),  # noqa: NAR001
            }
        )  # noqa: NAR001
    return rows


def main() -> None:
    """Run the studies named on the command line and write their CSVs."""
    parser = argparse.ArgumentParser(description="Audit and measure the streaming token pipeline.")
    parser.add_argument("study", choices=["audit", "shuffle", "resume", "stall", "all"])  # noqa: NAR001
    args = parser.parse_args()
    merges = build_shards()
    source = TokenSource(root=str(SHARDS_DIR))  # noqa: NAR001
    num_samples = (source.num_tokens - 1) // SEQ_LEN
    studies = {
        "audit": lambda: audit(num_samples=num_samples),
        "shuffle": lambda: shuffle_quality(source=source),
        "resume": lambda: resume(source=source),
        "stall": lambda: stall(source=source, merges=merges),
    }
    for name, study in studies.items():
        if args.study in (name, "all"):
            rows = study()
            write_csv(name=name, rows=rows)
            print(name, *rows[:6], sep="\n")  # noqa: NAR001


if __name__ == "__main__":
    main()
