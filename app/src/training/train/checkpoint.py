"""checkpoint.py -- what a checkpoint holds, and sharded, atomic, async saves of it.

A resume is only exact if the checkpoint holds everything the next step reads: model
and optimizer state, the scheduler, the loader's position, every RNG of every rank,
and the step. `capture` and `restore` define that list once; `save_state` and `load`
(one file, one process) and `Checkpointer` (sharded, any world size) both use them.

```
Checkpointer.save(state, step)
    ├─ get_state_dict(model, optimizer)    sharded DTensors stay sharded
    ├─ + scheduler, loader, step, config   replicated: written once
    ├─ + rank{r}/rng                       per rank: each rank writes its own
    ├─ dcp.save / dcp.async_save           every rank writes its shards in parallel
    └─ barrier → rank 0 writes COMMITTED    the marker last: a half-written step is
                                             never "latest"
Checkpointer.load_latest(state)
    └─ newest step_* with COMMITTED → dcp.load (reshards to this world size) → restore
```

With `async_save=True`, `dcp.async_save` copies the state to CPU before returning and
writes it on a background thread, so training stalls only for the copy; the copy must
finish before the next optimizer step changes the weights, which it does because it
happens inside the call. The commit marker is written at the next `save` or at
`wait`, from the main thread, so no collective ever runs off the main thread.

The Young/Daly interval `sqrt(2 x delta x MTBF)` balances the blocking cost of a
checkpoint (delta) against the work lost to a failure (half an interval on average).
"""

import math
import pickle
import random
import shutil
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import async_save, save
from training.train.model import TrainState
from training.utils.git import get_git_commit

COMMIT_MARKER = "COMMITTED"


def _rank() -> int:
    """This process's rank, 0 without a process group."""
    return dist.get_rank() if dist.is_initialized() else 0


def rng_state() -> bytes:
    """Every RNG this process draws from, pickled: torch CPU and CUDA, Python, NumPy."""
    state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    return pickle.dumps(obj=state)


def set_rng_state(blob: bytes) -> None:
    """Restore what `rng_state` captured.

    Side effects:
        Overwrites the process's global RNG states.
    """
    state = pickle.loads(blob)  # noqa: NAR001
    torch.set_rng_state(new_state=state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(new_states=state["cuda"])
    random.setstate(state["python"])  # noqa: NAR001
    np.random.set_state(state["numpy"])  # noqa: NAR001


def capture(state: TrainState, step: int) -> dict[str, Any]:
    """Everything besides model and optimizer that a bitwise resume needs."""
    payload: dict[str, Any] = {"step": step, f"rank{_rank()}/rng": rng_state()}
    if state.scheduler is not None:
        payload["scheduler"] = state.scheduler.state_dict()
    if state.loader is not None:
        payload["loader"] = state.loader.state_dict()
    return payload


def restore(state: TrainState, payload: dict[str, Any]) -> int:
    """Put `capture`'s items back; return the step. A rank with no saved RNG keeps its own.

    Side effects:
        Restores the scheduler, the loader position, and this rank's RNGs.
    """
    if state.scheduler is not None and "scheduler" in payload:
        state.scheduler.load_state_dict(state_dict=payload["scheduler"])
    if state.loader is not None and "loader" in payload:
        state.loader.load_state_dict(state_dict=payload["loader"])
    blob = payload.get(f"rank{_rank()}/rng")  # noqa: NAR001
    if blob is not None:
        set_rng_state(blob=blob)
    return int(payload.get("step", 0))  # noqa: NAR001


def latest_committed(root: Path) -> Optional[Path]:
    """The newest `step_*` directory under `root` that has a commit marker."""
    committed = (
        [p for p in root.glob(pattern="step_*") if (p / COMMIT_MARKER).exists()]
        if root.exists()
        else []
    )
    return max(committed, default=None)  # noqa: NAR001


def young_daly_interval(checkpoint_s: float, mtbf_s: float) -> float:
    """Optimal seconds between checkpoints: sqrt(2 x delta x MTBF)."""
    return math.sqrt(2 * checkpoint_s * mtbf_s)  # noqa: NAR001


def expected_goodput(
    interval_s: float, checkpoint_s: float, mtbf_s: float, restart_s: float
) -> float:
    """1 - (delta / tau + (tau / 2 + R) / MTBF): useful share of wall clock, first order."""
    return 1 - (checkpoint_s / interval_s + (interval_s / 2 + restart_s) / mtbf_s)


class Checkpointer:
    """Sharded checkpoints under `root/step_<n>`, committed atomically, optionally async."""

    def __init__(
        self, root: Path, async_save: bool = False, config: Optional[dict] = None, keep: int = 3
    ) -> None:
        """Write under `root`; keep the newest `keep` committed checkpoints.

        Args:
            config: Recorded in every checkpoint with the git commit, for provenance.
        """
        self.root, self.async_save, self.keep = root, async_save, keep
        self.meta = {"config": config or {}, "git_commit": get_git_commit()}
        self.pending: Optional[tuple[Future, Path]] = None

    def _state_dict(self, state: TrainState, step: int) -> dict[str, Any]:
        """Model and optimizer as DCP sees them (sharded if sharded), plus `capture`."""
        model_sd, optim_sd = get_state_dict(model=state.model, optimizers=state.optimizer)
        return {
            "model": model_sd,
            "optimizer": optim_sd,
            "meta": self.meta,
            **capture(state=state, step=step),
        }

    def save(self, state: TrainState, step: int) -> None:
        """Write a checkpoint for `step`; every rank must call it.

        Side effects:
            Commits any pending async save first, then writes `root/step_<n>`
            (synchronously, or starts it in the background), and prunes old ones.
        """
        self.wait()
        path = self.root / f"step_{step:08d}"
        state_dict = self._state_dict(state=state, step=step)
        if self.async_save:
            future = async_save(state_dict=state_dict, checkpoint_id=path)
            self.pending = (future, path)  # type: ignore[assignment]
        else:
            save(state_dict=state_dict, checkpoint_id=path)
            self._commit(path=path)

    def wait(self) -> None:
        """Finish and commit the pending async save, if any."""
        if self.pending is not None:
            future, path = self.pending
            future.result()
            self.pending = None
            self._commit(path=path)

    def _commit(self, path: Path) -> None:
        """After every rank has written, rank 0 writes the marker (temp file, then rename)."""
        if dist.is_initialized():
            dist.barrier()
        if _rank() == 0:
            temp = path / f".{COMMIT_MARKER}.tmp"
            temp.write_text(data=path.name)
            temp.rename(target=path / COMMIT_MARKER)
            committed = sorted(  # noqa: NAR001
                p for p in self.root.glob(pattern="step_*") if (p / COMMIT_MARKER).exists()
            )  # noqa: NAR001
            for old in committed[: -self.keep]:
                shutil.rmtree(path=old, ignore_errors=True)
        if dist.is_initialized():
            dist.barrier()

    def load_latest(self, state: TrainState) -> Optional[int]:
        """Load the newest committed checkpoint into `state`, resharding as needed; return its step.

        Returns None, changing nothing, when no committed checkpoint exists.
        """
        path = latest_committed(root=self.root)
        if path is None:
            return None
        saved = FileSystemReader(path=path).read_metadata().state_dict_metadata
        template = self._state_dict(state=state, step=0)
        template = {
            k: v
            for k, v in template.items()
            if k in saved or any(s.startswith(f"{k}.") for s in saved)  # noqa: NAR001
        }  # noqa: NAR001
        dcp_load(state_dict=template, checkpoint_id=path)
        set_state_dict(
            model=state.model,
            optimizers=state.optimizer,
            model_state_dict=template["model"],
            optim_state_dict=template["optimizer"],
        )
        return restore(state=state, payload=template)


def save_full_on_rank0(state: TrainState, path: Path) -> None:
    """The baseline: gather the full model and optimizer on rank 0 and `torch.save` them.

    Every rank participates in the gather; only rank 0 holds the full copy, which is the
    memory spike and the serial write that sharded checkpoints avoid.

    Side effects:
        Writes `path` from rank 0.
    """
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd, optim_sd = get_state_dict(
        model=state.model, optimizers=state.optimizer, options=options
    )
    if _rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj={"model": model_sd, "optimizer": optim_sd}, f=path)
