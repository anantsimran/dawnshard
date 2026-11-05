"""
subthreshold.py
Can a parity test catch a bug whose error each step is smaller than floating-point
noise? Tiny Qwen3 on 2 gloo ranks, bf16 autocast with fp32 weights and AdamW, every
candidate checked against one process on the same batches.

Two ways to run each candidate:

- restart: every step loads the reference's weights and Adam state into the
  candidate (sharded as the candidate holds them), takes one step on each, and
  records three statistics per step: the weight-relative error (the per-tensor
  criterion of single-iteration differential testing), the update-relative error,
  and the per-step update error itself, for `drift_test`.
- free: both train freely from the same init; the per-step absolute loss difference
  (the loss-parity test).

Candidates without bugs (naive and bucketed DDP, small buckets, ZeRO-2, hand TP) set
the noise, and bugs planted in bucketed DDP are what the checks should find:

- `grad_bf16`: the averaged gradient is rounded to bf16 before the optimizer.
- `microbatch_bf16`: micro-batch gradients accumulate in bf16.
- `bf16_master`: weights are stored in bf16 although the recipe says fp32.
- `decay_on_norms`: weight decay also hits the norm gains the recipe excludes.
- `adam_step_offset`: Adam's step counter runs one ahead, so bias correction is off.

Writes CSVs under experiments/subthreshold-bugs/:

    uv run python app/src/training/experiments/subthreshold.py
"""

import copy
import csv
import math
import os
import socket
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.multiprocessing.spawn import spawn
from training.constants import REPO_ROOT
from training.model.qwen3 import QWEN3_TINY, Qwen3, parallelize_tensor_parallel
from training.parallel.ddp import BucketedDDP, NaiveDDP
from training.parallel.parity import drift_test, flatten_shards, global_dot
from training.parallel.zero import Zero2AdamW

RESULTS_DIR = REPO_ROOT / "experiments" / "subthreshold-bugs"
STEPS = 200
CHECKPOINTS = (10, 25, 50, 100, 200)
LR, WEIGHT_DECAY = 1e-3, 0.1
ROWS, SEQ_LEN, MICRO_BATCHES, WORLD = 8, 32, 2, 2
ALPHA, NUM_FLIPS = 0.01, 4000
CONTROLS = ("naive_ddp", "bucketed_ddp", "bucketed_ddp_small", "zero2", "tp")
BUGS = ("grad_bf16", "microbatch_bf16", "bf16_master", "decay_on_norms", "adam_step_offset")
CALIBRATION_SEED, EVAL_SEEDS = 0, (1, 2)


@dataclass
class Candidate:
    """A parallel implementation under test, as one rank holds it.

    `params` are this rank's tensors in the dense model's parameter order; `shard`
    maps dense tensors in that order to what this rank holds; `rows` selects this
    rank's share of a batch.
    """

    model: nn.Module
    params: list[nn.Parameter]
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    shard: Callable[[Sequence[torch.Tensor]], list[torch.Tensor]]
    replicated: list[bool]
    rows: slice
    loss_is_replicated: bool


def markov_batches(steps: int, seed: int) -> list[torch.Tensor]:
    """Token batches `(ROWS, SEQ_LEN + 1)` from a random bigram chain with 4 successors."""
    generator = torch.Generator().manual_seed(seed)  # noqa: NAR001
    vocab = QWEN3_TINY.vocab_size
    successors = torch.randint(low=0, high=vocab, size=(vocab, 4), generator=generator)
    batches = []
    for _ in range(steps):  # noqa: NAR001
        tokens = torch.empty(size=(ROWS, SEQ_LEN + 1), dtype=torch.long)
        tokens[:, 0] = torch.randint(low=0, high=vocab, size=(ROWS,), generator=generator)
        for position in range(SEQ_LEN):  # noqa: NAR001
            choice = torch.randint(low=0, high=4, size=(ROWS,), generator=generator)
            tokens[:, position + 1] = successors[tokens[:, position], choice]
        batches.append(tokens)  # noqa: NAR001
    return batches


def make_adamw(params: list[nn.Parameter], decay_norms: bool) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices, and on 1-D gains too when `decay_norms`."""
    decayed = [p for p in params if decay_norms or p.ndim > 1]
    kept = [p for p in params if not (decay_norms or p.ndim > 1)]
    groups = [{"params": decayed, "weight_decay": WEIGHT_DECAY}]
    if kept:
        groups.append({"params": kept, "weight_decay": 0.0})  # noqa: NAR001
    return torch.optim.AdamW(params=groups, lr=LR)


def shard_tensor_parallel(template: Qwen3, tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """This rank's TP shards of dense `tensors`, cut by the same plan as the model."""
    model = copy.deepcopy(template)  # noqa: NAR001
    with torch.no_grad():
        for param, tensor in zip(model.parameters(), tensors, strict=True):  # noqa: NAR001
            param.copy_(tensor)  # noqa: NAR001
    parallelize_tensor_parallel(model=model, group=None)
    return [p.detach() for p in model.parameters()]


def build_candidate(kind: str, bug: str, dense: Qwen3, rank: int) -> Candidate:
    """Shard or wrap a copy of `dense` as `kind` does, with `bug` in its optimizer recipe."""
    model = copy.deepcopy(dense)  # noqa: NAR001
    decay_norms = kind == "zero2" or bug == "decay_on_norms"
    if kind == "tp":
        criterion = parallelize_tensor_parallel(model=model, group=None)
        params = list(model.parameters())  # noqa: NAR001
        shapes = [p.shape for p in dense.parameters()]
        return Candidate(
            model=model,
            params=params,
            criterion=criterion,
            optimizer=make_adamw(params=params, decay_norms=decay_norms),
            shard=partial(shard_tensor_parallel, dense),  # noqa: NAR001
            replicated=[p.shape == s for p, s in zip(params, shapes, strict=True)],  # noqa: NAR001
            rows=slice(None),  # noqa: NAR001
            loss_is_replicated=True,
        )
    params = list(model.parameters())  # noqa: NAR001
    if kind == "zero2":
        wrapped, optimizer = model, Zero2AdamW(params=params, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        bucket_mb = 0.01 if kind == "bucketed_ddp_small" else 25
        wrapped = (
            NaiveDDP(module=model)
            if kind == "naive_ddp"
            else BucketedDDP(module=model, bucket_mb=bucket_mb)
        )
        optimizer = make_adamw(params=params, decay_norms=decay_norms)
    share = ROWS // WORLD
    return Candidate(
        model=wrapped,
        params=params,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        shard=list,
        replicated=[True] * len(params),  # noqa: NAR001
        rows=slice(rank * share, (rank + 1) * share),  # noqa: NAR001
        loss_is_replicated=False,
    )


@torch.no_grad()
def round_to_bf16(tensors: Sequence[torch.Tensor | None]) -> None:
    """Round every tensor in place to the nearest bf16 value, keeping its dtype."""
    for tensor in tensors:
        if tensor is not None:
            tensor.copy_(tensor.bfloat16())  # noqa: NAR001


@torch.no_grad()
def load_state(
    candidate: Candidate, reference: nn.Module, optimizer: torch.optim.AdamW, bug: str
) -> None:
    """Overwrite the candidate's weights and Adam state with the reference's, sharded.

    Before the reference's first step there is no Adam state to copy, and only the
    weights are loaded (and, under `adam_step_offset`, a step counter of 1).

    Side effects:
        Overwrites `candidate.params` and `candidate.optimizer`'s state.
    """
    dense = list(reference.parameters())  # noqa: NAR001
    for param, tensor in zip(candidate.params, candidate.shard(dense), strict=True):  # noqa: NAR001
        param.copy_(tensor)  # noqa: NAR001
    state = optimizer.state
    offset = 1 if bug == "adam_step_offset" else 0
    if state:
        step = state[dense[0]]["step"].item() + offset
        moments = [
            candidate.shard([state[p][key] for p in dense])  # noqa: NAR001
            for key in ("exp_avg", "exp_avg_sq")
        ]
    else:
        step = offset
        moments = [[torch.zeros_like(input=p) for p in candidate.params]] * 2
    target = candidate.optimizer
    if isinstance(target, Zero2AdamW):  # noqa: NAR001
        for i, master in target.master.items():
            master.copy_(candidate.params[i])  # noqa: NAR001
            target.moments[i][0].copy_(moments[0][i])  # noqa: NAR001
            target.moments[i][1].copy_(moments[1][i])  # noqa: NAR001
        target.steps = int(step)  # noqa: NAR001
    elif state or offset:
        for param, m, v in zip(candidate.params, *moments, strict=True):  # noqa: NAR001
            target.state[param] = {
                "step": torch.tensor(data=float(step)),  # noqa: NAR001
                "exp_avg": m.clone(),
                "exp_avg_sq": v.clone(),
            }
    if bug == "bf16_master":
        round_to_bf16(tensors=candidate.params)  # noqa: NAR001


def candidate_step(candidate: Candidate, tokens: torch.Tensor, bug: str) -> float:
    """One optimizer step over `MICRO_BATCHES` micro-batches; return the global mean loss.

    Side effects:
        Updates `candidate.params` and its optimizer state.
    """
    chunks = tokens[candidate.rows].chunk(chunks=MICRO_BATCHES)
    candidate.optimizer.zero_grad(set_to_none=True)
    no_sync = getattr(candidate.model, "no_sync", None)  # noqa: NAR001
    total = 0.0
    for index, chunk in enumerate(chunks):  # noqa: NAR001
        last = index == len(chunks) - 1  # noqa: NAR001
        with no_sync() if no_sync is not None and not last else nullcontext():
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits = candidate.model(chunk[:, :-1])  # noqa: NAR001
                loss = candidate.criterion(logits, chunk[:, 1:].reshape(-1))  # noqa: NAR001
            (loss / len(chunks)).backward()  # noqa: NAR001
        total += loss.item() / len(chunks)  # noqa: NAR001
        if bug == "microbatch_bf16":
            round_to_bf16(tensors=[p.grad for p in candidate.params])
    if bug == "grad_bf16":
        round_to_bf16(tensors=[p.grad for p in candidate.params])
    candidate.optimizer.step()
    if bug == "bf16_master":
        round_to_bf16(tensors=candidate.params)  # noqa: NAR001
    if candidate.loss_is_replicated:
        return total
    summed = torch.tensor(data=total)
    dist.all_reduce(tensor=summed)
    return summed.item() / dist.get_world_size()


def reference_step(model: nn.Module, optimizer: torch.optim.AdamW, tokens: torch.Tensor) -> float:
    """One optimizer step on the whole batch in one process; return its loss.

    Side effects:
        Updates `model` and `optimizer`'s state.
    """
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(tokens[:, :-1])  # noqa: NAR001
        targets = tokens[:, 1:].reshape(-1)  # noqa: NAR001
        loss = nn.functional.cross_entropy(input=logits, target=targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def restart_run(
    candidate: Candidate, reference: nn.Module, optimizer: torch.optim.AdamW, bug: str, seed: int
) -> tuple[list[dict], list[dict]]:
    """Step-restart comparison over `STEPS` batches; per-step rows and drift tests."""
    flat = partial(  # noqa: NAR001
        flatten_shards, replicated=candidate.replicated, world_size=WORLD
    )
    dense = list(reference.parameters())  # noqa: NAR001
    errors, rows = [], []
    for step, tokens in enumerate(markov_batches(steps=STEPS, seed=seed)):  # noqa: NAR001
        load_state(candidate=candidate, reference=reference, optimizer=optimizer, bug=bug)
        before_candidate = flat(tensors=candidate.params)
        before_reference = flat(tensors=candidate.shard(dense))  # noqa: NAR001
        candidate_step(candidate=candidate, tokens=tokens, bug=bug)
        reference_step(model=reference, optimizer=optimizer, tokens=tokens)
        after_reference = flat(tensors=candidate.shard(dense))  # noqa: NAR001
        after_candidate = flat(tensors=candidate.params)
        update = after_reference - before_reference
        error = (after_candidate - before_candidate) - update
        weight_diff = after_candidate - after_reference
        update_energy = global_dot(a=update, b=update)
        weight_energy = global_dot(a=after_reference, b=after_reference)
        weight_rel = math.sqrt(  # noqa: NAR001
            global_dot(a=weight_diff, b=weight_diff) / weight_energy
        )
        update_rel = math.sqrt(global_dot(a=error, b=error) / update_energy)  # noqa: NAR001
        errors.append(error)  # noqa: NAR001
        rows.append(  # noqa: NAR001
            {
                "step": step,
                "weight_rel": weight_rel,
                "update_rel": update_rel,
                "update_bias": global_dot(a=error, b=update) / update_energy,
            }
        )
    tests = []
    for k in CHECKPOINTS:
        ratio, p_value = drift_test(errors=errors[:k], num_flips=NUM_FLIPS, seed=seed)
        tests.append({"steps": k, "drift_ratio": ratio, "p_value": p_value})  # noqa: NAR001
    return rows, tests


def free_run(
    candidate: Candidate, reference: nn.Module, optimizer: torch.optim.AdamW, bug: str, seed: int
) -> list[dict]:
    """Free training from the same init; per-step loss of each and their difference."""
    load_state(candidate=candidate, reference=reference, optimizer=optimizer, bug=bug)
    rows = []
    for step, tokens in enumerate(markov_batches(steps=STEPS, seed=seed)):  # noqa: NAR001
        loss = candidate_step(candidate=candidate, tokens=tokens, bug=bug)
        reference_loss = reference_step(model=reference, optimizer=optimizer, tokens=tokens)
        rows.append(  # noqa: NAR001
            {
                "step": step,
                "loss_candidate": loss,
                "loss_reference": reference_loss,
                "abs_loss_diff": abs(loss - reference_loss),  # noqa: NAR001
            }
        )
    return rows


def _worker(rank: int, port: int, mode: str, kind: str, bug: str, seed: int, queue) -> None:
    """Join the gloo group as `rank`, run one configuration, and queue rank 0's result."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))  # noqa: NAR001
    dist.init_process_group(backend="gloo", rank=rank, world_size=WORLD)
    torch.set_num_threads(1)  # noqa: NAR001
    torch.manual_seed(seed=seed)
    reference = Qwen3(config=QWEN3_TINY)
    params = list(reference.parameters())  # noqa: NAR001
    optimizer = make_adamw(params=params, decay_norms=kind == "zero2")
    candidate = build_candidate(kind=kind, bug=bug, dense=reference, rank=rank)
    run = restart_run if mode == "restart" else free_run
    result = run(candidate=candidate, reference=reference, optimizer=optimizer, bug=bug, seed=seed)
    if rank == 0:
        queue.put(result)  # noqa: NAR001
    dist.destroy_process_group()


def run_config(mode: str, kind: str, bug: str, seed: int):
    """Run one configuration on `WORLD` processes and return rank 0's result."""
    queue = mp.get_context(method="spawn").Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))  # noqa: NAR001
        port = sock.getsockname()[1]
    spawn(  # noqa: NAR001
        _worker, args=(port, mode, kind, bug, seed, queue), nprocs=WORLD, join=True
    )
    return queue.get()


def write_csv(name: str, rows: list[dict]) -> None:
    """Overwrite `RESULTS_DIR/<name>.csv` with `rows`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(file=RESULTS_DIR / f"{name}.csv", mode="w", newline="") as handle:
        writer = csv.DictWriter(f=handle, fieldnames=list(rows[0]))  # noqa: NAR001
        writer.writeheader()
        writer.writerows(rowdicts=rows)


def first_above(values: list[float], threshold: float) -> int | None:
    """Index of the first value strictly above `threshold`, or None."""
    return next((i for i, v in enumerate(values) if v > threshold), None)  # noqa: NAR001


def summarize(runs: list[dict]) -> tuple[list[dict], dict[str, float]]:
    """Per evaluation run, the first step each check flags, with limits from seed 0.

    A per-step check flags a step whose statistic exceeds the largest value any
    bug-free candidate reached on the calibration seed: one tolerance for the whole
    suite, as a test would set it. The drift test needs no limit; it flags at the
    first checkpoint with p-value at most `ALPHA`.

    Returns:
        `(rows, limits)`: one summary row per evaluation run, and the limit per statistic.
    """
    sources = {"weight_rel": "restart", "update_rel": "restart", "abs_loss_diff": "free"}
    calibration = [r for r in runs if r["bug"] == "none" and r["seed"] == CALIBRATION_SEED]
    limits = {
        key: max(row[key] for run in calibration for row in run[source])  # noqa: NAR001
        for key, source in sources.items()
    }
    rows = []
    for run in (r for r in runs if r["seed"] != CALIBRATION_SEED):
        first = {
            key: first_above(values=[row[key] for row in run[source]], threshold=limits[key])
            for key, source in sources.items()
        }
        tail = run["free"][-20:]
        gaps = [r["loss_candidate"] - r["loss_reference"] for r in tail]
        gap = sum(gaps) / len(gaps)  # noqa: NAR001
        rows.append(  # noqa: NAR001
            {
                "kind": run["kind"],
                "bug": run["bug"],
                "seed": run["seed"],
                "first_step_weight_rel": first["weight_rel"],
                "first_step_update_rel": first["update_rel"],
                "first_step_loss_parity": first["abs_loss_diff"],
                "drift_flagged_by_step": next(  # noqa: NAR001
                    (t["steps"] for t in run["drift"] if t["p_value"] <= ALPHA), None
                ),
                "drift_ratio_at_end": round(run["drift"][-1]["drift_ratio"], 2),  # noqa: NAR001
                "drift_p_at_end": run["drift"][-1]["p_value"],
                "final_loss_gap": gap,
            }
        )
    return rows, limits


def run_both(kind: str, bug: str, seed: int) -> dict:
    """Run one configuration in restart mode and in free mode."""
    restart_rows, tests = run_config(mode="restart", kind=kind, bug=bug, seed=seed)
    free_rows = run_config(mode="free", kind=kind, bug=bug, seed=seed)
    print(kind, bug, seed, tests[-1], flush=True)  # noqa: NAR001
    return {
        "kind": kind,
        "bug": bug,
        "seed": seed,
        "restart": restart_rows,
        "free": free_rows,
        "drift": tests,
    }


def main() -> None:
    """Run every control and planted bug in both modes, two at a time, and write the CSVs."""
    configs = [
        (kind, "none", seed) for kind in CONTROLS for seed in (CALIBRATION_SEED, *EVAL_SEEDS)
    ]
    configs += [("bucketed_ddp", bug, seed) for bug in BUGS for seed in EVAL_SEEDS]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = pool.map(lambda c: run_both(kind=c[0], bug=c[1], seed=c[2]), configs)  # noqa: NAR001
        runs = list(jobs)  # noqa: NAR001
    steps, drift = [], []
    for run in runs:
        label = {"kind": run["kind"], "bug": run["bug"], "seed": run["seed"]}
        pairs = zip(run["restart"], run["free"], strict=True)  # noqa: NAR001
        steps += [label | r | f for r, f in pairs]
        drift += [label | t for t in run["drift"]]
    summary, limits = summarize(runs=runs)
    write_csv(name="steps", rows=steps)
    write_csv(name="drift", rows=drift)
    write_csv(name="detection", rows=summary)
    write_csv(name="thresholds", rows=[limits])
    print(limits, *summary, sep="\n")  # noqa: NAR001


if __name__ == "__main__":
    main()
