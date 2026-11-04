"""
tp_pp.py
Tensor- and pipeline-parallel checks that run on CPU: loss parity of hand TP against
one process, the pipeline bubble against (p - 1) / (m + p - 1), stage balancing by
FLOPs for Qwen3-1.7B, and the TP communication-time prediction per interconnect.

- `parity`: tiny Qwen3, TP=2 over 2 gloo processes (hand plan, and DTensor), 50 fp32
  AdamW steps from the same full init and the same batches as a 1-process run; the
  max absolute per-step loss difference.
- `bubble`: simulated GPipe and 1F1B timelines for p = 4 and m in {4, 8, 16, 32},
  backward = 2 x forward, with the activations each stage holds at its peak.
- `balance`: per-stage cost of Qwen3-1.7B (28 blocks, tied head worth about 6.2
  blocks) split by layer count, then by `partition`.
- `comm`: predicted per-micro-step TP all-reduce time at NVLink and PCIe bandwidth.

Each writes a CSV under experiments/t7-tp-pp/:

    uv run python app/src/training/experiments/tp_pp.py all
"""

import argparse
import csv
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing.spawn import spawn
from training.constants import REPO_ROOT
from training.model.qwen3 import (
    QWEN3_1_7B,
    QWEN3_TINY,
    Qwen3,
    count_parameters,
    parallelize_dtensor,
    parallelize_tensor_parallel,
)
from training.parallel.pipeline import gpipe, one_f_one_b, partition, simulate, theoretical_bubble

RESULTS_DIR = REPO_ROOT / "experiments" / "t7-tp-pp"
STEPS = 50


def write_csv(name: str, rows: list[dict]) -> None:
    """Overwrite `RESULTS_DIR/<name>.csv` with `rows`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(file=RESULTS_DIR / f"{name}.csv", mode="w", newline="") as handle:
        writer = csv.DictWriter(f=handle, fieldnames=list(rows[0]))  # noqa: NAR001
        writer.writeheader()
        writer.writerows(rowdicts=rows)


def train_losses(model: nn.Module, criterion: nn.Module, steps: int) -> list[float]:
    """AdamW on the same seeded batches every call; return the loss of each step."""
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(7)  # noqa: NAR001
    losses = []
    for _ in range(steps):  # noqa: NAR001
        ids = torch.randint(low=0, high=QWEN3_TINY.vocab_size, size=(4, 33), generator=generator)
        loss = criterion(model(ids[:, :-1]), ids[:, 1:].reshape(-1))  # noqa: NAR001
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # noqa: NAR001
    return losses


def _parity_worker(rank: int, world: int, port: int, plan: str, queue) -> None:  # noqa: NAR002
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))  # noqa: NAR001
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    torch.manual_seed(seed=0)
    model = Qwen3(config=QWEN3_TINY)
    if plan == "hand":
        criterion = parallelize_tensor_parallel(model=model, group=None)
    else:
        parallelize_dtensor(
            model=model, tp_mesh=init_device_mesh(device_type="cpu", mesh_shape=(world,))
        )
        criterion = nn.CrossEntropyLoss()
    losses = train_losses(model=model, criterion=criterion, steps=STEPS)
    if rank == 0:
        queue.put(losses)  # noqa: NAR001
    dist.destroy_process_group()


def parity() -> list[dict]:
    """Per-step |loss(TP=2) - loss(1 process)| for the hand plan and the DTensor plan."""
    torch.manual_seed(seed=0)
    reference = train_losses(
        model=Qwen3(config=QWEN3_TINY), criterion=nn.CrossEntropyLoss(), steps=STEPS
    )
    rows = []
    for plan in ("hand", "dtensor"):
        context = mp.get_context(method="spawn")
        queue = context.Queue()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))  # noqa: NAR001
            port = sock.getsockname()[1]
        spawn(_parity_worker, args=(2, port, plan, queue), nprocs=2, join=True)  # noqa: NAR001
        losses = queue.get()
        diffs = [abs(a - b) for a, b in zip(losses, reference)]  # noqa: NAR001
        rows.append(  # noqa: NAR001
            {
                "plan": plan,
                "steps": STEPS,
                "step0_abs_loss_diff": f"{diffs[0]:.2e}",
                "max_abs_loss_diff": f"{max(diffs):.2e}",  # noqa: NAR001
                "final_loss_tp": round(losses[-1], 6),  # noqa: NAR001
                "final_loss_single": round(reference[-1], 6),  # noqa: NAR001
            }
        )  # noqa: NAR001
    return rows


def bubble() -> list[dict]:
    """Simulated bubble and peak in-flight micro-batches, GPipe vs 1F1B, p = 4."""
    rows = []
    for m in (4, 8, 16, 32):
        for name, make in (("GPipe", gpipe), ("1F1B", one_f_one_b)):
            schedule = make(num_stages=4, num_microbatches=m)
            timeline = simulate(schedule=schedule, forward_time=[1.0] * 4, backward_time=[2.0] * 4)
            peak = max(_peak_in_flight(actions=a) for a in schedule)  # noqa: NAR001
            rows.append(  # noqa: NAR001
                {
                    "schedule": name,
                    "p": 4,
                    "m": m,
                    "simulated_bubble": round(timeline.bubble, 4),  # noqa: NAR001
                    "formula": round(theoretical_bubble(num_stages=4, num_microbatches=m), 4),  # noqa: NAR001
                    "peak_activations_held": peak,
                }
            )  # noqa: NAR001
    return rows


def _peak_in_flight(actions: list[tuple[str, int]]) -> int:
    """Most micro-batches whose activations one stage holds at once."""
    held = peak = 0
    for kind, _ in actions:
        held += 1 if kind == "F" else -1
        peak = max(peak, held)  # noqa: NAR001
    return peak


def balance() -> list[dict]:
    """Qwen3-1.7B over 4 stages: equal layer counts vs `partition` by cost, with the head."""
    head = QWEN3_1_7B.vocab_size * QWEN3_1_7B.hidden_size
    block = (
        count_parameters(config=QWEN3_1_7B) - head - QWEN3_1_7B.hidden_size
    ) / QWEN3_1_7B.num_hidden_layers
    costs = [1.0] * QWEN3_1_7B.num_hidden_layers + [head / block]
    by_count = [range(0, 7), range(7, 14), range(14, 21), range(21, 29)]  # noqa: NAR001
    rows = []
    for method, stages in (
        ("equal layer count", by_count),
        ("partition by FLOPs", partition(costs=costs, num_stages=4)),
    ):
        loads = [sum(costs[i] for i in stage) for stage in stages]  # noqa: NAR001
        rows.append(  # noqa: NAR001
            {
                "method": method,
                "layers_per_stage": "/".join(str(len(s)) for s in stages),  # noqa: NAR001
                "cost_per_stage_in_blocks": "/".join(f"{x:.1f}" for x in loads),  # noqa: NAR001
                "slowest_over_mean": round(max(loads) / (sum(loads) / 4), 3),  # noqa: NAR001
            }
        )  # noqa: NAR001
    return rows


def comm() -> list[dict]:
    """Predicted TP all-reduce seconds per micro-step for Qwen3-1.7B, micro-batch 2 x 2048."""
    message = 2 * 2048 * QWEN3_1_7B.hidden_size * 2  # bf16 activations per all-reduce
    rows = []
    for tp in (2, 4):
        per_gpu = 2 * (tp - 1) / tp * message
        total = per_gpu * QWEN3_1_7B.num_hidden_layers * 4  # 2 forward + 2 backward per block
        for link, bandwidth in (("NVLink ~200 GB/s", 200e9), ("PCIe no P2P ~10 GB/s", 10e9)):
            rows.append(  # noqa: NAR001
                {
                    "tp": tp,
                    "interconnect": link,
                    "bytes_per_gpu_per_microstep_GB": round(total / 1e9, 2),  # noqa: NAR001
                    "predicted_comm_ms": round(1000 * total / bandwidth, 1),  # noqa: NAR001
                }
            )  # noqa: NAR001
    return rows


def main() -> None:
    """Run the named study (or all) and write its CSV."""
    parser = argparse.ArgumentParser(description="Tensor- and pipeline-parallel CPU checks.")
    parser.add_argument("study", choices=["parity", "bubble", "balance", "comm", "all"])  # noqa: NAR001
    args = parser.parse_args()
    for name, study in {
        "parity": parity,
        "bubble": bubble,
        "balance": balance,
        "comm": comm,
    }.items():
        if args.study in (name, "all"):
            rows = study()
            write_csv(name=name, rows=rows)
            print(name, *rows, sep="\n")  # noqa: NAR001


if __name__ == "__main__":
    main()
