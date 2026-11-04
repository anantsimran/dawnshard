"""Join the process group `torchrun` describes, and make logs say which rank wrote them.

`torchrun` sets RANK, LOCAL_RANK, and WORLD_SIZE for every process it starts.
`init_process_group` reads them, picks NCCL on CUDA and gloo otherwise, and leaves an
existing group alone (Ray Train creates its own). With one process and no group it is
a no-op that reports rank 0 of 1, so the same entry point runs alone or distributed.
"""

import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class DistInfo:
    """Where this process sits in the job."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        """Whether this is rank 0, the process that writes shared files."""
        return self.rank == 0


def init_process_group(cpu_device: torch.device | None = None) -> DistInfo:
    """Join the `torchrun` group if WORLD_SIZE > 1; return this process's place in it.

    Args:
        cpu_device: Device for a single process without CUDA, such as MPS. Distributed
            CPU runs always use the CPU.

    Side effects:
        May initialize the default process group, set the CUDA device, and add the
        rank to every loguru line.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))  # noqa: NAR001
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # noqa: NAR001
    if torch.cuda.is_available():
        torch.cuda.set_device(device=local_rank)
        device = torch.device("cuda", local_rank)  # noqa: NAR001
    else:
        device = cpu_device if world == 1 and cpu_device is not None else torch.device("cpu")  # noqa: NAR001
    if world > 1:
        logger.remove()
        logger.add(
            sink=sys.stderr, format=f"[rank {rank}] " + "{time:HH:mm:ss} | {level} | {message}"
        )
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world, device=device)


def parameter_checksum(model: torch.nn.Module) -> float:
    """Sum of every parameter as float64: equal on every rank iff the weights start equal.

    Sharded (DTensor) parameters are gathered first, so every rank must call it.
    """
    full = [p.full_tensor() if isinstance(p, DTensor) else p for p in model.parameters()]  # noqa: NAR001
    return float(sum(p.detach().double().sum() for p in full))  # noqa: NAR001
