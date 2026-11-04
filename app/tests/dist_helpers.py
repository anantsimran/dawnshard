"""Run a function on N CPU processes joined in a gloo group, for distributed tests."""

import os
import socket

import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _entry(rank: int, world: int, port: int, fn, args) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def run_distributed(fn, world: int = 2, args: tuple = ()) -> None:
    """Run `fn(rank, world, *args)` on `world` processes; a failure in any re-raises here."""
    mp.spawn(_entry, args=(world, _free_port(), fn, args), nprocs=world, join=True)
