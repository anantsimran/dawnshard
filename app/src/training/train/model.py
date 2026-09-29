"""model.py -- dataclasses shared by the training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass
class Metrics:
    # Subclass to add metrics; the step always fills in loss and count itself.
    loss: float = 0.0
    count: int = 0


# Generic over the Metrics subclass so a spec built from ClassificationMetrics functions
# type-checks: Callable parameters are contravariant, so a function that only accepts
# ClassificationMetrics is not a Callable[[Metrics, ...], ...].
type MetricsCalculateFn[M: Metrics] = Callable[[torch.Tensor, torch.Tensor], M]
type MetricsAccumulateFn[M: Metrics] = Callable[[M, M], None]
type MetricsReduceFn[M: Metrics] = Callable[[M], M]

# Called by the step on every batch after the forward pass with
# (model, input, target, predicted, epoch, batch_index, seed); `predicted` is
# detached. `seed` is the fixed val seed on a validation pass, so a probe that draws
# its own randomness draws the same thing on every validation pass, whatever model
# it runs on. It is None on a train pass: training randomness comes from the global
# RNG that fit seeds. The probe decides which calls to act on and where to store what
# it keeps; bind its settings with functools.partial.
Probe = Callable[
    [nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, int, int, Optional[int]], None
]


@dataclass
class EpochSpec[M: Metrics]:
    device: torch.device
    # Instantiated with no arguments at the start of every epoch as the accumulator.
    metrics_type: type[M]
    calculate_metrics: MetricsCalculateFn[M]
    accumulate_metrics: MetricsAccumulateFn[M]
    reduce_metrics: MetricsReduceFn[M]
    train_probe: Optional[Probe] = None
    eval_probe: Optional[Probe] = None


@dataclass
class TrainState:
    model: nn.Module
    optimizer: Optimizer
    criterion: nn.Module
    scheduler: Optional[LRScheduler] = None

    @classmethod
    def create(
        cls,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[LRScheduler] = None,
    ) -> "TrainState":
        """Build a TrainState, moving `model` to `device` in place."""
        # Build the optimizer from model.parameters() before calling create();
        # we move the model here so optimizer buffers land on device on first step.
        model.to(device=device)
        return cls(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler)


@dataclass
class SystemMetrics:
    cpu_usage_percent: float
    ram_used_gb: float
    # CUDA reports peak stats (reset each epoch); MPS only exposes current allocation.
    gpu_peak_memory_allocated_gb: Optional[float] = None
    gpu_peak_memory_reserved_gb: Optional[float] = None
    gpu_memory_allocated_gb: Optional[float] = None


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_metrics: Metrics
    epoch_duration_seconds: float
    system_metrics: SystemMetrics
    val_loss: Optional[float] = None
    val_metrics: Optional[Metrics] = None
