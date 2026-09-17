"""environment.py -- snapshot of host resource usage."""

import psutil
import torch
from training.train.model import SystemMetrics


def collect_system_metrics() -> SystemMetrics:
    """Return current CPU/RAM usage and, if available, GPU memory stats."""
    cpu_usage_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    metrics = SystemMetrics(cpu_usage_percent=cpu_usage_percent, ram_used_gb=ram.used / 1e9)
    if torch.cuda.is_available():
        metrics.gpu_peak_memory_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
        metrics.gpu_peak_memory_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    elif torch.backends.mps.is_available():
        metrics.gpu_memory_allocated_gb = torch.mps.current_allocated_memory() / 1e9
    return metrics
