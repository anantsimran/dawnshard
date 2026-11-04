"""The interface every compute backend implements, and what they share.

Four operations: `submit` a spec and get a job id, `status`, `logs`, `cancel`. Every
backend maps its own states onto `JobState`, so a caller polls the same way whether
the job is a local process, a JobSet, a RayJob, or a SageMaker training job.
"""

import enum
import shlex
import time
from collections.abc import Iterator
from typing import Protocol

from training.launch.spec import JobSpec

RDZV_PORT = 29500


class JobState(enum.Enum):
    """The states every backend's own states map onto."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class Backend(Protocol):
    """A place jobs run."""

    name: str

    def submit(self, spec: JobSpec) -> str:
        """Start the job; return its id."""
        ...

    def status(self, job_id: str) -> JobState:
        """Return the job's current state."""
        ...

    def logs(self, job_id: str) -> Iterator[str]:
        """Yield the job's log lines."""
        ...

    def cancel(self, job_id: str) -> None:
        """Stop the job and release its resources."""
        ...


def torchrun_command(spec: JobSpec, rdzv_endpoint: str, standalone: bool = False) -> list[str]:
    """Return the `torchrun` argument list that runs `spec.module` on one node.

    c10d rendezvous assigns node ranks itself, so the same command runs on every node
    and no backend has to number them.

    Args:
        rdzv_endpoint: `host:port` of the rendezvous store, normally node 0.
        standalone: Single-node mode with a private store; `rdzv_endpoint` is ignored.
    """
    launcher = ["torchrun", f"--nproc-per-node={spec.nproc_per_node}"]
    if standalone:
        launcher.append("--standalone")  # noqa: NAR001
    else:
        launcher += [
            f"--nnodes={spec.nodes}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={rdzv_endpoint}",
            f"--rdzv-id={spec.name}",
            f"--max-restarts={spec.max_restarts}",
        ]
    return [*launcher, "-m", spec.module, *spec.args]


def launch_env(spec: JobSpec, backend: str, job_id: str) -> dict[str, str]:
    """Return the spec's env plus what the training process records as launch telemetry."""
    return {
        **spec.env,
        "DAWNSHARD_BACKEND": backend,
        "DAWNSHARD_IMAGE": spec.image,
        "DAWNSHARD_RUN_ID": job_id,
        "DAWNSHARD_SUBMIT_TIME": f"{time.time():.3f}",
    }


def shell_join(command: list[str]) -> str:
    """Quote `command` for `sh -c`."""
    return shlex.join(split_command=command)


def run_cost(
    hourly_rates: list[float],
    hours: float,
    control_plane_per_hour: float = 0.0,
    rework_fraction: float = 0.0,
) -> float:
    """Return the dollars a run costs: instances, plus control plane, plus rework.

    Args:
        hourly_rates: $/hour of each instance the run holds, on-demand or spot.
        rework_fraction: Share of the run repeated after preemptions; spot only pays
            when this stays small, which is what checkpoints are for.
    """
    instances = sum(hourly_rates) * hours * (1 + rework_fraction)  # noqa: NAR001
    return instances + control_plane_per_hour * hours
