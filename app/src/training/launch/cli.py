"""`make launch BACKEND=<local|k8s|ray|sagemaker>`: submit a spec and follow it.

Loads a `JobSpec` from JSON, submits it to the named backend, streams its logs, and
exits with the job's final state. `--dry-run` prints the rendered manifest or request
instead of submitting, so a submission can be reviewed before it costs anything.

    uv run python -m training.launch.cli --backend local --spec infra/jobs/qwen3_tiny_local.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from training.launch.backend import Backend, JobState
from training.launch.k8s import K8sBackend, render_jobset, to_yaml
from training.launch.local import LocalBackend
from training.launch.ray_job import RayBackend, render_rayjob
from training.launch.sagemaker_job import SageMakerBackend, render_training_job
from training.launch.spec import JobSpec

TERMINAL = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


def build_backend(name: str) -> Backend:
    """Return the backend `name`; SageMaker reads its role and instance from the environment."""
    if name == "local":
        return LocalBackend()
    if name == "k8s":
        return K8sBackend()
    if name == "ray":
        return RayBackend()
    return SageMakerBackend(
        role_arn=os.environ["SAGEMAKER_ROLE_ARN"],
        instance_type=os.environ.get("SAGEMAKER_INSTANCE_TYPE", "ml.g5.xlarge"),  # noqa: NAR001
        spot=os.environ.get("SAGEMAKER_SPOT", "1") == "1",  # noqa: NAR001
    )


def render(name: str, spec: JobSpec) -> str:
    """Return what `name` would submit for `spec`, as text."""
    if name == "k8s":
        return to_yaml(manifest=render_jobset(spec=spec))
    if name == "ray":
        return to_yaml(manifest=render_rayjob(spec=spec))
    if name == "sagemaker":
        request = render_training_job(
            spec=spec, role_arn="<role>", instance_type="<instance>", spot=True
        )
        return json.dumps(obj=request, indent=2)
    return " ".join(LocalBackend().command(spec=spec))  # noqa: NAR001


def main() -> None:
    """Submit, stream logs, poll to a terminal state, and exit nonzero unless it succeeded."""
    parser = argparse.ArgumentParser(description="Submit a JobSpec to a compute backend.")
    parser.add_argument("--backend", required=True, choices=["local", "k8s", "ray", "sagemaker"])  # noqa: NAR001
    parser.add_argument("--spec", type=Path, required=True)  # noqa: NAR001
    parser.add_argument("--dry-run", action="store_true")  # noqa: NAR001
    args = parser.parse_args()
    spec = JobSpec.from_json(path=args.spec)
    if args.dry_run:
        print(render(name=args.backend, spec=spec))  # noqa: NAR001
        return
    backend = build_backend(name=args.backend)
    job_id = backend.submit(spec=spec)
    print(f"submitted {job_id} to {backend.name}")  # noqa: NAR001
    for line in backend.logs(job_id=job_id):
        print(line)  # noqa: NAR001
    state = backend.status(job_id=job_id)
    while state not in TERMINAL:
        time.sleep(5)  # noqa: NAR001
        state = backend.status(job_id=job_id)
    print(f"{job_id}: {state.value}")  # noqa: NAR001
    sys.exit(0 if state == JobState.SUCCEEDED else 1)  # noqa: NAR001


if __name__ == "__main__":
    main()
