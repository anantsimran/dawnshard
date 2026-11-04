"""Run a job on KubeRay as a RayJob whose entrypoint is Ray Train's `TorchTrainer`.

`render_rayjob` turns a `JobSpec` into a RayJob: a Ray cluster of one head and
`nodes` workers from the spec's image, created for the job and shut down after it,
and an entrypoint that runs `training.launch.ray_train`. That module starts a
`TorchTrainer` with one worker per process; Ray Train creates the process group and
restarts failed workers from the last checkpoint in `storage_path`, which is why the
spec's checkpoint URI is required here. The training module sees an initialized
process group and does not create its own.

Status, logs, and cancel go through `kubectl`, as for JobSets.
"""

from collections.abc import Iterator
from typing import Any

from training.launch.backend import JobState, launch_env, shell_join
from training.launch.k8s import GPU_RESOURCE, Kubectl
from training.launch.spec import JobSpec

# RayJob.status.jobStatus values, from the Ray job API.
RAY_STATES = {
    "PENDING": JobState.PENDING,
    "RUNNING": JobState.RUNNING,
    "SUCCEEDED": JobState.SUCCEEDED,
    "FAILED": JobState.FAILED,
    "STOPPED": JobState.CANCELLED,
}


def ray_container(spec: JobSpec, role: str, job_id: str) -> dict[str, Any]:
    """Return a Ray head or worker container; Ray starts its own processes in it."""
    limits: dict[str, Any] = {"cpu": str(spec.cpus_per_node), "memory": f"{spec.memory_gb}Gi"}  # noqa: NAR001
    if role == "worker" and spec.gpus_per_node:
        limits[GPU_RESOURCE] = spec.gpus_per_node
    env = launch_env(spec=spec, backend="ray", job_id=job_id)
    env.pop("DAWNSHARD_SUBMIT_TIME")  # noqa: NAR001
    return {
        "name": f"ray-{role}",
        "image": spec.image,
        "env": [{"name": key, "value": value} for key, value in sorted(env.items())],  # noqa: NAR001
        "resources": {"limits": limits, "requests": dict(limits)},  # noqa: NAR001
        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
    }


def group_template(spec: JobSpec, role: str) -> dict[str, Any]:
    """Return a head or worker group's pod template."""
    return {
        "spec": {
            "containers": [ray_container(spec=spec, role=role, job_id=spec.name)],
            "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
        }
    }


def render_rayjob(spec: JobSpec) -> dict[str, Any]:
    """Return the RayJob manifest for `spec`.

    Raises:
        ValueError: If `spec.checkpoint_uri` is empty; Ray Train needs shared storage
            to resume a restarted worker.
    """
    if not spec.checkpoint_uri:
        raise ValueError("a RayJob needs checkpoint_uri as Ray Train's storage_path")  # noqa: NAR001
    entrypoint = [
        "python", "-m", "training.launch.ray_train",
        f"--num-workers={spec.world_size}",
        f"--use-gpu={int(spec.gpus_per_node > 0)}",  # noqa: NAR001
        f"--storage-path={spec.checkpoint_uri}",
        f"--max-failures={spec.max_restarts}",
        "--module", spec.module, "--", *spec.args,
    ]  # fmt: skip
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {"name": spec.name, "labels": {"app.kubernetes.io/managed-by": "dawnshard"}},
        "spec": {
            "entrypoint": shell_join(command=entrypoint),
            "shutdownAfterJobFinishes": True,
            "submissionMode": "K8sJobMode",
            "rayClusterSpec": {
                "headGroupSpec": {
                    "rayStartParams": {},
                    "template": group_template(spec=spec, role="head"),
                },
                "workerGroupSpecs": [
                    {
                        "groupName": "workers",
                        "replicas": spec.nodes,
                        "minReplicas": spec.nodes,
                        "maxReplicas": spec.nodes,
                        "rayStartParams": {},
                        "template": group_template(spec=spec, role="worker"),
                    }
                ],
            },
        },
    }


class RayBackend:
    """RayJobs on a cluster with the KubeRay operator installed."""

    name = "ray"

    def __init__(self, kubectl: Kubectl | None = None) -> None:
        """Use `kubectl`, or a default one on the current context."""
        self.kubectl = kubectl or Kubectl()

    def submit(self, spec: JobSpec) -> str:
        """Validate, render, and apply the RayJob; the job id is its name."""
        spec.validate()
        self.kubectl.apply(manifest=render_rayjob(spec=spec))
        return spec.name

    def status(self, job_id: str) -> JobState:
        """Map `status.jobStatus` onto `JobState`; no status yet is pending."""
        status = self.kubectl.get_json(kind="rayjob", name=job_id).get("status", {})  # noqa: NAR001
        return RAY_STATES.get(status.get("jobStatus", "PENDING"), JobState.UNKNOWN)  # noqa: NAR001

    def logs(self, job_id: str) -> Iterator[str]:
        """Yield the submitter Job's log, which streams the Ray job's driver output."""
        yield from self.kubectl.run(args=["logs", f"job/{job_id}"]).splitlines()

    def cancel(self, job_id: str) -> None:
        """Delete the RayJob; KubeRay tears down its cluster."""
        self.kubectl.run(args=["delete", "rayjob", job_id, "--wait=false"])
