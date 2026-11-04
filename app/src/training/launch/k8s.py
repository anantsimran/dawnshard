"""Run a job on Kubernetes as a JobSet, and read its state, logs, and cold start.

`render_jobset` turns a `JobSpec` into manifests; nothing is hand-edited, and the
rendered YAML is checked against golden files. One replicated Job of `nodes` indexed
pods, each running `torchrun` with c10d rendezvous at pod 0's stable DNS name, which
JobSet's headless Service provides (`enableDNSHostnames`). JobSet's failure policy
restarts the whole group when any pod fails, which is what `torchrun` needs: a
half-dead process group cannot continue.

Pods wait for pod 0's DNS record before starting `torchrun`, because rendezvous that
starts before the headless Service publishes it fails instead of retrying. `/dev/shm`
is a memory-backed emptyDir, because the container default is 64 MB and DataLoader
workers and NCCL both use it. GPU jobs request `nvidia.com/gpu` and tolerate the
GPU taint. A spec with a `queue` gets the Kueue label and starts suspended, so Kueue
admits the whole JobSet or none of it.

Every call goes through `kubectl`, injected as `runner` so tests can see the exact
commands without a cluster.
"""

import json
import subprocess
import time
from collections.abc import Callable, Iterator
from typing import Any

import yaml
from training.launch.backend import RDZV_PORT, JobState, launch_env, shell_join, torchrun_command
from training.launch.spec import JobSpec

type Runner = Callable[..., subprocess.CompletedProcess]
REPLICATED_JOB = "workers"
GPU_RESOURCE = "nvidia.com/gpu"


def leader_host(name: str) -> str:
    """Pod 0's DNS name under JobSet's headless Service: `<js>-<rjob>-0-0.<js>`."""
    return f"{name}-{REPLICATED_JOB}-0-0.{name}"


def container(spec: JobSpec, job_id: str, submit_time: str | None) -> dict[str, Any]:
    """Return the training container: wait for leader DNS, then `torchrun`.

    Args:
        submit_time: Recorded as launch telemetry; None leaves it out, which keeps
            the rendered manifest deterministic for golden-file tests.
    """
    host = leader_host(name=spec.name)
    command = torchrun_command(spec=spec, rdzv_endpoint=f"{host}:{RDZV_PORT}")
    script = f"until getent hosts {host}; do sleep 1; done; exec {shell_join(command=command)}"
    limits: dict[str, Any] = {"cpu": str(spec.cpus_per_node), "memory": f"{spec.memory_gb}Gi"}  # noqa: NAR001
    if spec.gpus_per_node:
        limits[GPU_RESOURCE] = spec.gpus_per_node
    env = launch_env(spec=spec, backend="k8s", job_id=job_id)
    if submit_time is None:
        env.pop("DAWNSHARD_SUBMIT_TIME")  # noqa: NAR001
    else:
        env["DAWNSHARD_SUBMIT_TIME"] = submit_time
    return {
        "name": "trainer",
        "image": spec.image,
        "command": ["sh", "-c", script],
        "env": [{"name": key, "value": value} for key, value in sorted(env.items())],  # noqa: NAR001
        "ports": [{"containerPort": RDZV_PORT}],
        "resources": {"limits": limits, "requests": dict(limits)},  # noqa: NAR001
        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
    }


def pod_spec(spec: JobSpec, job_id: str, submit_time: str | None) -> dict[str, Any]:
    """Return the pod spec: one trainer container, memory-backed /dev/shm, GPU toleration."""
    pod: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container(spec=spec, job_id=job_id, submit_time=submit_time)],
        "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
    }
    if spec.gpus_per_node:
        pod["tolerations"] = [{"key": GPU_RESOURCE, "operator": "Exists", "effect": "NoSchedule"}]
    return pod


def render_jobset(spec: JobSpec, submit_time: str | None = None) -> dict[str, Any]:
    """Return the JobSet manifest for `spec`; the job id is the spec name."""
    labels = {"app.kubernetes.io/managed-by": "dawnshard"}
    if spec.queue:
        labels["kueue.x-k8s.io/queue-name"] = spec.queue
    job = {
        "parallelism": spec.nodes,
        "completions": spec.nodes,
        "completionMode": "Indexed",
        "backoffLimit": 0,
        "template": {"spec": pod_spec(spec=spec, job_id=spec.name, submit_time=submit_time)},
    }
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {"name": spec.name, "labels": labels},
        "spec": {
            "suspend": bool(spec.queue),  # noqa: NAR001
            "network": {"enableDNSHostnames": True},
            "failurePolicy": {"maxRestarts": spec.max_restarts},
            "replicatedJobs": [{"name": REPLICATED_JOB, "replicas": 1, "template": {"spec": job}}],
        },
    }


def to_yaml(manifest: dict[str, Any]) -> str:
    """Dump a manifest as YAML with keys in insertion order."""
    return yaml.safe_dump(data=manifest, sort_keys=False)


def jobset_state(jobset: dict[str, Any]) -> JobState:
    """Map a JobSet object's status onto `JobState`."""
    status = jobset.get("status", {})  # noqa: NAR001
    conditions = {c["type"]: c["status"] == "True" for c in status.get("conditions", [])}  # noqa: NAR001
    if conditions.get("Completed"):  # noqa: NAR001
        return JobState.SUCCEEDED
    if conditions.get("Failed"):  # noqa: NAR001
        return JobState.FAILED
    if conditions.get("Suspended") or jobset.get("spec", {}).get("suspend"):  # noqa: NAR001
        return JobState.PENDING
    active = sum(job.get("active", 0) for job in status.get("replicatedJobsStatus", []))  # noqa: NAR001
    return JobState.RUNNING if active else JobState.PENDING


def cold_start(events: list[dict[str, Any]]) -> dict[str, str]:
    """Return the first time each cold-start phase happened, from `kubectl get events`.

    Phases by event reason: Scheduled (pod placed), Pulling and Pulled (image), Started
    (container running). Process-group init and the first step are recorded by the
    training process itself.
    """
    first: dict[str, str] = {}
    for event in sorted(events, key=lambda e: e.get("firstTimestamp") or e.get("eventTime") or ""):  # noqa: NAR001
        reason = event.get("reason")  # noqa: NAR001
        stamp = event.get("firstTimestamp") or event.get("eventTime")  # noqa: NAR001
        if reason in ("Scheduled", "Pulling", "Pulled", "Started") and reason not in first:
            first[reason] = str(stamp)  # noqa: NAR001
    return first


class Kubectl:
    """Thin `kubectl` wrapper; `runner` stands in for `subprocess.run`."""

    def __init__(self, runner: Runner = subprocess.run, namespace: str = "default") -> None:
        """Bind the runner and namespace."""
        self.runner = runner
        self.namespace = namespace

    def run(self, args: list[str], stdin: str | None = None) -> str:
        """Run `kubectl -n <ns> <args>` and return stdout.

        Raises:
            subprocess.CalledProcessError: If kubectl fails.
        """
        result = self.runner(
            args=["kubectl", "-n", self.namespace, *args],
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def apply(self, manifest: dict[str, Any]) -> None:
        """`kubectl apply -f -` the manifest."""
        self.run(args=["apply", "-f", "-"], stdin=to_yaml(manifest=manifest))

    def get_json(self, kind: str, name: str) -> dict[str, Any]:
        """Return one object as parsed JSON."""
        return json.loads(s=self.run(args=["get", kind, name, "-o", "json"]))


class K8sBackend:
    """JobSets on whatever cluster the current kube context points at."""

    name = "k8s"

    def __init__(self, kubectl: Kubectl | None = None) -> None:
        """Use `kubectl`, or a default one on the current context."""
        self.kubectl = kubectl or Kubectl()

    def submit(self, spec: JobSpec) -> str:
        """Validate, render, and apply the JobSet; the job id is its name."""
        spec.validate()
        self.kubectl.apply(manifest=render_jobset(spec=spec, submit_time=f"{time.time():.3f}"))
        return spec.name

    def status(self, job_id: str) -> JobState:
        """Read the JobSet and map its conditions."""
        return jobset_state(jobset=self.kubectl.get_json(kind="jobset", name=job_id))

    def logs(self, job_id: str) -> Iterator[str]:
        """Yield every pod's log lines, prefixed with the pod name."""
        selector = f"jobset.sigs.k8s.io/jobset-name={job_id}"
        yield from self.kubectl.run(
            args=["logs", "-l", selector, "--all-containers", "--prefix"]
        ).splitlines()

    def cancel(self, job_id: str) -> None:
        """Delete the JobSet and its pods."""
        self.kubectl.run(args=["delete", "jobset", job_id, "--wait=false"])

    def cold_start(self, job_id: str) -> dict[str, str]:
        """Return the cold-start timestamps of the job's pods; see `cold_start`."""
        events = json.loads(s=self.kubectl.run(args=["get", "events", "-o", "json"]))["items"]  # noqa: NAR001
        mine = [e for e in events if e.get("involvedObject", {}).get("name", "").startswith(job_id)]  # noqa: NAR001
        return cold_start(events=mine)
