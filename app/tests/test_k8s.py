from pathlib import Path
import json
import subprocess

import pytest
from training.launch.backend import JobState
from training.launch.k8s import K8sBackend, Kubectl, cold_start, jobset_state, render_jobset, to_yaml
from training.launch.spec import JobSpec

GOLDEN = Path(__file__).parent / "golden"
IMAGE = "ghcr.io/x/dawnshard@sha256:" + "a" * 64
CPU = JobSpec(name="qwen3-cpu", image=IMAGE, module="training.model.qwen3", args=("--steps", "200"), nodes=2)
GPU = JobSpec(
    name="qwen3-gpu", image=IMAGE, module="training.model.qwen3", nodes=2, gpus_per_node=1,
    cpus_per_node=4, memory_gb=16, queue="team-a",
)


@pytest.mark.parametrize(("spec", "golden"), [(CPU, "jobset_cpu.yaml"), (GPU, "jobset_gpu.yaml")])
def test_rendered_jobset_matches_golden(spec, golden):
    assert to_yaml(manifest=render_jobset(spec=spec)) == (GOLDEN / golden).read_text()


def test_gpu_jobset_has_gpu_limit_toleration_shm_and_queue():
    manifest = render_jobset(spec=GPU)
    pod = manifest["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert pod["tolerations"][0]["key"] == "nvidia.com/gpu"
    assert pod["volumes"] == [{"name": "dshm", "emptyDir": {"medium": "Memory"}}]
    assert manifest["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "team-a"
    assert manifest["spec"]["suspend"] is True


def test_cpu_jobset_has_no_gpu_fields():
    pod = render_jobset(spec=CPU)["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    assert "nvidia.com/gpu" not in pod["containers"][0]["resources"]["limits"]
    assert "tolerations" not in pod


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"status": {"conditions": [{"type": "Completed", "status": "True"}]}}, JobState.SUCCEEDED),
        ({"status": {"conditions": [{"type": "Failed", "status": "True"}]}}, JobState.FAILED),
        ({"spec": {"suspend": True}}, JobState.PENDING),
        ({"status": {"replicatedJobsStatus": [{"active": 1}]}}, JobState.RUNNING),
        ({}, JobState.PENDING),
    ],
)
def test_status_mapping(status, expected):
    assert jobset_state(jobset=status) == expected


def test_cold_start_takes_the_first_of_each_phase():
    events = [
        {"reason": "Started", "firstTimestamp": "2026-01-01T00:01:10Z"},
        {"reason": "Scheduled", "firstTimestamp": "2026-01-01T00:00:01Z"},
        {"reason": "Pulled", "firstTimestamp": "2026-01-01T00:01:05Z"},
        {"reason": "Scheduled", "firstTimestamp": "2026-01-01T00:00:03Z"},
        {"reason": "BackOff", "firstTimestamp": "2026-01-01T00:00:02Z"},
    ]
    assert cold_start(events=events) == {
        "Scheduled": "2026-01-01T00:00:01Z",
        "Pulled": "2026-01-01T00:01:05Z",
        "Started": "2026-01-01T00:01:10Z",
    }


def test_backend_applies_reads_and_deletes_through_kubectl():
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs.get("input")))
        stdout = json.dumps({"status": {"replicatedJobsStatus": [{"active": 2}]}}) if "get" in args else ""
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

    backend = K8sBackend(kubectl=Kubectl(runner=runner))
    assert backend.submit(spec=CPU) == "qwen3-cpu"
    assert calls[0][0][-3:] == ["apply", "-f", "-"] and "DAWNSHARD_SUBMIT_TIME" in calls[0][1]
    assert backend.status(job_id="qwen3-cpu") == JobState.RUNNING
    backend.cancel(job_id="qwen3-cpu")
    assert calls[-1][0][3:6] == ["delete", "jobset", "qwen3-cpu"]
