import json
import subprocess

import pytest
from training.launch.backend import JobState
from training.launch.k8s import Kubectl
from training.launch.ray_job import RayBackend, render_rayjob
from training.launch.spec import JobSpec

IMAGE = "ghcr.io/x/dawnshard@sha256:" + "a" * 64
SPEC = JobSpec(
    name="qwen3-ray", image=IMAGE, module="training.model.qwen3", args=("--steps", "200"),
    nodes=2, gpus_per_node=1, checkpoint_uri="s3://b/ckpt",
)


def test_rayjob_runs_torch_trainer_with_one_worker_per_process():
    manifest = render_rayjob(spec=SPEC)
    entrypoint = manifest["spec"]["entrypoint"]
    assert "training.launch.ray_train" in entrypoint
    assert "--num-workers=2" in entrypoint and "--use-gpu=1" in entrypoint
    assert "--storage-path=s3://b/ckpt" in entrypoint
    assert entrypoint.endswith("--module training.model.qwen3 -- --steps 200")
    workers = manifest["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]
    assert workers["replicas"] == 2
    assert manifest["spec"]["shutdownAfterJobFinishes"] is True


def test_rayjob_needs_shared_storage():
    with pytest.raises(ValueError):
        render_rayjob(spec=JobSpec(name="x", image=IMAGE, module="m"))


@pytest.mark.parametrize(
    ("job_status", "expected"),
    [("RUNNING", JobState.RUNNING), ("SUCCEEDED", JobState.SUCCEEDED), ("STOPPED", JobState.CANCELLED), (None, JobState.PENDING)],
)
def test_status_against_a_fake_kubectl(job_status, expected):
    status = {} if job_status is None else {"status": {"jobStatus": job_status}}

    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(status), stderr="")

    assert RayBackend(kubectl=Kubectl(runner=runner)).status(job_id="qwen3-ray") == expected
