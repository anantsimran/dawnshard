import pytest
from training.launch.backend import JobState
from training.launch.sagemaker_job import SageMakerBackend, render_training_job
from training.launch.spec import JobSpec

IMAGE = "123.dkr.ecr.us-east-1.amazonaws.com/dawnshard@sha256:" + "a" * 64
SPEC = JobSpec(
    name="qwen3-sm", image=IMAGE, module="training.model.qwen3", args=("--steps", "200"),
    nodes=2, gpus_per_node=1, data_uri="s3://b/data/", checkpoint_uri="s3://b/ckpt/",
)


class FakeSageMaker:
    def __init__(self):
        self.requests, self.stopped = [], []

    def create_training_job(self, **request):
        self.requests.append(request)

    def describe_training_job(self, TrainingJobName):
        return {"TrainingJobStatus": "InProgress"}

    def stop_training_job(self, TrainingJobName):
        self.stopped.append(TrainingJobName)


def test_spot_request_has_checkpoints_and_a_long_enough_wait():
    request = render_training_job(spec=SPEC, role_arn="arn:role", instance_type="ml.g5.xlarge", spot=True)
    assert request["EnableManagedSpotTraining"] is True
    stopping = request["StoppingCondition"]
    assert stopping["MaxWaitTimeInSeconds"] >= stopping["MaxRuntimeInSeconds"]
    assert request["CheckpointConfig"] == {"S3Uri": "s3://b/ckpt/", "LocalPath": "/opt/ml/checkpoints"}
    assert request["ResourceConfig"]["InstanceCount"] == 2
    assert request["AlgorithmSpecification"]["ContainerArguments"] == ["1", "training.model.qwen3", "--steps", "200"]


def test_on_demand_request_has_no_wait_time():
    request = render_training_job(spec=SPEC, role_arn="arn:role", instance_type="ml.g5.xlarge", spot=False)
    assert "MaxWaitTimeInSeconds" not in request["StoppingCondition"]


def test_non_s3_uris_are_rejected():
    with pytest.raises(ValueError):
        render_training_job(
            spec=JobSpec(name="x", image=IMAGE, module="m", data_uri="gs://b/d", checkpoint_uri="s3://b/c"),
            role_arn="r", instance_type="i", spot=False,
        )


def test_backend_against_a_fake_client():
    fake = FakeSageMaker()
    backend = SageMakerBackend(role_arn="arn:role", instance_type="ml.g5.xlarge", spot=True, client=fake, logs_client=object())
    assert backend.submit(spec=SPEC) == "qwen3-sm"
    assert fake.requests[0]["TrainingJobName"] == "qwen3-sm"
    assert backend.status(job_id="qwen3-sm") == JobState.RUNNING
    backend.cancel(job_id="qwen3-sm")
    assert fake.stopped == ["qwen3-sm"]
