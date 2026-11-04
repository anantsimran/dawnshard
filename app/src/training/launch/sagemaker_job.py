"""Run a job as a SageMaker training job, through boto3's `CreateTrainingJob`.

`render_training_job` turns a `JobSpec` into the request: the spec's image as a
bring-your-own container whose entrypoint is `training.launch.sagemaker_entry`, the
data URI as the `train` input channel, the checkpoint URI synced to
`/opt/ml/checkpoints`, and managed spot when asked. SageMaker builds a fresh cluster
for every job, so there is nothing to tear down afterwards.

A stopped spot job resumes only through its checkpoint location: a new job with the
same `checkpoint_uri` starts with that directory filled. The training module has to
look there; the job itself restarts from scratch.

`client` is a boto3 `sagemaker` client, injected so tests run against a fake.
"""

import importlib
from collections.abc import Iterator
from typing import Any

from training.launch.backend import JobState, launch_env
from training.launch.spec import JobSpec

CHECKPOINT_DIR = "/opt/ml/checkpoints"
SAGEMAKER_STATES = {
    "InProgress": JobState.RUNNING,
    "Completed": JobState.SUCCEEDED,
    "Failed": JobState.FAILED,
    "Stopping": JobState.CANCELLED,
    "Stopped": JobState.CANCELLED,
}


def render_training_job(
    spec: JobSpec, role_arn: str, instance_type: str, spot: bool, max_run_s: int = 3600
) -> dict[str, Any]:
    """Return the `CreateTrainingJob` request for `spec`.

    Args:
        spot: Use managed spot. The maximum wait is set to twice the maximum run, and
            must be at least the run, or SageMaker rejects the request.

    Raises:
        ValueError: If the data or checkpoint URI is not on S3.
    """
    on_s3 = [uri.startswith("s3://") for uri in (spec.data_uri, spec.checkpoint_uri)]  # noqa: NAR001
    if not all(on_s3):  # noqa: NAR001
        raise ValueError("SageMaker reads data and checkpoints from s3:// URIs")  # noqa: NAR001
    env = launch_env(spec=spec, backend="sagemaker", job_id=spec.name)
    stopping: dict[str, int] = {"MaxRuntimeInSeconds": max_run_s}
    if spot:
        stopping["MaxWaitTimeInSeconds"] = 2 * max_run_s
    return {
        "TrainingJobName": spec.name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": spec.image,
            "TrainingInputMode": "FastFile",
            "ContainerEntrypoint": ["python", "-m", "training.launch.sagemaker_entry"],
            "ContainerArguments": [str(spec.nproc_per_node), spec.module, *spec.args],  # noqa: NAR001
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": spec.data_uri}},
            }
        ],
        "OutputDataConfig": {"S3OutputPath": f"{spec.checkpoint_uri.rstrip('/')}/output"},  # noqa: NAR001
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": spec.nodes,
            "VolumeSizeInGB": 50,
        },
        "StoppingCondition": stopping,
        "EnableManagedSpotTraining": spot,
        "CheckpointConfig": {"S3Uri": spec.checkpoint_uri, "LocalPath": CHECKPOINT_DIR},
        "Environment": env,
    }


class SageMakerBackend:
    """Training jobs on SageMaker in the client's region."""

    name = "sagemaker"

    def __init__(
        self,
        role_arn: str,
        instance_type: str,
        spot: bool,
        client: Any = None,
        logs_client: Any = None,
    ) -> None:
        """Bind the role, instance type, and clients; default clients come from boto3."""
        if client is None or logs_client is None:
            boto3 = importlib.import_module(name="boto3")
            client = client or boto3.client("sagemaker")  # noqa: NAR001
            logs_client = logs_client or boto3.client("logs")  # noqa: NAR001
        self.client, self.logs_client = client, logs_client
        self.role_arn, self.instance_type, self.spot = role_arn, instance_type, spot

    def submit(self, spec: JobSpec) -> str:
        """Validate and create the training job; the job id is its name."""
        spec.validate()
        request = render_training_job(
            spec=spec, role_arn=self.role_arn, instance_type=self.instance_type, spot=self.spot
        )
        self.client.create_training_job(**request)
        return spec.name

    def status(self, job_id: str) -> JobState:
        """Map `TrainingJobStatus` onto `JobState`."""
        job = self.client.describe_training_job(TrainingJobName=job_id)
        return SAGEMAKER_STATES.get(job["TrainingJobStatus"], JobState.UNKNOWN)  # noqa: NAR001

    def logs(self, job_id: str) -> Iterator[str]:
        """Yield every log stream's messages for the job from CloudWatch."""
        group = "/aws/sagemaker/TrainingJobs"
        streams = self.logs_client.describe_log_streams(
            logGroupName=group, logStreamNamePrefix=job_id
        )
        for stream in streams["logStreams"]:
            events = self.logs_client.get_log_events(
                logGroupName=group, logStreamName=stream["logStreamName"], startFromHead=True
            )
            yield from (event["message"] for event in events["events"])

    def cancel(self, job_id: str) -> None:
        """Stop the training job."""
        self.client.stop_training_job(TrainingJobName=job_id)
