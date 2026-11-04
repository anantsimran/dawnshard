import sys

from training.launch.backend import JobState
from training.launch.local import LocalBackend
from training.launch.spec import JobSpec

SPEC = JobSpec(name="fake", image="local", module="training.model.qwen3", nodes=1, procs_per_node=2)


def test_command_is_standalone_torchrun_with_one_proc_per_rank(tmp_path):
    command = LocalBackend(logs_dir=tmp_path, command_prefix=()).command(spec=SPEC)
    assert command[:3] == ["torchrun", "--nproc-per-node=2", "--standalone"]


def test_streams_logs_of_a_fake_command_and_reports_success(tmp_path):
    backend = LocalBackend(logs_dir=tmp_path, command_prefix=())
    code = "import os; print('rank ready'); print(os.environ['DAWNSHARD_BACKEND'])"
    job_id = backend.submit(spec=SPEC, command=[sys.executable, "-c", code])
    assert list(backend.logs(job_id=job_id)) == ["rank ready", "local"]
    assert backend.status(job_id=job_id) == JobState.SUCCEEDED


def test_cancel_marks_the_job_cancelled(tmp_path):
    backend = LocalBackend(logs_dir=tmp_path, command_prefix=())
    job_id = backend.submit(spec=SPEC, command=[sys.executable, "-c", "import time; time.sleep(30)"])
    backend.cancel(job_id=job_id)
    assert backend.status(job_id=job_id) == JobState.CANCELLED
