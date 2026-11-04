"""Run a job as a local `torchrun --standalone` subprocess.

The baseline every other backend is compared against: no image, no scheduler, no
network; the cold start is interpreter start plus process-group init. The image in
the spec is ignored, so validation does not require a digest. Logs go to a file
under the history directory, so `logs` can replay them after the process ends.
"""

import os
import signal
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from training.constants import HISTORY_PATH, REPO_ROOT
from training.launch.backend import JobState, launch_env, torchrun_command
from training.launch.spec import JobSpec

LOGS_PATH = HISTORY_PATH / "launch_logs"


class LocalBackend:
    """Jobs as child processes of this one, each in its own process group."""

    name = "local"

    def __init__(
        self, logs_dir: Path = LOGS_PATH, command_prefix: tuple[str, ...] = ("uv", "run")
    ) -> None:
        """Keep each job's log under `logs_dir`.

        Args:
            command_prefix: Prepended to the `torchrun` command, so it runs in the
                project's environment. Tests pass `()` and a command of their own.
        """
        self.logs_dir = logs_dir
        self.command_prefix = command_prefix
        self.processes: dict[str, subprocess.Popen] = {}

    def command(self, spec: JobSpec) -> list[str]:
        """Return the full command `submit` runs for `spec`."""
        return [
            *self.command_prefix,
            *torchrun_command(spec=spec, rdzv_endpoint="", standalone=True),
        ]

    def submit(self, spec: JobSpec, command: list[str] | None = None) -> str:
        """Start `spec` (or an explicit `command`) and return its job id.

        Side effects:
            Starts a process group and creates its log file.
        """
        spec.validate(require_digest=False)
        job_id = f"{spec.name}-{uuid.uuid4().hex[:8]}"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT / "app" / "src"),  # noqa: NAR001
            **launch_env(spec=spec, backend=self.name, job_id=job_id),
        }  # noqa: NAR001
        with open(file=self.logs_dir / f"{job_id}.log", mode="wb") as log:
            self.processes[job_id] = subprocess.Popen(
                args=command or self.command(spec=spec),
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return job_id

    def status(self, job_id: str) -> JobState:
        """Map the process's return code onto `JobState`."""
        returncode = self.processes[job_id].poll()
        if returncode is None:
            return JobState.RUNNING
        if returncode == 0:
            return JobState.SUCCEEDED
        return JobState.CANCELLED if returncode == -signal.SIGTERM else JobState.FAILED

    def logs(self, job_id: str) -> Iterator[str]:
        """Yield log lines as they are written, until the process exits."""
        with open(file=self.logs_dir / f"{job_id}.log", encoding="utf-8", errors="replace") as log:
            while True:
                line = log.readline()
                if line:
                    yield line.rstrip("\n")  # noqa: NAR001
                elif self.processes[job_id].poll() is not None:
                    yield from (rest.rstrip("\n") for rest in log.readlines())  # noqa: NAR001
                    return
                else:
                    time.sleep(0.1)  # noqa: NAR001

    def cancel(self, job_id: str) -> None:
        """SIGTERM the job's process group and wait for it."""
        process = self.processes[job_id]
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)  # noqa: NAR001
            process.wait()
