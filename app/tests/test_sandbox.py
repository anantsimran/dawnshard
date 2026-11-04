import subprocess
import sys
import time
from pathlib import Path

import pytest
from training.env.sandbox import DockerSandbox, Limits, SubprocessSandbox, WarmPool


@pytest.fixture
def sandbox():
    box = SubprocessSandbox()
    box.start()
    yield box
    box.destroy()


def _alive(pid: int) -> bool:
    """A zombie counts as dead: it runs nothing and only waits to be reaped."""
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return False
    return "\tZ" not in status.read_text().split("State:")[1].splitlines()[0]


def test_runs_files_and_captures_output(sandbox):
    result = sandbox.run(
        files={"main.py": "import sys\nprint('hi')\nprint('err', file=sys.stderr)\nsys.exit(3)"},
        command=[sys.executable, "main.py"],
        limits=Limits(),
    )
    assert (result.exit_code, result.stdout, result.stderr) == (3, "hi\n", "err\n")
    assert not result.timed_out


def test_timeout_escalates_to_sigkill_of_the_whole_group(sandbox):
    code = (
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child = subprocess.Popen(['sleep', '100'])\n"
        "open('child.pid', 'w').write(str(child.pid))\n"
        "time.sleep(100)\n"
    )
    start = time.perf_counter()
    result = sandbox.run(
        files={"main.py": code},
        command=[sys.executable, "main.py"],
        limits=Limits(timeout_s=1.0, kill_grace_s=0.5),
    )
    assert result.timed_out
    assert result.exit_code == 137
    assert time.perf_counter() - start < 5
    child_pid = int((sandbox.work_dir / "child.pid").read_text())
    time.sleep(0.2)
    assert not _alive(pid=child_pid)


def test_output_is_capped(sandbox):
    result = sandbox.run(
        files={"main.py": "print('x' * 10**6)"},
        command=[sys.executable, "main.py"],
        limits=Limits(output_bytes=1000),
    )
    assert len(result.stdout) <= 1000


def test_reset_empties_the_work_directory(sandbox):
    sandbox.run(files={"a.txt": "data"}, command=["true"], limits=Limits())
    sandbox.reset()
    assert list(sandbox.work_dir.iterdir()) == []


def test_run_before_start_reports_start_failure():
    result = SubprocessSandbox().run(files={}, command=["true"], limits=Limits())
    assert result.start_failed


def test_docker_start_command_isolates_the_container():
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="abc123\n", stderr="")

    box = DockerSandbox(runtime="runsc", image="python:3.12-slim", limits=Limits(), runner=runner)
    box.start()
    command = calls[0]
    for flag in ("--runtime=runsc", "--network=none", "--read-only", "--cap-drop=ALL",
                 "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=512m"):
        assert flag in command
    assert "--privileged" not in command
    assert not any("docker.sock" in part for part in command)
    assert box.container_id == "abc123"


def test_docker_run_reports_timeout_and_cgroup_oom_kill():
    oom_counts = iter(["oom_kill 0\n", "oom_kill 1\n"])

    def runner(args, **kwargs):
        if "/sys/fs/cgroup/memory.events" in args:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=next(oom_counts), stderr="")
        return subprocess.CompletedProcess(args=args, returncode=124, stdout="", stderr="")

    box = DockerSandbox(runtime="runc", image="img", limits=Limits(), runner=runner)
    box.container_id = "c1"
    result = box.run(files={"m.py": "x"}, command=["python", "m.py"], limits=Limits())
    assert result.timed_out
    assert result.oom_killed


def test_warm_pool_reuses_and_resets():
    pool = WarmPool(factory=SubprocessSandbox, size=1)
    first = pool.acquire()
    first.run(files={"a.txt": "data"}, command=["true"], limits=Limits())
    pool.release(sandbox=first)
    second = pool.acquire()
    assert second is first
    assert list(second.work_dir.iterdir()) == []
    second.destroy()
