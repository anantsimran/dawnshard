"""Sandboxes that run untrusted programs under limits, behind one interface.

Every runtime implements `Sandbox`: `start`, `run` files and a command under `Limits`,
`reset` to a clean work directory, and `destroy`. `run` never raises for anything the
program does; it reports it in `RunResult`, so the failure classifier can tell a
program that misbehaved from infrastructure that did.

- `SubprocessSandbox` runs on the host with rlimits and a process-group kill. It is
  the unsafe baseline: it has no network, filesystem, or pid isolation.
- `DockerSandbox` runs each sandbox as a long-lived container with no network, a
  read-only root, a tmpfs work directory, cgroup limits, all capabilities dropped,
  and `no-new-privileges`. `runtime="runc"` shares the host kernel;
  `runtime="runsc"` puts gVisor's user-space kernel between the program and the
  host.
- `WarmPool` keeps started sandboxes ready and resets them between episodes.
"""

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Exit code a process reports when SIGKILL ends it, as shells and Docker print it.
SIGKILL_EXIT_CODE = 128 + signal.SIGKILL
WORK_DIR = "/work"
SANDBOX_USER = "65534:65534"  # nobody


@dataclass(frozen=True)
class Limits:
    """Per-command limits. `timeout_s` is wall time; `cpu_seconds` is CPU time."""

    timeout_s: float = 10.0
    cpu_seconds: int = 10
    memory_mb: int = 512
    pids: int = 64
    output_bytes: int = 64 * 1024
    kill_grace_s: float = 1.0


@dataclass
class RunResult:
    """What one command did. `run` fills this instead of raising."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    oom_killed: bool = False
    start_failed: bool = False


class Sandbox(Protocol):
    """One isolated place to run programs, reusable across commands.

    Attributes:
        python: The interpreter command inside the sandbox.
    """

    python: str

    def start(self) -> None:
        """Make the sandbox ready to run commands."""
        ...

    def run(
        self, files: dict[str, str], command: list[str], limits: Limits, stdin: str = ""
    ) -> RunResult:
        """Write `files` into the work directory, then run `command` there."""
        ...

    def reset(self) -> None:
        """Empty the work directory and end every process left running."""
        ...

    def destroy(self) -> None:
        """Release everything the sandbox holds."""
        ...


def _read_capped(path: Path, limit: int) -> str:
    """Return at most `limit` bytes of `path`, decoded with replacement."""
    with open(file=path, mode="rb") as handle:
        return handle.read(limit).decode(encoding="utf-8", errors="replace")  # noqa: NAR001


def _kill_group(process: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM the process group, then SIGKILL it if it outlives `grace_s`.

    Killing the group, not the pid, also ends every child the program forked.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)  # noqa: NAR001
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_s)
            if sig == signal.SIGKILL:
                return
        except subprocess.TimeoutExpired:
            continue
    # The leader may have exited while children ignore SIGTERM; SIGKILL the group.
    try:
        os.killpg(process.pid, signal.SIGKILL)  # noqa: NAR001
    except ProcessLookupError:
        pass


class SubprocessSandbox:
    """Run commands as host processes in a temporary directory. Unsafe baseline.

    Limits are rlimits (address space, CPU time, file size) applied in the child
    before exec, plus a wall-clock timeout that kills the whole process group. Output
    goes to files capped by RLIMIT_FSIZE, so a program cannot fill host memory by
    printing. Nothing stops it reading host files or opening sockets.
    """

    def __init__(self) -> None:
        """Create no directory until `start`; run the host's own interpreter."""
        self.python = sys.executable
        self.work_dir: Path | None = None

    def start(self) -> None:
        """Create the work directory.

        Side effects:
            Creates a temporary directory.
        """
        self.work_dir = Path(tempfile.mkdtemp(prefix="sandbox-"))  # noqa: NAR001

    def run(
        self, files: dict[str, str], command: list[str], limits: Limits, stdin: str = ""
    ) -> RunResult:
        """Run `command` in the work directory; see `Sandbox.run`."""
        if self.work_dir is None:
            return RunResult(
                exit_code=-1, stdout="", stderr="not started", duration_s=0.0, start_failed=True
            )
        for name, content in files.items():
            (self.work_dir / name).write_text(data=content, encoding="utf-8")
        out_path, err_path = self.work_dir / ".stdout", self.work_dir / ".stderr"
        start = time.perf_counter()
        with open(file=out_path, mode="wb") as out, open(file=err_path, mode="wb") as err:
            process = subprocess.Popen(
                args=command,
                cwd=self.work_dir,
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                start_new_session=True,
                preexec_fn=lambda: _apply_rlimits(limits=limits),
            )
            timed_out = False
            try:
                process.communicate(input=stdin.encode(), timeout=limits.timeout_s)  # noqa: NAR001
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(process=process, grace_s=limits.kill_grace_s)
        return RunResult(
            exit_code=_shell_exit_code(returncode=process.wait()),
            stdout=_read_capped(path=out_path, limit=limits.output_bytes),
            stderr=_read_capped(path=err_path, limit=limits.output_bytes),
            duration_s=time.perf_counter() - start,
            timed_out=timed_out,
        )

    def reset(self) -> None:
        """Replace the work directory with an empty one."""
        self.destroy()
        self.start()

    def destroy(self) -> None:
        """Delete the work directory."""
        if self.work_dir is not None:
            shutil.rmtree(path=self.work_dir, ignore_errors=True)
            self.work_dir = None


def _shell_exit_code(returncode: int) -> int:
    """Map Popen's `-signal` for a signalled child to the shell's `128 + signal`."""
    return 128 - returncode if returncode < 0 else returncode


def _apply_rlimits(limits: Limits) -> None:
    """Set the child's rlimits. Runs between fork and exec, so it must not raise."""
    memory = limits.memory_mb * 2**20
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))  # noqa: NAR001
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))  # noqa: NAR001
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.output_bytes, limits.output_bytes))  # noqa: NAR001


# Runs inside the container: write the JSON {name: content} on stdin into the cwd.
_WRITE_FILES = "import json,sys\nfor n,c in json.load(sys.stdin).items():\n open(n,'w').write(c)"
type Runner = Callable[..., subprocess.CompletedProcess]


class DockerSandbox:
    """A long-lived container per sandbox; each command is a `docker exec`.

    The container runs `sleep infinity` as its main process, so a program that
    crashes or is killed never takes the sandbox with it. Commands run as `nobody`
    inside `WORK_DIR`, a size-capped tmpfs; the root filesystem is read-only. A
    command is wrapped in coreutils `timeout`, which sends SIGTERM and then SIGKILL
    after `kill_grace_s`, and the host-side `docker exec` has a backstop timeout.
    An OOM kill is read from the cgroup's `memory.events` counter, before and after.

    `runner` stands in for `subprocess.run`, so tests can check the exact commands.
    """

    def __init__(
        self, runtime: str, image: str, limits: Limits, runner: Runner = subprocess.run
    ) -> None:
        """Configure, but do not create, the container.

        Args:
            runtime: Docker runtime name: "runc" (default kernel sharing) or "runsc"
                (gVisor), as registered with the Docker daemon.
            limits: Container-wide cgroup limits; each `run` adds its own timeout.
        """
        self.python = "python"
        self.runtime = runtime
        self.image = image
        self.limits = limits
        self.runner = runner
        self.container_id: str | None = None

    def start_command(self) -> list[str]:
        """Return the `docker run` argument list that `start` executes."""
        memory = f"{self.limits.memory_mb}m"
        return [
            "docker", "run", "--detach", "--rm",
            f"--runtime={self.runtime}",
            "--network=none", "--read-only",
            f"--tmpfs={WORK_DIR}:rw,exec,size=64m,mode=1777",
            f"--memory={memory}", f"--memory-swap={memory}",
            "--cpus=1", f"--pids-limit={self.limits.pids}",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            f"--workdir={WORK_DIR}", self.image, "sleep", "infinity",
        ]  # fmt: skip

    def start(self) -> None:
        """Start the container and remember its id; leave it None on failure.

        Side effects:
            Creates a container.
        """
        result = self.runner(args=self.start_command(), capture_output=True, text=True)
        self.container_id = result.stdout.strip() if result.returncode == 0 else None

    def _exec(self, args: list[str], stdin: str, timeout_s: float) -> subprocess.CompletedProcess:
        """`docker exec` `args` as nobody in WORK_DIR, with a host-side backstop timeout."""
        assert self.container_id is not None
        command = ["docker", "exec", "-i", f"--user={SANDBOX_USER}", self.container_id, *args]
        return self.runner(
            args=command, input=stdin, capture_output=True, text=True, timeout=timeout_s
        )

    def _oom_kills(self) -> int:
        """Return the container cgroup's `oom_kill` count."""
        events = self._exec(args=["cat", "/sys/fs/cgroup/memory.events"], stdin="", timeout_s=5.0)
        counts = dict(line.split() for line in events.stdout.splitlines() if line.strip())  # noqa: NAR001
        return int(counts.get("oom_kill", 0))  # noqa: NAR001

    def run(
        self, files: dict[str, str], command: list[str], limits: Limits, stdin: str = ""
    ) -> RunResult:
        """Write `files` and run `command`; see `Sandbox.run`."""
        if self.container_id is None:
            return RunResult(
                exit_code=-1,
                stdout="",
                stderr="container did not start",
                duration_s=0.0,
                start_failed=True,
            )
        self._exec(args=["python", "-c", _WRITE_FILES], stdin=json.dumps(obj=files), timeout_s=10.0)
        oom_before, start = self._oom_kills(), time.perf_counter()
        wrapped = [
            "timeout",
            f"--kill-after={limits.kill_grace_s}",
            str(limits.timeout_s),  # noqa: NAR001
            *command,
        ]  # noqa: NAR001
        try:
            result = self._exec(args=wrapped, stdin=stdin, timeout_s=limits.timeout_s + 10)
        except subprocess.TimeoutExpired:
            return RunResult(
                exit_code=-1,
                stdout="",
                stderr="docker exec hung",
                duration_s=time.perf_counter() - start,
                start_failed=True,
            )
        return RunResult(
            exit_code=result.returncode,
            stdout=result.stdout[: limits.output_bytes],
            stderr=result.stderr[: limits.output_bytes],
            duration_s=time.perf_counter() - start,
            timed_out=result.returncode == 124,
            oom_killed=self._oom_kills() > oom_before,
        )

    def reset(self) -> None:
        """Kill every `nobody` process and empty WORK_DIR; the container stays up."""
        self._exec(args=["kill", "-9", "-1"], stdin="", timeout_s=5.0)
        self._exec(args=["find", WORK_DIR, "-mindepth", "1", "-delete"], stdin="", timeout_s=5.0)

    def destroy(self) -> None:
        """Remove the container.

        Side effects:
            Runs `docker rm --force`.
        """
        if self.container_id is not None:
            self.runner(args=["docker", "rm", "--force", self.container_id], capture_output=True)
            self.container_id = None


class WarmPool:
    """Keep `size` started sandboxes ready; `acquire` takes one, `release` resets it.

    Starting a container costs hundreds of milliseconds; resetting a running one
    costs two `docker exec`s. A pool trades idle memory for that difference. A lock
    guards the idle list, so concurrent episodes can share one pool.
    """

    def __init__(self, factory: Callable[[], Sandbox], size: int) -> None:
        """Start `size` sandboxes from `factory`.

        Side effects:
            Starts `size` sandboxes.
        """
        self.factory = factory
        self.lock = threading.Lock()
        self.idle: list[Sandbox] = []
        for _ in range(size):  # noqa: NAR001
            sandbox = factory()
            sandbox.start()
            self.idle.append(sandbox)  # noqa: NAR001

    def acquire(self) -> Sandbox:
        """Return an idle sandbox, or start a new one when none is idle."""
        with self.lock:
            if self.idle:
                return self.idle.pop()
        sandbox = self.factory()
        sandbox.start()
        return sandbox

    def release(self, sandbox: Sandbox) -> None:
        """Reset `sandbox` and return it to the idle list."""
        sandbox.reset()
        with self.lock:
            self.idle.append(sandbox)  # noqa: NAR001

    def close(self) -> None:
        """Destroy every idle sandbox."""
        for sandbox in self.idle:
            sandbox.destroy()
        self.idle = []
