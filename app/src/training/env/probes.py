"""Adversarial programs that check what a sandbox actually contains.

Each probe prints `ESCAPED` when it gets something a sandboxed program should not:
a network connection, the host's canary file, other tenants' processes, or a host
that is still responsive after a resource bomb. A probe is contained when its output
has no `ESCAPED` and the sandbox still answers afterwards.

Probes that run as a submission (test tampering) live with the verifier's tests
instead, because what they attack is the reward, not the sandbox.
"""

from dataclasses import dataclass

from training.env.sandbox import Limits, Sandbox


@dataclass(frozen=True)
class Probe:
    """One adversarial program.

    Attributes:
        host_safe: Whether it may run on the unisolated subprocess runtime. A fork
            bomb as root has no pid limit there and would take the host down.
    """

    name: str
    expected: str
    code: str
    host_safe: bool = True


def probes(canary_path: str) -> list[Probe]:
    """Return the probe suite; `canary_path` is a host file no sandbox should read."""
    return [
        Probe(
            name="network_egress",
            expected="connection refused or unreachable",
            code=(
                "import socket\n"
                "s = socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
                "print('ESCAPED')"
            ),
        ),
        Probe(
            name="host_file_read",
            expected="canary not visible",
            code=f"print('ESCAPED', open({canary_path!r}).read())",
        ),
        Probe(
            name="proc_snooping",
            expected="only the sandbox's own processes in /proc",
            code=(
                "import os\npids = [p for p in os.listdir('/proc') if p.isdigit()]\n"
                "print('ESCAPED' if len(pids) > 10 else 'contained', len(pids))"
            ),
        ),
        Probe(
            name="memory_bomb",
            expected="MemoryError or OOM kill within the limit",
            code="x = bytearray(16 * 2**30)\nprint('ESCAPED')",
        ),
        Probe(
            name="fork_bomb",
            expected="pids limit stops it; sandbox still answers",
            code="import os\nwhile True:\n    os.fork()",
            host_safe=False,
        ),
    ]


@dataclass(frozen=True)
class ProbeOutcome:
    """What one probe did: its exit code, a short observation, and the verdict."""

    name: str
    expected: str
    observed: str
    contained: bool


def run_probe(probe: Probe, sandbox: Sandbox, limits: Limits) -> ProbeOutcome:
    """Run `probe`, then check the sandbox still runs a trivial command.

    Side effects:
        Resets `sandbox` afterwards, killing anything the probe left running.
    """
    result = sandbox.run(
        files={"probe.py": probe.code}, command=[sandbox.python, "probe.py"], limits=limits
    )
    sandbox.reset()
    alive = sandbox.run(files={}, command=[sandbox.python, "-c", "print('ok')"], limits=limits)
    escaped = "ESCAPED" in result.stdout
    tail = (result.stderr.strip().splitlines() or [result.stdout.strip()])[-1][:120]
    observed = f"exit {result.exit_code}{', timed out' if result.timed_out else ''}: {tail}"
    return ProbeOutcome(
        name=probe.name,
        expected=probe.expected,
        observed=observed,
        contained=not escaped and alive.stdout.strip() == "ok",
    )
