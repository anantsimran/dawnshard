"""The verifier: reward 1 only for a genuine pass of tests the model cannot touch.

The verifier keeps each task's tests in its own directory, read-only, with a sha256
taken at registration. `verify` re-hashes the file before every run and refuses a
mismatch, then runs the candidate in a fresh sandbox the agent never had, so nothing
the agent did during the episode can reach the tests.

Inside that sandbox, three defenses catch the common reward hacks:

- **A nonce.** The harness reads a random nonce from stdin before importing the
  solution and prints it only after the last assert. A solution that calls
  `sys.exit(0)` or `os._exit(0)` at import exits 0 without it.
- **Trusted comparisons.** `assert a == b` lines are rewritten to `_check(a, b)`,
  which refuses any value whose exact type, or whose elements' exact types, is not
  on an allowlist of builtin and standard-library value types. An object whose
  `__eq__` always returns True is not on it, and setting a class's `__module__` to
  "builtins" does not change its type.
- **The harness is written last and run by path**, after `solution.py`, so a
  solution cannot replace it on disk before it runs.

What it does not stop: a solution that searches the harness's memory for the nonce
(`gc.get_objects()`) and prints it. In-process checks are defense in depth; the hash
and the fresh sandbox are the boundary.
"""

import ast
import hashlib
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from training.env.failures import FailureClass, InfraError, classify
from training.env.sandbox import Limits, RunResult, Sandbox

HARNESS = """import sys
_nonce = sys.stdin.readline().strip()
_out = sys.stdout


import collections, decimal, fractions
_TRUSTED = {{
    int, float, complex, bool, str, bytes, type(None), list, tuple, dict, set, frozenset,
    collections.Counter, collections.OrderedDict, collections.defaultdict, collections.deque,
    decimal.Decimal, fractions.Fraction,
}}


def _trusted(value):
    if type(value) not in _TRUSTED:
        return False
    if isinstance(value, dict):
        return all(_trusted(k) and _trusted(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset, collections.deque)):
        return all(_trusted(item) for item in value)
    return True


def _check(actual, expected):
    if not (_trusted(actual) and expected == actual):
        raise AssertionError(f"{{actual!r}} != {{expected!r}}")


sys.path.insert(0, ".")
from solution import *  # noqa: F401,F403

{tests}

_out.write(_nonce + "\\n")
_out.flush()
"""


class _AssertToCheck(ast.NodeTransformer):
    """Rewrite `assert a == b` into `_check(a, b)`; leave other asserts alone."""

    def visit_Assert(self, node: ast.Assert) -> ast.AST:  # noqa: NAR002
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):  # noqa: NAR001
            call = ast.Call(
                func=ast.Name(id="_check", ctx=ast.Load()),
                args=[test.left, test.comparators[0]],
                keywords=[],
            )
            return ast.copy_location(new_node=ast.Expr(value=call), old_node=node)
        return node


def harden_tests(source: str) -> str:
    """Return `source` with every `assert a == b` rewritten to `_check(a, b)`."""
    tree = _AssertToCheck().visit(ast.parse(source=source))  # noqa: NAR001
    return ast.unparse(ast_obj=ast.fix_missing_locations(node=tree))


@dataclass(frozen=True)
class Verdict:
    """The reward and why. `reason` is "pass", "fail", or "tests_tampered"."""

    reward: float
    reason: str
    result: RunResult | None = None


class Verifier:
    """Owns each task's tests and turns a submission into a reward."""

    def __init__(
        self, tests_dir: Path, sandbox_factory: Callable[[], Sandbox], limits: Limits
    ) -> None:
        """Keep tests under `tests_dir`; run each check in a new sandbox from `sandbox_factory`."""
        self.tests_dir = tests_dir
        self.sandbox_factory = sandbox_factory
        self.limits = limits
        self.hashes: dict[str, str] = {}

    def _path(self, task_id: str) -> Path:
        """Where `task_id`'s tests live."""
        return self.tests_dir / f"{task_id}.py"

    def register(self, task_id: str, tests: str) -> None:
        """Store hardened `tests` read-only and remember their hash.

        Side effects:
            Writes `tests_dir/<task_id>.py` with mode 0o444.
        """
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(task_id=task_id)
        if path.exists():
            path.chmod(mode=0o644)
        source = harden_tests(source=tests)
        path.write_text(data=source, encoding="utf-8")
        path.chmod(mode=0o444)
        self.hashes[task_id] = hashlib.sha256(source.encode()).hexdigest()  # noqa: NAR001

    def verify(self, task_id: str, code: str) -> Verdict:
        """Run `code` against `task_id`'s tests in a fresh sandbox.

        Raises:
            InfraError: If the sandbox failed rather than the code.
        """
        source = self._path(task_id=task_id).read_text(encoding="utf-8")
        if hashlib.sha256(source.encode()).hexdigest() != self.hashes[task_id]:  # noqa: NAR001
            return Verdict(reward=0.0, reason="tests_tampered")
        nonce = secrets.token_hex(nbytes=16)
        sandbox = self.sandbox_factory()
        sandbox.start()
        try:
            result = sandbox.run(
                files={"solution.py": code, "verify.py": HARNESS.format(tests=source)},
                command=[sandbox.python, "verify.py"],
                limits=self.limits,
                stdin=nonce + "\n",
            )
        finally:
            sandbox.destroy()
        if classify(result=result) == FailureClass.INFRA:
            raise InfraError(f"verifier sandbox failed: {result.stderr[:200]}")  # noqa: NAR001
        lines = result.stdout.strip().splitlines()
        passed = result.exit_code == 0 and bool(lines) and lines[-1] == nonce  # noqa: NAR001
        return Verdict(
            reward=1.0 if passed else 0.0, reason="pass" if passed else "fail", result=result
        )
