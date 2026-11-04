import os

import pytest
from training.env.sandbox import Limits, SubprocessSandbox
from training.env.verify import Verifier, harden_tests

TESTS = "assert add(1, 2) == 3\nassert add(-1, 1) == 0"


@pytest.fixture
def verifier(tmp_path):
    verifier = Verifier(tests_dir=tmp_path / "tests", sandbox_factory=SubprocessSandbox, limits=Limits())
    verifier.register(task_id="t", tests=TESTS)
    return verifier


def test_genuine_pass_and_fail(verifier):
    assert verifier.verify(task_id="t", code="def add(a, b):\n    return a + b").reward == 1.0
    assert verifier.verify(task_id="t", code="def add(a, b):\n    return a - b").reward == 0.0


def test_rejects_a_modified_test_file(verifier, tmp_path):
    path = tmp_path / "tests" / "t.py"
    os.chmod(path, 0o644)
    path.write_text("pass\n")
    verdict = verifier.verify(task_id="t", code="def add(a, b):\n    return 0")
    assert (verdict.reward, verdict.reason) == (0.0, "tests_tampered")


def test_exit_zero_before_the_asserts_is_not_a_pass(verifier):
    for code in ("import sys\nsys.exit(0)", "import os\nos._exit(0)"):
        assert verifier.verify(task_id="t", code=code).reward == 0.0


def test_an_always_equal_object_is_not_a_pass(verifier):
    code = (
        "class Anything:\n"
        "    def __eq__(self, other):\n        return True\n"
        "def add(a, b):\n    return Anything()"
    )
    assert verifier.verify(task_id="t", code=code).reward == 0.0
    nested = code.replace("return Anything()", "return [Anything()]")
    verifier.register(task_id="list", tests="assert add(1, 2) == [3]")
    assert verifier.verify(task_id="list", code=nested).reward == 0.0


def test_hardening_rewrites_only_equality_asserts():
    hardened = harden_tests(source="assert f(1) == 2\nassert g(3)\nassert 1 < 2")
    assert "_check(f(1), 2)" in hardened
    assert "assert g(3)" in hardened
    assert "assert 1 < 2" in hardened


def test_a_spoofed_builtins_module_is_not_a_pass(verifier):
    code = (
        "class Anything:\n    __module__ = 'builtins'\n"
        "    def __eq__(self, other):\n        return True\n"
        "def add(a, b):\n    return Anything()"
    )
    assert verifier.verify(task_id="t", code=code).reward == 0.0


def test_standard_library_values_are_trusted(verifier):
    verifier.register(task_id="count", tests="assert count('aab') == {'a': 2, 'b': 1}")
    code = "from collections import Counter\ndef count(s):\n    return Counter(s)"
    assert verifier.verify(task_id="count", code=code).reward == 1.0
