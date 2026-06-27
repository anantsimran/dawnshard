"""Pre-commit hook: flag function calls that pass positional arguments.

Any call with one or more positional args (excluding *unpacks) is a violation.
Suppress a line with `# noqa: named-args` on the call's opening line.
"""

import ast
import sys
from pathlib import Path

NOQA_TAG = "NAR001"


def noqa_lines(source: str) -> set[int]:
    result: set[int] = set()
    for line_number, line in enumerate(source.splitlines(), start=1):
        if f"# noqa: {NOQA_TAG}" in line:
            result.add(line_number)
    return result


def check_file(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source=source, filename=str(path))
    suppressed = noqa_lines(source=source)
    violations: list[str] = []
    for node in ast.walk(node=tree):
        if not isinstance(node, ast.Call):
            continue
        if node.lineno in suppressed:
            continue
        positional_count = sum(1 for arg in node.args if not isinstance(arg, ast.Starred))
        if positional_count:
            violations.append(
                f"{path}:{node.lineno}: {positional_count} positional arg(s) — use keyword arguments"
            )
    return violations


def main() -> None:
    files = [Path(file) for file in sys.argv[1:] if file.endswith(".py")]
    violations: list[str] = []
    for file in files:
        violations.extend(check_file(path=file))
    for violation in violations:
        print(violation)  # noqa: NAR001
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
