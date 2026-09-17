"""Flag functions and methods missing a docstring.

Skips test files. Suppress a specific function with `# noqa: NAR002` on its
`def` line.
"""

import ast
import sys
from pathlib import Path

NOQA_TAG = "NAR002"


def is_test_file(path: Path) -> bool:
    return path.name.startswith("test_") or "tests" in path.parts


def noqa_lines(source: str) -> set[int]:
    result: set[int] = set()
    for line_number, line in enumerate(source.splitlines(), start=1):
        if f"# noqa: {NOQA_TAG}" in line:
            result.add(line_number)
    return result


def check_file(path: Path) -> list[str]:
    if is_test_file(path=path):
        return []
    source = path.read_text()
    tree = ast.parse(source=source, filename=str(path))
    suppressed = noqa_lines(source=source)
    violations: list[str] = []
    for node in ast.walk(node=tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.lineno in suppressed:
            continue
        if ast.get_docstring(node=node) is None:
            violations.append(f"{path}:{node.lineno}: {node.name} is missing a docstring")
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
