#!/usr/bin/env python3
import json
import subprocess
import sys


def block(reason: str) -> None:
    """Stop the create-pr skill and tell Claude why."""
    print(json.dumps(obj={"continue": False, "stopReason": reason}))
    sys.exit(0)


def base_ref() -> str:
    """Prefer origin/main, fall back to main."""
    for ref in ("origin/main", "main"):
        result = subprocess.run(
            args=["git", "rev-parse", "--verify", "--quiet", ref],
            capture_output=True,
        )
        if result.returncode == 0:
            return ref
    return "main"


def changed_files() -> set[str]:
    """Files changed on this branch vs the base, including uncommitted edits."""
    ref = base_ref()
    files: set[str] = set()
    committed = subprocess.run(
        args=["git", "diff", "--name-only", f"{ref}...HEAD"],
        capture_output=True,
        text=True,
    )
    files.update(line for line in committed.stdout.splitlines() if line)
    working_tree = subprocess.run(
        args=["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    # porcelain lines look like "XY path"; the path starts at column 3.
    files.update(line[3:] for line in working_tree.stdout.splitlines() if line)
    return files


def check_readme_updated() -> None:
    """If source under app/src changed but README.md did not, block."""
    files = changed_files()
    source_changed = any(
        path.startswith("app/src/") and path.endswith(".py") for path in files
    )
    if source_changed and "README.md" not in files:
        block(
            reason=(
                "Source under app/src/ changed but README.md was not updated. "
                "Review README.md and update it (setup, running, project "
                "structure, models) before creating the PR."
            )
        )


data = json.load(fp=sys.stdin)
skill = data.get("tool_input", {}).get("skill", "")
if skill == "create-pr":
    named_args_result = subprocess.run(
        args=["bash", "-c", "find app/src -name '*.py' | xargs /Users/anantsimran/.local/bin/uv run python scripts/check_named_args.py"],
    )
    if named_args_result.returncode != 0:
        block(reason="Named-arg violations found — fix before creating PR")

    check_readme_updated()

    tests_result = subprocess.run(args=[".venv/bin/pytest", "app/tests/", "-q"])
    if tests_result.returncode != 0:
        block(reason="Tests failed — fix before creating PR")
