#!/usr/bin/env python3
import json
import subprocess
import sys

data = json.load(sys.stdin)
skill = data.get("tool_input", {}).get("skill", "")
if skill == "create-pr":
    result = subprocess.run(args=[".venv/bin/pytest", "app/tests/", "-q"])
    if result.returncode != 0:
        print(
            json.dumps(
                {
                    "continue": False,
                    "stopReason": "Tests failed — fix before creating PR",
                }
            )
        )
