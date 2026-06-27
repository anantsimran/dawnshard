import subprocess
from typing import Optional


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            args=["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
