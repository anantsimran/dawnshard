import subprocess
from typing import Optional


def get_git_commit() -> Optional[str]:
    """Return the current commit hash, or None if git is unavailable or fails."""
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
