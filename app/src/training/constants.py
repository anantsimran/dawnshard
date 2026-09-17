from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]  # noqa: NAR001

CHECKPOINTS_PATH = REPO_ROOT / "app" / "checkpoints"
HISTORY_PATH = REPO_ROOT / "app" / "history"
TRACES_PATH = HISTORY_PATH / "traces"
ATTENTION_PROBES_PATH = HISTORY_PATH / "attention_probes"
