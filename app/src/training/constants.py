from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]  # noqa: NAR001

CHECKPOINTS_PATH = REPO_ROOT / "app" / "checkpoints"
HISTORY_PATH = REPO_ROOT / "app" / "history"
TRACES_PATH = HISTORY_PATH / "traces"
ATTENTION_PROBES_PATH = HISTORY_PATH / "attention_probes"

# Eval randomness (masking, subsampling, augmentation) is fixed repo-wide so that
# val and test numbers compare across runs; only the train seed varies per run.
VAL_SEED = 1000000
TEST_SEED = 2000000

CROSS_ENTROPY_IGNORE_INDEX = -100  # CrossEntropyLoss default ignore_index
