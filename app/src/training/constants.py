from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]  # noqa: NAR001

HISTORY_PATH = REPO_ROOT / "app" / "history"
CHECKPOINTS_PATH = HISTORY_PATH / "checkpoints"
TRACES_PATH = HISTORY_PATH / "traces"
ATTENTION_PROBES_PATH = HISTORY_PATH / "attention_probes"
# One YAML per runnable model, read by its `run/` module; ablation studies pin theirs
# under `ablations/`.
CONFIGS_PATH = REPO_ROOT / "app" / "src" / "config"

# Eval randomness (masking, subsampling, augmentation) is fixed repo-wide so that
# val and test numbers compare across runs; only the train seed varies per run.
VAL_SEED = 1000000
TEST_SEED = 2000000

CROSS_ENTROPY_IGNORE_INDEX = -100  # CrossEntropyLoss default ignore_index
