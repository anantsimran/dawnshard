from pathlib import Path

DATASETS_CACHE_DIR = Path(__file__).parents[3] / "datasets" / "cache"  # noqa: NAR001
CACHE_PATH = DATASETS_CACHE_DIR / "cache.sqlite"  # noqa: NAR001

TOKENIZERS_PATH = DATASETS_CACHE_DIR.parent / "tokenizers"  # merges files, one per tokenizer
