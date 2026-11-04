"""
mbpp.py
MBPP (Mostly Basic Python Problems) as coding tasks for the sandboxed agent benchmark.

The full 974-problem jsonl is downloaded once to data_dir/mbpp/. `load_mbpp_tasks`
draws a fixed sample by seed, so a benchmark's task list is a function of the seed and
can be committed as ids. The prompt shows the first test so the agent knows the
function's name and signature; the verifier runs every test, with the problem's setup
code first. Each task also carries the reference solution, so the harness can be
checked without a model ("oracle" mode).
"""

import json
import random
import urllib.request
from pathlib import Path

from training.dataload.constants import DATASETS_CACHE_DIR

MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
)


def load_mbpp_tasks(
    num_tasks: int, seed: int, data_dir: Path = DATASETS_CACHE_DIR
) -> list[dict[str, str]]:
    """Return `num_tasks` MBPP problems sampled by `seed`, sorted by task id.

    Returns:
        Dicts with `task_id` ("mbpp_<n>"), `prompt`, `tests` (setup code, then one
        assert per line), and `solution` (the reference code).

    Side effects:
        Downloads mbpp.jsonl on first use.
    """
    path = data_dir / "mbpp" / "mbpp.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url=MBPP_URL, filename=path)
    rows = [json.loads(s=line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    sample = random.Random(x=seed).sample(population=rows, k=num_tasks)  # noqa: NAR001
    return [
        {
            "task_id": f"mbpp_{row['task_id']}",
            "prompt": f"{row['text']}\nYour code must pass tests like:\n{row['test_list'][0]}",
            "tests": "\n".join([row["test_setup_code"], *row["test_list"]]),  # noqa: NAR001
            "solution": row["code"],
        }
        for row in sorted(sample, key=lambda row: row["task_id"])  # noqa: NAR001
    ]
