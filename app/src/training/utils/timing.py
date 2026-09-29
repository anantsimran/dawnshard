import time
from collections.abc import Generator
from contextlib import contextmanager

from loguru import logger


@contextmanager
def log_elapsed(label: str) -> Generator[None]:
    """Log how long the `with` block took, as "<label> in 1.2s".

    Args:
        label: What the block did, e.g. "Downloaded the dataset".

    Side effects:
        Logs one INFO line when the block exits. Nothing is logged if it raises,
        so a failed step never reports a duration.
    """
    start = time.perf_counter()
    yield
    logger.info("{} in {:.1f}s", label, time.perf_counter() - start)  # noqa: NAR001
