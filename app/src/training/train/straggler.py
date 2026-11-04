"""straggler.py -- find the rank that makes every other rank wait.

In synchronous data parallelism a step ends when the slowest rank reaches the
collective. The slow rank looks busy (high compute time); every healthy rank looks idle
(high time waiting in the collective). So the signature is inverted from intuition:
the rank with the least waiting is the problem.

`StragglerDetector` keeps a window of per-rank `(compute_s, wait_s)` per step and flags
a rank whose median compute time exceeds the median of the other ranks' by more than
`tolerance`, while those other ranks spend at least `min_wait_share` of their step
waiting. Requiring both avoids flagging a rank that is merely busier on a balanced
step (heterogeneous batch) when nobody is actually waiting for it.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StragglerDetector:
    """Rolling per-rank step timings and the straggler rule."""

    num_ranks: int
    window: int = 50
    tolerance: float = 0.2
    min_wait_share: float = 0.1
    steps: deque = field(default_factory=deque)

    def record(self, compute_s: list[float], wait_s: list[float]) -> None:
        """Add one step's timings, one entry per rank, gathered from every rank.

        Side effects:
            Appends to the window, dropping the oldest step past `window`.
        """
        self.steps.append((list(compute_s), list(wait_s)))  # noqa: NAR001
        while len(self.steps) > self.window:  # noqa: NAR001
            self.steps.popleft()

    def stragglers(self) -> list[int]:
        """Ranks that compute slower than the rest while the rest wait."""
        if not self.steps:
            return []
        compute = np.median(a=np.array(object=[c for c, _ in self.steps]), axis=0)
        wait = np.median(a=np.array(object=[w for _, w in self.steps]), axis=0)
        flagged = []
        for rank in range(self.num_ranks):  # noqa: NAR001
            others = [r for r in range(self.num_ranks) if r != rank]  # noqa: NAR001
            others_compute = float(np.median(a=compute[others]))  # noqa: NAR001
            others_wait_share = float(np.median(a=wait[others] / (wait[others] + compute[others])))  # noqa: NAR001
            if (
                compute[rank] > (1 + self.tolerance) * others_compute
                and others_wait_share >= self.min_wait_share
            ):
                flagged.append(rank)  # noqa: NAR001
        return flagged
