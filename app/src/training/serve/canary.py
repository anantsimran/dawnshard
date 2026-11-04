"""Canary rollout: shift traffic to a new version in stages, roll back on an SLO breach.

Traffic is split by a stable hash of the request id, so a request (and its retries)
always sees one version. The controller moves through `stages` (5% → 25% → 100% by
default). At each stage it waits for `min_requests` canary requests, then compares the
canary's SLO-miss rate with the baseline's: worse by more than `tolerance` means
rollback to 0%, otherwise advance. The decision is a pure function of the counts, so a
deliberately degraded version can be tested to roll back within a known number of
requests (the detection window).
"""

import enum
import hashlib
from dataclasses import dataclass, field


class Decision(enum.Enum):
    """What the controller does after an evaluation."""

    HOLD = "hold"
    ADVANCE = "advance"
    ROLLBACK = "rollback"
    DONE = "done"


@dataclass
class CanaryController:
    """Stage index, per-version SLO counts for the current stage, and the action log."""

    stages: tuple[float, ...] = (0.05, 0.25, 1.0)
    min_requests: int = 50
    tolerance: float = 0.02
    stage: int = 0
    rolled_back: bool = False
    counts: dict[str, list[int]] = field(
        default_factory=lambda: {"baseline": [0, 0], "canary": [0, 0]}
    )
    log: list[str] = field(default_factory=list)

    @property
    def canary_share(self) -> float:
        """Share of traffic the canary gets now; 0 after a rollback."""
        return 0.0 if self.rolled_back else self.stages[self.stage]

    def route(self, request_id: str) -> str:
        """`"canary"` or `"baseline"`, stable for a given request id."""
        bucket = int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:4], "big") / 2**32  # noqa: NAR001
        return "canary" if bucket < self.canary_share else "baseline"

    def observe(self, version: str, met_slo: bool) -> Decision:
        """Record one finished request and evaluate the stage.

        Side effects:
            Updates the counts, and on a decision the stage, the rollback flag, and `log`.
        """
        self.counts[version][0] += 1
        self.counts[version][1] += 0 if met_slo else 1
        return self.evaluate()

    def evaluate(self) -> Decision:
        """Advance, roll back, or hold, from the current stage's counts."""
        if self.rolled_back or (
            self.stage == len(self.stages) - 1 and self.counts["canary"][0] >= self.min_requests  # noqa: NAR001
        ):  # noqa: NAR001
            return Decision.DONE
        canary_n, canary_miss = self.counts["canary"]
        if canary_n < self.min_requests:
            return Decision.HOLD
        base_n, base_miss = self.counts["baseline"]
        miss_rate = lambda n, m: m / n if n else 0.0  # noqa: E731
        if miss_rate(canary_n, canary_miss) > miss_rate(base_n, base_miss) + self.tolerance:  # noqa: NAR001
            self.rolled_back = True
            self.log.append(  # noqa: NAR001
                f"rollback at {self.stages[self.stage]:.0%}: canary misses {canary_miss}/{canary_n}"
            )  # noqa: NAR001
            return Decision.ROLLBACK
        self.stage += 1
        self.counts = {"baseline": [0, 0], "canary": [0, 0]}
        self.log.append(f"advance to {self.stages[self.stage]:.0%}")  # noqa: NAR001
        return Decision.ADVANCE
