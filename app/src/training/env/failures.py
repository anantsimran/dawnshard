"""Infra vs task failures, and retries that count an episode once.

A reward is a statement about the model's code. A failure the infrastructure caused
(a host OOM kill, a sandbox that never started, a 503 from the model server) says
nothing about the code, so scoring it 0 teaches the model that correct answers were
wrong. `classify` separates the two; `run_with_retries` retries infra failures under
the same episode id and gives up rather than scoring them.

The rules, in order:

- The sandbox did not start, or the model server returned 5xx or was unreachable:
  infra.
- The program hit its own timeout or its sandbox's memory limit: task. The program
  chose to loop forever or allocate too much.
- Exit 137 (SIGKILL) with neither: infra. Something outside the sandbox's limits
  killed it, such as the host OOM killer.
- Any other nonzero exit: task.
"""

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from training.env.sandbox import SIGKILL_EXIT_CODE, RunResult


class FailureClass(enum.Enum):
    """What a run's outcome says about the model's code."""

    NONE = "none"
    TASK = "task"
    INFRA = "infra"


class InfraError(Exception):
    """Raised by episode code when infrastructure, not the program, failed."""


class ModelServerError(InfraError):
    """The model server answered with an HTTP error, or not at all. Always infra:
    no server error says anything about the model's code."""

    def __init__(self, status: Optional[int]) -> None:
        """Record the HTTP status, or None when the connection failed."""
        super().__init__(f"model server error: {status}")  # noqa: NAR001
        self.status = status


def classify(result: RunResult) -> FailureClass:
    """Label one sandbox run; see the module docstring for the rules."""
    if result.start_failed:
        return FailureClass.INFRA
    if result.timed_out or result.oom_killed:
        return FailureClass.TASK
    if result.exit_code == SIGKILL_EXIT_CODE:
        return FailureClass.INFRA
    return FailureClass.NONE if result.exit_code == 0 else FailureClass.TASK


@dataclass
class EpisodeLedger:
    """Final outcome per episode id. A retry overwrites; it never adds a row."""

    rewards: dict[str, float]
    attempts: dict[str, int]
    dropped: set[str]

    @classmethod
    def empty(cls) -> "EpisodeLedger":
        """Return a ledger with no episodes."""
        return cls(rewards={}, attempts={}, dropped=set())  # noqa: NAR001


def run_with_retries(
    episode_id: str,
    episode: Callable[[], float],
    ledger: EpisodeLedger,
    max_attempts: int = 2,
) -> Optional[float]:
    """Run `episode` until it returns a reward, retrying on `InfraError`.

    Args:
        episode: Runs the whole episode and returns its reward; raises `InfraError`
            when infrastructure failed. Must be safe to call again with the same id.
        max_attempts: Attempts before the episode is dropped, unscored.

    Returns:
        The reward, or None when every attempt hit infra and the episode was dropped.

    Side effects:
        Records the reward or the drop, and the attempt count, under `episode_id`.
    """
    for attempt in range(1, max_attempts + 1):  # noqa: NAR001
        ledger.attempts[episode_id] = attempt
        try:
            reward = episode()
        except InfraError:
            continue
        ledger.rewards[episode_id] = reward
        ledger.dropped.discard(episode_id)  # noqa: NAR001
        return reward
    ledger.dropped.add(episode_id)  # noqa: NAR001
    ledger.rewards.pop(episode_id, None)  # noqa: NAR001
    return None
