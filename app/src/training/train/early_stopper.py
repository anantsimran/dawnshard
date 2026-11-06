"""early_stopper.py -- decide when `fit` stops, from its validation losses."""

from typing import Optional

from training.dataclass.config import EarlyStoppingConfig


class EarlyStopper:
    """Tracks the best validation loss and says when `patience` checks passed without a gain."""

    def __init__(self, early_stopping: Optional[EarlyStoppingConfig]) -> None:
        """Build a stopper; with `early_stopping=None`, `update` never asks to stop."""
        self._config = early_stopping
        self._best = float("inf")  # noqa: NAR001
        self._checks_without_gain = 0

    def update(self, val_loss: float) -> bool:
        """Record one validation loss; return True when the run should stop.

        A loss counts as a gain only if it beats the best so far by more than
        `min_delta`.

        Side effects:
            Updates the best loss and the count of checks without a gain.
        """
        if self._config is None:
            return False
        if val_loss < self._best - self._config.min_delta:
            self._best = val_loss
            self._checks_without_gain = 0
        else:
            self._checks_without_gain += 1
        return self._checks_without_gain >= self._config.patience
