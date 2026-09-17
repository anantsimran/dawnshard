from torch import Tensor
from training.train.model import Metrics


def calculate_metrics(predicted: Tensor, target: Tensor) -> Metrics:
    """Return this batch's metrics; the default adds nothing beyond loss and count."""
    return Metrics()


def accumulate_metrics(acc: Metrics, batch: Metrics) -> None:
    """Add this batch into `acc` in place.

    `acc.loss` holds the batch-size-weighted loss sum until `reduce_metrics` runs.
    """
    acc.loss += batch.loss * batch.count
    acc.count += batch.count


def reduce_metrics(acc: Metrics) -> Metrics:
    """Return the epoch mean loss from `acc`, or 0.0 if nothing was accumulated."""
    return Metrics(loss=acc.loss / acc.count if acc.count else 0.0, count=acc.count)
