import math
from dataclasses import dataclass

from torch import Tensor
from training.constants import CROSS_ENTROPY_IGNORE_INDEX
from training.train.model import Metrics


@dataclass
class MaskedTokenMetrics(Metrics):
    # For per-position targets where only some positions are scored, the rest set to
    # CROSS_ENTROPY_IGNORE_INDEX. The step sets `count` to the target's first axis, but
    # the loss is a mean over scored positions, so `masked` is what weights the loss and
    # the accuracy. `accuracy` and `perplexity` are filled in by `reduce_metrics` only.
    masked: int = 0
    correct: int = 0
    accuracy: float = 0.0
    perplexity: float = 0.0


def calculate_metrics(predicted: Tensor, target: Tensor) -> MaskedTokenMetrics:
    """Count scored positions and correct top-1 predictions among them.

    Args:
        predicted: Logits with the vocab axis last, such as (batch, seq_len, vocab) or
            flattened (batch*seq_len, vocab).
        target: Token ids shaped like `predicted` without its last axis,
            `CROSS_ENTROPY_IGNORE_INDEX` at every position that isn't scored.
    """
    keep = target != CROSS_ENTROPY_IGNORE_INDEX
    hits = predicted.argmax(dim=-1)[keep] == target[keep]
    return MaskedTokenMetrics(
        masked=int(keep.sum().item()),  # noqa: NAR001
        correct=int(hits.sum().item()),  # noqa: NAR001
    )


def accumulate_metrics(acc: MaskedTokenMetrics, batch: MaskedTokenMetrics) -> None:
    """Add this batch into `acc` in place.

    `acc.loss` holds the loss sum over scored positions until `reduce_metrics` runs:
    the batch loss is a mean over `batch.masked` positions, so weighting by `count`
    would skew the epoch loss toward batches with few scored positions.
    """
    acc.loss += batch.loss * batch.masked
    acc.count += batch.count
    acc.masked += batch.masked
    acc.correct += batch.correct


def reduce_metrics(acc: MaskedTokenMetrics) -> MaskedTokenMetrics:
    """Return the per-position mean loss, accuracy, and perplexity for the epoch.

    Returns zeros if no position was scored.
    """
    if not acc.masked:
        return MaskedTokenMetrics(count=acc.count)
    loss = acc.loss / acc.masked
    return MaskedTokenMetrics(
        loss=loss,
        count=acc.count,
        masked=acc.masked,
        correct=acc.correct,
        accuracy=acc.correct / acc.masked,
        perplexity=math.exp(loss),  # noqa: NAR001
    )
