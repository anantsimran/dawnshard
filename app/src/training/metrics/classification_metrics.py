from dataclasses import dataclass, field

import torch
from torch import Tensor
from training.train.model import Metrics


@dataclass
class ClassificationMetrics(Metrics):
    # Per-class counts are indexed by class id and sized from the logits on the first
    # batch, so nothing here depends on a particular dataset's number of classes.
    # `reduce_metrics` turns them into the scalar fields and leaves the lists empty.
    correct: int = 0
    true_positives: list[int] = field(default_factory=list)
    false_positives: list[int] = field(default_factory=list)
    false_negatives: list[int] = field(default_factory=list)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


def calculate_metrics(predicted: Tensor, target: Tensor) -> ClassificationMetrics:
    """Count correct predictions and per-class TP/FP/FN in this batch.

    Args:
        predicted: Logits of shape (batch, num_classes).
        target: Class ids of shape (batch,).
    """
    num_classes = predicted.size(dim=-1)
    predicted_ids = predicted.argmax(dim=-1)
    hit = predicted_ids == target
    predicted_counts = torch.bincount(input=predicted_ids, minlength=num_classes)
    target_counts = torch.bincount(input=target, minlength=num_classes)
    true_positives = torch.bincount(input=predicted_ids[hit], minlength=num_classes)
    return ClassificationMetrics(
        correct=int(hit.sum().item()),  # noqa: NAR001
        true_positives=true_positives.tolist(),
        false_positives=(predicted_counts - true_positives).tolist(),
        false_negatives=(target_counts - true_positives).tolist(),
    )


def accumulate_metrics(acc: ClassificationMetrics, batch: ClassificationMetrics) -> None:
    """Add this batch into `acc` in place.

    `acc.loss` holds the batch-size-weighted loss sum until `reduce_metrics` runs.
    """
    acc.loss += batch.loss * batch.count
    acc.count += batch.count
    acc.correct += batch.correct
    if not acc.true_positives:
        num_classes = len(batch.true_positives)  # noqa: NAR001
        acc.true_positives = [0] * num_classes
        acc.false_positives = [0] * num_classes
        acc.false_negatives = [0] * num_classes
    for class_id in range(len(batch.true_positives)):  # noqa: NAR001
        acc.true_positives[class_id] += batch.true_positives[class_id]
        acc.false_positives[class_id] += batch.false_positives[class_id]
        acc.false_negatives[class_id] += batch.false_negatives[class_id]


def _macro_average(hits: list[int], misses: list[int]) -> float:
    """Return the mean over classes of hits / (hits + misses), counting 0/0 as 0.0."""
    if not hits:
        return 0.0
    ratios = [h / (h + m) if h + m else 0.0 for h, m in zip(hits, misses)]  # noqa: NAR001
    return sum(ratios) / len(ratios)  # noqa: NAR001


def reduce_metrics(acc: ClassificationMetrics) -> ClassificationMetrics:
    """Return epoch mean loss, accuracy, and macro-averaged precision and recall.

    Returns zeros if nothing was accumulated.
    """
    return ClassificationMetrics(
        loss=acc.loss / acc.count if acc.count else 0.0,
        count=acc.count,
        correct=acc.correct,
        accuracy=acc.correct / acc.count if acc.count else 0.0,
        precision=_macro_average(hits=acc.true_positives, misses=acc.false_positives),
        recall=_macro_average(hits=acc.true_positives, misses=acc.false_negatives),
    )
