import pytest
import torch
from training.metrics.classification_metrics import (
    ClassificationMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)


def test_calculate_metrics_counts_per_class():
    # argmax predictions: [0, 1, 0]; targets: [0, 1, 1]
    predicted = torch.tensor(data=[[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.7, 0.3, 0.0]])
    target = torch.tensor(data=[0, 1, 1])
    assert calculate_metrics(predicted=predicted, target=target) == ClassificationMetrics(
        correct=2,
        true_positives=[1, 1, 0],
        false_positives=[1, 0, 0],
        false_negatives=[0, 1, 0],
    )


def test_accumulate_metrics_accumulates_across_batches():
    accumulator = ClassificationMetrics()
    accumulate_metrics(
        acc=accumulator,
        batch=ClassificationMetrics(
            loss=1.0,
            count=3,
            correct=2,
            true_positives=[1, 1],
            false_positives=[1, 0],
            false_negatives=[0, 1],
        ),
    )
    accumulate_metrics(
        acc=accumulator,
        batch=ClassificationMetrics(
            loss=2.0,
            count=3,
            correct=3,
            true_positives=[2, 1],
            false_positives=[0, 0],
            false_negatives=[0, 0],
        ),
    )
    assert accumulator == ClassificationMetrics(
        loss=9.0,
        count=6,
        correct=5,
        true_positives=[3, 2],
        false_positives=[1, 0],
        false_negatives=[0, 1],
    )


def test_reduce_metrics_returns_macro_precision_and_recall():
    reduced = reduce_metrics(
        acc=ClassificationMetrics(
            loss=12.0,
            count=4,
            correct=3,
            true_positives=[1, 2],
            false_positives=[1, 0],
            false_negatives=[0, 1],
        )
    )
    assert reduced.loss == 3.0
    assert reduced.accuracy == 0.75
    assert reduced.precision == pytest.approx(expected=(1 / 2 + 2 / 2) / 2)
    assert reduced.recall == pytest.approx(expected=(1 / 1 + 2 / 3) / 2)
    assert reduced.true_positives == []


def test_reduce_metrics_counts_unseen_class_as_zero():
    reduced = reduce_metrics(
        acc=ClassificationMetrics(
            count=1,
            correct=1,
            true_positives=[1, 0],
            false_positives=[0, 0],
            false_negatives=[0, 0],
        )
    )
    assert reduced.precision == 0.5
    assert reduced.recall == 0.5


def test_reduce_metrics_returns_zero_when_empty():
    assert reduce_metrics(acc=ClassificationMetrics()) == ClassificationMetrics()
