import torch

from training.metrics.loss_only_metrics import (
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.train.model import Metrics


def test_calculate_metrics_returns_empty_batch_metrics():
    metrics = calculate_metrics(predicted=torch.zeros(4, 1), target=torch.zeros(4, 1))  # noqa: NAR001
    assert metrics == Metrics(loss=0.0, count=0)


def test_accumulate_metrics_adds_weighted_loss():
    accumulator = Metrics()
    accumulate_metrics(acc=accumulator, batch=Metrics(loss=2.0, count=4))
    assert accumulator.loss == 8.0
    assert accumulator.count == 4


def test_accumulate_metrics_accumulates_across_batches():
    accumulator = Metrics()
    accumulate_metrics(acc=accumulator, batch=Metrics(loss=1.0, count=3))
    accumulate_metrics(acc=accumulator, batch=Metrics(loss=2.0, count=3))
    assert accumulator.loss == 9.0
    assert accumulator.count == 6


def test_reduce_metrics_returns_mean():
    assert reduce_metrics(acc=Metrics(loss=12.0, count=4)) == Metrics(loss=3.0, count=4)


def test_reduce_metrics_returns_zero_when_empty():
    assert reduce_metrics(acc=Metrics()) == Metrics(loss=0.0, count=0)
