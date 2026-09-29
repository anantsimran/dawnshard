import math

import pytest
import torch
from torch.nn import functional as F
from training.constants import CROSS_ENTROPY_IGNORE_INDEX
from training.metrics.masked_token_metrics import (
    MaskedTokenMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)

IGNORE = CROSS_ENTROPY_IGNORE_INDEX


def test_calculate_metrics_scores_only_unignored_positions():
    # argmax per position: [0, 1, 0]; position 1 is ignored, position 2 is wrong.
    predicted = torch.tensor(data=[[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]])
    target = torch.tensor(data=[[0, IGNORE, 1]])
    assert calculate_metrics(predicted=predicted, target=target) == MaskedTokenMetrics(
        masked=2, correct=1
    )


def test_accumulate_metrics_weights_loss_by_masked_not_count():
    accumulator = MaskedTokenMetrics()
    accumulate_metrics(
        acc=accumulator, batch=MaskedTokenMetrics(loss=1.0, count=2, masked=3, correct=2)
    )
    accumulate_metrics(
        acc=accumulator, batch=MaskedTokenMetrics(loss=2.0, count=2, masked=1, correct=1)
    )
    assert accumulator == MaskedTokenMetrics(loss=5.0, count=4, masked=4, correct=3)


def test_reduce_metrics_returns_per_position_means_and_perplexity():
    reduced = reduce_metrics(acc=MaskedTokenMetrics(loss=8.0, count=2, masked=4, correct=3))
    assert reduced.loss == 2.0
    assert reduced.count == 2
    assert reduced.accuracy == 0.75
    assert reduced.perplexity == pytest.approx(expected=math.exp(2.0))  # noqa: NAR001


def test_reduce_metrics_returns_zero_when_nothing_scored():
    assert reduce_metrics(acc=MaskedTokenMetrics(count=3)) == MaskedTokenMetrics(count=3)


def test_epoch_loss_equals_cross_entropy_over_every_scored_position():
    # Two batches with different numbers of scored positions: weighting each batch
    # mean by `masked` must give the same loss as one mean over the whole epoch.
    generator = torch.Generator().manual_seed(0)  # noqa: NAR001
    logits = [torch.randn(2, 3, 5, generator=generator) for _ in range(2)]  # noqa: NAR001
    targets = [
        torch.tensor(data=[[1, IGNORE, IGNORE], [IGNORE, IGNORE, IGNORE]]),
        torch.tensor(data=[[0, 2, IGNORE], [4, 3, 1]]),
    ]
    accumulator = MaskedTokenMetrics()
    for predicted, target in zip(logits, targets):  # noqa: NAR001
        batch = calculate_metrics(predicted=predicted, target=target)
        batch.loss = F.cross_entropy(
            input=predicted.transpose(dim0=1, dim1=2), target=target
        ).item()
        batch.count = target.size(dim=0)
        accumulate_metrics(acc=accumulator, batch=batch)

    expected = F.cross_entropy(
        input=torch.cat(tensors=logits).transpose(dim0=1, dim1=2),
        target=torch.cat(tensors=targets),
    )
    assert reduce_metrics(acc=accumulator).loss == pytest.approx(expected=expected.item())
