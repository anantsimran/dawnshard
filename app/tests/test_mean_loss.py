from train.mean_loss import accumulate_loss, compute_reduced_loss


def test_accumulate_loss_adds_weighted_loss():
    accumulator = {"accumulated_loss": 0.0, "count": 0}
    accumulate_loss(acc=accumulator, loss=2.0, batch_size=4)
    assert accumulator["accumulated_loss"] == 8.0
    assert accumulator["count"] == 4


def test_accumulate_loss_accumulates_across_batches():
    accumulator = {"accumulated_loss": 0.0, "count": 0}
    accumulate_loss(acc=accumulator, loss=1.0, batch_size=3)
    accumulate_loss(acc=accumulator, loss=2.0, batch_size=3)
    assert accumulator["accumulated_loss"] == 9.0
    assert accumulator["count"] == 6


def test_compute_reduced_loss_returns_mean():
    accumulator = {"accumulated_loss": 12.0, "count": 4}
    assert compute_reduced_loss(acc=accumulator) == 3.0


def test_compute_reduced_loss_returns_zero_when_empty():
    accumulator = {"accumulated_loss": 0.0, "count": 0}
    assert compute_reduced_loss(acc=accumulator) == 0.0
