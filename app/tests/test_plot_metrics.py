from training.viz.plot_metrics import metric_series


def test_train_and_val_loss_share_a_panel():
    history = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 3.0},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 3.5},
    ]
    assert metric_series(history=history) == {
        "loss": {"train": [(1, 2.0), (2, 1.0)], "val": [(1, 3.0), (2, 3.5)]},
    }


def test_nested_rates_are_plotted_and_counts_and_lists_are_not():
    history = [
        {
            "epoch": 1,
            "train_metrics": {"accuracy": 0.5, "count": 10, "true_positives": [1, 2]},
            "val_metrics": {"accuracy": 0.4, "count": 5, "true_positives": [0, 1]},
        }
    ]
    assert metric_series(history=history) == {
        "accuracy": {"train": [(1, 0.5)], "val": [(1, 0.4)]},
    }


def test_nested_loss_does_not_duplicate_the_top_level_loss():
    history = [{"epoch": 1, "train_loss": 2.0, "train_metrics": {"loss": 2.0}}]
    assert metric_series(history=history) == {"loss": {"train": [(1, 2.0)]}}


def test_epochs_without_validation_are_skipped():
    history = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": None, "val_metrics": None},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 3.0, "val_metrics": {"accuracy": 0.4}},
    ]
    assert metric_series(history=history) == {
        "loss": {"train": [(1, 2.0), (2, 1.0)], "val": [(2, 3.0)]},
        "accuracy": {"val": [(2, 0.4)]},
    }


def test_unsplit_fields_get_their_own_panel():
    history = [{"epoch": 1, "epoch_duration_seconds": 50.0, "ram_used_gb": 9}]
    assert metric_series(history=history) == {
        "epoch_duration_seconds": {"epoch_duration_seconds": [(1, 50.0)]},
        "ram_used_gb": {"ram_used_gb": [(1, 9)]},
    }
