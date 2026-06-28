"""Plot per-epoch metrics from a training history JSON file."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(history_path: Path) -> list[dict]:
    with open(file=history_path) as history_file:
        data = json.load(fp=history_file)
    if isinstance(data, list):  # noqa: NAR001
        return data
    return data["history"]


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from a history JSON file."
    )
    parser.add_argument("history_file", type=Path, help="Path to history JSON file.")  # noqa: NAR001
    args = parser.parse_args()

    history = load_history(history_path=args.history_file)

    metric_names = [
        key
        for key in history[0]
        if key != "epoch" and isinstance(history[0][key], (int, float))  # noqa: NAR001
    ]
    epochs = [record["epoch"] for record in history]

    num_metrics = len(metric_names)  # noqa: NAR001
    fig, axes = plt.subplots(
        nrows=num_metrics, ncols=1, figsize=(10, 4 * num_metrics), squeeze=False
    )

    for axis, metric_name in zip(axes[:, 0], metric_names):  # noqa: NAR001
        values = [record.get(metric_name) for record in history]  # noqa: NAR001
        present = [
            (epoch, value)
            for epoch, value in zip(epochs, values)  # noqa: NAR001
            if value is not None
        ]
        if present:
            plot_epochs, plot_values = zip(*present)  # noqa: NAR001
            axis.plot(plot_epochs, plot_values, marker="o", label=metric_name)  # noqa: NAR001
        axis.set_title(label=metric_name)
        axis.set_xlabel(xlabel="Epoch")
        axis.set_ylabel(ylabel=metric_name)
        axis.legend()
        axis.grid(visible=True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
