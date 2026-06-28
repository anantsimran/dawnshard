"""Compare per-epoch metrics from two training runs on the same graph."""

import argparse
import base64
import io
import json
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _save_and_open_html(fig: Figure, output_path: Path) -> None:
    buffer = io.BytesIO()
    fig.savefig(fname=buffer, format="png", bbox_inches="tight")
    buffer.seek(0)  # noqa: NAR001
    encoded = base64.b64encode(s=buffer.read()).decode(encoding="utf-8")
    html = f'<html><body><img src="data:image/png;base64,{encoded}"></body></html>'
    output_path.write_text(data=html)
    webbrowser.open(url=output_path.as_uri())


def load_history(history_path: Path) -> list[dict]:
    with open(file=history_path) as history_file:
        data = json.load(fp=history_file)
    if isinstance(data, list):  # noqa: NAR001
        return data
    return data["history"]


def main():
    parser = argparse.ArgumentParser(
        description="Compare metrics from two training history JSON files."
    )
    parser.add_argument(  # noqa: NAR001
        "history_file_a", type=Path, help="Path to first history JSON file."
    )
    parser.add_argument(  # noqa: NAR001
        "history_file_b", type=Path, help="Path to second history JSON file."
    )
    args = parser.parse_args()

    history_a = load_history(history_path=args.history_file_a)
    history_b = load_history(history_path=args.history_file_b)

    label_a = args.history_file_a.stem
    label_b = args.history_file_b.stem

    numeric_keys_a = {
        key
        for key in history_a[0]
        if key != "epoch" and isinstance(history_a[0][key], (int, float))  # noqa: NAR001
    }
    numeric_keys_b = {
        key
        for key in history_b[0]
        if key != "epoch" and isinstance(history_b[0][key], (int, float))  # noqa: NAR001
    }
    shared_metric_names = sorted(numeric_keys_a & numeric_keys_b)  # noqa: NAR001

    epochs_a = [record["epoch"] for record in history_a]
    epochs_b = [record["epoch"] for record in history_b]

    num_metrics = len(shared_metric_names)  # noqa: NAR001
    fig, axes = plt.subplots(
        nrows=num_metrics, ncols=1, figsize=(10, 4 * num_metrics), squeeze=False
    )

    for axis, metric_name in zip(axes[:, 0], shared_metric_names):  # noqa: NAR001
        values_a = [record.get(metric_name) for record in history_a]  # noqa: NAR001
        values_b = [record.get(metric_name) for record in history_b]  # noqa: NAR001

        present_a = [
            (epoch, value)
            for epoch, value in zip(epochs_a, values_a)  # noqa: NAR001
            if value is not None
        ]
        present_b = [
            (epoch, value)
            for epoch, value in zip(epochs_b, values_b)  # noqa: NAR001
            if value is not None
        ]

        if present_a:
            plot_epochs_a, plot_values_a = zip(*present_a)  # noqa: NAR001
            axis.plot(  # noqa: NAR001
                plot_epochs_a,
                plot_values_a,
                marker="o",
                color="steelblue",
                label=label_a,
            )
        if present_b:
            plot_epochs_b, plot_values_b = zip(*present_b)  # noqa: NAR001
            axis.plot(  # noqa: NAR001
                plot_epochs_b,
                plot_values_b,
                marker="s",
                color="tomato",
                label=label_b,
            )

        axis.set_title(label=metric_name)
        axis.set_xlabel(xlabel="Epoch")
        axis.set_ylabel(ylabel=metric_name)
        axis.legend()
        axis.grid(visible=True)

    fig.tight_layout()
    _save_and_open_html(
        fig=fig,
        output_path=Path("/tmp/compare_runs.html"),  # noqa: NAR001
    )


if __name__ == "__main__":
    main()
