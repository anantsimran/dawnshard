"""Plot per-epoch metrics from a training history JSON file.

One panel per metric. A metric recorded for both splits, as `train_loss` and
`val_loss` or as the same field of `train_metrics` and `val_metrics`, gets one panel
with a train line and a val line, so the gap between them is the plot.
"""

import argparse
import base64
import io
import json
import math
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

SPLITS = ("train", "val")
PANEL_COLUMNS = 2


def _save_and_open_html(fig: Figure, output_path: Path) -> None:
    """Write `fig` as a base64-embedded PNG in an HTML file and open it in the browser."""
    buffer = io.BytesIO()
    fig.savefig(fname=buffer, format="png", bbox_inches="tight")
    buffer.seek(0)  # noqa: NAR001
    encoded = base64.b64encode(s=buffer.read()).decode(encoding="utf-8")
    html = f'<html><body><img src="data:image/png;base64,{encoded}"></body></html>'
    output_path.write_text(data=html)
    webbrowser.open(url=output_path.as_uri())


def load_history(history_path: Path) -> list[dict]:
    """Load per-epoch records, accepting either a bare JSON list or `{"history": [...]}`."""
    with open(file=history_path) as history_file:
        data = json.load(fp=history_file)
    if isinstance(data, list):  # noqa: NAR001
        return data
    return data["history"]


def _is_number(value: object) -> bool:
    """Return whether `value` is an int or float; bool is excluded."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)  # noqa: NAR001


def metric_series(history: list[dict]) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Return `{panel: {line: [(epoch, value), ...]}}` for every plottable field.

    `<split>_<name>` and the fields of `<split>_metrics` go to panel `<name>` as line
    `<split>`, for each split in SPLITS. Inside a metrics dict only floats are
    plotted: ints there are accumulator counts and lists are per-class tallies,
    neither of which reads as a curve. Any other numeric top-level field is its own
    panel and line. When a nested field repeats a top-level one, as `train_metrics`'s
    `loss` repeats `train_loss`, the first in record order is kept. None values,
    such as validation on an epoch that skipped it, are left out.
    """
    points: dict[str, dict[str, dict[int, float]]] = {}

    def add(panel: str, line: str, epoch: int, value: float) -> None:
        """Record `value` unless this panel's line already has one for `epoch`."""
        panel_lines = points.setdefault(panel, {})  # noqa: NAR001
        panel_lines.setdefault(line, {}).setdefault(epoch, value)  # noqa: NAR001

    for record in history:
        epoch = record["epoch"]
        for key, value in record.items():
            if key == "epoch" or value is None:
                continue
            split, _, name = key.partition("_")  # noqa: NAR001
            if split in SPLITS and name == "metrics" and isinstance(value, dict):  # noqa: NAR001
                for field, field_value in value.items():
                    if isinstance(field_value, float):  # noqa: NAR001
                        add(panel=field, line=split, epoch=epoch, value=field_value)
            elif split in SPLITS and name and _is_number(value=value):
                add(panel=name, line=split, epoch=epoch, value=value)
            elif _is_number(value=value):
                add(panel=key, line=key, epoch=epoch, value=value)

    return {
        panel: {line: sorted(by_epoch.items()) for line, by_epoch in lines.items()}  # noqa: NAR001
        for panel, lines in points.items()
    }


def main():
    """CLI entry point: plot metrics from a history file and open the result.

    Writes the chart to `/tmp/plot_metrics.html` and opens it in the default
    browser.
    """
    parser = argparse.ArgumentParser(description="Plot training metrics from a history JSON file.")
    parser.add_argument("history_file", type=Path, help="Path to history JSON file.")  # noqa: NAR001
    args = parser.parse_args()

    series = metric_series(history=load_history(history_path=args.history_file))

    num_rows = math.ceil(len(series) / PANEL_COLUMNS)  # noqa: NAR001
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=PANEL_COLUMNS,
        figsize=(7 * PANEL_COLUMNS, 4 * num_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for axis, (panel, lines) in zip(flat_axes, series.items()):  # noqa: NAR001
        for line, line_points in lines.items():
            plot_epochs, plot_values = zip(*line_points)  # noqa: NAR001
            axis.plot(plot_epochs, plot_values, marker="o", markersize=3, label=line)  # noqa: NAR001
        axis.set_title(label=panel)
        axis.set_xlabel(xlabel="Epoch")
        axis.set_ylabel(ylabel=panel)
        axis.legend()
        axis.grid(visible=True)
    for axis in flat_axes[len(series) :]:  # noqa: NAR001
        axis.set_visible(b=False)

    fig.tight_layout()
    _save_and_open_html(
        fig=fig,
        output_path=Path("/tmp/plot_metrics.html"),  # noqa: NAR001
    )


if __name__ == "__main__":
    main()
