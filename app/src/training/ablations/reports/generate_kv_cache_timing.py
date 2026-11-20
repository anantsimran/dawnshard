"""Render the KV-cache timing report's figures from its checked-in CSVs.

Run from the repository root:
    uv run python app/src/training/ablations/reports/generate_kv_cache_timing.py

Reads `sentences.csv` and `positions.csv`, which
`run/multi30k_translator/time_decoding.py` writes from the best translator
checkpoint. Needs neither the checkpoint nor the device the timings came from.
"""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use(backend="Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ABLATIONS = Path(__file__).resolve().parents[1]  # noqa: NAR001
ASSETS = ABLATIONS / "reports" / "assets" / "kv_cache_timing"
# The CSVs carry no device column; time_decoding.py prints the device it timed on.
DEVICE = "MPS"
INDIGO = "#4f46e5"
AMBER = "#b45309"
MUTED = "#64748b"


def read_csv(name: str) -> list[dict[str, str]]:
    """Return the rows of one CSV under ASSETS, values as strings."""
    with (ASSETS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))  # noqa: NAR001


def save_figure(figure: Figure, name: str) -> None:
    """Save a figure as a shareable PNG and an SVG with selectable text."""
    figure.tight_layout(pad=1.4)
    for extension in ("png", "svg"):
        figure.savefig(fname=ASSETS / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig=figure)


def add_title(axis: Axes, title: str, subtitle: str) -> None:
    """Put a bold left-aligned title above the axes and a muted subtitle under it."""
    axis.set_title(label=title, loc="left", pad=30, fontweight="bold")
    axis.text(  # noqa: NAR001
        0, 1.025, subtitle, transform=axis.transAxes, color=MUTED, fontsize=10
    )


def plot_sentences(rows: list[dict[str, str]]) -> None:
    """Scatter each sentence's cached time against its uncached time.

    Points below the diagonal are sentences the cache made faster.
    """
    uncached = [float(row["uncached_ms"]) for row in rows]  # noqa: NAR001
    cached = [float(row["cached_ms"]) for row in rows]  # noqa: NAR001
    median_uncached = statistics.median(uncached)  # noqa: NAR001
    median_cached = statistics.median(cached)  # noqa: NAR001
    top = max(uncached) * 1.05  # noqa: NAR001

    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    axis.plot(  # noqa: NAR001
        [0, top], [0, top], color=MUTED, linewidth=1, linestyle="--", zorder=1
    )
    axis.annotate(
        text="same time both ways",
        xy=(top * 0.62, top * 0.62),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=MUTED,
        rotation=45,
        rotation_mode="anchor",
    )
    axis.scatter(  # noqa: NAR001
        uncached,
        cached,
        s=22,
        color=INDIGO,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.6,
        zorder=2,
    )
    axis.scatter(  # noqa: NAR001
        [median_uncached],
        [median_cached],
        s=90,
        color=AMBER,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )
    axis.annotate(
        text=(
            f"median sentence: {median_uncached:.0f} → {median_cached:.0f} ms,\n"
            f"{median_uncached / median_cached:.2f}× faster"
        ),
        xy=(median_uncached, median_cached),
        xytext=(top * 0.97, top * 0.12),
        textcoords="data",
        ha="right",
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": AMBER, "linewidth": 1},
    )
    same = sum(row["same_output"] == "True" for row in rows)  # noqa: NAR001
    add_title(
        axis=axis,
        title="The cache made every sentence faster",
        subtitle=(
            f"{len(rows)} test captions · greedy decoding, batch 1, {DEVICE} · "  # noqa: NAR001
            f"{same} of {len(rows)} translations identical"  # noqa: NAR001
        ),
    )
    axis.set_xlabel(xlabel="Uncached: ms to translate the sentence")
    axis.set_ylabel(ylabel="Cached: ms to translate the sentence")
    axis.set_xlim(left=0, right=top)
    axis.set_ylim(bottom=0, top=top)
    axis.set_aspect(aspect="equal")
    save_figure(figure=figure, name="sentence_latency")


def plot_positions(rows: list[dict[str, str]]) -> None:
    """Draw the median time of one decode step against output position, both ways."""
    positions = [int(row["position"]) for row in rows]  # noqa: NAR001
    figure, axis = plt.subplots(figsize=(9, 4.6))
    for key, label, color in (
        ("uncached_ms", "Uncached: re-run the whole prefix", AMBER),
        ("cached_ms", "Cached: one new token", INDIGO),
    ):
        axis.plot(  # noqa: NAR001
            positions,
            [float(row[key]) for row in rows],  # noqa: NAR001
            color=color,
            marker="o",
            markersize=4,
            linewidth=2,
            label=label,
        )
    first = rows[0]
    axis.annotate(
        text="first cached step also\nprojects the source keys",
        xy=(1, float(first["cached_ms"])),  # noqa: NAR001
        xytext=(2.5, 2.2),
        textcoords="data",
        color=MUTED,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )
    add_title(
        axis=axis,
        title="Each cached step is cheaper, and the gap barely grows",
        subtitle=(
            f"One {DEVICE} decode step per output position · median of 20 runs · 11-token source"
        ),
    )
    axis.set_xlabel(xlabel="Output position (tokens decoded so far + 1)")
    axis.set_ylabel(ylabel="Median ms per decode step")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, loc="lower right")
    save_figure(figure=figure, name="step_latency")


def main() -> None:
    """Write both figures as PNG and SVG next to the CSVs."""
    plt.rcParams.update(  # noqa: NAR001
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e2e8f0",
            "grid.alpha": 0.8,
            "svg.fonttype": "none",
            "svg.hashsalt": "kv-cache-timing",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    plot_sentences(rows=read_csv(name="sentences.csv"))
    plot_positions(rows=read_csv(name="positions.csv"))
    print(f"Exported two PNG/SVG figures to {ASSETS}")  # noqa: NAR001


if __name__ == "__main__":
    main()
