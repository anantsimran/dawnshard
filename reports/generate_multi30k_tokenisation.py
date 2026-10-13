"""Export the notebook's saved BPE statistics and render the report's figures.

Run from the repository root:
    uv run python reports/generate_multi30k_tokenisation.py

Reads saved stream output only; does not download data or retrain tokenizers.
The CSV and plots preserve the notebook's two-decimal precision and vocabulary
counts, including its historical count of four protected token ids.
"""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use(backend="Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # noqa: NAR001
NOTEBOOK = ROOT / "notebooks" / "multi30k_dataload.ipynb"
ASSETS = ROOT / "reports" / "assets" / "multi30k_tokenisation"
LANGUAGES = {"en": ("English", "#4f46e5", "o"), "de": ("German", "#b45309", "s")}
BUDGETS = (500, 1000, 2000, 3000, 5000, 8000, 10000)


@dataclass(frozen=True)
class Stats:
    """One row of the notebook's printed training-sample summary."""

    config: str
    language: str
    merges: int
    vocab: int
    tokens_per_word: float
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    p99_9: float
    max: int
    roundtrip: float


def read_stats() -> list[Stats]:
    """Parse and validate the fourteen saved configurations used by this article."""
    notebook = json.loads(s=NOTEBOOK.read_text(encoding="utf-8"))
    rows = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):  # noqa: NAR001
            if output.get("output_type") != "stream":  # noqa: NAR001
                continue
            saved_text = output.get("text", [])  # noqa: NAR001
            if isinstance(saved_text, list):  # noqa: NAR001
                saved_text = "".join(saved_text)  # noqa: NAR001
            for line in saved_text.splitlines():
                fields = line.split()
                if not fields or not fields[0].startswith("train_"):  # noqa: NAR001
                    continue
                if len(fields) != 12:  # noqa: NAR001
                    raise ValueError(f"Unexpected notebook statistics row: {line}")  # noqa: NAR001
                _, language, budget = fields[0].split(sep="_")
                values = [float(value) for value in fields[1:]]  # noqa: NAR001
                row = Stats(
                    config=fields[0],
                    language=language,
                    merges=int(values[0]),  # noqa: NAR001
                    vocab=int(values[1]),  # noqa: NAR001
                    tokens_per_word=values[2],
                    mean=values[3],
                    p50=values[4],
                    p90=values[5],
                    p95=values[6],
                    p99=values[7],
                    p99_9=values[8],
                    max=int(values[9]),  # noqa: NAR001
                    roundtrip=values[10],
                )
                if row.merges != int(budget) or row.vocab != row.merges + 260:  # noqa: NAR001
                    raise ValueError("Vocabulary layout changed; update the article.")  # noqa: NAR001
                if not row.p50 <= row.p90 <= row.p95 <= row.p99 <= row.p99_9 <= row.max:
                    raise ValueError(f"Unordered percentiles in {row.config}")  # noqa: NAR001
                rows.append(row)  # noqa: NAR001
    expected = {(language, budget) for language in LANGUAGES for budget in BUDGETS}
    if len(rows) != 14 or {(row.language, row.merges) for row in rows} != expected:  # noqa: NAR001
        raise ValueError("Expected fourteen saved BPE rows; execute the notebook first.")  # noqa: NAR001
    return sorted(rows, key=lambda row: (row.language, row.merges))  # noqa: NAR001


def save_figure(figure: Figure, name: str) -> None:
    """Save a figure as a shareable PNG and an SVG with selectable text."""
    figure.tight_layout(pad=1.4)
    for extension in ("png", "svg"):
        figure.savefig(fname=ASSETS / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig=figure)


def plot_curve(rows: list[Stats], metric: str, name: str, title: str, ylabel: str) -> None:
    """Plot each language's saved values against its recorded vocabulary size."""
    figure, axis = plt.subplots(figsize=(9, 4.6))
    for language, (label, color, marker) in LANGUAGES.items():
        selected = [row for row in rows if row.language == language]
        axis.plot(  # noqa: NAR001
            [row.vocab / 1000 for row in selected],
            [getattr(row, metric) for row in selected],  # noqa: NAR001
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.3,
            label=label,
        )
        last = selected[-1]
        axis.annotate(
            text=f"{getattr(last, metric):.2f}",  # noqa: NAR001
            xy=(last.vocab / 1000, getattr(last, metric)),  # noqa: NAR001
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontweight="bold",
        )
    axis.set_title(label=title, loc="left", pad=30, fontweight="bold")
    axis.text(  # noqa: NAR001
        0,
        1.025,
        "English and German image captions · 5,000 samples per language",
        transform=axis.transAxes,
        color="#64748b",
        fontsize=10,
    )
    axis.set_xlabel(xlabel="Vocabulary size (thousands of tokens)")
    axis.set_ylabel(ylabel=ylabel)
    axis.set_xlim(left=0, right=11)
    axis.set_xticks(ticks=[0, 2, 4, 6, 8, 10])
    if metric == "mean":
        axis.set_ylim(bottom=10, top=27)
    else:
        axis.set_ylim(bottom=1, top=2.4)
    axis.legend(frameon=False, loc="upper right")
    save_figure(figure=figure, name=name)


def plot_tail(rows: list[Stats]) -> None:
    """Compare averages and upper percentiles at a recorded vocabulary of 5,260."""
    figure, axis = plt.subplots(figsize=(9, 4.8))
    metrics = ("mean", "p90", "p99")
    labels = ("Average", "90th percentile", "99th percentile")
    for language, (label, color, _) in LANGUAGES.items():
        row = next(row for row in rows if row.language == language and row.merges == 5000)  # noqa: NAR001
        offset = -0.19 if language == "en" else 0.19
        bars = axis.bar(
            x=[i + offset for i in range(len(metrics))],  # noqa: NAR001
            height=[getattr(row, metric) for metric in metrics],  # noqa: NAR001
            width=0.35,
            color=color,
            label=label,
        )
        axis.bar_label(container=bars, fmt="%g", padding=3, fontsize=11)
    axis.set_title(label="The average hides longer inputs", loc="left", pad=30, fontweight="bold")
    axis.text(  # noqa: NAR001
        0,
        1.025,
        "Vocabulary: 5,260 tokens · 5,000 captions per language",
        transform=axis.transAxes,
        color="#64748b",
        fontsize=10,
    )
    axis.set_xticks(ticks=list(range(len(metrics))), labels=labels)  # noqa: NAR001
    axis.set_ylabel(ylabel="Tokens per caption")
    axis.set_ylim(bottom=0, top=38)
    axis.legend(frameon=False, loc="upper left")
    save_figure(figure=figure, name="length_tail")


def main() -> None:
    """Write the CSV and all three figures from the notebook's saved output."""
    rows = read_stats()
    ASSETS.mkdir(parents=True, exist_ok=True)
    with (ASSETS / "stats.csv").open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(  # noqa: NAR001
            handle, fieldnames=[field for field in Stats.__dataclass_fields__]
        )
        writer.writeheader()
        writer.writerows(rowdicts=[asdict(obj=row) for row in rows])
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
            "svg.hashsalt": "multi30k-tokenisation",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    plot_curve(
        rows=rows,
        metric="mean",
        name="mean_length",
        title="A larger vocabulary produces shorter sequences",
        ylabel="Mean tokens per caption",
    )
    plot_curve(
        rows=rows,
        metric="tokens_per_word",
        name="tokens_per_word",
        title="Tokens per word differ between languages",
        ylabel="Tokens per word (total tokens / total words)",
    )
    plot_tail(rows=rows)
    print(f"Exported {len(rows)} rows and three PNG/SVG figures to {ASSETS}")  # noqa: NAR001


if __name__ == "__main__":
    main()
