"""Build the ablation report's figures and summary from checked-in study CSVs.

Run with ``uv run python reports/generate_multi30k_ablations.py``. No training,
downloads, or gitignored history files are needed. The 100-epoch translator pilot
is excluded from the three-seed comparisons.
"""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use(backend="Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # noqa: NAR001
ASSETS = ROOT / "reports/assets/multi30k_ablations"
INDIGO = "#4f46e5"
AMBER = "#b45309"
MUTED = "#64748b"


@dataclass(frozen=True)
class Study:
    """Identify a comparison group within a study's exported records."""

    key: str
    directory: str
    experiment: str
    arms: int
    epochs: int


STUDIES = (
    Study(
        key="translator_128",
        directory="multi30k_translator/2026-10-13",
        experiment="ablation_30ep",
        arms=10,
        epochs=30,
    ),
    Study(
        key="translator_scale",
        directory="multi30k_translator/2026-10-15",
        experiment="scale_25ep",
        arms=9,
        epochs=25,
    ),
    Study(
        key="masked_token",
        directory="multi30ken_predictor/2026-10-15",
        experiment="ablation_100ep",
        arms=3,
        epochs=100,
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read the exported records without changing their stored precision."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(f=handle))  # noqa: NAR001


def load_study(study: Study) -> dict[str, list[dict[str, str]]]:
    """Validate complete three-seed arms against their per-epoch records."""
    directory = ROOT / "experiments" / study.directory
    runs = [
        row
        for row in read_csv(path=directory / "runs.csv")
        if row["experiment"] == study.experiment
    ]
    epochs_by_run = defaultdict(list)  # noqa: NAR001
    for row in read_csv(path=directory / "epochs.csv"):
        epochs_by_run[row["run_id"]].append(row)  # noqa: NAR001
    groups = defaultdict(list)  # noqa: NAR001
    for run in runs:
        epochs = epochs_by_run[run["run_id"]]
        numbers = {int(row["epoch"]) for row in epochs}  # noqa: NAR001
        if len(epochs) != study.epochs or numbers != set(range(1, study.epochs + 1)):  # noqa: NAR001
            raise ValueError(f"Incomplete epochs: {run['run_id']}")  # noqa: NAR001
        best = min(epochs, key=lambda row: float(row["val_loss"]))  # noqa: NAR001
        # CSV rounding can tie several epochs; validate the recorded epoch too.
        recorded_best = next(  # noqa: NAR001
            row for row in epochs if row["epoch"] == run["best_epoch"]
        )
        if not math.isclose(
            a=float(best["val_loss"]),  # noqa: NAR001
            b=float(run["best_val_loss"]),  # noqa: NAR001
            abs_tol=0.0001,
        ) or not math.isclose(
            a=float(recorded_best["val_loss"]),  # noqa: NAR001
            b=float(run["best_val_loss"]),  # noqa: NAR001
            abs_tol=0.0001,
        ):
            raise ValueError(f"Best epoch disagrees with runs.csv: {run['run_id']}")  # noqa: NAR001
        groups[run["arm"]].append(run)  # noqa: NAR001
    if len(groups) != study.arms or len({run["run_id"] for run in runs}) != study.arms * 3:  # noqa: NAR001
        raise ValueError(f"Unexpected run or arm count: {study.key}")  # noqa: NAR001
    for arm, rows in groups.items():
        if len(rows) != 3 or len({row["train_seed"] for row in rows}) != 3:  # noqa: NAR001
            raise ValueError(f"Expected three distinct seeds: {study.key}/{arm}")  # noqa: NAR001
    return dict(groups)  # noqa: NAR001


def losses(rows: list[dict[str, str]]) -> list[float]:
    """Return each seed's own minimum validation loss."""
    return [float(row["best_val_loss"]) for row in rows]  # noqa: NAR001


def save_figure(figure: Figure, name: str) -> None:
    """Save shareable raster and vector figures with deterministic SVG IDs."""
    figure.tight_layout(pad=1.4)
    for extension in ("png", "svg"):
        figure.savefig(
            fname=ASSETS / f"{name}.{extension}",
            dpi=180,
            bbox_inches="tight",
            metadata={"Date": None},
        )
    plt.close(fig=figure)


def plot_arms(
    groups: dict[str, list[dict[str, str]]], arms: dict[str, str], title: str, name: str
) -> None:
    """Show every seed and its arm mean against the observed control range."""
    figure, axis = plt.subplots(figsize=(9, 4.8))
    control = losses(rows=groups["control"])
    axis.axvspan(xmin=min(control), xmax=max(control), color=INDIGO, alpha=0.08)  # noqa: NAR001
    for position, (arm, label) in enumerate(arms.items()):  # noqa: NAR001
        values = losses(rows=groups[arm])
        color = INDIGO if arm == "control" else AMBER
        axis.scatter(
            x=values, y=[position - 0.13, position, position + 0.13], color=color, s=38, zorder=3
        )
        axis.plot(  # noqa: NAR001
            [mean(data=values)],
            [position],
            marker="|",
            color="#17213a",
            markersize=18,
            markeredgewidth=2.2,
            zorder=4,
        )
        axis.annotate(
            text=f"{mean(data=values):.3f}",
            xy=(max(values), position),  # noqa: NAR001
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=10,
            fontweight="bold",
        )
    axis.set_yticks(ticks=list(range(len(arms))), labels=list(arms.values()))  # noqa: NAR001
    axis.invert_yaxis()
    axis.margins(x=0.22, y=0.16)
    axis.set_title(label=title, loc="left", pad=34, fontweight="bold")
    axis.text(
        x=0,
        y=1.04,
        s="Dots: individual seeds · tick: mean · band: control's observed range",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=9,
    )
    axis.set_xlabel(xlabel="Best validation loss per run · lower is better")
    save_figure(figure=figure, name=name)


def plot_scale(
    narrow: dict[str, list[dict[str, str]]], scale: dict[str, list[dict[str, str]]]
) -> None:
    """Compare constant and warmup-cosine schedules at each translator width."""
    figure, axis = plt.subplots(figsize=(9, 4.8))
    series = (
        (
            "Constant learning rate",
            INDIGO,
            -0.12,
            [narrow["control"], scale["control"], scale["d_model=512"]],
        ),
        (
            "Warmup + cosine",
            AMBER,
            0.12,
            [
                narrow["schedule=warmup_cosine"],
                scale["schedule=warmup_cosine"],
                scale["d_model=512, schedule=warmup_cosine"],
            ],
        ),
    )
    for label, color, offset, groups in series:
        values = [losses(rows=rows) for rows in groups]
        means = [mean(data=items) for items in values]
        axis.errorbar(
            x=[i + offset for i in range(3)],  # noqa: NAR001
            y=means,  # noqa: NAR001
            yerr=[
                [avg - min(items) for avg, items in zip(means, values)],  # noqa: NAR001
                [max(items) - avg for avg, items in zip(means, values)],  # noqa: NAR001
            ],  # noqa: NAR001
            fmt="o-",
            color=color,
            label=label,
            capsize=5,
            linewidth=2,
        )
        for position, value in enumerate(means):  # noqa: NAR001
            axis.annotate(
                text=f"{value:.3f}",
                xy=(position + offset, value),
                xytext=(-12 if offset < 0 else 12, -18 if position == 1 and offset < 0 else 9),
                textcoords="offset points",
                ha="right" if offset < 0 else "left",
                color=color,
                fontweight="bold",
            )
    axis.set_title(
        label="A wider translator changes the schedule result",
        loc="left",
        pad=34,
        fontweight="bold",
    )
    axis.text(
        x=0,
        y=1.04,
        s="Three-seed means and observed ranges · not confidence intervals",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=9,
    )
    axis.set_xticks(
        ticks=[0, 1, 2],
        labels=[
            "128 wide\n7.51M parameters\n≈51 s / epoch",
            "256 wide\n22.1M parameters\n≈118 s / epoch",
            "512 wide\n72.5M parameters\n≈330 s / epoch",
        ],
    )
    axis.set_ylabel(ylabel="Best validation loss per run")
    axis.set_ylim(bottom=2.03, top=2.27)
    axis.set_xlim(left=-0.55, right=2.55)
    axis.legend(loc="upper left", frameon=False)
    save_figure(figure=figure, name="width_and_schedule")


def write_summary(studies: dict[str, dict[str, list[dict[str, str]]]]) -> None:
    """Export all 22 arms, retaining their study-specific controls and seed ranges."""
    records = []
    for study, groups in studies.items():
        control = losses(rows=groups["control"])
        for arm, rows in groups.items():
            values = losses(rows=rows)
            records.append(  # noqa: NAR001
                {
                    "study": study,
                    "arm": arm,
                    "runs": len(rows),  # noqa: NAR001
                    "parameters": rows[0]["num_parameters"],
                    "epochs": rows[0]["epochs"],
                    "best_val_mean": f"{mean(data=values):.6f}",
                    "best_val_min": min(values),  # noqa: NAR001
                    "best_val_max": max(values),  # noqa: NAR001
                    "delta_vs_control": f"{mean(data=values) - mean(data=control):.6f}",
                    "overlaps_control": (
                        max(min(values), min(control)) <= min(max(values), max(control))  # noqa: NAR001
                    ),
                }
            )
    with (ASSETS / "summary.csv").open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(f=handle, fieldnames=list(records[0]))  # noqa: NAR001
        writer.writeheader()
        writer.writerows(rowdicts=records)


def main() -> None:
    """Validate source data and build the report without rerunning experiments."""
    studies = {study.key: load_study(study=study) for study in STUDIES}
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(  # noqa: NAR001
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e2e8f0",
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "multi30k-ablations",
        }
    )
    plot_arms(
        groups=studies["translator_128"],
        arms={
            "control": "Control · 6 encoder + 6 decoder blocks",
            "norm=rmsnorm": "Use RMSNorm",
            "qk_norm=False": "Remove QK normalization",
            "num_encoders=2": "Keep 2 encoder blocks",
            "num_decoders=2": "Keep 2 decoder blocks",
            "dropout_rate=0.0": "Remove dropout",
        },
        title="At width 128, dropout and depth carry the clearest effects",
        name="translator_components",
    )
    plot_scale(narrow=studies["translator_128"], scale=studies["translator_scale"])
    plot_arms(
        groups=studies["masked_token"],
        arms={
            "schedule=cosine": "Cosine decay",
            "control": "Control · constant learning rate",
            "dropout_rate=0.0": "Remove dropout",
        },
        title="At 100 epochs, removing dropout hurts masked-token prediction",
        name="masked_token_convergence",
    )
    write_summary(studies=studies)
    print(f"Validated 66 runs / 2,475 epochs; exported 22 arms and 3 PNG/SVG figures to {ASSETS}")  # noqa: NAR001


if __name__ == "__main__":
    main()
