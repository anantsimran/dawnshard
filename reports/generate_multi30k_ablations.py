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
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # noqa: NAR001
ASSETS = ROOT / "reports/assets/multi30k_ablations"
INDIGO = "#4f46e5"
AMBER = "#b45309"
MUTED = "#64748b"
BLUE = "#0284c7"
type Histories = dict[str, list[dict[str, str]]]


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


def load_histories(study: Study) -> Histories:
    """Group epoch records in chronological order, including the translator pilot."""
    histories = defaultdict(list)  # noqa: NAR001
    directory = ROOT / "experiments" / study.directory
    for row in read_csv(path=directory / "epochs.csv"):
        histories[row["run_id"]].append(row)  # noqa: NAR001
    return {
        run_id: sorted(rows, key=lambda row: int(row["epoch"]))  # noqa: NAR001
        for run_id, rows in histories.items()
    }


def load_study(study: Study) -> dict[str, list[dict[str, str]]]:
    """Validate complete three-seed arms against their per-epoch records."""
    directory = ROOT / "experiments" / study.directory
    runs = [
        row
        for row in read_csv(path=directory / "runs.csv")
        if row["experiment"] == study.experiment
    ]
    epochs_by_run = load_histories(study=study)
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
        path = ASSETS / f"{name}.{extension}"
        figure.savefig(
            fname=path,
            dpi=180,
            bbox_inches="tight",
            metadata={"Date": None},
        )
        if extension == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            cleaned = "\n".join(line.rstrip() for line in lines)  # noqa: NAR001
            path.write_text(
                data=f"{cleaned}\n",
                encoding="utf-8",
            )
    plt.close(fig=figure)


def draw_arms(
    axis: Axes, groups: dict[str, list[dict[str, str]]], arms: dict[str, str], title: str
) -> None:
    """Show every seed and its arm mean against the observed control range."""
    control = losses(rows=groups["control"])
    axis.axvspan(xmin=min(control), xmax=max(control), color=INDIGO, alpha=0.08)  # noqa: NAR001
    for position, (arm, label) in enumerate(arms.items()):  # noqa: NAR001
        values = losses(rows=groups[arm])
        color = INDIGO if arm == "control" else BLUE if arm == "schedule=cosine" else AMBER
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


def plot_arms(
    groups: dict[str, list[dict[str, str]]], arms: dict[str, str], title: str, name: str
) -> None:
    """Export a standalone comparison of each arm's three seed minima."""
    figure, axis = plt.subplots(figsize=(9, 4.8))
    draw_arms(axis=axis, groups=groups, arms=arms, title=title)
    save_figure(figure=figure, name=name)


def draw_curve(
    axis: Axes,
    histories: Histories,
    rows: list[dict[str, str]],
    metric: str,
    label: str,
    color: str,
    start_epoch: int = 1,
    linestyle: str = "-",
    multiplier: float = 1,
) -> None:
    """Draw an epoch-wise mean and observed seed range, without smoothing."""
    records = [histories[row["run_id"]] for row in rows]
    numbers = [int(row["epoch"]) for row in records[0]]  # noqa: NAR001
    if any([int(row["epoch"]) for row in run] != numbers for run in records):  # noqa: NAR001
        raise ValueError("Curves require identical epoch coverage across seeds")  # noqa: NAR001
    values = [
        [float(row[metric]) * multiplier for row in epoch]  # noqa: NAR001
        for epoch in zip(*records)  # noqa: NAR001
        if int(epoch[0]["epoch"]) >= start_epoch  # noqa: NAR001
    ]
    numbers = [epoch for epoch in numbers if epoch >= start_epoch]
    axis.plot(  # noqa: NAR001
        numbers,
        [mean(data=items) for items in values],
        color=color,
        label=label,
        linewidth=2,
        linestyle=linestyle,
    )
    if len(rows) > 1:  # noqa: NAR001
        axis.fill_between(
            x=numbers,
            y1=[min(items) for items in values],  # noqa: NAR001
            y2=[max(items) for items in values],  # noqa: NAR001
            color=color,
            alpha=0.10,
            linewidth=0,
        )


def plot_training_budget(
    translator: Histories,
    masked: Histories,
    masked_control: list[dict[str, str]],
) -> None:
    """Explain overfitting and slower convergence using the recorded controls."""
    runs = read_csv(path=ROOT / "experiments" / STUDIES[0].directory / "runs.csv")
    pilot = [row for row in runs if row["experiment"] == "first_control_100ep"]
    if len(pilot) != 1 or len(translator[pilot[0]["run_id"]]) != 100:  # noqa: NAR001
        raise ValueError("Expected one complete 100-epoch translator pilot")  # noqa: NAR001
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.7))
    for axis, histories, rows, title in (
        (axes[0], translator, pilot, "Translator · one pilot run"),
        (axes[1], masked, masked_control, "Masked-token model · three seeds"),
    ):
        for metric, label, color, style in (
            ("val_loss", "Validation", INDIGO, "-"),
            ("train_loss", "Training", AMBER, "--"),
        ):
            draw_curve(
                axis=axis,
                histories=histories,
                rows=rows,
                metric=metric,
                label=label,
                color=color,
                linestyle=style,
            )
        axis.set_title(label=title, loc="left", fontsize=11, fontweight="bold", pad=14)
        axis.set_xlabel(xlabel="Epoch · one pass through the training set")
        axis.set_ylabel(ylabel="Token prediction loss · lower is better")
        axis.set_xlim(left=1, right=100)
        axis.set_ylim(bottom=0, top=5.6)
        axis.legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].axvline(x=9, color=MUTED, linestyle=":")
    axes[0].annotate(
        text="Best validation loss\nat epoch 9",
        xy=(9, float(pilot[0]["best_val_loss"])),  # noqa: NAR001
        xytext=(28, 2.5),
        arrowprops={"arrowstyle": "->", "color": MUTED},
        fontsize=9,
    )
    axes[1].axvspan(xmin=87, xmax=93, color=INDIGO, alpha=0.08)
    axes[1].text(x=38, y=1.0, s="Best epochs: 87, 87, 93", fontsize=9, color=INDIGO)
    figure.suptitle(t="The training budget follows the validation curve", fontweight="bold")
    save_figure(figure=figure, name="training_budget")


def plot_schedule_dynamics(histories: Histories, groups: dict[str, list[dict[str, str]]]) -> None:
    """Relate the configured 25-epoch schedule to the width-512 validation curves."""
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.6))
    epochs = list(range(1, 26))  # noqa: NAR001
    # Reconstruct build_scheduler's 5-epoch LinearLR (4% start), then cosine.
    # Epoch 6 uses the peak; fit steps the schedule after each training pass.
    rates = [
        0.04 + 0.96 * (epoch - 1) / 5
        if epoch <= 6
        else (1 + math.cos(math.pi * (epoch - 6) / 20)) / 2  # noqa: NAR001
        for epoch in epochs
    ]
    for label, color, values in (
        ("Constant", INDIGO, [1.0] * 25),
        ("Warmup + cosine", AMBER, rates),
    ):
        axes[0].step(x=epochs, y=values, where="post", label=label, color=color, linewidth=2)
    axes[0].set_title(label="1. Change the step size", loc="left", fontweight="bold", pad=14)
    axes[0].set_ylabel(ylabel="Learning rate / peak rate (0.001)")
    axes[0].set_ylim(bottom=0, top=1.12)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    for arm, label, color in (
        ("d_model=512", "Constant", INDIGO),
        ("d_model=512, schedule=warmup_cosine", "Warmup + cosine", AMBER),
    ):
        draw_curve(
            axis=axes[1],
            histories=histories,
            rows=groups[arm],
            metric="val_loss",
            label=label,
            color=color,
        )
    axes[1].set_title(label="2. Measure validation loss", loc="left", fontweight="bold", pad=14)
    axes[1].set_ylabel(ylabel="Validation loss · width 512")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    for axis in axes:
        axis.set_xlabel(xlabel="Epoch")
        axis.set_xlim(left=1, right=25)
        axis.axvspan(xmin=7, xmax=8, color=MUTED, alpha=0.12)
    axes[0].text(
        x=0.60,
        y=0.73,
        s="Best epochs: 7–8\n98–99% of peak",
        transform=axes[0].transAxes,
        fontsize=9,
    )
    figure.suptitle(t="The schedule changes the path the wider model takes", fontweight="bold")
    save_figure(figure=figure, name="schedule_dynamics")


def plot_masked_convergence(histories: Histories, groups: dict[str, list[dict[str, str]]]) -> None:
    """Connect late-training trajectories with the three-seed minimum comparison."""
    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7.6))
    for arm, label, color in (
        ("control", "Control · constant rate", INDIGO),
        ("dropout_rate=0.0", "No dropout", AMBER),
        ("schedule=cosine", "Cosine decay", BLUE),
    ):
        draw_curve(
            axis=axes[0],
            histories=histories,
            rows=groups[arm],
            metric="val_loss",
            label=label,
            color=color,
            start_epoch=20,
        )
    axes[0].set_title(
        label="Longer training reveals which changes keep helping",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    axes[0].set_xlabel(xlabel="Epoch · zoom into epochs 20–100")
    axes[0].set_ylabel(ylabel="Masked-token validation loss")
    axes[0].set_xlim(left=20, right=100)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    draw_arms(
        axis=axes[1],
        groups=groups,
        arms={
            "schedule=cosine": "Cosine decay",
            "control": "Control · constant rate",
            "dropout_rate=0.0": "No dropout",
        },
        title="A lower mean still needs a repeatable advantage",
    )
    save_figure(figure=figure, name="masked_token_convergence")


def plot_loss_and_accuracy(histories: Histories, groups: dict[str, list[dict[str, str]]]) -> None:
    """Show why loss and top-1 accuracy can select different checkpoints."""
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.6))
    for axis, metric, title, ylabel, multiplier in (
        (axes[0], "val_loss", "Loss starts rising again", "Validation loss · lower is better", 1),
        (
            axes[1],
            "val_accuracy",
            "Accuracy keeps improving",
            "Correct top tokens (%) · higher is better",
            100,
        ),
    ):
        draw_curve(
            axis=axis,
            histories=histories,
            rows=groups["schedule=warmup_cosine"],
            metric=metric,
            label="Width 256 · warmup + cosine",
            color=INDIGO,
            start_epoch=5,
            multiplier=multiplier,
        )
        axis.axvspan(xmin=7, xmax=9, color=AMBER, alpha=0.13)
        axis.set_title(label=title, loc="left", fontweight="bold", pad=14)
        axis.set_xlabel(xlabel="Epoch · zoom into epochs 5–25")
        axis.set_ylabel(ylabel=ylabel)
        axis.set_xlim(left=5, right=25)
    figure.suptitle(
        t="The same three runs, scored two ways · band: best-loss epochs",
        fontweight="bold",
    )
    save_figure(figure=figure, name="loss_and_accuracy")


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
        label="More width helps only with a suitable training recipe",
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
    histories = {study.key: load_histories(study=study) for study in STUDIES}
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
    plot_training_budget(
        translator=histories["translator_128"],
        masked=histories["masked_token"],
        masked_control=studies["masked_token"]["control"],
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
        title="Removing dropout or depth raises validation loss",
        name="translator_components",
    )
    plot_scale(narrow=studies["translator_128"], scale=studies["translator_scale"])
    plot_schedule_dynamics(
        histories=histories["translator_scale"], groups=studies["translator_scale"]
    )
    plot_masked_convergence(
        histories=histories["masked_token"],
        groups=studies["masked_token"],
    )
    plot_loss_and_accuracy(
        histories=histories["translator_scale"], groups=studies["translator_scale"]
    )
    write_summary(studies=studies)
    print(  # noqa: NAR001
        "Validated 66 comparison runs / 2,475 epochs; "
        f"exported 22 arms and 6 PNG/SVG figures to {ASSETS}"
    )


if __name__ == "__main__":
    main()
