# crossmodel.py — the figures and tables that compare the models against each other.
# Author: Colile Sibanda
#
# Everything here reads only files that a finished run leaves behind, so the
# comparison can be rebuilt on any machine once the run directories are present.

from __future__ import annotations

from pathlib import Path

import pandas as pd

# One colour per architectural family, so a reader can tell the families apart
# at a glance across every figure.
FAMILY_COLOURS = {
    "CNN from scratch": "#4e79a7",
    "Pretrained convolutional": "#f28e2b",
    "Pretrained transformer": "#59a14f",
}
_FALLBACK_COLOUR = "#9c755f"


def _pyplot():
    """Purpose: the headless pyplot every figure in this module draws with."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _colour_for(family: object) -> str:
    """Purpose: the bar colour for one family. Output: a hex colour."""
    return FAMILY_COLOURS.get(str(family), _FALLBACK_COLOUR)


def collect_results(runs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Purpose: gather every model's recorded result files into one place.

    Input: a mapping of model name to the run directory holding its `metrics/`.
    Output: a dict with the frames that were found — `comparison`, `per_class`,
    `history` and `errors` — each carrying a `model` column. A model whose files
    are missing is skipped rather than failing the whole comparison, because the
    runs finish at different times on different machines.
    """
    aggregate: dict[str, list[pd.DataFrame]] = {
        "comparison": [],
        "per_class": [],
        "history": [],
        "errors": [],
    }
    # The efficiency numbers are written to their own file, but the cost figure
    # and the report's recommendation need them beside the scores, so they are
    # merged onto the comparison row for each model.
    efficiency_columns = (
        "median_ms_per_image",
        "p90_ms_per_image",
        "total_parameters",
        "size_mb",
        "device",
    )
    for model, directory in runs.items():
        metrics = Path(directory) / "metrics"
        sources = {
            "comparison": metrics / f"metrics_{model}.csv",
            "per_class": metrics / f"per_class_{model}.csv",
            "history": metrics / f"history_{model}.csv",
            "errors": metrics / f"errors_{model}.csv",
        }
        for key, path in sources.items():
            if not path.is_file():
                continue
            try:
                frame = pd.read_csv(path)
            except (pd.errors.EmptyDataError, OSError):
                continue
            if frame.empty:
                continue
            frame = frame.copy()
            frame["model"] = model
            if key == "comparison":
                efficiency_path = metrics / f"efficiency_{model}.csv"
                if efficiency_path.is_file():
                    try:
                        efficiency = pd.read_csv(efficiency_path)
                    except (pd.errors.EmptyDataError, OSError):
                        efficiency = pd.DataFrame()
                    if not efficiency.empty:
                        row = efficiency.iloc[0]
                        for column in efficiency_columns:
                            if column in efficiency.columns:
                                frame[column] = row[column]
            aggregate[key].append(frame)
    return {
        key: pd.concat(frames, ignore_index=True)
        for key, frames in aggregate.items()
        if frames
    }


def plot_metric_comparison(
    comparison: pd.DataFrame,
    path: str | Path,
    metrics: tuple[str, ...] = ("accuracy", "macro_f1"),
) -> Path:
    """Purpose: the headline chart — each model's aggregate scores side by side.

    Inputs: the comparison table (one row per model, with `family` when known),
    the destination file, and which metric columns to draw.
    Output: the path written. Raises KeyError when a requested metric is absent.
    """
    missing = [name for name in metrics if name not in comparison.columns]
    if missing:
        raise KeyError(
            f"The comparison table has no column(s) {', '.join(missing)}; "
            f"cannot draw the metric comparison."
        )
    plt = _pyplot()
    frame = comparison.sort_values(metrics[-1], ascending=False).reset_index(drop=True)
    models = frame["model"].tolist()
    families = (
        frame["family"].tolist() if "family" in frame.columns else [""] * len(models)
    )
    width = 0.8 / len(metrics)
    figure, axes = plt.subplots(figsize=(max(7.0, 1.9 * len(models)), 4.6))
    for index, metric in enumerate(metrics):
        offsets = [position + index * width for position in range(len(models))]
        values = [float(value) for value in frame[metric]]
        axes.bar(
            offsets,
            values,
            width,
            label=metric.replace("_", " "),
            color=[_colour_for(family) for family in families],
            alpha=1.0 if index == 0 else 0.55,
            edgecolor="white",
        )
        for position, value in zip(offsets, values):
            axes.text(
                position,
                value + 0.004,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes.set_xticks(
        [position + width * (len(metrics) - 1) / 2 for position in range(len(models))]
    )
    axes.set_xticklabels(models)
    axes.set_ylabel("score on the held-out test split")
    axes.set_title("Held-out test performance by model")
    lowest = min(float(frame[metric].min()) for metric in metrics)
    axes.set_ylim(max(0.0, lowest - 0.08), 1.0)
    axes.legend(title="metric", loc="lower right")
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_per_class_comparison(per_class: pd.DataFrame, path: str | Path) -> Path:
    """Purpose: show where each model is strong and weak, class by class.

    This is the figure that answers "which model is better, and at what": the
    aggregate charts cannot, because one model can win overall while losing on
    the classes that matter most.

    Inputs: the stacked per-class frame (`class_name`, `f1`, `model`) and the
    destination. Output: the path written.
    """
    for column in ("class_name", "f1", "model"):
        if column not in per_class.columns:
            raise KeyError(
                f"The per-class frame has no {column!r} column; cannot draw the "
                f"per-class comparison."
            )
    plt = _pyplot()
    pivot = per_class.pivot_table(
        index="class_name", columns="model", values="f1", aggfunc="first"
    )
    # Hardest classes first, by the mean across models, so the eye lands on the
    # problem cases rather than on the easy ones.
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    models = list(pivot.columns)
    width = 0.8 / max(len(models), 1)
    figure, axes = plt.subplots(figsize=(max(8.0, 1.1 * len(pivot)), 5.0))
    for index, model in enumerate(models):
        offsets = [position + index * width for position in range(len(pivot))]
        axes.bar(
            offsets,
            [float(value) for value in pivot[model]],
            width,
            label=model,
            edgecolor="white",
        )
    axes.set_xticks(
        [position + width * (len(models) - 1) / 2 for position in range(len(pivot))]
    )
    axes.set_xticklabels(pivot.index, rotation=30, ha="right")
    axes.set_ylabel("per-class F1 on the held-out split")
    axes.set_title("Per-class F1 by model, hardest classes first")
    axes.set_ylim(0.0, 1.0)
    axes.legend(title="model")
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_history_comparison(history: pd.DataFrame, path: str | Path) -> Path:
    """Purpose: put every model's validation curve on one pair of axes.

    This is what shows the effect of pretraining: where each model starts, how
    fast it climbs, and whether it was still improving when training stopped.

    Inputs: the stacked history frame (`epoch`, `val_macro_f1`, `val_loss`,
    `model`) and the destination. Output: the path written.
    """
    for column in ("epoch", "val_macro_f1", "model"):
        if column not in history.columns:
            raise KeyError(
                f"The history frame has no {column!r} column; cannot draw the "
                f"curve comparison."
            )
    plt = _pyplot()
    has_loss = "val_loss" in history.columns
    figure, axes = plt.subplots(
        1, 2 if has_loss else 1, figsize=(12.0 if has_loss else 7.0, 4.6)
    )
    first, second = (axes[0], axes[1]) if has_loss else (axes, None)
    for model, group in history.groupby("model"):
        group = group.sort_values("epoch")
        first.plot(
            group["epoch"],
            group["val_macro_f1"],
            marker="o",
            markersize=3,
            label=model,
        )
        if second is not None:
            second.plot(
                group["epoch"], group["val_loss"], marker="o", markersize=3, label=model
            )
    first.set_xlabel("epoch")
    first.set_ylabel("validation macro-F1")
    first.set_title("Validation macro-F1 per epoch")
    first.grid(alpha=0.3)
    first.legend()
    if second is not None:
        second.set_xlabel("epoch")
        second.set_ylabel("validation loss")
        second.set_title("Validation loss per epoch")
        second.grid(alpha=0.3)
        second.legend()
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_accuracy_against_cost(comparison: pd.DataFrame, path: str | Path) -> Path:
    """Purpose: the deployment argument — accuracy against inference cost.

    The brief asks for a recommendation on accuracy, inference time and model
    size together, which is what this figure shows: macro-F1 against latency,
    with the marker area standing for the parameter count.

    Inputs: the comparison table and the destination. Output: the path written.
    """
    needed = ("macro_f1", "median_ms_per_image")
    missing = [name for name in needed if name not in comparison.columns]
    if missing:
        raise KeyError(
            f"The comparison table has no column(s) {', '.join(missing)}; "
            f"cannot draw accuracy against cost."
        )
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.4, 5.0))
    has_params = "total_parameters" in comparison.columns
    largest = (
        float(max(comparison["total_parameters"].astype(float)))
        if has_params and comparison["total_parameters"].notna().any()
        else 0.0
    )
    for _, row in comparison.reset_index(drop=True).iterrows():
        area = 120.0
        known = has_params and pd.notna(row.get("total_parameters")) and largest > 0
        if known:
            # Area proportional to the parameter count, scaled so the smallest
            # model stays visible next to an 86 M-parameter transformer.
            area = 80.0 + 520.0 * float(row["total_parameters"]) / largest
        axes.scatter(
            float(row["median_ms_per_image"]),
            float(row["macro_f1"]),
            s=area,
            alpha=0.75,
            edgecolor="white",
            color=_colour_for(row.get("family", "")),
            zorder=3,
        )
        label = str(row["model"])
        if known:
            label += f"\n{float(row['total_parameters']) / 1e6:.1f} M params"
        axes.annotate(
            label,
            (float(row["median_ms_per_image"]), float(row["macro_f1"])),
            textcoords="offset points",
            xytext=(12, -4),
            fontsize=8,
        )
    axes.set_xlabel("median inference time per image (ms, lower is better)")
    axes.set_ylabel("macro-F1 on the held-out split (higher is better)")
    axes.set_title("Accuracy against inference cost; marker area is parameter count")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def strengths_table(per_class: pd.DataFrame) -> pd.DataFrame:
    """Purpose: say plainly which model wins each class, and by how much.

    Input: the stacked per-class frame. Output: one row per class with the best
    model, its F1, the runner-up and the margin — the table the discussion needs
    in order to claim more than "model X scored highest overall".
    """
    for column in ("class_name", "f1", "model"):
        if column not in per_class.columns:
            raise KeyError(f"The per-class frame has no {column!r} column.")
    rows: list[dict[str, object]] = []
    for class_name, group in per_class.groupby("class_name"):
        ordered = group.sort_values("f1", ascending=False).reset_index(drop=True)
        best = ordered.iloc[0]
        runner_up = ordered.iloc[1] if len(ordered) > 1 else None
        rows.append(
            {
                "class_name": class_name,
                "best_model": best["model"],
                "best_f1": round(float(best["f1"]), 4),
                "runner_up": runner_up["model"] if runner_up is not None else "",
                "runner_up_f1": (
                    round(float(runner_up["f1"]), 4) if runner_up is not None else ""
                ),
                "margin": (
                    round(float(best["f1"]) - float(runner_up["f1"]), 4)
                    if runner_up is not None
                    else ""
                ),
                "models_compared": len(ordered),
            }
        )
    return pd.DataFrame(rows).sort_values("class_name").reset_index(drop=True)
