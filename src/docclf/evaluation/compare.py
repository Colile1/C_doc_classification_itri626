# compare.py — the one table that puts all three models side by side.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from docclf.evaluation.efficiency import EfficiencySummary
from docclf.evaluation.metrics import MetricSet

COMPARISON_FILENAME = "comparison.csv"


def build_comparison_table(
    results: Mapping[str, MetricSet],
    efficiency: Mapping[str, EfficiencySummary] | None = None,
    families: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Purpose: assemble the comparison every conclusion in the report rests on.

    Inputs: metrics per model name, optional efficiency summaries, and optional
    architectural family labels.
    Output: a DataFrame with one row per model, ordered by macro-F1 descending.
    Raises ValueError when the models were not evaluated on the same number of
    pages, because a comparison across different sets is not a comparison.
    """
    if not results:
        raise ValueError("No results were supplied; the comparison table would be empty.")
    sizes = {name: metrics.n_samples for name, metrics in results.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(
            f"Models were evaluated on different numbers of pages: {sizes}. All three "
            f"must be evaluated on the same held-out split for the comparison to mean "
            f"anything."
        )
    rows: list[dict[str, object]] = []
    for name, metrics in results.items():
        row = metrics.as_row()
        row["model"] = name
        if families and name in families:
            row["family"] = families[name]
        if efficiency and name in efficiency:
            summary = efficiency[name].as_row()
            row["median_ms_per_image"] = summary["median_ms_per_image"]
            row["total_parameters"] = summary["total_parameters"]
            row["size_mb"] = summary["size_mb"]
            row["device"] = summary["device"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    ordered = ["model"] + (["family"] if "family" in frame.columns else [])
    remaining = [name for name in frame.columns if name not in ordered]
    return (
        frame[ordered + remaining]
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )


def save_comparison(frame: pd.DataFrame, directory: str | Path) -> Path:
    """Purpose: write `comparison.csv`, the file the report's numbers must come from.

    Inputs: the comparison table and the destination directory.
    Output: the file written.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / COMPARISON_FILENAME
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def recommend(frame: pd.DataFrame, latency_budget_ms: float | None = None) -> str:
    """Purpose: state which model is recommended, and on what grounds.

    Inputs: the comparison table and an optional per-image latency budget.
    Output: a sentence naming the model and the numbers behind the choice. The
    recommendation is the best macro-F1 among the models inside the budget.
    """
    if frame.empty:
        raise ValueError("The comparison table is empty; there is nothing to recommend.")
    eligible = frame
    if latency_budget_ms is not None and "median_ms_per_image" in frame.columns:
        within = frame[frame["median_ms_per_image"] <= latency_budget_ms]
        if not within.empty:
            eligible = within
    best = eligible.sort_values("macro_f1", ascending=False).iloc[0]
    latency = (
        f", {best['median_ms_per_image']:.1f} ms per image"
        if "median_ms_per_image" in best.index
        else ""
    )
    return (
        f"Recommended model: {best['model']} (macro-F1 {best['macro_f1']:.4f}, "
        f"accuracy {best['accuracy']:.4f}{latency})"
    )
