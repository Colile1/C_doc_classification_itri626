# test_crossmodel.py — the cross-model comparison must read real run directories.
# Author: Colile Sibanda
#
# The comparison is what the report's conclusions rest on, so it has to combine
# runs that finished on different machines, and it has to be honest about the
# models it could not find rather than quietly dropping them.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from docclf.evaluation.crossmodel import (
    collect_results,
    plot_accuracy_against_cost,
    plot_history_comparison,
    plot_metric_comparison,
    plot_per_class_comparison,
    strengths_table,
)

CLASSES = ("invoice", "budget", "form", "letter")


def write_run(root: Path, model: str, scores: dict[str, float], epochs: int = 3) -> Path:
    """Purpose: lay out one model's run directory the way a finished run leaves it.

    Inputs: the parent directory, the model name, a per-class F1 mapping and the
    number of epochs to record. Output: the run directory.
    """
    directory = root / model
    metrics = directory / "metrics"
    metrics.mkdir(parents=True)
    macro = sum(scores.values()) / len(scores)
    pd.DataFrame(
        [{"model": model, "n_samples": 400, "accuracy": macro, "macro_f1": macro}]
    ).to_csv(metrics / f"metrics_{model}.csv", index=False)
    pd.DataFrame(
        [
            {"class_name": name, "support": 100, "precision": value,
             "recall": value, "f1": value}
            for name, value in scores.items()
        ]
    ).to_csv(metrics / f"per_class_{model}.csv", index=False)
    pd.DataFrame(
        [
            {"epoch": epoch, "train_loss": 1.0 / epoch, "val_loss": 1.1 / epoch,
             "val_accuracy": macro, "val_macro_f1": macro, "seconds": 10.0}
            for epoch in range(1, epochs + 1)
        ]
    ).to_csv(metrics / f"history_{model}.csv", index=False)
    pd.DataFrame(
        [{"model_name": model, "device": "cpu", "median_ms_per_image": 5.0,
          "total_parameters": 1_000_000, "size_mb": 4.0}]
    ).to_csv(metrics / f"efficiency_{model}.csv", index=False)
    return directory


@pytest.fixture
def two_runs(tmp_path: Path) -> dict[str, Path]:
    """Purpose: two finished runs that disagree about which class they win."""
    first = write_run(
        tmp_path, "model_a",
        {"invoice": 0.70, "budget": 0.80, "form": 0.60, "letter": 0.90},
    )
    second = write_run(
        tmp_path, "model_b",
        {"invoice": 0.75, "budget": 0.70, "form": 0.65, "letter": 0.85},
    )
    return {"model_a": first, "model_b": second}


def test_collect_results_stacks_every_model(two_runs) -> None:
    """Each frame must carry one block of rows per model, labelled by model."""
    frames = collect_results(two_runs)
    assert set(frames) == {"comparison", "per_class", "history", "errors"} - {"errors"}
    assert sorted(frames["comparison"]["model"].unique()) == ["model_a", "model_b"]
    assert len(frames["per_class"]) == 2 * len(CLASSES)
    assert len(frames["history"]) == 6


def test_efficiency_is_merged_onto_the_comparison_row(two_runs) -> None:
    """The cost figure needs latency and size beside the scores, not in a
    separate file, so collect_results must merge them."""
    comparison = collect_results(two_runs)["comparison"]
    for column in ("median_ms_per_image", "total_parameters", "size_mb"):
        assert column in comparison.columns
    assert comparison["median_ms_per_image"].notna().all()


def test_a_model_without_results_is_skipped_not_fatal(two_runs, tmp_path) -> None:
    """Runs finish at different times on different machines, so an unfinished
    model must not prevent the comparison of the ones that are done."""
    runs = dict(two_runs)
    runs["model_c"] = tmp_path / "model_c"  # never created
    comparison = collect_results(runs)["comparison"]
    assert sorted(comparison["model"].unique()) == ["model_a", "model_b"]


def test_collect_results_is_empty_when_nothing_has_finished(tmp_path) -> None:
    """With no results at all, the caller gets an empty dict to report on."""
    assert collect_results({"model_a": tmp_path / "missing"}) == {}


def test_strengths_table_names_the_winner_and_the_margin(two_runs) -> None:
    """The discussion needs per-class winners, not just an aggregate score."""
    per_class = collect_results(two_runs)["per_class"]
    strengths = strengths_table(per_class)
    assert len(strengths) == len(CLASSES)
    budget = strengths.set_index("class_name").loc["budget"]
    assert budget["best_model"] == "model_a"
    assert budget["runner_up"] == "model_b"
    assert budget["margin"] == pytest.approx(0.10, abs=1e-6)
    invoice = strengths.set_index("class_name").loc["invoice"]
    assert invoice["best_model"] == "model_b", (
        "a model can win overall and still lose a class; the table must say so"
    )


def test_every_figure_is_written(two_runs, tmp_path) -> None:
    """All four comparison figures must produce a non-empty file."""
    frames = collect_results(two_runs)
    figures = tmp_path / "figures"
    written = [
        plot_metric_comparison(frames["comparison"], figures / "metrics.png"),
        plot_per_class_comparison(frames["per_class"], figures / "per_class.png"),
        plot_history_comparison(frames["history"], figures / "history.png"),
        plot_accuracy_against_cost(frames["comparison"], figures / "cost.png"),
    ]
    for path in written:
        assert path.is_file()
        assert path.stat().st_size > 1000


def test_a_missing_metric_column_is_named(two_runs, tmp_path) -> None:
    """A blank figure would be worse than a failure, so the column is named."""
    comparison = collect_results(two_runs)["comparison"]
    with pytest.raises(KeyError, match="roc_auc"):
        plot_metric_comparison(
            comparison, tmp_path / "x.png", metrics=("accuracy", "roc_auc")
        )


def test_single_model_comparison_still_draws(tmp_path) -> None:
    """The comparison is built from the first finished model onward, so one
    model must not be a special case that crashes."""
    runs = {"only": write_run(tmp_path, "only", {"invoice": 0.8, "budget": 0.7})}
    frames = collect_results(runs)
    path = plot_per_class_comparison(frames["per_class"], tmp_path / "one.png")
    assert path.is_file()
    strengths = strengths_table(frames["per_class"])
    assert (strengths["models_compared"] == 1).all()
    assert (strengths["runner_up"] == "").all()
