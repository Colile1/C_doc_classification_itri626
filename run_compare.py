#!/usr/bin/env python
# run_compare.py — assemble the cross-model comparison from the recorded runs.
# Author: Colile Sibanda
#
# Reads only the files each finished run leaves behind, so the comparison can be
# rebuilt on any machine, and runs that finished on different machines combine
# without re-running inference. Nothing here touches a model or the test images.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from docclf.evaluation.compare import recommend  # noqa: E402
from docclf.evaluation.crossmodel import (  # noqa: E402
    collect_results,
    plot_accuracy_against_cost,
    plot_history_comparison,
    plot_metric_comparison,
    plot_per_class_comparison,
    strengths_table,
)
from docclf.models.registry import model_family  # noqa: E402
from docclf.utils.logging import add_file_handler, get_logger, log_values  # noqa: E402
from run_status import MODELS  # noqa: E402

LOGGER = get_logger("docclf.run_compare")

COMPARISON_DIR = ROOT / "outputs" / "comparison"


def build_comparison(
    models: list[str], destination: Path, strict: bool = False
) -> pd.DataFrame:
    """Purpose: write every cross-model table and figure the report needs.

    Inputs: the model names to include, the output directory, and `strict`,
    which fails when a requested model has no recorded result instead of
    skipping it.
    Output: the combined comparison table. Figures and CSVs are written beside
    it under `destination`.
    """
    runs = {name: ROOT / MODELS[name]["dir"] for name in models if name in MODELS}
    unknown = [name for name in models if name not in MODELS]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Valid names are "
            f"{', '.join(sorted(MODELS))}."
        )
    frames = collect_results(runs)
    comparison = frames.get("comparison")
    if comparison is None or comparison.empty:
        raise FileNotFoundError(
            "No evaluated model was found. Train and evaluate at least one model "
            "first; `python run_status.py` shows what is outstanding."
        )

    found = sorted(comparison["model"].unique())
    missing = [name for name in models if name not in found]
    if missing:
        message = (
            f"No held-out metrics for {', '.join(missing)}; the comparison covers "
            f"{', '.join(found)} only."
        )
        if strict:
            raise FileNotFoundError(message)
        LOGGER.warning("%s", message)

    comparison = comparison.copy()
    comparison["family"] = [model_family(name) for name in comparison["model"]]
    ordered = ["model", "family"] + [
        name for name in comparison.columns if name not in ("model", "family")
    ]
    comparison = (
        comparison[ordered]
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    destination.mkdir(parents=True, exist_ok=True)
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(destination / "comparison.csv", index=False, encoding="utf-8")
    written = [destination / "comparison.csv"]

    # Efficiency columns only exist once a model has been evaluated, and only
    # the cost figure needs them, so it is drawn conditionally.
    if "median_ms_per_image" in comparison.columns:
        written.append(
            plot_accuracy_against_cost(
                comparison, figures / "accuracy_against_cost.png"
            )
        )
    written.append(
        plot_metric_comparison(comparison, figures / "metric_comparison.png")
    )

    per_class = frames.get("per_class")
    if per_class is not None and not per_class.empty:
        per_class.to_csv(
            destination / "per_class_all_models.csv", index=False, encoding="utf-8"
        )
        written.append(destination / "per_class_all_models.csv")
        written.append(
            plot_per_class_comparison(per_class, figures / "per_class_comparison.png")
        )
        strengths = strengths_table(per_class)
        strengths.to_csv(
            destination / "class_strengths.csv", index=False, encoding="utf-8"
        )
        written.append(destination / "class_strengths.csv")
        LOGGER.info("Which model wins each class:")
        for _, row in strengths.iterrows():
            LOGGER.info(
                "  %-14s %-16s F1 %.4f (margin %s over %s)",
                row["class_name"], row["best_model"], row["best_f1"],
                row["margin"], row["runner_up"] or "nothing",
            )

    history = frames.get("history")
    if history is not None and not history.empty:
        history.to_csv(
            destination / "history_all_models.csv", index=False, encoding="utf-8"
        )
        written.append(destination / "history_all_models.csv")
        written.append(
            plot_history_comparison(history, figures / "history_comparison.png")
        )

    errors = frames.get("errors")
    if errors is not None and not errors.empty:
        errors.to_csv(
            destination / "errors_all_models.csv", index=False, encoding="utf-8"
        )
        written.append(destination / "errors_all_models.csv")

    log_values(
        LOGGER,
        {
            "Models compared": ", ".join(comparison["model"]),
            "Families represented": comparison["family"].nunique(),
            "Best macro-F1": f"{comparison.iloc[0]['macro_f1']:.4f} "
                             f"({comparison.iloc[0]['model']})",
            "Files written": len(written),
        },
    )
    if len(comparison) > 1:
        LOGGER.info("Recommendation: %s", recommend(comparison))
    for path in written:
        LOGGER.info("Wrote %s", path.relative_to(ROOT))
    return comparison


def main() -> int:
    """Purpose: the command-line entry point. Output: the process exit code."""
    parser = argparse.ArgumentParser(
        description="Build the cross-model comparison from the recorded runs."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"Models to include. Default: all of {', '.join(sorted(MODELS))}.",
    )
    parser.add_argument(
        "--out",
        default=str(COMPARISON_DIR),
        help="Destination directory (default outputs/comparison).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a requested model has no recorded held-out metrics.",
    )
    arguments = parser.parse_args()
    destination = Path(arguments.out)
    destination.mkdir(parents=True, exist_ok=True)
    add_file_handler(destination / "compare.log")
    build_comparison(
        arguments.models or list(MODELS), destination, strict=arguments.strict
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
