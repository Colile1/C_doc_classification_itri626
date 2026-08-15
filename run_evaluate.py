# run_evaluate.py — evaluate a trained model on the held-out split and compare.
# Author: Colile Sibanda

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from docclf.config.loader import load_config  # noqa: E402
from docclf.config.paths import ProjectPaths, resolve_paths  # noqa: E402
from docclf.config.schema import RunConfig  # noqa: E402
from docclf.config.seed import set_global_seed  # noqa: E402
from docclf.data.loaders import build_dataloaders  # noqa: E402
from docclf.evaluation.compare import COMPARISON_FILENAME, recommend, save_comparison  # noqa: E402
from docclf.evaluation.confusion import most_confused_pairs, plot_confusion  # noqa: E402
from docclf.evaluation.curves import plot_learning_curves  # noqa: E402
from docclf.evaluation.efficiency import measure_inference  # noqa: E402
from docclf.evaluation.errors import plot_error_examples, sample_errors  # noqa: E402
from docclf.evaluation.metrics import compute_metrics  # noqa: E402
from docclf.evaluation.predict import collect_predictions  # noqa: E402
from docclf.models.registry import build_model, model_family  # noqa: E402
from docclf.splits.io import load_splits  # noqa: E402
from docclf.splits.leakage import assert_no_group_overlap  # noqa: E402
from docclf.training.callbacks import Checkpoint  # noqa: E402
from docclf.utils.env import detect_environment, select_device  # noqa: E402
from docclf.utils.logging import add_file_handler, get_logger, log_values  # noqa: E402
from docclf.utils.timing import timed  # noqa: E402

LOGGER = get_logger("run_evaluate")
HELD_OUT_SPLIT = "test"


def evaluate_model(cfg: RunConfig, paths: ProjectPaths) -> pd.DataFrame:
    """Purpose: produce every number and figure the report needs for one model.

    Inputs: the RunConfig and the resolved paths.
    Output: the refreshed comparison table. The held-out split is touched here
    and nowhere else in the pipeline.
    """
    splits = load_splits(paths.splits)
    assert_no_group_overlap(splits, cfg.split.group_column)
    loaders = build_dataloaders(splits, cfg)
    device = select_device(cfg)
    model = build_model(cfg.train.model, cfg.n_classes, cfg)
    checkpoint = Checkpoint(paths.checkpoints, cfg.train.model)
    loaded = checkpoint.load_best(model, map_location=device)
    with timed(f"Evaluating {cfg.train.model}"):
        predictions = collect_predictions(
            model, loaders[HELD_OUT_SPLIT], device, cfg.classes
        )
    metrics = compute_metrics(predictions, cfg.train.model)
    efficiency = measure_inference(
        model,
        loaders[HELD_OUT_SPLIT],
        device,
        cfg.train.model,
        repeats=cfg.eval.latency_repeats,
    )
    written = _write_artefacts(cfg, paths, predictions, metrics, efficiency)
    log_values(
        LOGGER,
        {
            "Model": cfg.train.model,
            "Checkpoint epoch": loaded.get("epoch"),
            "Held-out pages": metrics.n_samples,
            "Accuracy": round(metrics.accuracy, 4),
            "Macro-F1": round(metrics.macro_f1, 4),
            "Weighted F1": round(metrics.weighted_f1, 4),
            "Median ms per image": efficiency.median_ms_per_image,
            "Model size MB": efficiency.size_mb,
            **written,
        },
    )
    for row in most_confused_pairs(predictions, top_n=5).to_dict("records"):
        LOGGER.info(
            "Most confused: %s read as %s, %d page(s), %.1f%% of the class",
            row["true"],
            row["predicted"],
            row["count"],
            100 * row["share_of_true"],
        )
    comparison = _refresh_comparison(paths)
    LOGGER.info("%s", recommend(comparison, cfg.eval.review_threshold))
    return comparison


def _write_artefacts(cfg, paths, predictions, metrics, efficiency) -> dict[str, Path]:
    """Purpose: write the metric tables and figures for one model.

    Inputs: the config, paths, predictions, metric set and efficiency summary.
    Output: a mapping of description to the file written.
    """
    files = metrics.save(paths.metrics)
    prediction_path = predictions.save(paths.metrics, cfg.train.model)
    efficiency_path = paths.metrics / f"efficiency_{cfg.train.model}.csv"
    pd.DataFrame([efficiency.as_row()]).to_csv(
        efficiency_path, index=False, encoding="utf-8"
    )
    errors_path = paths.metrics / f"errors_{cfg.train.model}.csv"
    sample_errors(predictions, cfg.eval.error_samples).to_csv(
        errors_path, index=False, encoding="utf-8"
    )
    confusion_path = plot_confusion(
        predictions, paths.figures / f"confusion_{cfg.train.model}.png", cfg.train.model
    )
    normalised_path = plot_confusion(
        predictions,
        paths.figures / f"confusion_normalised_{cfg.train.model}.png",
        cfg.train.model,
        normalise=True,
    )
    error_figure = plot_error_examples(
        predictions,
        paths.figures / f"errors_{cfg.train.model}.png",
        cfg.eval.error_samples,
        cfg.train.model,
    )
    written = {
        "Metrics": files["summary"],
        "Per-class metrics": files["per_class"],
        "Predictions": prediction_path,
        "Efficiency": efficiency_path,
        "Error table": errors_path,
        "Confusion figure": confusion_path,
        "Normalised confusion figure": normalised_path,
        "Error figure": error_figure,
    }
    history_path = paths.metrics / f"history_{cfg.train.model}.csv"
    if history_path.is_file():
        written["Learning curves"] = plot_learning_curves(
            pd.read_csv(history_path),
            paths.figures / f"curves_{cfg.train.model}.png",
            cfg.train.model,
        )
    return written


def _refresh_comparison(paths: ProjectPaths) -> pd.DataFrame:
    """Purpose: rebuild `comparison.csv` from every model evaluated so far.

    Input: the resolved paths.
    Output: the comparison table, also written to `outputs/metrics/`.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(paths.metrics.glob("metrics_*.csv")):
        row = pd.read_csv(path).to_dict("records")[0]
        name = str(row["model"])
        try:
            row["family"] = model_family(name)
        except ValueError:
            row["family"] = "unknown"
        efficiency_path = paths.metrics / f"efficiency_{name}.csv"
        if efficiency_path.is_file():
            row.update(
                pd.read_csv(efficiency_path)[
                    ["median_ms_per_image", "total_parameters", "size_mb", "device"]
                ].to_dict("records")[0]
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    sizes = set(frame["n_samples"]) if "n_samples" in frame.columns else set()
    if len(sizes) > 1:
        LOGGER.warning(
            "Models were evaluated on different page counts %s; the comparison is "
            "not valid until they are re-run on the same split",
            sorted(sizes),
        )
    frame = frame.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    path = save_comparison(frame, paths.metrics)
    LOGGER.info("Comparison table: %s", path)
    LOGGER.info("Comparison rows: %d", len(frame))
    return frame


def main() -> int:
    """Purpose: parse the command line and evaluate one trained model.

    Input: none (reads `sys.argv`). Output: the process exit code.
    """
    parser = argparse.ArgumentParser(
        description=f"Evaluate a trained model on the held-out split ({COMPARISON_FILENAME})."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument("--env", default=None, help="local, colab or hpc")
    parser.add_argument("--model", default=None, help="Override train.model")
    arguments = parser.parse_args()
    env = arguments.env or detect_environment()
    cfg = load_config(arguments.config, env=env)
    if arguments.model:
        import dataclasses

        cfg = dataclasses.replace(
            cfg, train=dataclasses.replace(cfg.train, model=arguments.model)
        )
    paths = resolve_paths(cfg, env).ensure()
    add_file_handler(paths.logs / f"evaluate_{cfg.train.model}.log")
    set_global_seed(cfg.seed)
    log_values(LOGGER, {"Config": arguments.config, "Environment": env})
    evaluate_model(cfg, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
