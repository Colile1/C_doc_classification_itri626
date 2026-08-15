# run_train.py — train one model on the committed split, and record the run.
# Author: Colile Sibanda

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from docclf.config.loader import load_config  # noqa: E402
from docclf.config.paths import ProjectPaths, resolve_paths  # noqa: E402
from docclf.config.schema import RunConfig  # noqa: E402
from docclf.config.seed import set_global_seed  # noqa: E402
from docclf.data.loaders import build_dataloaders  # noqa: E402
from docclf.models.registry import build_model, model_family  # noqa: E402
from docclf.models.summary import describe_model  # noqa: E402
from docclf.splits.io import load_splits  # noqa: E402
from docclf.splits.leakage import assert_no_group_overlap  # noqa: E402
from docclf.training.class_weights import compute_class_weights  # noqa: E402
from docclf.training.history import TrainingHistory  # noqa: E402
from docclf.training.trainer import fit  # noqa: E402
from docclf.utils.env import detect_environment, select_device  # noqa: E402
from docclf.utils.logging import add_file_handler, get_logger, log_values  # noqa: E402
from docclf.utils.timing import timed  # noqa: E402

LOGGER = get_logger("run_train")


def train_model(cfg: RunConfig, paths: ProjectPaths) -> TrainingHistory:
    """Purpose: train the configured model and write its history and checkpoints.

    Inputs: the RunConfig (`train.model` names the architecture) and the paths.
    Output: the TrainingHistory. The split is re-verified before training, so a
    hand-edited split file cannot quietly produce an invalid result.
    """
    splits = load_splits(paths.splits)
    assert_no_group_overlap(splits, cfg.split.group_column)
    loaders = build_dataloaders(splits, cfg)
    device = select_device(cfg)
    model = build_model(cfg.train.model, cfg.n_classes, cfg)
    summary = describe_model(model, cfg.train.model)
    weights = None
    if cfg.train.class_weights:
        weights = compute_class_weights(
            splits[splits["split"] == "train"], cfg.n_classes
        )
    log_values(
        LOGGER,
        {
            "Model": cfg.train.model,
            "Family": model_family(cfg.train.model),
            "Device": device,
            "Total parameters": summary.total_parameters,
            "Trainable parameters": summary.trainable_parameters,
            "Model size MB": summary.size_mb,
            "Epochs": cfg.train.epochs,
            "Batch size": cfg.train.batch_size,
            "Learning rate": cfg.train.learning_rate,
            "Class weighting": cfg.train.class_weights,
        },
    )
    with timed(f"Training {cfg.train.model}"):
        history = fit(model, loaders, cfg, device, paths.checkpoints, weights)
    history_path = history.save(paths.metrics)
    best = history.best(cfg.train.monitor)
    log_values(
        LOGGER,
        {
            "Epochs run": len(history.records),
            "Best epoch": best.epoch,
            "Best validation macro-F1": best.val_macro_f1,
            "Best validation accuracy": best.val_accuracy,
            "Mean seconds per epoch": round(
                sum(item.seconds for item in history.records) / len(history.records), 2
            ),
            "History": history_path,
            "Checkpoints": paths.checkpoints,
        },
    )
    return history


def main() -> int:
    """Purpose: parse the command line and run one training run.

    Input: none (reads `sys.argv`). Output: the process exit code.
    """
    parser = argparse.ArgumentParser(description="Train one model on the committed split.")
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument("--env", default=None, help="local, colab or hpc")
    parser.add_argument("--model", default=None, help="Override train.model")
    arguments = parser.parse_args()
    env = arguments.env or detect_environment()
    cfg = load_config(arguments.config, env=env)
    if arguments.model:
        cfg = _with_model(cfg, arguments.model)
    paths = resolve_paths(cfg, env).ensure()
    add_file_handler(paths.logs / f"train_{cfg.train.model}.log")
    set_global_seed(cfg.seed)
    log_values(LOGGER, {"Config": arguments.config, "Environment": env, "Seed": cfg.seed})
    train_model(cfg, paths)
    return 0


def _with_model(cfg: RunConfig, model: str) -> RunConfig:
    """Purpose: apply the `--model` override to a frozen config.

    Inputs: the loaded config and the model name.
    Output: a new RunConfig differing only in `train.model`.
    """
    import dataclasses

    return dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, model=model)
    )


if __name__ == "__main__":
    raise SystemExit(main())
