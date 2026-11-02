# run_train.py — train one model on the committed split, and record the run.
# Author: Colile Sibanda

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
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
import pandas as pd  # noqa: E402
import torch  # noqa: E402

LOGGER = get_logger("run_train")


def train_model(
    cfg: RunConfig,
    paths: ProjectPaths,
    config_path: str | None = None,
    environment: str | None = None,
    initial_checkpoint: str | None = None,
    resume: bool = False,
) -> TrainingHistory:
    """Purpose: train the configured model and write its history and checkpoints.

    Inputs: the RunConfig (`train.model` names the architecture) and the paths.
    Output: the TrainingHistory. The split is re-verified before training, so a
    hand-edited split file cannot quietly produce an invalid result. When an
    initial checkpoint is supplied, its model weights are loaded strictly;
    optimizer and scheduler state start fresh for this run. With `resume` the
    run instead continues from this model's own every-epoch checkpoint, keeping
    the optimizer, scheduler and early-stopping state, so a paused run can be
    finished on another machine.
    """
    run_started_at = datetime.now(timezone.utc)
    run_started = time.perf_counter()
    with timed("Split file loading") as split_load_time:
        splits = load_splits(paths.splits, root=paths.root, raw=paths.raw)
    with timed("Split leakage verification") as leakage_check_time:
        assert_no_group_overlap(splits, cfg.split.group_column)
    with timed("Data loader construction") as loader_time:
        loaders = build_dataloaders(splits, cfg)
    device = select_device(cfg)
    with timed("Model construction") as model_time:
        model = build_model(cfg.train.model, cfg.n_classes, cfg)
    if initial_checkpoint:
        checkpoint_path = Path(initial_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(
                f"Initial checkpoint {checkpoint_path} has no 'state_dict' entry."
            )
        model.load_state_dict(payload["state_dict"], strict=True)
        LOGGER.info(
            "Initial weights loaded from %s (source model %s, epoch %s)",
            checkpoint_path,
            payload.get("model_name", "unknown"),
            payload.get("epoch", "unknown"),
        )
    summary = describe_model(model, cfg.train.model)
    weights = None
    if cfg.train.class_weights:
        weights = compute_class_weights(
            splits[splits["split"] == "train"], cfg.n_classes
        )
    split_counts = splits.groupby("split", sort=True).size().to_dict()
    class_counts = {
        f"{split}/{class_name}": int(count)
        for (split, class_name), count in splits.groupby(
            ["split", "class_name"], sort=True
        ).size().items()
    }
    system_values = {
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Pandas version": pd.__version__,
        "Logical CPU cores": os.cpu_count(),
        "CUDA available": torch.cuda.is_available(),
        "CUDA device": torch.cuda.get_device_name(device) if device.type == "cuda" else "none",
    }
    log_values(LOGGER, {
            "Model": cfg.train.model,
            "Initial checkpoint": initial_checkpoint or "none; random initialization",
            "Family": model_family(cfg.train.model),
            "Device": device,
            "Total parameters": summary.total_parameters,
            "Trainable parameters": summary.trainable_parameters,
            "Model size MB": summary.size_mb,
            "Epochs": cfg.train.epochs,
            "Batch size": cfg.train.batch_size,
            "Validation batch size": cfg.eval.batch_size,
            "Training workers": cfg.train.num_workers,
            "Validation workers": cfg.eval.num_workers,
            "Learning rate": cfg.train.learning_rate,
            "Weight decay": cfg.train.weight_decay,
            "Optimiser": cfg.train.optimiser,
            "Scheduler": cfg.train.scheduler,
            "Image size": cfg.image.size,
            "Mixed precision": bool(cfg.train.amp and device.type == "cuda"),
            "Class weighting": cfg.train.class_weights,
            "Train examples": int(split_counts.get("train", 0)),
            "Validation examples": int(split_counts.get("val", 0)),
            "Test examples (not read by trainer)": int(split_counts.get("test", 0)),
            **{f"Split count {key}": int(value) for key, value in split_counts.items()},
            **{f"Class count {key}": value for key, value in class_counts.items()},
            **system_values,
    })
    if weights is not None:
        log_values(LOGGER, {
            f"Class weight {name}": round(float(weights[index]), 6)
            for index, name in enumerate(cfg.classes)
        })
    with timed(f"Training {cfg.train.model}") as training_time:
        history = fit(
            model, loaders, cfg, device, paths.checkpoints, weights,
            history_dir=paths.metrics, resume=resume,
        )
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
            "Mean training images per second": round(
                sum(item.train_images_per_second for item in history.records)
                / len(history.records), 2
            ),
            "Training seconds": round(training_time.seconds, 3),
            "Total run seconds": round(time.perf_counter() - run_started, 3),
            "Early stopped": len(history.records) < int(cfg.train.epochs),
            "History": history_path,
            "Checkpoints": paths.checkpoints,
        },
    )
    run_finished_at = datetime.now(timezone.utc)
    run_summary_path = paths.metrics / (
        f"run_summary_{cfg.train.model}_{run_started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    run_summary = {
        "run_id": run_started_at.strftime("%Y%m%dT%H%M%SZ"),
        "started_at_utc": run_started_at.isoformat(),
        "finished_at_utc": run_finished_at.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        "config_path": config_path,
        "environment": environment,
        "seed": cfg.seed,
        "config": dataclasses.asdict(cfg),
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "pandas": pd.__version__,
            "logical_cpu_cores": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "dataset": {
            "split_counts": {str(key): int(value) for key, value in split_counts.items()},
            "class_counts_by_split": class_counts,
            "training_batches_per_epoch": len(loaders["train"]),
            "validation_batches_per_epoch": len(loaders["val"]),
        },
        "timing_seconds": {
            "split_loading": round(split_load_time.seconds, 3),
            "leakage_verification": round(leakage_check_time.seconds, 3),
            "data_loader_construction": round(loader_time.seconds, 3),
            "model_construction": round(model_time.seconds, 3),
            "training": round(training_time.seconds, 3),
        },
        "model_summary": dataclasses.asdict(summary),
        "best_epoch": best.epoch,
        "best_monitor": cfg.train.monitor,
        "best_monitor_value": float(getattr(best, cfg.train.monitor)),
        "epochs_run": len(history.records),
        "epochs_requested": int(cfg.train.epochs),
        "history_csv": str(history_path),
        "checkpoints_dir": str(paths.checkpoints),
        "initial_checkpoint": str(Path(initial_checkpoint).expanduser().resolve())
        if initial_checkpoint
        else None,
        "epoch_history": history.to_frame().to_dict(orient="records"),
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    LOGGER.info("Run summary JSON: %s", run_summary_path)
    return history


def main() -> int:
    """Purpose: parse the command line and run one training run.

    Input: none (reads `sys.argv`). Output: the process exit code.
    """
    parser = argparse.ArgumentParser(description="Train one model on the committed split.")
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument("--env", default=None, help="local, colab or hpc")
    parser.add_argument("--model", default=None, help="Override train.model")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights from a saved checkpoint before this run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue this model's paused run from its own every-epoch checkpoint, "
            "restoring the optimiser, scheduler and early-stopping state. Use it to "
            "finish on another machine a run that was stopped part-way."
        ),
    )
    arguments = parser.parse_args()
    env = arguments.env or detect_environment()
    cfg = load_config(arguments.config, env=env)
    if arguments.model:
        cfg = _with_model(cfg, arguments.model)
    paths = resolve_paths(cfg, env).ensure()
    add_file_handler(paths.logs / f"train_{cfg.train.model}.log")
    set_global_seed(cfg.seed)
    log_values(LOGGER, {
        "Config": arguments.config,
        "Environment": env,
        "Seed": cfg.seed,
        "Project root": paths.root,
        "Initial checkpoint": arguments.init_checkpoint or "none; random initialization",
        "Run started UTC": datetime.now(timezone.utc).isoformat(),
    })
    train_model(
        cfg,
        paths,
        config_path=arguments.config,
        environment=env,
        initial_checkpoint=arguments.init_checkpoint,
        resume=arguments.resume,
    )
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
