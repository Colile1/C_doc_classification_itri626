# trainer.py — the one training loop, shared unchanged by all three architectures.
# Author: Colile Sibanda

from __future__ import annotations

import time
from typing import Any, Mapping

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from docclf.training.callbacks import Checkpoint, EarlyStopping
from docclf.training.history import EpochRecord, TrainingHistory
from docclf.training.optim import build_criterion, build_optimiser, build_scheduler
from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)

# This module reads two loaders only: `train` and `val`. Held-out evaluation is
# the evaluation module's job, so that no code path here can touch it.
TRAIN_KEY = "train"
VAL_KEY = "val"


def fit(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    cfg: Any,
    device: torch.device,
    checkpoint_dir: Any = None,
    class_weights: torch.Tensor | None = None,
) -> TrainingHistory:
    """Purpose: train one model and record what happened, epoch by epoch.

    Inputs: the model, the loaders (`train` and `val` are the only keys read),
    the RunConfig, the device, the checkpoint directory, and optional class weights.
    Output: a TrainingHistory. A checkpoint is written every epoch, and the best
    epoch by `train.monitor` is kept separately for evaluation to load.
    """
    for key in (TRAIN_KEY, VAL_KEY):
        if key not in loaders:
            raise KeyError(
                f"Loader {key!r} is required to train. Build the loaders with "
                f"data.loaders.build_dataloaders."
            )
    model = model.to(device)
    optimiser = build_optimiser(model, cfg)
    scheduler = build_scheduler(optimiser, cfg)
    criterion = build_criterion(
        cfg, class_weights.to(device) if class_weights is not None else None
    )
    stopper = EarlyStopping(
        patience=int(cfg.train.early_stopping_patience), monitor=str(cfg.train.monitor)
    )
    checkpoint = (
        Checkpoint(checkpoint_dir, cfg.train.model) if checkpoint_dir is not None else None
    )
    history = TrainingHistory(model_name=str(cfg.train.model), config_name=str(cfg.name))
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, int(cfg.train.epochs) + 1):
        started = time.perf_counter()
        train_loss = _train_one_epoch(
            model, loaders[TRAIN_KEY], criterion, optimiser, device, scaler, use_amp
        )
        val_loss, val_accuracy, val_macro_f1 = _validate(
            model, loaders[VAL_KEY], criterion, device
        )
        record = EpochRecord(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6),
            val_accuracy=round(val_accuracy, 6),
            val_macro_f1=round(val_macro_f1, 6),
            learning_rate=optimiser.param_groups[0]["lr"],
            seconds=round(time.perf_counter() - started, 3),
        )
        history.append(record)
        _log_epoch(record)
        _step_scheduler(scheduler, val_macro_f1)
        stop = stopper.update(getattr(record, str(cfg.train.monitor)))
        if checkpoint is not None:
            checkpoint.save(model, epoch, vars(record), best=stopper.is_best())
        if stop:
            break
    return history


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    """Purpose: run one pass over the training split.

    Inputs: model, loader, loss, optimiser, device, gradient scaler, AMP flag.
    Output: the mean training loss over the epoch.
    """
    model.train()
    total, seen = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total += float(loss.item()) * labels.size(0)
        seen += labels.size(0)
    if seen == 0:
        raise ValueError("The training loader yielded no batches; the split is empty.")
    return total / seen


@torch.no_grad()
def _validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, float]:
    """Purpose: measure the model on the validation split.

    Inputs: model, validation loader, loss, device.
    Output: `(mean loss, accuracy, macro F1)`. Macro F1 is computed here rather
    than imported, because training must not depend on the evaluation module.
    """
    model.eval()
    total, seen = 0.0, 0
    predictions: list[int] = []
    truths: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total += float(criterion(logits, labels).item()) * labels.size(0)
        seen += labels.size(0)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        truths.extend(labels.cpu().tolist())
    if seen == 0:
        raise ValueError("The validation loader yielded no batches; the split is empty.")
    correct = sum(int(a == b) for a, b in zip(predictions, truths))
    macro_f1 = float(f1_score(truths, predictions, average="macro", zero_division=0))
    return total / seen, correct / seen, macro_f1


def _step_scheduler(scheduler: Any, monitored: float) -> None:
    """Purpose: advance the schedule, passing the metric only to plateau schedulers.

    Inputs: the scheduler (or None) and the monitored validation metric.
    Output: None.
    """
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(monitored)
    else:
        scheduler.step()


def _log_epoch(record: EpochRecord) -> None:
    """Purpose: report one epoch in the project's output format.

    Input: the epoch record. Output: None.
    """
    _LOGGER.info(
        "Epoch %d: train loss %.4f | val loss %.4f | val accuracy %.4f | "
        "val macro-F1 %.4f | seconds %.1f",
        record.epoch,
        record.train_loss,
        record.val_loss,
        record.val_accuracy,
        record.val_macro_f1,
        record.seconds,
    )
