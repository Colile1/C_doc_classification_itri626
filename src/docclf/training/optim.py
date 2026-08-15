# optim.py — optimiser and scheduler builders, so a per-model rate is a config value.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim

_OPTIMISERS = ("adamw", "adam", "sgd")
_SCHEDULERS = ("cosine", "step", "plateau", "none")


def build_optimiser(model: nn.Module, cfg: Any) -> optim.Optimizer:
    """Purpose: build the optimiser named in the config, over trainable parameters only.

    Inputs: the model and the RunConfig.
    Output: a torch optimiser. Raises ValueError listing the valid names when
    `train.optimiser` is unknown. Frozen parameters are excluded, which is what
    makes the transfer-learning run cheaper as well as correct.
    """
    name = str(cfg.train.optimiser).lower()
    parameters = [item for item in model.parameters() if item.requires_grad]
    if not parameters:
        raise ValueError(
            "No trainable parameters: every layer is frozen. Check train.freeze_upto."
        )
    lr = float(cfg.train.learning_rate)
    weight_decay = float(cfg.train.weight_decay)
    if name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(
        f"Unknown optimiser: {cfg.train.optimiser!r}. Valid values are "
        f"{', '.join(_OPTIMISERS)}."
    )


def build_scheduler(optimiser: optim.Optimizer, cfg: Any) -> Any:
    """Purpose: build the learning-rate schedule named in the config.

    Inputs: the optimiser and the RunConfig.
    Output: a torch scheduler, or None when `train.scheduler` is `none`.
    A `plateau` scheduler is stepped with the monitored validation metric; the
    others are stepped once per epoch.
    """
    name = str(cfg.train.scheduler).lower()
    epochs = int(cfg.train.epochs)
    if name == "none":
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimiser, step_size=max(epochs // 3, 1), gamma=0.1
        )
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=0.5, patience=2
        )
    raise ValueError(
        f"Unknown scheduler: {cfg.train.scheduler!r}. Valid values are "
        f"{', '.join(_SCHEDULERS)}."
    )


def build_criterion(cfg: Any, class_weights: torch.Tensor | None) -> nn.Module:
    """Purpose: build the loss, weighted only when the config asks for it.

    Inputs: the RunConfig and the class weight tensor (or None).
    Output: a cross-entropy loss module.
    """
    if bool(cfg.train.class_weights) and class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()
