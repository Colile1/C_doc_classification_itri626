# pretrained_cnn.py — the transfer-learning entry, with the freezing decision recorded.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision import models

from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)

# Backbone name -> (torchvision constructor, weights enum, ordered stage names)
_BACKBONES = {
    "resnet50": ("resnet50", "ResNet50_Weights", ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")),
    "mobilenetv3": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", ()),
}


def build_pretrained_cnn(
    n_classes: int, cfg: Any, backbone: str = "resnet50"
) -> nn.Module:
    """Purpose: build the ImageNet-pretrained convolutional model (objective O4).

    Inputs: the class count, the RunConfig (`train.pretrained`, `train.freeze_upto`),
    and the backbone name — `resnet50` or `mobilenetv3`, whichever the config selects.
    Output: an `nn.Module` emitting `n_classes` logits. With
    `train.pretrained: false` no weights are downloaded, so the shape tests run
    offline.
    """
    if backbone not in _BACKBONES:
        raise ValueError(
            f"Unknown pretrained CNN backbone: {backbone!r}. Valid values are "
            f"{', '.join(sorted(_BACKBONES))}."
        )
    constructor_name, weights_name, _ = _BACKBONES[backbone]
    pretrained = bool(getattr(cfg.train, "pretrained", True)) if cfg else False
    weights = getattr(models, weights_name).DEFAULT if pretrained else None
    model = getattr(models, constructor_name)(weights=weights)
    _replace_head(model, backbone, n_classes)
    freeze_upto = getattr(cfg.train, "freeze_upto", None) if cfg else None
    if freeze_upto:
        freeze_backbone(model, freeze_upto)
    _LOGGER.info("Pretrained CNN backbone: %s", backbone)
    _LOGGER.info("Pretrained weights loaded: %s", pretrained)
    return model


def _replace_head(model: nn.Module, backbone: str, n_classes: int) -> None:
    """Purpose: swap the ImageNet head for one sized to the eight classes.

    Inputs: the model, its backbone name and the class count. Output: None.
    """
    if backbone == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, n_classes)


def freeze_backbone(model: nn.Module, upto: str) -> list[str]:
    """Purpose: freeze the early layers and record exactly which ones.

    Inputs: the model and the name of the last stage to keep frozen. For
    ResNet50 that is one of conv1, bn1, layer1..layer4; for MobileNetV3 it is
    `features.N`, meaning feature blocks 0 to N inclusive.
    Output: the list of frozen parameter-name prefixes, so the report can state
    the choice rather than assert it.
    """
    stages = _stage_order(model)
    if upto not in stages:
        raise ValueError(
            f"Cannot freeze up to {upto!r}: this model's freezable stages are "
            f"{', '.join(stages)}."
        )
    cutoff = stages.index(upto)
    frozen: list[str] = []
    for stage in stages[: cutoff + 1]:
        module = _resolve(model, stage)
        for parameter in module.parameters():
            parameter.requires_grad = False
        frozen.append(stage)
    _LOGGER.info("Frozen stages: %s", ", ".join(frozen))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _LOGGER.info("Trainable parameters: %d", trainable)
    return frozen


def _stage_order(model: nn.Module) -> list[str]:
    """Purpose: list the model's stages, earliest first.

    Input: the model. Output: stage names usable with `freeze_backbone`.
    """
    if hasattr(model, "layer4"):
        return list(_BACKBONES["resnet50"][2])
    if hasattr(model, "features"):
        return [f"features.{index}" for index in range(len(model.features))]
    raise ValueError(
        f"{type(model).__name__} exposes no recognised backbone stages to freeze."
    )


def _resolve(model: nn.Module, dotted: str) -> nn.Module:
    """Purpose: fetch a submodule by its dotted name.

    Inputs: the model and a name such as `layer2` or `features.3`.
    Output: the submodule.
    """
    module: nn.Module = model
    for part in dotted.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module
