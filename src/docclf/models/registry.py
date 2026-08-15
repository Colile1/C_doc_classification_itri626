# registry.py — the one entry point that maps a config name onto an architecture.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from docclf.models.baseline_cnn import build_baseline_cnn
from docclf.models.pretrained_cnn import build_pretrained_cnn
from docclf.models.transformer import build_transformer

# Three families, fixed by the brief. The alternatives inside the second and
# third rows are alternatives within one family; they are not a fourth model.
MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "baseline_cnn": build_baseline_cnn,
    "resnet50": lambda n, cfg: build_pretrained_cnn(n, cfg, backbone="resnet50"),
    "mobilenetv3": lambda n, cfg: build_pretrained_cnn(n, cfg, backbone="mobilenetv3"),
    "vit_b16": lambda n, cfg: build_transformer(n, cfg, backbone="vit_b16"),
    "swin_t": lambda n, cfg: build_transformer(n, cfg, backbone="swin_t"),
}

MODEL_FAMILIES: dict[str, str] = {
    "baseline_cnn": "CNN from scratch",
    "resnet50": "Pretrained convolutional",
    "mobilenetv3": "Pretrained convolutional",
    "vit_b16": "Pretrained transformer",
    "swin_t": "Pretrained transformer",
}


def build_model(name: str, n_classes: int, cfg: Any) -> nn.Module:
    """Purpose: construct one of the three architectures by name.

    Inputs: the model name from `train.model`, the class count, the RunConfig.
    Output: an `nn.Module` whose final layer emits `n_classes` logits.
    Raises ValueError listing the valid names when `name` is unknown.
    """
    builder = MODEL_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown model: {name!r}. Valid models are "
            f"{', '.join(sorted(MODEL_BUILDERS))}. Adding a fourth architecture is a "
            f"design change: the brief fixes three model families."
        )
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}.")
    return builder(n_classes, cfg)


def model_family(name: str) -> str:
    """Purpose: name the architectural family a model belongs to, for the report.

    Input: a model name. Output: the family label. Raises ValueError if unknown.
    """
    if name not in MODEL_FAMILIES:
        raise ValueError(
            f"Unknown model: {name!r}. Valid models are "
            f"{', '.join(sorted(MODEL_FAMILIES))}."
        )
    return MODEL_FAMILIES[name]
