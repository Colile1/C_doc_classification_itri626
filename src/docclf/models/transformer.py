# transformer.py — the transformer entry, a genuinely different architectural family.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Any

import timm
from torch import nn

from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)

# ViT-B/16 first, Swin-T if ViT will not fit the Colab session budget.
_BACKBONES = {
    "vit_b16": "vit_base_patch16_224",
    "swin_t": "swin_tiny_patch4_window7_224",
}

_NATIVE_SIZE = 224


def build_transformer(n_classes: int, cfg: Any, backbone: str = "vit_b16") -> nn.Module:
    """Purpose: build the pretrained transformer classifier (objective O5).

    Inputs: the class count, the RunConfig, and the backbone name — `vit_b16`
    or `swin_t`.
    Output: an `nn.Module` emitting `n_classes` logits. An image size other than
    224 is passed to timm as `img_size`, which is what lets the smoke config run
    the same code path on 64-pixel pages.
    """
    if backbone not in _BACKBONES:
        raise ValueError(
            f"Unknown transformer backbone: {backbone!r}. Valid values are "
            f"{', '.join(sorted(_BACKBONES))}."
        )
    pretrained = bool(getattr(cfg.train, "pretrained", True)) if cfg else False
    size = int(getattr(cfg.image, "size", _NATIVE_SIZE)) if cfg else _NATIVE_SIZE
    kwargs: dict[str, Any] = {"num_classes": n_classes, "pretrained": pretrained}
    if size != _NATIVE_SIZE:
        kwargs["img_size"] = size
        if pretrained:
            _LOGGER.warning(
                "Transformer input size %d differs from the pretrained 224; "
                "position embeddings are interpolated",
                size,
            )
    model = timm.create_model(_BACKBONES[backbone], **kwargs)
    _LOGGER.info("Transformer backbone: %s", _BACKBONES[backbone])
    _LOGGER.info("Pretrained weights loaded: %s", pretrained)
    return model
