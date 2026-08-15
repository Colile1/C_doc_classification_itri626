# __init__.py — public surface of the models module.
# Author: Colile Sibanda

from docclf.models.baseline_cnn import build_baseline_cnn
from docclf.models.pretrained_cnn import build_pretrained_cnn, freeze_backbone
from docclf.models.registry import MODEL_BUILDERS, build_model
from docclf.models.summary import ModelSummary, describe_model
from docclf.models.transformer import build_transformer

__all__ = [
    "MODEL_BUILDERS",
    "ModelSummary",
    "build_baseline_cnn",
    "build_model",
    "build_pretrained_cnn",
    "build_transformer",
    "describe_model",
    "freeze_backbone",
]
