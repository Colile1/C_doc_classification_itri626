# schema.py — the closed set of settings a run may specify.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Mapping

CLASS_NAMES: tuple[str, ...] = (
    "invoice",
    "budget",
    "form",
    "letter",
    "memo",
    "handwritten",
    "resume",
    "advertisement",
)


@dataclass(frozen=True)
class PathsConfig:
    """Directory names, relative to `root` unless already absolute."""

    root: str = "."
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    splits: str = "data/splits"
    checkpoints: str = "outputs/checkpoints"
    figures: str = "outputs/figures"
    metrics: str = "outputs/metrics"
    logs: str = "outputs/logs"


@dataclass(frozen=True)
class DataConfig:
    """How the raw archive is located, inventoried and reduced to a working set."""

    archive_name: str = "rvl-cdip.tar.gz"
    archive_sha256: str | None = None
    source_url: str | None = None
    n_per_class: int = 2000
    min_side_pixels: int = 64
    hash_distance: int = 4
    group_pattern: str = r"^(?P<group>[^_]+)"


@dataclass(frozen=True)
class SplitConfig:
    """The grouped split. `ratios` are train / validation / test and must sum to 1."""

    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    group_column: str = "doc_group_id"
    stratify_column: str = "class_name"


@dataclass(frozen=True)
class ImageConfig:
    """Input geometry and normalisation. No horizontal flip is offered, by design."""

    size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    rotation_degrees: float = 5.0
    brightness: float = 0.15
    contrast: float = 0.15
    blur_probability: float = 0.20
    crop_scale: tuple[float, float] = (0.85, 1.0)
    preserve_aspect_ratio: bool = False


@dataclass(frozen=True)
class TrainConfig:
    """Everything the shared training loop reads. Per-architecture values live here."""

    model: str = "baseline_cnn"
    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimiser: str = "adamw"
    scheduler: str = "cosine"
    early_stopping_patience: int = 5
    monitor: str = "val_macro_f1"
    class_weights: bool = False
    freeze_upto: str | None = None
    pretrained: bool = True
    device: str = "auto"
    amp: bool = False


@dataclass(frozen=True)
class EvalConfig:
    """Test-time settings, kept apart so evaluation never inherits training batching."""

    batch_size: int = 32
    num_workers: int = 0
    latency_batch_size: int = 1
    latency_repeats: int = 50
    error_samples: int = 12
    review_threshold: float | None = None


@dataclass(frozen=True)
class RunConfig:
    """One fully specified run. Every result in the report traces back to one of these."""

    name: str = "base"
    seed: int = 626
    classes: tuple[str, ...] = CLASS_NAMES
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def n_classes(self) -> int:
        """Number of target classes. Purpose: keep the count in one place."""
        return len(self.classes)

    @property
    def class_to_id(self) -> dict[str, int]:
        """Map class name to its fixed integer id, ordered as `classes` is."""
        return {name: index for index, name in enumerate(self.classes)}


def dataclass_field_names(cls: type) -> tuple[str, ...]:
    """Purpose: list the accepted keys of a config dataclass.

    Input: a dataclass type. Output: its field names in declaration order.
    """
    if not is_dataclass(cls):
        raise TypeError(f"Not a configuration dataclass: {cls!r}")
    return tuple(item.name for item in fields(cls))


def has_default(cls: type, name: str) -> bool:
    """Purpose: report whether a config field may be omitted from the YAML.

    Input: a dataclass type and a field name. Output: True when a default exists.
    """
    for item in fields(cls):
        if item.name == name:
            return item.default is not MISSING or item.default_factory is not MISSING
    raise KeyError(f"No such configuration field: {cls.__name__}.{name}")


SECTION_TYPES: Mapping[str, Any] = {
    "paths": PathsConfig,
    "data": DataConfig,
    "split": SplitConfig,
    "image": ImageConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}
