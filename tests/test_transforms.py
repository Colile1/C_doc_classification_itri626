# test_transforms.py — no horizontal flip, and no randomness outside training.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from docclf.config.loader import load_config
from docclf.data.dataset import DocumentPageDataset
from docclf.data.loaders import build_dataloaders
from docclf.data.manifest import build_manifest
from docclf.data.transforms import FORBIDDEN_TRANSFORMS, build_transforms, transform_names
from docclf.splits.grouped import make_grouped_split

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SRC = Path(__file__).resolve().parents[1] / "src" / "docclf"
CLASSES = (
    "invoice",
    "budget",
    "form",
    "letter",
    "memo",
    "handwritten",
    "resume",
    "advertisement",
)


def smoke_config():
    """Purpose: the laptop config every transform test runs against."""
    return load_config(CONFIGS / "smoke.yaml")


def test_no_horizontal_flip_in_the_training_pipeline() -> None:
    """A mirrored page of text does not occur in practice and must not be taught."""
    names = transform_names(build_transforms(smoke_config(), train=True))
    for forbidden in FORBIDDEN_TRANSFORMS:
        assert forbidden not in names, f"{forbidden} is in the training pipeline"


def test_no_flip_appears_anywhere_in_the_source() -> None:
    """The guarantee is stronger if no module can reintroduce a flip elsewhere."""
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name == "transforms.py":
            continue  # names the forbidden transforms in order to exclude them
        text = path.read_text(encoding="utf-8")
        if "RandomHorizontalFlip" in text or "hflip" in text:
            offenders.append(path.name)
    assert not offenders, f"a horizontal flip appears in: {', '.join(offenders)}"


def test_evaluation_transforms_contain_no_random_operation() -> None:
    """Validation and test results must be deterministic, run to run."""
    names = transform_names(build_transforms(smoke_config(), train=False))
    assert not [name for name in names if name.startswith("Random")], names


def test_evaluation_transform_is_deterministic(tmp_path: Path) -> None:
    """Asserted by running it twice on the same page, not only by inspection."""
    page = Image.new("L", (300, 400), color=200)
    for x in range(0, 300, 7):
        page.putpixel((x, 150), 0)
    pipeline = build_transforms(smoke_config(), train=False)
    assert torch.equal(pipeline(page), pipeline(page))


def test_training_transform_actually_augments() -> None:
    """Augmentation that never changes anything would be a silent no-op."""
    torch.manual_seed(0)
    page = Image.new("L", (300, 400), color=255)
    for x in range(20, 280, 3):
        for y in range(20, 380, 9):
            page.putpixel((x, y), 0)
    pipeline = build_transforms(smoke_config(), train=True)
    assert not torch.equal(pipeline(page), pipeline(page))


def test_transforms_produce_three_channels_at_the_configured_size() -> None:
    """Pretrained backbones expect three channels; pages are greyscale."""
    cfg = smoke_config()
    page = Image.new("L", (140, 300), color=128)
    for train in (True, False):
        tensor = build_transforms(cfg, train=train)(page)
        assert tensor.shape == (3, cfg.image.size, cfg.image.size)
        assert tensor.dtype == torch.float32


def test_dataset_returns_a_tensor_and_a_label(image_dir: Path) -> None:
    """The Dataset is the boundary between the manifest and torch."""
    cfg = smoke_config()
    frame = build_manifest(image_dir, CLASSES)
    frame = frame[frame["readable"]].reset_index(drop=True)
    dataset = DocumentPageDataset(frame, build_transforms(cfg, train=False))
    tensor, label = dataset[0]
    assert tensor.shape == (3, cfg.image.size, cfg.image.size)
    assert isinstance(label, int) and 0 <= label < 8
    assert len(dataset.paths) == len(dataset)


def test_only_the_training_loader_shuffles_and_augments(image_dir: Path) -> None:
    """Shuffling the held-out split would break the path-to-prediction alignment."""
    cfg = smoke_config()
    frame = build_manifest(image_dir, CLASSES)
    frame = frame[frame["readable"]].reset_index(drop=True)
    loaders = build_dataloaders(make_grouped_split(frame, cfg.split.ratios, cfg.seed), cfg)
    assert type(loaders["train"].sampler).__name__ == "RandomSampler"
    for name in ("val", "test"):
        assert type(loaders[name].sampler).__name__ == "SequentialSampler"
        names = transform_names(loaders[name].dataset.transform)
        assert not [item for item in names if item.startswith("Random")]
