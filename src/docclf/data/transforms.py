# transforms.py — the augmentation policy. No horizontal flip exists here, by design.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Any, Callable

from PIL import Image
from torchvision import transforms

# A mirrored page of text does not occur in client onboarding, so a horizontal
# flip would teach an invariance that is wrong. The name is listed here so the
# test suite can assert that no transform in the pipeline is one.
FORBIDDEN_TRANSFORMS = ("RandomHorizontalFlip", "RandomVerticalFlip")

PAGE_FILL = 255  # scanned pages have white margins; rotation must not pad black


class ResizeAndPad:
    """Resize a page without changing its aspect ratio, then center it on white."""

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"size must be positive, got {size}.")
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(self.size / width, self.size / height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (self.size, self.size), color=(PAGE_FILL,) * 3)
        canvas.paste(
            resized,
            ((self.size - resized.width) // 2, (self.size - resized.height) // 2),
        )
        return canvas


def build_transforms(cfg: Any, train: bool) -> Callable:
    """Purpose: build the image pipeline for one split.

    Inputs: the RunConfig and whether this is the training split.
    Output: a torchvision `Compose`. With `train=False` the pipeline contains no
    random operation at all, so validation and test results are deterministic.
    """
    image = cfg.image
    size = int(image.size)
    resize = (
        ResizeAndPad(size)
        if bool(getattr(image, "preserve_aspect_ratio", False))
        else transforms.Resize((size, size))
    )
    to_tensor = [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(image.mean), std=list(image.std)),
    ]
    if not train:
        return transforms.Compose([resize] + to_tensor)
    augmentation = [
        resize,
        transforms.RandomResizedCrop(
            size, scale=tuple(image.crop_scale), ratio=(0.95, 1.05), antialias=True
        ),
        transforms.RandomRotation(
            degrees=float(image.rotation_degrees), fill=PAGE_FILL, expand=False
        ),
        transforms.ColorJitter(
            brightness=float(image.brightness), contrast=float(image.contrast)
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=float(image.blur_probability),
        ),
    ]
    return transforms.Compose(augmentation + to_tensor)


def transform_names(pipeline: Callable) -> list[str]:
    """Purpose: list the transform class names in a pipeline, nested ones included.

    Input: a `Compose`, a `RandomApply` or a single transform.
    Output: the class names, so a test can prove what is and is not present.
    """
    names: list[str] = []
    inner = getattr(pipeline, "transforms", None)
    if inner is None:
        return [type(pipeline).__name__]
    names.append(type(pipeline).__name__)
    for item in inner:
        names.extend(transform_names(item))
    return names
