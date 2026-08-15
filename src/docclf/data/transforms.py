# transforms.py — the augmentation policy. No horizontal flip exists here, by design.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Any, Callable

from torchvision import transforms

# A mirrored page of text does not occur in client onboarding, so a horizontal
# flip would teach an invariance that is wrong. The name is listed here so the
# test suite can assert that no transform in the pipeline is one.
FORBIDDEN_TRANSFORMS = ("RandomHorizontalFlip", "RandomVerticalFlip")

PAGE_FILL = 255  # scanned pages have white margins; rotation must not pad black


def build_transforms(cfg: Any, train: bool) -> Callable:
    """Purpose: build the image pipeline for one split.

    Inputs: the RunConfig and whether this is the training split.
    Output: a torchvision `Compose`. With `train=False` the pipeline contains no
    random operation at all, so validation and test results are deterministic.
    """
    image = cfg.image
    size = int(image.size)
    to_tensor = [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(image.mean), std=list(image.std)),
    ]
    if not train:
        return transforms.Compose([transforms.Resize((size, size))] + to_tensor)
    augmentation = [
        transforms.Resize((size, size)),
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
