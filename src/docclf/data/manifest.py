# manifest.py — one row per page image, carrying the doc_group_id the split needs.
# Author: Colile Sibanda

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pandas as pd
from PIL import Image, UnidentifiedImageError

from docclf.utils.logging import get_logger
from docclf.utils.retry import is_transient_io_error, retry_io

_LOGGER = get_logger(__name__)
_PROGRESS_EVERY = 10_000

MANIFEST_COLUMNS = (
    "path",
    "class_name",
    "class_id",
    "doc_group_id",
    "width",
    "height",
    "bytes",
    "readable",
)

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif")


def build_manifest(
    root: Path, classes: Sequence[str], group_pattern: str = r"^(?P<group>[^_]+)"
) -> pd.DataFrame:
    """Purpose: inventory every page image under a class-per-directory root.

    Inputs: the dataset root, the fixed class names in their canonical order,
    and the regular expression used to derive a group id from a flat filename.
    Output: a DataFrame with MANIFEST_COLUMNS, one row per image. Unreadable
    files are kept with `readable` False so `data.clean` can report them.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(
            f"Dataset root is not a directory: {root}. Run data.acquire.fetch_dataset "
            f"first, or point paths.raw at the extracted dataset."
        )
    missing = [name for name in classes if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing class director{'y' if len(missing) == 1 else 'ies'} under "
            f"{root}: {', '.join(missing)}. Every one of the eight fixed classes "
            f"must have its own directory."
        )
    rows: list[dict[str, object]] = []
    for class_id, class_name in enumerate(classes):
        class_dir = root / class_name
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows.append(
                _describe_image(path, class_dir, class_name, class_id, group_pattern)
            )
            if len(rows) % _PROGRESS_EVERY == 0:
                _LOGGER.info("Manifest progress: %d images described", len(rows))
    frame = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    _LOGGER.info("Manifest rows: %d", len(frame))
    _LOGGER.info("Manifest classes: %d", frame["class_name"].nunique())
    return frame


def _describe_image(
    path: Path, class_dir: Path, class_name: str, class_id: int, group_pattern: str
) -> dict[str, object]:
    """Purpose: build the manifest row for one image file.

    Inputs: the image path, its class directory, class name and id, the pattern.
    Output: a dictionary keyed by MANIFEST_COLUMNS.
    """
    width, height, readable = 0, 0, True

    def read_size() -> tuple[int, int]:
        with Image.open(path) as image:
            return image.size

    try:
        width, height = retry_io(read_size, path=path)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        if is_transient_io_error(error, path):
            raise
        readable = False
    return {
        "path": str(path),
        "class_name": class_name,
        "class_id": class_id,
        "doc_group_id": derive_group_id(path, class_dir, group_pattern),
        "width": width,
        "height": height,
        "bytes": retry_io(lambda: path.stat().st_size, path=path),
        "readable": readable,
    }


def derive_group_id(path: Path, class_dir: Path, group_pattern: str) -> str:
    """Purpose: name the source document a page belongs to.

    Inputs: the image path, the class directory it sits under, and the regular
    expression applied to the file stem when the layout is flat.
    Output: a group id, prefixed by the class so two classes cannot share one.

    Nested layouts (RVL-CDIP ships pages of one document in one directory) use
    the containing directory. Flat layouts use the pattern's `group` capture,
    falling back to the whole stem when the pattern does not match.
    """
    path, class_dir = Path(path), Path(class_dir)
    parent = path.parent
    if parent != class_dir:
        return f"{class_dir.name}/{parent.name}"
    match = re.match(group_pattern, path.stem)
    if match is None:
        return f"{class_dir.name}/{path.stem}"
    captured = match.groupdict().get("group") or match.group(0)
    if not captured:
        raise ValueError(
            f"Group pattern {group_pattern!r} matched an empty group id for "
            f"{path.name}. Fix data.group_pattern so every page yields a group."
        )
    return f"{class_dir.name}/{captured}"
