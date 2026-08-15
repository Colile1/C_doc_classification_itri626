# dataset.py — a torch Dataset reading rows of the manifest.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Callable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

REQUIRED_COLUMNS = ("path", "class_id")


class DocumentPageDataset(Dataset):
    """One page image and its class id, addressed by manifest row order."""

    def __init__(self, df: pd.DataFrame, transform: Callable | None = None) -> None:
        """Purpose: hold the manifest rows this dataset serves.

        Inputs: a manifest frame with `path` and `class_id`, and the transform
        pipeline for the split.
        Output: None. Raises KeyError when a required column is missing.
        """
        missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
        if missing:
            raise KeyError(
                f"Manifest is missing column(s) {', '.join(missing)}; "
                f"DocumentPageDataset cannot serve it."
            )
        self.frame = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        """Purpose: the number of pages. Output: the row count."""
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Purpose: load one page and return it with its label.

        Input: the row index.
        Output: `(image tensor, class id)`. Raises ValueError naming the file
        when it cannot be read, rather than returning a blank image silently.
        """
        row = self.frame.iloc[index]
        path = str(row["path"])
        try:
            with Image.open(path) as image:
                page = image.convert("L")
                page.load()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Page image not found: {path}. The split files reference images that "
                f"are no longer on disk; re-run run_prepare_data.py."
            ) from None
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(
                f"Cannot read page image {path}: {error}. It should have been removed "
                f"by data.clean; re-run the cleaning stage."
            ) from error
        tensor = self.transform(page) if self.transform is not None else page
        return tensor, int(row["class_id"])

    @property
    def paths(self) -> list[str]:
        """Purpose: the image paths in serving order, for the error analysis figures."""
        return [str(item) for item in self.frame["path"]]

    @property
    def targets(self) -> list[int]:
        """Purpose: the labels in serving order, for class weighting and sanity checks."""
        return [int(item) for item in self.frame["class_id"]]
