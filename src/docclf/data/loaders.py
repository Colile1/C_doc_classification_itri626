# loaders.py — build the DataLoaders, with batching chosen by the environment.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import DataLoader

from docclf.data.dataset import DocumentPageDataset
from docclf.data.transforms import build_transforms
from docclf.splits.grouped import SPLIT_NAMES
from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)


def build_dataloaders(
    splits: pd.DataFrame | Mapping[str, pd.DataFrame], cfg: Any
) -> dict[str, DataLoader]:
    """Purpose: turn the split assignment into one DataLoader per split.

    Inputs: the split frame (or a mapping of split name to frame) and the RunConfig.
    Output: a dict keyed `train`, `val`, `test`. Only the training loader shuffles
    and only the training split receives augmentation; both are properties of this
    function, so no caller can accidentally augment the test set.
    """
    frames = _as_mapping(splits)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loaders: dict[str, DataLoader] = {}
    for name in SPLIT_NAMES:
        if name not in frames:
            continue
        is_train = name == "train"
        dataset = DocumentPageDataset(frames[name], build_transforms(cfg, train=is_train))
        batch_size = cfg.train.batch_size if is_train else cfg.eval.batch_size
        workers = cfg.train.num_workers if is_train else cfg.eval.num_workers
        loaders[name] = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=is_train,
            num_workers=int(workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=generator if is_train else None,
            persistent_workers=bool(workers) and is_train,
        )
        _LOGGER.info("Loader %s pages: %d", name, len(dataset))
    if not loaders:
        raise ValueError(
            "No splits were supplied to build_dataloaders. Expected at least one of "
            f"{', '.join(SPLIT_NAMES)}."
        )
    return loaders


def _as_mapping(
    splits: pd.DataFrame | Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Purpose: accept either a labelled frame or a mapping of split name to frame.

    Input: the split assignment in either form.
    Output: a mapping of split name to its rows.
    """
    if isinstance(splits, pd.DataFrame):
        if "split" not in splits.columns:
            raise KeyError(
                "Split frame has no 'split' column; pass the output of "
                "make_grouped_split or splits.io.load_splits."
            )
        return {
            str(name): frame.reset_index(drop=True)
            for name, frame in splits.groupby("split")
        }
    return {str(name): frame.reset_index(drop=True) for name, frame in splits.items()}
