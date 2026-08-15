# subsample.py — apply the compute-budget cap without splitting a source document.
# Author: Colile Sibanda

from __future__ import annotations

import random

import pandas as pd

from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)


def cap_per_class(
    df: pd.DataFrame,
    n_per_class: int,
    seed: int,
    group_column: str = "doc_group_id",
) -> pd.DataFrame:
    """Purpose: reduce each class to at most `n_per_class` pages, whole groups only.

    Inputs: the cleaned manifest, the per-class cap, the global seed, and the
    grouping column.
    Output: the capped manifest. Groups are added whole, in a seed-determined
    order, until the next group would exceed the cap, so no `doc_group_id` is
    ever split between the working set and the discarded remainder.
    """
    if n_per_class < 1:
        raise ValueError(f"n_per_class must be at least 1, got {n_per_class}.")
    if group_column not in df.columns:
        raise KeyError(
            f"Manifest has no column {group_column!r}; cap_per_class cannot protect "
            f"group integrity. Rebuild the manifest with data.manifest.build_manifest."
        )
    selected: list[pd.DataFrame] = []
    for class_name, class_frame in df.groupby("class_name", sort=True):
        selected.append(_cap_one_class(class_frame, class_name, n_per_class, seed, group_column))
    capped = pd.concat(selected, ignore_index=True) if selected else df.head(0).copy()
    capped = capped.sort_values("path", kind="mergesort").reset_index(drop=True)
    _LOGGER.info("Capped working-set rows: %d", len(capped))
    _LOGGER.info("Capped per-class cap: %d", n_per_class)
    return capped


def _cap_one_class(
    class_frame: pd.DataFrame,
    class_name: str,
    n_per_class: int,
    seed: int,
    group_column: str,
) -> pd.DataFrame:
    """Purpose: cap one class by taking whole groups in a reproducible order.

    Inputs: the class's rows, its name, the cap, the seed and the group column.
    Output: the retained rows for that class.
    """
    groups = sorted(class_frame[group_column].unique())
    rng = random.Random(f"{seed}:{class_name}")
    rng.shuffle(groups)
    sizes = class_frame[group_column].value_counts().to_dict()
    taken: list[str] = []
    total = 0
    for group in groups:
        size = int(sizes[group])
        if total + size > n_per_class:
            continue
        taken.append(group)
        total += size
        if total == n_per_class:
            break
    if not taken and groups:
        smallest = min(groups, key=lambda group: (int(sizes[group]), group))
        taken = [smallest]
        _LOGGER.warning(
            "Class %s has no group smaller than the cap; kept one group of %d pages",
            class_name,
            int(sizes[smallest]),
        )
    return class_frame[class_frame[group_column].isin(taken)].copy()
