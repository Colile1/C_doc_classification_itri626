# grouped.py — assign whole source documents to one split each (correctness-critical).
# Author: Colile Sibanda

from __future__ import annotations

import random
from typing import Sequence

import pandas as pd

from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def make_grouped_split(
    df: pd.DataFrame,
    ratios: Sequence[float],
    seed: int,
    group_column: str = "doc_group_id",
    stratify_column: str = "class_name",
) -> pd.DataFrame:
    """Purpose: divide the working set so no source document spans two splits.

    Inputs: the cleaned manifest, the train/val/test ratios, the global seed,
    and the grouping and stratification column names.
    Output: the manifest with an added `split` column, sorted by path so the
    result is byte-identical between runs with the same seed.

    Whole groups are placed, largest first, into the split with the largest
    remaining page deficit; the assignment is done per class, which is what
    keeps the class distribution close to the ratios. Group integrity is a hard
    constraint and the ratios are the thing that bends to it.
    """
    _validate_inputs(df, ratios, group_column, stratify_column)
    assignments: dict[str, str] = {}
    for class_name, class_frame in df.groupby(stratify_column, sort=True):
        assignments.update(
            _assign_one_class(class_frame, class_name, ratios, seed, group_column)
        )
    result = df.copy()
    result["split"] = result[group_column].map(assignments)
    unassigned = result["split"].isna().sum()
    if unassigned:
        raise RuntimeError(
            f"{unassigned} page(s) were left without a split assignment. This is a "
            f"bug in make_grouped_split; do not train on this split."
        )
    result = result.sort_values("path", kind="mergesort").reset_index(drop=True)
    for name in SPLIT_NAMES:
        _LOGGER.info("Split %s pages: %d", name, int((result["split"] == name).sum()))
    return result


def _validate_inputs(
    df: pd.DataFrame, ratios: Sequence[float], group_column: str, stratify_column: str
) -> None:
    """Purpose: refuse to split a manifest that cannot be split safely.

    Inputs: the manifest, the ratios and the two column names.
    Output: None. Raises KeyError or ValueError naming what is wrong.
    """
    for column in (group_column, stratify_column, "path"):
        if column not in df.columns:
            raise KeyError(
                f"Manifest has no column {column!r}. The grouped split cannot run "
                f"without it; rebuild the manifest with data.manifest.build_manifest."
            )
    if len(ratios) != 3:
        raise ValueError(f"Expected three ratios (train, val, test), got {len(ratios)}.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}.")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError(f"Split ratios must be strictly positive, got {tuple(ratios)}.")
    duplicated = df["path"].duplicated().sum()
    if duplicated:
        raise ValueError(
            f"Manifest contains {duplicated} duplicated path(s). Deduplicate before "
            f"splitting, otherwise one page can land in two splits."
        )


def _assign_one_class(
    class_frame: pd.DataFrame,
    class_name: str,
    ratios: Sequence[float],
    seed: int,
    group_column: str,
) -> dict[str, str]:
    """Purpose: place one class's groups across the three splits.

    Inputs: the class's rows, its name, the ratios, the seed, the group column.
    Output: a mapping of group id to split name.
    """
    sizes = class_frame[group_column].value_counts().to_dict()
    groups = sorted(sizes)
    rng = random.Random(f"{seed}:{class_name}")
    rng.shuffle(groups)
    groups.sort(key=lambda group: -int(sizes[group]))  # stable: ties keep shuffled order
    total = len(class_frame)
    targets = {name: total * ratio for name, ratio in zip(SPLIT_NAMES, ratios)}
    placed = {name: 0 for name in SPLIT_NAMES}
    assignment: dict[str, str] = {}
    for group in groups:
        chosen = max(SPLIT_NAMES, key=lambda name: (targets[name] - placed[name], name))
        assignment[group] = chosen
        placed[chosen] += int(sizes[group])
    _repair_empty_splits(assignment, sizes, placed, class_name)
    return assignment


def _repair_empty_splits(
    assignment: dict[str, str],
    sizes: dict[str, int],
    placed: dict[str, int],
    class_name: str,
) -> None:
    """Purpose: guarantee every class is represented in every split where possible.

    Inputs: the group-to-split assignment, group sizes, per-split page counts,
    and the class name (used in the warning).
    Output: None; the assignment is repaired in place by moving the smallest
    group out of the most over-filled split. A class with fewer than three
    groups cannot be repaired and is warned about instead.
    """
    empty = [name for name in SPLIT_NAMES if placed[name] == 0]
    if not empty:
        return
    if len(sizes) < len(SPLIT_NAMES):
        _LOGGER.warning(
            "Class %s has only %d source document(s); split(s) %s are empty",
            class_name,
            len(sizes),
            ", ".join(empty),
        )
        return
    for name in empty:
        donor = max(SPLIT_NAMES, key=lambda item: placed[item])
        candidates = [
            group
            for group, split in assignment.items()
            if split == donor and placed[donor] - int(sizes[group]) > 0
        ]
        if not candidates:
            continue
        moved = min(candidates, key=lambda group: (int(sizes[group]), group))
        assignment[moved] = name
        placed[donor] -= int(sizes[moved])
        placed[name] += int(sizes[moved])
        _LOGGER.warning(
            "Class %s: moved group %s from %s to %s to keep every split populated",
            class_name,
            moved,
            donor,
            name,
        )
