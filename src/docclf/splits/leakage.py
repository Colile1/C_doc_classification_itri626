# leakage.py — prove the split is clean: no shared group, no shared appearance.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from docclf.utils.hashing import hamming_distance
from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)


class LeakageError(AssertionError):
    """Raised when a source document appears in more than one split."""


@dataclass(frozen=True)
class Violation:
    """One pair of visually identical pages sitting on opposite sides of a split."""

    left_path: str
    left_split: str
    right_path: str
    right_split: str
    distance: int

    def __str__(self) -> str:
        """Purpose: a one-line, actionable description of the violation."""
        return (
            f"{self.left_path} ({self.left_split}) matches {self.right_path} "
            f"({self.right_split}) at distance {self.distance}"
        )


def as_frame(splits: pd.DataFrame | Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Purpose: accept either a labelled frame or a mapping of split name to frame.

    Input: a DataFrame carrying a `split` column, or a mapping of name to frame.
    Output: a single DataFrame with a `split` column.
    """
    if isinstance(splits, pd.DataFrame):
        if "split" not in splits.columns:
            raise KeyError(
                "Split frame has no 'split' column. Pass the output of "
                "make_grouped_split, or a mapping of split name to DataFrame."
            )
        return splits
    frames = []
    for name, frame in splits.items():
        labelled = frame.copy()
        labelled["split"] = name
        frames.append(labelled)
    if not frames:
        raise ValueError("No splits were supplied; nothing to check.")
    return pd.concat(frames, ignore_index=True)


def assert_no_group_overlap(
    splits: pd.DataFrame | Mapping[str, pd.DataFrame],
    group_column: str = "doc_group_id",
) -> None:
    """Purpose: enforce the project's central invariant.

    Input: the split assignment.
    Output: None. Raises LeakageError naming every offending `doc_group_id`
    when a source document appears under more than one split label.
    """
    frame = as_frame(splits)
    if group_column not in frame.columns:
        raise KeyError(
            f"Split frame has no column {group_column!r}; group overlap cannot be "
            f"checked. Do not train on this split."
        )
    per_group = frame.groupby(group_column)["split"].nunique()
    offenders = sorted(per_group[per_group > 1].index)
    if offenders:
        shown = ", ".join(str(item) for item in offenders[:10])
        more = "" if len(offenders) <= 10 else f" and {len(offenders) - 10} more"
        raise LeakageError(
            f"Leakage: {len(offenders)} source document(s) appear in more than one "
            f"split: {shown}{more}. Rebuild the split with "
            f"splits.grouped.make_grouped_split; results from this split are invalid."
        )
    _LOGGER.info("Group overlap check: passed on %d groups", int(per_group.size))


def check_near_duplicates(
    splits: pd.DataFrame | Mapping[str, pd.DataFrame],
    max_distance: int = 0,
    hash_column: str = "phash",
) -> list[Violation]:
    """Purpose: catch visually identical pages that carry different group ids.

    Inputs: the split assignment, the maximum Hamming distance that counts as a
    duplicate, and the column holding the perceptual hash.
    Output: a list of Violations, empty when the split is clean. This is the
    second line of defence and it reports rather than raises, so the notebook
    can show what was found.
    """
    frame = as_frame(splits)
    if hash_column not in frame.columns:
        raise KeyError(
            f"Split frame has no column {hash_column!r}. Run data.clean.clean_manifest "
            f"first; it adds the perceptual hash the duplicate check needs."
        )
    records = frame[["path", "split", hash_column]].to_dict("records")
    violations: list[Violation] = []
    if max_distance == 0:
        by_hash: dict[str, list[dict]] = {}
        for record in records:
            by_hash.setdefault(record[hash_column], []).append(record)
        for group in by_hash.values():
            violations.extend(_pairs_across_splits(group, 0, hash_column))
    else:
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                if left["split"] == right["split"]:
                    continue
                distance = hamming_distance(left[hash_column], right[hash_column])
                if distance <= max_distance:
                    violations.extend(
                        _pairs_across_splits([left, right], distance, hash_column)
                    )
    _LOGGER.info("Near-duplicate violations: %d", len(violations))
    return violations


def _pairs_across_splits(
    records: list[dict], distance: int, hash_column: str
) -> list[Violation]:
    """Purpose: turn a set of matching records into cross-split violation pairs.

    Inputs: records sharing a hash, their distance, the hash column name.
    Output: one Violation per pair that straddles two splits.
    """
    found: list[Violation] = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            found.append(
                Violation(
                    left_path=str(left["path"]),
                    left_split=str(left["split"]),
                    right_path=str(right["path"]),
                    right_split=str(right["split"]),
                    distance=int(distance),
                )
            )
    return found
