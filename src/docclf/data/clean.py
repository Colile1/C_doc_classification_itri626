# clean.py — remove corrupt, undersized and near-duplicate pages, and say why.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from docclf.utils.hashing import distances_to, hashes_to_array, perceptual_hash
from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)
_PROGRESS_EVERY = 10_000

REASON_CORRUPT = "corrupt_file"
REASON_TOO_SMALL = "below_size_floor"
REASON_DUPLICATE = "near_duplicate"


@dataclass
class CleaningReport:
    """Every image lost between the raw count and the working count, with a reason."""

    raw_count: int
    removals: list[dict[str, Any]] = field(default_factory=list)

    @property
    def kept_count(self) -> int:
        """Number of rows surviving cleaning. Purpose: close the arithmetic."""
        return self.raw_count - len(self.removals)

    def record(self, row: pd.Series, reason: str, detail: str = "") -> None:
        """Purpose: register one removal.

        Inputs: the manifest row, the reason code, an optional detail string.
        Output: None.
        """
        self.removals.append(
            {
                "path": row["path"],
                "class_name": row["class_name"],
                "doc_group_id": row["doc_group_id"],
                "reason": reason,
                "detail": detail,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        """Purpose: the per-removal table written to `data/splits/cleaning_report.csv`.

        Input: none. Output: a DataFrame, empty with the right columns if nothing went.
        """
        columns = ["path", "class_name", "doc_group_id", "reason", "detail"]
        return pd.DataFrame(self.removals, columns=columns)

    def summary(self) -> dict[str, int]:
        """Purpose: the counts that must reconcile raw against working.

        Input: none. Output: raw, kept and one count per removal reason.
        """
        counts = {"raw_count": self.raw_count, "kept_count": self.kept_count}
        for reason in (REASON_CORRUPT, REASON_TOO_SMALL, REASON_DUPLICATE):
            counts[reason] = sum(1 for item in self.removals if item["reason"] == reason)
        return counts


def clean_manifest(df: pd.DataFrame, cfg: Any) -> tuple[pd.DataFrame, CleaningReport]:
    """Purpose: reduce the raw manifest to a working set, accounting for every loss.

    Inputs: the manifest from `build_manifest` and the RunConfig.
    Output: the cleaned manifest with a `phash` column, and the CleaningReport.
    Rows are removed for being unreadable, smaller than `data.min_side_pixels`,
    or within `data.hash_distance` of a page already kept in the same class.
    """
    report = CleaningReport(raw_count=len(df))
    keep_index: list[int] = []
    hashes: dict[int, str] = {}
    kept_by_class: dict[str, _KeptHashes] = {}
    for position, (index, row) in enumerate(df.iterrows(), start=1):
        if position % _PROGRESS_EVERY == 0:
            _LOGGER.info(
                "Cleaning progress: %d of %d checked, %d kept",
                position, len(df), len(keep_index),
            )
        if not bool(row.get("readable", True)):
            report.record(row, REASON_CORRUPT, "Pillow could not open the file")
            continue
        if min(int(row["width"]), int(row["height"])) < cfg.data.min_side_pixels:
            report.record(
                row, REASON_TOO_SMALL, f"{int(row['width'])}x{int(row['height'])}"
            )
            continue
        try:
            digest = perceptual_hash(row["path"])
        except (ValueError, FileNotFoundError) as error:
            report.record(row, REASON_CORRUPT, str(error).split(":")[0])
            continue
        kept = kept_by_class.setdefault(row["class_name"], _KeptHashes())
        duplicate_of = kept.find_duplicate(digest, cfg.data.hash_distance)
        if duplicate_of is not None:
            report.record(row, REASON_DUPLICATE, f"duplicate of {df.at[duplicate_of, 'path']}")
            continue
        hashes[index] = digest
        keep_index.append(index)
        kept.add(index, digest)
    cleaned = df.loc[keep_index].copy()
    cleaned["phash"] = [hashes[index] for index in keep_index]
    cleaned = cleaned.reset_index(drop=True)
    _log_summary(report)
    return cleaned, report


class _KeptHashes:
    """The pages kept so far in one class, packed for a one-pass distance check.

    Comparing a page against every kept page one pair at a time is quadratic in
    Python calls and does not finish at 25 000 pages per class. Holding the
    hashes as a uint64 array turns each check into a single vectorised XOR and
    bit count, with the same result: the earliest kept page within the distance.
    """

    def __init__(self) -> None:
        self._indices: list[int] = []
        self._packed = np.empty(1024, dtype=np.uint64)

    def find_duplicate(self, digest: str, max_distance: int) -> int | None:
        """Return the index of the first kept page within max_distance, or None."""
        count = len(self._indices)
        if count == 0:
            return None
        target = int(hashes_to_array([digest])[0])
        close = np.flatnonzero(distances_to(target, self._packed[:count]) <= max_distance)
        return self._indices[int(close[0])] if close.size else None

    def add(self, index: int, digest: str) -> None:
        """Record a kept page, growing the packed array when it is full."""
        count = len(self._indices)
        if count == len(self._packed):
            self._packed = np.resize(self._packed, count * 2)
        self._packed[count] = hashes_to_array([digest])[0]
        self._indices.append(index)


def _log_summary(report: CleaningReport) -> None:
    """Purpose: log the cleaning arithmetic. Input: the report. Output: None."""
    for description, value in report.summary().items():
        _LOGGER.info("Cleaning %s: %s", description.replace("_", " "), value)
