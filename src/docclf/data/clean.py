# clean.py — remove corrupt, undersized and near-duplicate pages, and say why.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from docclf.utils.hashing import hamming_distance, perceptual_hash
from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)

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
    kept_by_class: dict[str, list[int]] = {}
    for index, row in df.iterrows():
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
        duplicate_of = _find_duplicate(
            digest, kept_by_class.get(row["class_name"], []), hashes, cfg.data.hash_distance
        )
        if duplicate_of is not None:
            report.record(row, REASON_DUPLICATE, f"duplicate of {df.at[duplicate_of, 'path']}")
            continue
        hashes[index] = digest
        keep_index.append(index)
        kept_by_class.setdefault(row["class_name"], []).append(index)
    cleaned = df.loc[keep_index].copy()
    cleaned["phash"] = [hashes[index] for index in keep_index]
    cleaned = cleaned.reset_index(drop=True)
    _log_summary(report)
    return cleaned, report


def _find_duplicate(
    digest: str, candidates: list[int], hashes: dict[int, str], max_distance: int
) -> int | None:
    """Purpose: find an already-kept page this one duplicates.

    Inputs: the candidate hash, the indices already kept for the class, the hash
    table, and the maximum Hamming distance that still counts as a duplicate.
    Output: the index duplicated, or None.
    """
    for index in candidates:
        if hamming_distance(digest, hashes[index]) <= max_distance:
            return index
    return None


def _log_summary(report: CleaningReport) -> None:
    """Purpose: log the cleaning arithmetic. Input: the report. Output: None."""
    for description, value in report.summary().items():
        _LOGGER.info("Cleaning %s: %s", description.replace("_", " "), value)
