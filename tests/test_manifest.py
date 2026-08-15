# test_manifest.py — the manifest must carry a group id for every page, always.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import pytest

from docclf.config.loader import load_config
from docclf.data.clean import REASON_CORRUPT, clean_manifest
from docclf.data.manifest import MANIFEST_COLUMNS, build_manifest, derive_group_id
from docclf.data.subsample import cap_per_class

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CLASSES = (
    "invoice",
    "budget",
    "form",
    "letter",
    "memo",
    "handwritten",
    "resume",
    "advertisement",
)


def test_manifest_has_the_expected_columns(image_dir: Path) -> None:
    """Downstream code indexes these by name, so the set is part of the contract."""
    frame = build_manifest(image_dir, CLASSES)
    assert list(frame.columns) == list(MANIFEST_COLUMNS)
    assert len(frame) == 8 * 8 + 1  # eight classes x eight pages, plus the corrupt file


def test_every_row_has_a_group_id(image_dir: Path) -> None:
    """The central invariant depends on this column being populated everywhere."""
    frame = build_manifest(image_dir, CLASSES)
    assert frame["doc_group_id"].notna().all()
    assert (frame["doc_group_id"].astype(str).str.len() > 0).all()


def test_pages_of_one_document_share_a_group(image_dir: Path) -> None:
    """Two pages of doc0 must group together and apart from doc1."""
    frame = build_manifest(image_dir, CLASSES)
    invoices = frame[frame["class_name"] == "invoice"]
    doc0 = set(invoices[invoices["path"].str.contains("doc0_")]["doc_group_id"])
    doc1 = set(invoices[invoices["path"].str.contains("doc1_")]["doc_group_id"])
    assert len(doc0) == 1 and len(doc1) == 1
    assert doc0 != doc1


def test_group_ids_never_collide_across_classes(image_dir: Path) -> None:
    """`doc0` exists in all eight classes; they must not be treated as one document."""
    frame = build_manifest(image_dir, CLASSES)
    pairs = frame.groupby("doc_group_id")["class_name"].nunique()
    assert (pairs == 1).all(), "a group id spans two classes"


def test_class_ids_follow_the_fixed_class_order(image_dir: Path) -> None:
    """class_id is the model's output index; it must match the configured order."""
    frame = build_manifest(image_dir, CLASSES)
    for class_id, class_name in enumerate(CLASSES):
        subset = frame[frame["class_name"] == class_name]
        assert set(subset["class_id"]) == {class_id}


def test_corrupt_file_is_flagged_then_removed_with_a_reason(image_dir: Path) -> None:
    """A file that will not open must be reported, not silently dropped."""
    cfg = load_config(CONFIGS / "smoke.yaml")
    frame = build_manifest(image_dir, CLASSES)
    assert (~frame["readable"]).sum() == 1
    cleaned, report = clean_manifest(frame, cfg)
    reasons = report.to_frame()
    assert REASON_CORRUPT in set(reasons["reason"])
    assert "doc9_p0" in " ".join(reasons["path"])
    assert "doc9_p0" not in " ".join(cleaned["path"])


def test_cleaning_report_accounts_for_every_image(image_dir: Path) -> None:
    """The build order's check: raw count = kept count + removals, exactly."""
    cfg = load_config(CONFIGS / "smoke.yaml")
    frame = build_manifest(image_dir, CLASSES)
    cleaned, report = clean_manifest(frame, cfg)
    assert report.raw_count == len(frame)
    assert len(cleaned) + len(report.to_frame()) == report.raw_count
    assert report.summary()["kept_count"] == len(cleaned)


def test_cleaning_adds_the_perceptual_hash(image_dir: Path) -> None:
    """The leakage check's second line of defence needs this column."""
    cfg = load_config(CONFIGS / "smoke.yaml")
    cleaned, _ = clean_manifest(build_manifest(image_dir, CLASSES), cfg)
    assert "phash" in cleaned.columns
    assert cleaned["phash"].notna().all()


def test_missing_class_directory_is_an_error(tmp_path: Path) -> None:
    """Seven of eight classes present is a broken download, not a small dataset."""
    (tmp_path / "invoice").mkdir()
    with pytest.raises(FileNotFoundError, match="budget"):
        build_manifest(tmp_path, CLASSES)


def test_cap_per_class_never_splits_a_group(manifest) -> None:
    """The cap is a compute decision and must not compromise group integrity."""
    capped = cap_per_class(manifest, n_per_class=4, seed=626)
    for group, rows in capped.groupby("doc_group_id"):
        assert len(rows) == len(manifest[manifest["doc_group_id"] == group])
    assert (capped["class_name"].value_counts() <= 4).all()


def test_cap_per_class_is_reproducible(manifest) -> None:
    """The same seed must select the same documents."""
    first = cap_per_class(manifest, n_per_class=4, seed=626)
    second = cap_per_class(manifest, n_per_class=4, seed=626)
    assert list(first["path"]) == list(second["path"])


def test_cap_larger_than_the_data_keeps_everything(manifest) -> None:
    """A cap above the available count must not lose pages."""
    assert len(cap_per_class(manifest, n_per_class=10_000, seed=626)) == len(manifest)


def test_derive_group_id_uses_the_containing_directory(tmp_path: Path) -> None:
    """RVL-CDIP ships a document's pages in one directory; that is the group."""
    class_dir = tmp_path / "invoice"
    page = class_dir / "aaa06d00" / "page1.tif"
    page.parent.mkdir(parents=True)
    page.touch()
    assert derive_group_id(page, class_dir, r"^(?P<group>[^_]+)") == "invoice/aaa06d00"


def test_derive_group_id_falls_back_to_the_filename_pattern(tmp_path: Path) -> None:
    """A flat layout groups by the stem prefix, so page suffixes stay together."""
    class_dir = tmp_path / "memo"
    class_dir.mkdir()
    first = class_dir / "abc123_p1.png"
    second = class_dir / "abc123_p2.png"
    first.touch()
    second.touch()
    pattern = r"^(?P<group>[^_]+)"
    assert derive_group_id(first, class_dir, pattern) == derive_group_id(
        second, class_dir, pattern
    ) == "memo/abc123"
