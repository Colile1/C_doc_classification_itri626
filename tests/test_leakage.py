# test_leakage.py — the grouped split must never put one document in two splits.
# Author: Colile Sibanda
#
# Correctness-critical. Written before splits/ was implemented.
# A leaked split produces high test scores that mean nothing and that no
# downstream code will flag, so these tests are the project's central evidence.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from docclf.splits.grouped import make_grouped_split
from docclf.splits.io import load_splits, resolve_paths_against, save_splits, to_relative_paths
from docclf.splits.leakage import (
    LeakageError,
    assert_no_group_overlap,
    check_near_duplicates,
)
from docclf.splits.report import describe_splits

RATIOS = (0.70, 0.15, 0.15)
SEED = 626


def test_no_group_spans_two_splits(manifest: pd.DataFrame) -> None:
    """Every doc_group_id must appear under exactly one split label."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    per_group = split.groupby("doc_group_id")["split"].nunique()
    offenders = per_group[per_group > 1]
    assert offenders.empty, f"Groups in more than one split: {list(offenders.index)}"
    assert_no_group_overlap(split)


def test_poisoned_split_is_rejected(manifest: pd.DataFrame) -> None:
    """The check must reject a split that has been deliberately poisoned."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    poisoned = split.copy()
    victim = poisoned.loc[poisoned["split"] == "train", "doc_group_id"].iloc[0]
    rows = poisoned.index[poisoned["doc_group_id"] == victim]
    poisoned.loc[rows[0], "split"] = "test"
    if len(rows) == 1:  # a single-page group needs a second page to straddle
        extra = poisoned.loc[[rows[0]]].copy()
        extra["split"] = "train"
        poisoned = pd.concat([poisoned, extra], ignore_index=True)
    with pytest.raises(LeakageError) as error:
        assert_no_group_overlap(poisoned)
    assert victim in str(error.value)


def test_every_row_is_assigned_exactly_once(manifest: pd.DataFrame) -> None:
    """The split covers the working set: no row lost, no row duplicated."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    assert len(split) == len(manifest)
    assert sorted(split["path"]) == sorted(manifest["path"])
    assert set(split["split"]) == {"train", "val", "test"}


def test_split_summary_totals_equal_the_working_set(manifest: pd.DataFrame) -> None:
    """describe_splits must account for every page in the working set."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    summary = describe_splits(split)
    assert int(summary["count"].sum()) == len(manifest)
    for class_name, class_frame in manifest.groupby("class_name"):
        rows = summary[summary["class_name"] == class_name]
        assert int(rows["count"].sum()) == len(class_frame)


def test_ratios_are_approximately_honoured(manifest: pd.DataFrame) -> None:
    """Whole groups cannot hit 70/15/15 exactly, but must land close to it."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    shares = split["split"].value_counts(normalize=True)
    for name, target in zip(("train", "val", "test"), RATIOS):
        assert abs(shares[name] - target) < 0.12, f"{name} share {shares[name]:.3f}"


def test_every_class_appears_in_every_split(manifest: pd.DataFrame) -> None:
    """A class missing from the test split would make its metrics undefined."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    for name in ("train", "val", "test"):
        present = set(split.loc[split["split"] == name, "class_name"])
        assert present == set(manifest["class_name"]), f"{name} is missing classes"


def test_split_is_reproducible_from_the_seed(manifest: pd.DataFrame) -> None:
    """The same seed must give the identical assignment; a different seed need not."""
    first = make_grouped_split(manifest, RATIOS, SEED)
    second = make_grouped_split(manifest, RATIOS, SEED)
    pd.testing.assert_frame_equal(first, second)
    other = make_grouped_split(manifest, RATIOS, SEED + 1)
    assert len(other) == len(first)


def test_near_duplicate_across_splits_is_reported(manifest: pd.DataFrame) -> None:
    """The second line of defence: identical pages carrying different group ids."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    train_row = split.index[split["split"] == "train"][0]
    test_row = split.index[split["split"] == "test"][0]
    planted = split.copy()
    planted.loc[test_row, "phash"] = planted.loc[train_row, "phash"]
    violations = check_near_duplicates(planted, max_distance=0)
    assert violations, "A planted cross-split duplicate was not reported"
    assert any(
        item.left_path == planted.loc[train_row, "path"]
        or item.right_path == planted.loc[train_row, "path"]
        for item in violations
    )


def test_clean_split_has_no_near_duplicate_violations(manifest: pd.DataFrame) -> None:
    """The fixture's hashes are distinct, so the clean split must report nothing."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    assert check_near_duplicates(split, max_distance=0) == []


def test_splits_round_trip_through_csv(manifest: pd.DataFrame, tmp_path) -> None:
    """Committed split files must reload as the same assignment."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    written = save_splits(split, tmp_path)
    assert set(written) == {"train", "val", "test"}
    reloaded = load_splits(tmp_path)
    pd.testing.assert_frame_equal(
        split.sort_values("path").reset_index(drop=True),
        reloaded.sort_values("path").reset_index(drop=True),
        check_dtype=False,
    )


def test_relative_round_trip_restores_absolute_paths(
    manifest: pd.DataFrame, tmp_path
) -> None:
    """Paths stored relative to the project root must come back absolute, so the
    same split files load on any machine."""
    split = make_grouped_split(manifest, RATIOS, SEED)
    root = tmp_path / "project"
    (root / "data" / "splits").mkdir(parents=True)
    # The fixture's paths live under the root, which is what makes them relative.
    split = split.copy()
    split["path"] = [str(root / f"data/raw/page_{i}.tif") for i in range(len(split))]

    save_splits(split, root / "data" / "splits", root=root)
    stored = pd.read_csv(root / "data" / "splits" / "train.csv")
    assert not stored["path"].str.startswith("/").any()
    assert not stored["path"].str.contains(":").any()

    reloaded = load_splits(root / "data" / "splits", root=root)
    assert all(Path(value).is_absolute() for value in reloaded["path"])
    pd.testing.assert_frame_equal(
        split.sort_values("path").reset_index(drop=True),
        reloaded.sort_values("path").reset_index(drop=True),
        check_dtype=False,
    )


def test_absolute_paths_still_load_unchanged(manifest: pd.DataFrame) -> None:
    """Split files written before paths became relative must keep working."""
    elsewhere = Path("/elsewhere/data/raw/page.tif").resolve()
    absolute = pd.DataFrame({"path": [str(elsewhere)]})
    resolved = resolve_paths_against(absolute, Path("/some/root").resolve())
    assert resolved["path"].iloc[0] == str(elsewhere)


def test_path_outside_the_root_is_left_absolute() -> None:
    """A path that cannot be expressed relative to the root stays absolute,
    rather than gaining `..` segments that break on another machine."""
    outside = pd.DataFrame({"path": [str(Path("/outside/page.tif").resolve())]})
    stored = to_relative_paths(outside, Path("/project").resolve())
    assert Path(stored["path"].iloc[0]).is_absolute()


def test_missing_group_column_is_an_error(manifest: pd.DataFrame) -> None:
    """A manifest without doc_group_id cannot be split safely, so it must raise."""
    with pytest.raises(KeyError, match="doc_group_id"):
        make_grouped_split(manifest.drop(columns=["doc_group_id"]), RATIOS, SEED)


def test_ratios_must_sum_to_one(manifest: pd.DataFrame) -> None:
    """A malformed ratio triple must fail loudly rather than silently rescale."""
    with pytest.raises(ValueError, match="sum to 1"):
        make_grouped_split(manifest, (0.7, 0.2, 0.2), SEED)


def test_save_splits_round_trips_on_a_foreign_machine(manifest, tmp_path) -> None:
    """The committed split must load on a machine whose project lives elsewhere.

    This is the failure that silently invalidates everything: a split file whose
    paths resolve on one machine only, rewritten on another, producing a
    different set of pages under the same file name.
    """
    split = make_grouped_split(manifest, RATIOS, SEED)
    here = tmp_path / "machine_a"
    there = tmp_path / "machine_b"
    for root in (here, there):
        (root / "data" / "splits").mkdir(parents=True)

    split = split.copy()
    split["path"] = [str(here / f"data/raw/page_{i}.tif") for i in range(len(split))]
    save_splits(split, here / "data" / "splits", root=here)

    # The same file, read on a machine with a different project root.
    for name in ("train", "val", "test"):
        (there / "data" / "splits" / f"{name}.csv").write_bytes(
            (here / "data" / "splits" / f"{name}.csv").read_bytes()
        )
    reloaded = load_splits(there / "data" / "splits", root=there)

    assert all(str(value).startswith(str(there)) for value in reloaded["path"])
    relative = [Path(value).relative_to(there).as_posix() for value in reloaded["path"]]
    original = [Path(value).relative_to(here).as_posix() for value in split["path"]]
    assert sorted(relative) == sorted(original), (
        "the same split file must name the same pages on both machines"
    )


def test_images_outside_the_project_still_resolve() -> None:
    """On Colab the project sits on Drive while the images are unpacked to
    local disk, because reading them over the Drive mount is slower than the
    GPU they feed. The same split file must work in both layouts."""
    stored = pd.DataFrame({"path": ["data/raw/resume/page.tif"]})
    project = Path("/drive/project").resolve()

    inside = resolve_paths_against(stored, project, raw=project / "data" / "raw")
    assert Path(inside["path"].iloc[0]) == project / "data" / "raw" / "resume" / "page.tif"

    elsewhere = Path("/local/data/raw").resolve()
    outside = resolve_paths_against(stored, project, raw=elsewhere)
    assert Path(outside["path"].iloc[0]) == elsewhere / "resume" / "page.tif", (
        "a split path must follow the configured raw directory, not the project root"
    )


def test_omitting_raw_keeps_the_root_behaviour() -> None:
    """Callers that pass no raw directory must be unaffected."""
    stored = pd.DataFrame({"path": ["data/raw/memo/page.tif"]})
    project = Path("/drive/project").resolve()
    assert (
        resolve_paths_against(stored, project)["path"].iloc[0]
        == str(project / "data" / "raw" / "memo" / "page.tif")
    )
