# __init__.py — public surface of the split module.
# Author: Colile Sibanda

from docclf.splits.grouped import SPLIT_NAMES, make_grouped_split
from docclf.splits.io import load_splits, save_splits
from docclf.splits.leakage import (
    LeakageError,
    Violation,
    assert_no_group_overlap,
    check_near_duplicates,
)
from docclf.splits.report import describe_splits

__all__ = [
    "LeakageError",
    "SPLIT_NAMES",
    "Violation",
    "assert_no_group_overlap",
    "check_near_duplicates",
    "describe_splits",
    "load_splits",
    "make_grouped_split",
    "save_splits",
]
