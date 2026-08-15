# report.py — the per-split, per-class table the report has to contain.
# Author: Colile Sibanda

from __future__ import annotations

import pandas as pd

from docclf.splits.grouped import SPLIT_NAMES
from docclf.splits.leakage import as_frame


def describe_splits(
    splits: pd.DataFrame, group_column: str = "doc_group_id"
) -> pd.DataFrame:
    """Purpose: count pages and source documents per split and per class.

    Input: the split assignment from `make_grouped_split`.
    Output: a long-form DataFrame with one row per (split, class): `count`,
    `groups`, `share_of_split` and `share_of_class`. The `count` column sums to
    the working-set total exactly, which is the check the build order requires.
    """
    frame = as_frame(splits)
    grouped = (
        frame.groupby(["split", "class_name"], sort=True)
        .agg(count=("path", "size"), groups=(group_column, "nunique"))
        .reset_index()
    )
    per_split = grouped.groupby("split")["count"].transform("sum")
    per_class = grouped.groupby("class_name")["count"].transform("sum")
    grouped["share_of_split"] = (grouped["count"] / per_split).round(4)
    grouped["share_of_class"] = (grouped["count"] / per_class).round(4)
    order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    grouped["_order"] = grouped["split"].map(order).fillna(len(order))
    return (
        grouped.sort_values(["_order", "class_name"], kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def split_totals(splits: pd.DataFrame) -> pd.DataFrame:
    """Purpose: the one-line-per-split summary used in the slides.

    Input: the split assignment.
    Output: a DataFrame with `split`, `pages`, `groups` and `share`.
    """
    frame = as_frame(splits)
    totals = (
        frame.groupby("split", sort=True)
        .agg(pages=("path", "size"), groups=("doc_group_id", "nunique"))
        .reset_index()
    )
    totals["share"] = (totals["pages"] / totals["pages"].sum()).round(4)
    order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    totals["_order"] = totals["split"].map(order).fillna(len(order))
    return totals.sort_values("_order").drop(columns="_order").reset_index(drop=True)
