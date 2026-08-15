# class_weights.py — inverse-frequency weights for the imbalance the cap leaves behind.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch


def compute_class_weights(
    train_split: pd.DataFrame, n_classes: int, column: str = "class_id"
) -> torch.Tensor:
    """Purpose: weight each class inversely to how often it appears.

    Inputs: the training split rows, the number of classes, and the label column.
    Output: a float tensor of length `n_classes`, normalised to mean 1.0 so the
    loss scale stays comparable to the unweighted run. A class absent from the
    training split receives weight 1.0 rather than an infinite one.
    """
    if column not in train_split.columns:
        raise KeyError(
            f"Training split has no column {column!r}; class weights cannot be "
            f"computed."
        )
    counts = train_split[column].value_counts().to_dict()
    present = [float(counts.get(index, 0)) for index in range(n_classes)]
    total = sum(present)
    if total == 0:
        raise ValueError("The training split is empty; class weights are undefined.")
    weights = [
        (total / (n_classes * count)) if count > 0 else 1.0 for count in present
    ]
    tensor = torch.tensor(weights, dtype=torch.float32)
    return tensor / tensor.mean()


def class_distribution(split: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    """Purpose: the per-class count and share, for the dataset section of the report.

    Inputs: a split's rows and the ordered class names.
    Output: a DataFrame with `class_name`, `count` and `share`.
    """
    counts = split["class_name"].value_counts()
    frame = pd.DataFrame(
        {"class_name": list(classes), "count": [int(counts.get(name, 0)) for name in classes]}
    )
    total = frame["count"].sum()
    frame["share"] = (frame["count"] / total).round(4) if total else 0.0
    return frame
