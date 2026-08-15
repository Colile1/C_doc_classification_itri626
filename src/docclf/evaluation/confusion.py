# confusion.py — rows are true classes, columns predicted, said on the figure itself.
# Author: Colile Sibanda  (correctness-critical: see tests/test_metrics.py)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from docclf.evaluation.predict import Predictions


def confusion_frame(preds: Predictions, normalise: bool = False) -> pd.DataFrame:
    """Purpose: the confusion matrix, with its orientation fixed and labelled.

    Inputs: the Predictions and whether to normalise each row to sum to 1.
    Output: a DataFrame whose index is the true class (`true`) and whose columns
    are the predicted class (`predicted`). The orientation is asserted by the
    test suite, because a transposed matrix reads as a plausible result.
    """
    labels = list(range(len(preds.class_names)))
    matrix = confusion_matrix(preds.y_true, preds.y_pred, labels=labels)
    if normalise:
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix, totals, out=np.zeros(matrix.shape, dtype=float), where=totals != 0
        )
    frame = pd.DataFrame(
        matrix, index=list(preds.class_names), columns=list(preds.class_names)
    )
    frame.index.name = "true"
    frame.columns.name = "predicted"
    return frame


def most_confused_pairs(preds: Predictions, top_n: int = 10) -> pd.DataFrame:
    """Purpose: rank the off-diagonal cells, which drives report section 7.

    Inputs: the Predictions and how many pairs to return.
    Output: a DataFrame with `true`, `predicted`, `count` and `share_of_true`,
    ordered by count. The direction is kept: invoice read as budget is not the
    same error as budget read as invoice.
    """
    counts = confusion_frame(preds)
    rows: list[dict[str, object]] = []
    per_true = counts.sum(axis=1)
    for true_name in counts.index:
        for predicted_name in counts.columns:
            if true_name == predicted_name:
                continue
            count = int(counts.loc[true_name, predicted_name])
            if count == 0:
                continue
            total = int(per_true[true_name])
            rows.append(
                {
                    "true": true_name,
                    "predicted": predicted_name,
                    "count": count,
                    "share_of_true": round(count / total, 4) if total else 0.0,
                }
            )
    frame = pd.DataFrame(rows, columns=["true", "predicted", "count", "share_of_true"])
    if frame.empty:
        return frame
    return (
        frame.sort_values(["count", "true", "predicted"], ascending=[False, True, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_confusion(
    preds: Predictions,
    path: str | Path,
    model_name: str = "model",
    normalise: bool = False,
) -> Path:
    """Purpose: draw the confusion matrix with both axes named on the figure.

    Inputs: the Predictions, the destination file, the model name for the title,
    and whether to normalise by true class.
    Output: the path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = confusion_frame(preds, normalise=normalise)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8.5, 7.0))
    sns.heatmap(
        frame,
        annot=True,
        fmt=".2f" if normalise else "d",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.4,
        ax=axes,
    )
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("True class")
    suffix = " (row-normalised)" if normalise else " (page counts)"
    axes.set_title(f"Confusion matrix — {model_name}{suffix}\nrows: true, columns: predicted")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
