# errors.py — the qualitative half of the error analysis (objective O7).
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import pandas as pd

from docclf.evaluation.confusion import most_confused_pairs
from docclf.evaluation.predict import Predictions


def sample_errors(preds: Predictions, n: int = 12) -> pd.DataFrame:
    """Purpose: the most confidently wrong pages, which is where the story is.

    Inputs: the Predictions and how many rows to return.
    Output: a DataFrame of misclassified pages ordered by descending confidence,
    with the path, the true and predicted class, and the confidence. An empty
    frame with the same columns is returned when nothing was misclassified.
    """
    frame = preds.to_frame()
    wrong = frame[~frame["correct"]].copy()
    if wrong.empty:
        return wrong.head(0)
    return (
        wrong.sort_values("confidence", ascending=False)
        .head(n)
        .reset_index(drop=True)[
            ["path", "true_class", "predicted_class", "confidence"]
        ]
    )


def sample_pair_examples(
    preds: Predictions, true_class: str, predicted_class: str, n: int = 4
) -> pd.DataFrame:
    """Purpose: example images for one confusion direction, for the report figure.

    Inputs: the Predictions, the true and predicted class names, and how many.
    Output: a DataFrame of matching pages, most confident first.
    """
    frame = preds.to_frame()
    subset = frame[
        (frame["true_class"] == true_class) & (frame["predicted_class"] == predicted_class)
    ]
    return (
        subset.sort_values("confidence", ascending=False)
        .head(n)
        .reset_index(drop=True)[["path", "true_class", "predicted_class", "confidence"]]
    )


def plot_error_examples(
    preds: Predictions, path: str | Path, n: int = 8, model_name: str = "model"
) -> Path:
    """Purpose: a contact sheet of the most confidently wrong pages.

    Inputs: the Predictions, the destination file, how many pages, the model name.
    Output: the path written. Pages whose image file is missing are skipped with
    a placeholder panel rather than aborting the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    errors = sample_errors(preds, n)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = min(4, max(len(errors), 1))
    rows = max(1, (len(errors) + columns - 1) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.6 * rows))
    panels = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for panel in panels:
        panel.axis("off")
    for panel, (_, row) in zip(panels, errors.iterrows()):
        try:
            with Image.open(row["path"]) as image:
                panel.imshow(image.convert("L"), cmap="gray")
        except (FileNotFoundError, OSError):
            panel.text(0.5, 0.5, "image unavailable", ha="center", va="center")
        panel.set_title(
            f"true: {row['true_class']}\npredicted: {row['predicted_class']} "
            f"({row['confidence']:.2f})",
            fontsize=9,
        )
    figure.suptitle(f"Most confident errors — {model_name}")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def error_summary(preds: Predictions, top_n: int = 5) -> pd.DataFrame:
    """Purpose: the confused-pair table the report quotes in section 7.

    Inputs: the Predictions and how many pairs to keep.
    Output: the ranked pair table from `most_confused_pairs`.
    """
    return most_confused_pairs(preds, top_n=top_n)
