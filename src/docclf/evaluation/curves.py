# curves.py — the learning curves drawn from the recorded training history.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_learning_curves(
    history: pd.DataFrame, path: str | Path, model_name: str = "model"
) -> Path:
    """Purpose: draw training and validation loss, and the validation macro-F1.

    Inputs: the history frame from `TrainingHistory.to_frame()` (or the CSV it
    writes), the destination file, and the model name for the title.
    Output: the path written. Raises KeyError naming any column that is missing,
    because a silently blank curve would be worse than a failure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = ("epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1")
    missing = [name for name in required if name not in history.columns]
    if missing:
        raise KeyError(
            f"History is missing column(s) {', '.join(missing)}; the learning curves "
            f"cannot be drawn. Pass TrainingHistory.to_frame() or the history CSV."
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, (loss_axes, score_axes) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    loss_axes.plot(history["epoch"], history["train_loss"], marker="o", label="Training loss")
    loss_axes.plot(history["epoch"], history["val_loss"], marker="s", label="Validation loss")
    loss_axes.set_xlabel("Epoch")
    loss_axes.set_ylabel("Cross-entropy loss")
    loss_axes.set_title("Loss")
    loss_axes.legend()
    loss_axes.grid(alpha=0.3)
    score_axes.plot(
        history["epoch"], history["val_accuracy"], marker="o", label="Validation accuracy"
    )
    score_axes.plot(
        history["epoch"], history["val_macro_f1"], marker="s", label="Validation macro-F1"
    )
    score_axes.set_xlabel("Epoch")
    score_axes.set_ylabel("Score")
    score_axes.set_ylim(0.0, 1.0)
    score_axes.set_title("Validation score")
    score_axes.legend()
    score_axes.grid(alpha=0.3)
    figure.suptitle(f"Learning curves — {model_name}")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
