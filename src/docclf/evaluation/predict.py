# predict.py — gather predictions once, and let everything below reuse them.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Predictions:
    """Logits, predicted labels, true labels and image paths for one split."""

    y_true: np.ndarray
    y_pred: np.ndarray
    logits: np.ndarray
    paths: Sequence[str]
    class_names: Sequence[str]

    def __post_init__(self) -> None:
        """Purpose: refuse a malformed prediction set, loudly and by name."""
        lengths = {
            "y_true": len(self.y_true),
            "y_pred": len(self.y_pred),
            "logits": len(self.logits),
            "paths": len(self.paths),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"y_true, y_pred, logits and paths must have the same length, got "
                f"{lengths}."
            )
        n_classes = len(self.class_names)
        for name, values in (("y_true", self.y_true), ("y_pred", self.y_pred)):
            if len(values) and (int(np.min(values)) < 0 or int(np.max(values)) >= n_classes):
                raise ValueError(
                    f"{name} contains a label outside the {n_classes}-class set "
                    f"(0 to {n_classes - 1}). Check the class ordering in the manifest."
                )

    @property
    def probabilities(self) -> np.ndarray:
        """Purpose: softmax scores, for the confidence-based error analysis."""
        shifted = self.logits - self.logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    @property
    def confidence(self) -> np.ndarray:
        """Purpose: the probability assigned to the predicted class, per page."""
        return self.probabilities[np.arange(len(self.y_pred)), self.y_pred]

    def to_frame(self) -> pd.DataFrame:
        """Purpose: the per-page prediction table written to `outputs/metrics/`.

        Output: a DataFrame with path, true and predicted class, and confidence.
        """
        return pd.DataFrame(
            {
                "path": list(self.paths),
                "true_id": self.y_true,
                "true_class": [self.class_names[index] for index in self.y_true],
                "predicted_id": self.y_pred,
                "predicted_class": [self.class_names[index] for index in self.y_pred],
                "confidence": self.confidence.round(6),
                "correct": self.y_true == self.y_pred,
            }
        )

    def save(self, directory: str | Path, model_name: str) -> Path:
        """Purpose: persist the per-page predictions for the error analysis.

        Inputs: the destination directory and the model name.
        Output: the file written.
        """
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"predictions_{model_name}.csv"
        self.to_frame().to_csv(path, index=False, encoding="utf-8")
        return path


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Predictions:
    """Purpose: run the model over a loader once and keep everything downstream needs.

    Inputs: the model, a DataLoader, the device, and the ordered class names.
    Output: a Predictions object. The loader must not shuffle, otherwise the
    paths would not line up with the rows; this is checked rather than assumed.
    """
    sampler = getattr(loader, "sampler", None)
    if type(sampler).__name__ == "RandomSampler":
        raise ValueError(
            "collect_predictions requires a non-shuffling loader, otherwise the "
            "recorded image paths do not correspond to the predictions."
        )
    model = model.to(device).eval()
    logits: list[np.ndarray] = []
    truths: list[int] = []
    for images, labels in loader:
        batch = model(images.to(device))
        logits.append(batch.detach().float().cpu().numpy())
        truths.extend(int(item) for item in labels)
    stacked = (
        np.concatenate(logits, axis=0)
        if logits
        else np.zeros((0, len(class_names)), dtype=float)
    )
    dataset = loader.dataset
    paths = list(getattr(dataset, "paths", [f"row_{index}" for index in range(len(truths))]))
    return Predictions(
        y_true=np.array(truths, dtype=int),
        y_pred=stacked.argmax(axis=1).astype(int) if len(stacked) else np.array([], dtype=int),
        logits=stacked,
        paths=paths,
        class_names=tuple(class_names),
    )
