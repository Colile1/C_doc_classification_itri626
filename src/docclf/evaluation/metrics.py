# metrics.py — the metric set, with the averaging mode explicit in every call.
# Author: Colile Sibanda  (correctness-critical: see tests/test_metrics.py)

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from docclf.evaluation.predict import Predictions

# Every average below is passed explicitly. `zero_division=0` makes a class the
# model never predicts score 0 rather than raise, which is the behaviour the
# imbalanced test case pins down.
_ZERO_DIVISION = 0


@dataclass(frozen=True)
class MetricSet:
    """Every number the report quotes for one model on one split."""

    model_name: str
    n_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class: pd.DataFrame = field(repr=False)

    def as_row(self) -> dict[str, object]:
        """Purpose: the single-row summary used by the comparison table.

        Output: a dictionary of the scalar metrics, rounded for reporting.
        """
        return {
            "model": self.model_name,
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_precision": round(self.weighted_precision, 4),
            "weighted_recall": round(self.weighted_recall, 4),
            "weighted_f1": round(self.weighted_f1, 4),
        }

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Purpose: write the summary and the per-class table to `outputs/metrics/`.

        Input: the destination directory. Output: the two files written.
        """
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        summary_path = destination / f"metrics_{self.model_name}.csv"
        per_class_path = destination / f"per_class_{self.model_name}.csv"
        pd.DataFrame([self.as_row()]).to_csv(summary_path, index=False, encoding="utf-8")
        self.per_class.round(
            {"precision": 4, "recall": 4, "f1": 4}
        ).to_csv(per_class_path, index=False, encoding="utf-8")
        return {"summary": summary_path, "per_class": per_class_path}


def compute_metrics(preds: Predictions, model_name: str = "model") -> MetricSet:
    """Purpose: compute the full required metric set from a prediction set.

    Inputs: the Predictions and the model name to label the row with.
    Output: a MetricSet holding accuracy, macro and weighted precision, recall
    and F1, and the per-class table. Macro treats every document type equally,
    which is the honest view when the classes are unevenly represented; weighted
    is reported alongside it so the two can be compared rather than confused.
    """
    if len(preds.y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty prediction set.")
    labels = list(range(len(preds.class_names)))
    macro = precision_recall_fscore_support(
        preds.y_true,
        preds.y_pred,
        labels=labels,
        average="macro",
        zero_division=_ZERO_DIVISION,
    )
    weighted = precision_recall_fscore_support(
        preds.y_true,
        preds.y_pred,
        labels=labels,
        average="weighted",
        zero_division=_ZERO_DIVISION,
    )
    return MetricSet(
        model_name=model_name,
        n_samples=int(len(preds.y_true)),
        accuracy=float(accuracy_score(preds.y_true, preds.y_pred)),
        macro_precision=float(macro[0]),
        macro_recall=float(macro[1]),
        macro_f1=float(macro[2]),
        weighted_precision=float(weighted[0]),
        weighted_recall=float(weighted[1]),
        weighted_f1=float(weighted[2]),
        per_class=per_class_table(preds),
    )


def per_class_table(preds: Predictions) -> pd.DataFrame:
    """Purpose: the per-class precision, recall, F1 and support table.

    Input: the Predictions.
    Output: a DataFrame with one row per class, in the fixed class order, so a
    class the model never predicts still appears with zeros rather than vanishing.
    Values are kept at full precision here and rounded only when written out.
    """
    labels = list(range(len(preds.class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        preds.y_true,
        preds.y_pred,
        labels=labels,
        average=None,
        zero_division=_ZERO_DIVISION,
    )
    return pd.DataFrame(
        {
            "class_name": list(preds.class_names),
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
