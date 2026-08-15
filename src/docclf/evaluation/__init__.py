# __init__.py — public surface of the evaluation module.
# Author: Colile Sibanda

from docclf.evaluation.compare import build_comparison_table
from docclf.evaluation.confusion import confusion_frame, most_confused_pairs, plot_confusion
from docclf.evaluation.curves import plot_learning_curves
from docclf.evaluation.efficiency import EfficiencySummary, measure_inference
from docclf.evaluation.errors import sample_errors
from docclf.evaluation.metrics import MetricSet, compute_metrics
from docclf.evaluation.predict import Predictions, collect_predictions

__all__ = [
    "EfficiencySummary",
    "MetricSet",
    "Predictions",
    "build_comparison_table",
    "collect_predictions",
    "compute_metrics",
    "confusion_frame",
    "measure_inference",
    "most_confused_pairs",
    "plot_confusion",
    "plot_learning_curves",
    "sample_errors",
]
