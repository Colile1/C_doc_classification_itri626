# src/docclf/evaluation/  (correctness-critical)

Produces every number and figure the report and the presentation need,
from the held-out test set. Correctness-critical because a subtly wrong
averaging mode or a mislabelled confusion axis produces a plausible table
that says the opposite of the truth.

## Planned files

| File | Responsibility |
|------|----------------|
| `predict.py` | `collect_predictions(model, loader, device) -> Predictions` — logits, predicted labels, true labels and image paths, gathered once and reused by everything below |
| `metrics.py` | `compute_metrics(preds) -> MetricSet` — accuracy, macro and weighted precision/recall/F1, and the per-class table. Averaging mode is explicit in every call |
| `confusion.py` | `confusion_frame(preds, class_names) -> DataFrame` and `plot_confusion(...)` — rows are true classes, columns predicted, stated on the figure itself |
| `curves.py` | `plot_learning_curves(history)` — training and validation loss, and the validation accuracy or macro-F1 curve |
| `errors.py` | `sample_errors(preds, n)` — the most confidently wrong and the most confused pairs, with image paths, for the qualitative figures |
| `efficiency.py` | `measure_inference(model, loader) -> EfficiencySummary` — median per-image latency on a fixed batch size, plus model size |
| `compare.py` | `build_comparison_table(results: dict[str, MetricSet]) -> DataFrame` — the one table that puts all three models side by side |

## Integration

Consumes `models` and a test DataLoader. Writes to `outputs/metrics/` and
`outputs/figures/`, which `report/` and `slides/` then reference. Never
imports `training`.

Done when: `pytest tests/test_metrics.py` reproduces hand-computed
metrics on a small fixed confusion fixture, including a deliberately
imbalanced case where accuracy and macro-F1 disagree.
