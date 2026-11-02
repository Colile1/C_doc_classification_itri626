# src/docclf/evaluation/  (correctness-critical)

Produces every number and figure the report and the presentation need,
from the held-out test set. Correctness-critical because a subtly wrong
averaging mode or a mislabelled confusion axis produces a plausible table
that says the opposite of the truth.

## The modules

| File | Responsibility |
|------|----------------|
| `predict.py` | `collect_predictions(model, loader, device, class_names) -> Predictions` — logits, predicted labels, true labels and image paths, gathered once and reused by everything below. A shuffling loader is rejected, because the recorded paths would no longer line up with the predictions |
| `metrics.py` | `compute_metrics(preds, model_name) -> MetricSet` — accuracy, macro and weighted precision/recall/F1, and the per-class table. Averaging mode is explicit in every call, and `zero_division=0` is stated once at the top of the file |
| `confusion.py` | `confusion_frame(preds, class_names) -> DataFrame` and `plot_confusion(...)` — rows are true classes, columns predicted, stated on the figure itself |
| `curves.py` | `plot_learning_curves(history)` — training and validation loss, and the validation accuracy or macro-F1 curve |
| `errors.py` | `sample_errors(preds, n)` — the most confidently wrong and the most confused pairs, with image paths, for the qualitative figures |
| `efficiency.py` | `measure_inference(model, loader) -> EfficiencySummary` — median per-image latency on a fixed batch size, plus model size |
| `compare.py` | `build_comparison_table(results: dict[str, MetricSet]) -> DataFrame` — the one table that puts all three models side by side |

## Integration

Consumes `models` and a test DataLoader. Writes to `outputs/metrics/` and
`outputs/figures/`, which `report/` and `slides/` then reference. Never
imports `training`.

The tests hold this module to one guarantee: `pytest tests/test_metrics.py` reproduces hand-computed metrics on a small fixed confusion fixture, including a deliberately imbalanced case where accuracy and macro-F1 disagree.

Built at steps 10 and 11, tests written first. Twelve tests pass. The
imbalanced fixture is eight correct pages of one class against one page
each of two classes the model never predicts: accuracy 0.8, macro-F1
0.2963. Values in `per_class` are kept at full precision and rounded only
when written to CSV, so the hand-computed comparison is exact.
`confusion.py` names its axes on the figure and its index and columns
(`true`, `predicted`) in the frame itself.
