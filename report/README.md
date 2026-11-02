# report/

The written report is submitted as a PDF, separately from this repository.
This folder holds the material the report is built from.

## Where the report's evidence lives

| Evidence | Location |
|----------|----------|
| The twelve report figures | [`outputs/report_figures/`](../outputs/report_figures/) |
| The three-model comparison table | [`outputs/comparison/comparison.csv`](../outputs/comparison/comparison.csv) |
| Per-class F1 for every model, and the winner of each class | [`outputs/comparison/per_class_all_models.csv`](../outputs/comparison/per_class_all_models.csv), [`class_strengths.csv`](../outputs/comparison/class_strengths.csv) |
| Each model's confusion matrix, curves and misclassified pages | `runs/<model>/figures/` |
| Each model's metrics, predictions and training log | `runs/<model>/metrics/`, `runs/<model>/logs/` |
| Dataset source, licence and counts | [`docs/dataset-card.md`](../docs/dataset-card.md) |

## The rule the results section follows

Every number in the report appears in a file under `outputs/` or `runs/`, and
every claim names the run that produced it. The comparison table, per-class
scores and error counts are read from those files rather than transcribed by
hand, which is what makes them checkable.

`report/figures/` is where frozen copies of cited figures go. `outputs/` is
regenerable, and re-running a model overwrites its figures, so a copy keeps
the document from changing meaning when a run is repeated.
