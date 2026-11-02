# outputs/

The cross-model results. Nothing here is edited by hand: `run_compare.py`
rebuilds `comparison/` from the three runs' recorded metrics.

| Path | Contents |
|------|----------|
| `comparison/` | The three-model comparison: `comparison.csv` (accuracy, macro-F1, latency, size), `per_class_all_models.csv`, `class_strengths.csv` (which model wins each class), `history_all_models.csv`, `errors_all_models.csv`, and four figures in `comparison/figures/` |
| `report_figures/` | The twelve figures the report cites, from the headline scores to the confusion matrices of all three models |
| `checkpoints/`, `figures/`, `logs/`, `metrics/` | Where a run writes when its config does not name its own directory, as the smoke test does. Empty in git |

Each model's own results are in [`runs/<model>/`](../runs/), one directory per
model, so re-running one model never overwrites another's files. Model weights
are never committed: they run to hundreds of megabytes.
