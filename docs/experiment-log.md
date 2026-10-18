# Experiment log

One row per training run, written at the time, not reconstructed
afterwards. This is the second external memory: when the report asks why
a hyperparameter was chosen, the answer is here.

| # | Date | Model | Config | Commit | Environment | Epochs run | Best val macro-F1 | Test accuracy | Test macro-F1 | Wall-clock | What was learned |
|---|------|-------|--------|--------|-------------|-----------|-------------------|---------------|---------------|-----------|------------------|
| | | | | | | | | | | | |

## Environment, pinned 2026-08-15

The report has to state the library versions, so they are recorded once
here and repeated per run only when they change.

| Library | Version |
|---------|---------|
| Python | 3.11.9 |
| torch | 2.13.0 (+cpu on the laptop) |
| torchvision | 0.28.0 |
| timm | 1.0.28 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| pillow | 12.3.0 |
| imagehash | 4.3.2 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |

The full resolved environment, transitive dependencies included, is in
`requirements.lock.txt`. Colab supplies its own torch build; record the
version it reports at the top of each training notebook.

No training runs have happened yet: the table above the fold is empty
because the dataset is not in place. The first row is the baseline CNN's full run.

## What to record beyond the table

For each run, note in prose underneath:

- Anything that differed from the config on purpose, and why.
- Anything that went wrong: a disconnect, an out-of-memory, a restart.
- The single most useful thing the run revealed, even when the run failed.

A failed run with a recorded reason is worth more than a successful run
with no notes.
