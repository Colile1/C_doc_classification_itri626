# Experiment log

One row per training run, written at the time, not reconstructed
afterwards. This is the second external memory: when the report asks why
a hyperparameter was chosen, the answer is here.

| # | Date | Model | Config | Commit | Environment | Epochs run | Best val macro-F1 | Test accuracy | Test macro-F1 | Wall-clock | What was learned |
|---|------|-------|--------|--------|-------------|-----------|-------------------|---------------|---------------|-----------|------------------|
| | | | | | | | | | | | |

## What to record beyond the table

For each run, note in prose underneath:

- Anything that differed from the config on purpose, and why.
- Anything that went wrong: a disconnect, an out-of-memory, a restart.
- The single most useful thing the run revealed, even when the run failed.

A failed run with a recorded reason is worth more than a successful run
with no notes.
