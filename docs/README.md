# docs/

Project documents that are not the report. They record the process
behind the results: where the data came from and every training run.

| File | Purpose |
|------|---------|
| `dataset-card.md` | Source, licence, usage conditions, class definitions, counts before and after cleaning |
| `experiment-log.md` | One row per training run: config, commit, environment, result, what was learned |
| `results-and-comparison.md` | Which comparison figure answers which question, and how to read latency figures |
| `colab.md` | Training a model on a Colab GPU from the prepared bundle |
| `multi-machine.md` | Training different models on several machines at once without collisions |
| `images/` | Dashboard screenshots used in the main README |

A reader new to the project can read `dataset-card.md` and
`experiment-log.md` and know exactly what was trained, on what, and how.
