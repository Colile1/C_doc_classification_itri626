# src/docclf/

The project library. Every module here is pure enough to test offline: no
network calls at import time, no global mutable state, and no printing.
Randomness is driven by the single seed set in `config.seed`.

## Module map

| Module | Responsibility |
|--------|----------------|
| `config/` | Load and validate the run configuration, resolve environment-specific paths, set the global seed |
| `data/` | Acquire, inventory, clean and transform the page images |
| `splits/` | Build the grouped train, validation and test split, and prove it has no leakage |
| `models/` | Construct each architecture behind one common interface |
| `training/` | The training loop, early stopping, checkpointing, class weighting, and resuming a paused run |
| `evaluation/` | Metrics, confusion matrices, curves, error analysis, efficiency measures, and the cross-model comparison |
| `utils/` | Device detection, structured logging, timing helpers |

## Dependency direction

The graph is acyclic, and deliberately so:

```
utils  <-  config  <-  data  <-  splits
                          \        /
                           v      v
                   models -> training -> evaluation
```

Nothing depends on `training` except `evaluation` and the root orchestrators.
`splits` never imports `models`. `evaluation` never imports the data loaders
directly; it receives an already-built DataLoader. `test_no_test_leakage.py`
enforces this, so a future change that introduces a cycle fails the suite
rather than passing unnoticed.
