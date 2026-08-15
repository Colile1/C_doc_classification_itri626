# src/docclf/

The project library. Every module here is pure enough to unit-test
offline: no network calls at import time, no global mutable state, and no
printing. Randomness is always driven by the one seed set in
`config.seed`.

## Module map

| Module | Responsibility |
|--------|----------------|
| `config/` | Load and validate the run configuration; resolve environment-specific paths; set the global seed |
| `data/` | Acquire, inventory, clean and transform the page images |
| `splits/` | Build the grouped train/validation/test split and prove it has no leakage |
| `models/` | Construct the three architectures behind one common interface |
| `training/` | The training loop, early stopping, checkpointing, class weighting |
| `evaluation/` | Metrics, confusion matrix, curves, error analysis, efficiency measures |
| `utils/` | Device detection, structured logging, timing helpers |

## Dependency direction (acyclic)

```
utils  <-  config  <-  data  <-  splits
                          \        /
                           v      v
                   models -> training -> evaluation
```

Nothing depends on `training` except `evaluation` and the root
orchestrators. `splits` never imports `models`. `evaluation` never
imports `data` loaders directly; it receives an already-built DataLoader.

Done when: `pytest tests/` passes and an import-cycle check over
`src/docclf` reports none.
