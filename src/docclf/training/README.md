# src/docclf/training/

The training loop, written once and shared by all three models. Fairness
in the comparison comes from this module being architecture-blind:
whatever differs between models must differ in the config, where it can
be reported and justified, never in the loop.

## The modules

| File | Responsibility |
|------|----------------|
| `trainer.py` | `fit(model, loaders, cfg, device, checkpoint_dir, class_weights, history_dir) -> TrainingHistory` — the epoch loop; returns per-epoch losses, validation accuracy and macro-F1, separate train/validation durations, and throughput. It checkpoints and flushes the history CSV after every epoch. Device and output paths are passed in rather than resolved here |
| `optim.py` | `build_optimiser(model, cfg)` and `build_scheduler(optimiser, cfg)` — kept separate so a per-architecture learning rate is a config value, not a code branch |
| `class_weights.py` | `compute_class_weights(train_split) -> Tensor` — inverse-frequency weights, applied only when the config enables them |
| `callbacks.py` | `EarlyStopping(patience, monitor)` and `Checkpoint(dir)` — checkpoint every epoch, because a Colab disconnect must cost one epoch and not a whole run |
| `history.py` | `TrainingHistory` — the record written to `outputs/metrics/`, and the source of the learning-curve figures |

## Integration

Consumes `models`, `data` loaders and `config`. Writes checkpoints and
history to `outputs/`. Training logs include split and class counts, system
details, configuration, per-epoch metrics, throughput and elapsed time. The
held-out test loader is never read during training; test metrics belong to
`evaluation`.

The tests hold this module to one guarantee: a two-epoch run on the smoke subset produces a history with monotonically recorded epochs, a checkpoint on disk, and no reference to the test loader anywhere in the module.

The loop reads two loader keys, `train` and `val`, and
they are declared as constants at the top of `trainer.py` so the claim can
be checked mechanically. Validation macro-F1 is computed here with
scikit-learn rather than imported from `evaluation`, because the
dependency direction forbids the import. Evidence:
`tests/test_no_test_leakage.py`, which reads this package's source, and
the end-to-end run in `tests/test_pipeline.py`.
