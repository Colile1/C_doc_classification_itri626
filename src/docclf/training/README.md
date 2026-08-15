# src/docclf/training/

The training loop, written once and shared by all three models. Fairness
in the comparison comes from this module being architecture-blind:
whatever differs between models must differ in the config, where it can
be reported and justified, never in the loop.

## Planned files

| File | Responsibility |
|------|----------------|
| `trainer.py` | `fit(model, loaders, cfg, device, checkpoint_dir, class_weights) -> TrainingHistory` — the epoch loop; returns per-epoch training loss, validation loss, validation accuracy and validation macro-F1. The device and the checkpoint directory are passed in rather than resolved here, so the loop stays free of environment knowledge |
| `optim.py` | `build_optimiser(model, cfg)` and `build_scheduler(optimiser, cfg)` — kept separate so a per-architecture learning rate is a config value, not a code branch |
| `class_weights.py` | `compute_class_weights(train_split) -> Tensor` — inverse-frequency weights, applied only when the config enables them |
| `callbacks.py` | `EarlyStopping(patience, monitor)` and `Checkpoint(dir)` — checkpoint every epoch, because a Colab disconnect must cost one epoch and not a whole run |
| `history.py` | `TrainingHistory` — the record written to `outputs/metrics/`, and the source of the learning-curve figures |

## Integration

Consumes `models`, `data` loaders and `config`. Writes checkpoints and
history to `outputs/`. Does not compute test metrics — that belongs to
`evaluation`, so that no code path can touch the test set during
training.

Done when: a two-epoch run on the smoke subset produces a history with
monotonically recorded epochs, a checkpoint on disk, and no reference to
the test loader anywhere in the module.

Built at step 9. The loop reads two loader keys, `train` and `val`, and
they are declared as constants at the top of `trainer.py` so the claim can
be checked mechanically. Validation macro-F1 is computed here with
scikit-learn rather than imported from `evaluation`, because the
dependency direction forbids the import. Evidence:
`tests/test_no_test_leakage.py`, which reads this package's source, and
the end-to-end run in `tests/test_pipeline.py`.
