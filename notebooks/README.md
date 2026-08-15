# notebooks/

The submitted notebooks. These are presentation surfaces, not the place
where logic lives: each one imports `docclf` and shows outputs. If a
notebook cell grows past a few lines of real logic, that logic belongs in
`src/` and the notebook calls it.

## Planned files

| File | Responsibility |
|------|----------------|
| `01_dataset_and_cleaning.ipynb` | Class definitions, counts before and after cleaning, what was removed and why, representative images from all eight classes |
| `02_splits_and_leakage.ipynb` | The grouped split, per-split class distribution, the seed, and the executed leakage check with its output visible |
| `03_train_baseline_cnn.ipynb` | Baseline training run and its curves |
| `04_train_resnet50.ipynb` | Transfer-learning run, including the frozen-layer justification |
| `05_train_vit.ipynb` | Transformer run |
| `06_evaluation_and_comparison.ipynb` | Test metrics for all three, confusion matrices, error analysis, efficiency table, final recommendation |

## Integration

Notebooks 03 to 05 are the ones that run on Colab; 01, 02 and 06 run
comfortably on the laptop. Each begins with the same four lines: install,
`sys.path` setup, `load_config`, `set_global_seed`.

Done when: every notebook has been re-run top to bottom in a fresh
kernel, and its committed copy has all outputs, figures and metric tables
visible, as the submission specification requires.
