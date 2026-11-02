# tests/

Unit tests, each written before the module it covers. Two of the suites are not
routine: they are the evidence behind the project's central claims, and they
are marked below.

## The suites

| File | What it checks |
|------|----------------|
| `test_config.py` | An unknown configuration key is rejected by name, and the same seed reproduces the same split |
| `test_manifest.py` | `doc_group_id` is parsed correctly from the archive structure, and a corrupt file is detected and reported |
| `test_leakage.py` | **Central.** No `doc_group_id` spans two splits, a deliberately poisoned fixture is rejected, per-class counts sum to the working-set total, and a split file written on one machine names the same pages when read on another |
| `test_transforms.py` | Validation and test transforms apply no randomness, and no horizontal flip reaches the training pipeline |
| `test_models.py` | Every builder emits `(batch, 8)` logits from a `(batch, 3, H, W)` input, without needing data |
| `test_metrics.py` | **Central.** Metrics match hand-computed values on a fixed confusion fixture, including an imbalanced case where accuracy and macro-F1 disagree |
| `test_crossmodel.py` | The cross-model comparison reads real run directories, merges efficiency data, names per-class winners, and skips a model whose results are absent rather than failing |
| `test_no_test_leakage.py` | The training module never references the test loader, and the package has no import cycles |
| `test_pipeline.py` | The four orchestrators run end to end over a synthetic dataset in a temporary directory, and a paused run resumes at the right epoch rather than restarting |
| `conftest.py` | Puts `src/` on the import path and holds the shared fixtures |

## Running them

```bash
pytest tests/ -q
```

The suite uses small synthetic fixtures and never touches RVL-CDIP, so it runs
in under a minute on a laptop with no dataset and no network access. That makes
it the fast feedback loop the hours-long training runs cannot provide.

Current state: 122 tests pass from a clean checkout.
