# tests/

Unit tests, written before the module they cover. Two of these suites are
not routine: they are the evidence that the project's central claims hold.

## Planned files

| File | Responsibility |
|------|----------------|
| `test_config.py` | An unknown key is rejected by name; the same seed reproduces the same split |
| `test_manifest.py` | `doc_group_id` is parsed correctly from the archive's structure; a corrupt file is detected and reported |
| `test_leakage.py` | **Critical.** No `doc_group_id` spans two splits; a deliberately poisoned fixture is rejected; per-class counts sum to the working-set total |
| `test_transforms.py` | Validation and test transforms apply no randomness; no horizontal flip appears in the training pipeline |
| `test_models.py` | All three builders emit `(batch, 8)` logits from a `(batch, 3, H, W)` input, with no data required |
| `test_metrics.py` | **Critical.** Metrics match hand-computed values on a fixed confusion fixture, including an imbalanced case where accuracy and macro-F1 disagree |
| `test_no_test_leakage.py` | The training module contains no reference to the test loader; the dependency direction in `src/docclf/README.md` holds and the package has no import cycles |
| `test_pipeline.py` | The four orchestrators run end to end — prepare, split, train, evaluate — over a synthetic dataset written into a temporary directory. The real smoke run needs RVL-CDIP, and the suite must never require it |
| `conftest.py` | Puts `src/` on the import path and holds the two shared fixtures: a synthetic manifest, and a small class-per-directory dataset on disk |

## Integration

Run with `pytest tests/ -q` from the repository root. They use small
synthetic fixtures and never touch the real dataset, so they run in
seconds on the laptop and are the fast feedback loop the slow training
runs cannot provide.

Done when: the suite passes from a clean checkout with no dataset
present.

State: 95 tests pass in about 20 seconds on the laptop, with no dataset
present and no network access. The two critical suites are 12 tests each.
