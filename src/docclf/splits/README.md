# src/docclf/splits/  (correctness-critical)

Divides the manifest into training, validation and test sets so that no
source document appears on both sides of the divide. This is the single
module that can silently invalidate the entire project: a leaked split
produces high test scores that mean nothing, and nothing downstream will
complain. It is therefore built test-first and reviewed before use.

## Planned files

| File | Responsibility |
|------|----------------|
| `grouped.py` | `make_grouped_split(df, ratios, seed) -> DataFrame` — assign whole `doc_group_id` groups to one split each, stratified by class as closely as group sizes permit |
| `leakage.py` | `assert_no_group_overlap(splits) -> None` — raise `LeakageError` on any `doc_group_id` present in more than one split. `check_near_duplicates(splits, max_distance) -> list[Violation]` — the second line of defence, catching visually identical pages that carry different group identifiers. The hashes are read from the frame's `phash` column, which `data.clean` has already computed, rather than being passed separately |
| `report.py` | `describe_splits(splits) -> DataFrame` — per-split, per-class counts and percentages, the table the report must contain |
| `io.py` | `save_splits(splits, path)` / `load_splits(path)` — CSV files committed to `data/splits/`, so a reader can recreate the exact experiment from the seed |

## Integration

Consumes the cleaned manifest from `data`; produces the split assignment
every model in `training` must use. Imports nothing from `models` or
`training` — the split is fixed once and is not permitted to vary per
architecture.

Done when: `pytest tests/test_leakage.py` passes, including a
deliberately poisoned fixture that the check must reject, and
`describe_splits` totals equal the working-set count exactly.

The tests were written before the code. Twelve tests pass, including the
poisoned fixture, the planted cross-split near-duplicate, reproducibility
from the seed, and the CSV round trip. The assignment places whole groups
largest-first into the split with the largest remaining page deficit, per
class; group integrity is the hard constraint and the ratios bend to it,
so the realised shares are close to but not exactly 70/15/15. A class
whose groups would leave a split empty is repaired by moving its smallest
group, and a class with fewer than three source documents is warned about
rather than silently split.
