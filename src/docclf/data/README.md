# src/docclf/data/

Everything between "a public dataset exists somewhere" and "a DataLoader
yields tensors". This module owns the evidence the report needs for
dataset preparation: counts before and after cleaning, what was removed
and why, and representative examples per class.

## The modules

| File | Responsibility |
|------|----------------|
| `acquire.py` | `fetch_dataset(cfg, raw_dir: Path) -> Path` — locate or extract RVL-CDIP under the resolved raw directory, verify the archive checksum, never re-download silently. The directory is passed in rather than re-derived, so `data` does not depend on `config.paths` |
| `manifest.py` | `build_manifest(root: Path, classes: list[str], group_pattern: str) -> DataFrame` — one row per page: `path`, `class_name`, `class_id`, `doc_group_id`, `width`, `height`, `bytes`, `readable`. The `doc_group_id` is what makes the grouped split possible; `readable` carries a corrupt file through to `clean` so the removal is reported rather than silent |
| `clean.py` | `clean_manifest(df, cfg) -> tuple[DataFrame, CleaningReport]` — drop corrupt files, drop near-duplicates by perceptual hash, drop images below a size floor; the report records every removal with a reason |
| `subsample.py` | `cap_per_class(df, n_per_class, seed) -> DataFrame` — apply the compute-budget cap without breaking `doc_group_id` integrity |
| `transforms.py` | `build_transforms(cfg, train: bool) -> Callable` — resize, grayscale-to-3-channel, normalise with pretraining statistics; augmentation only when `train=True`, and never a horizontal flip |
| `dataset.py` | `DocumentPageDataset(df, transforms)` — a `torch.utils.data.Dataset` reading rows of the manifest |
| `loaders.py` | `build_dataloaders(splits, cfg) -> dict[str, DataLoader]` — batch size, workers and pin-memory chosen from the resolved environment |

## Integration

Consumes `config`; produces the manifest that `splits` divides. Holds no
knowledge of which model will consume the batches — input size and
normalisation arrive through the config, not through a model import.

The tests hold this module to one guarantee: the cleaning report accounts for every image lost between the raw count and the working count, and `build_transforms(train=False)` provably applies no random operation.

Built at steps 4, 5 and 7. `clean_manifest` adds a `phash` column, which
is what lets `splits.leakage` run its second line of defence without
re-reading the images. `build_dataloaders` takes the whole split frame,
not one split at a time, because only it is allowed to decide that the
training loader is the only one that shuffles and augments. Evidence:
`tests/test_manifest.py` and `tests/test_transforms.py`.
