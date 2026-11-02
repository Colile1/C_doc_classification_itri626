# data/

This is the working area for the dataset. Image files are not committed; only
the manifests and split files are, because those are small and are what makes
the experiment reproducible.

| Path | Contents | In git |
|------|----------|--------|
| `raw/` | The dataset as downloaded | No |
| `interim/` | Extracted and reorganised page images | No |
| `processed/` | The capped working set that training reads | No |
| `splits/` | `train.csv`, `val.csv`, `test.csv`, `cleaning_report.csv`, `split_summary.csv` | Yes |

## The dataset

Eight of the sixteen RVL-CDIP classes: invoice, budget, form, letter, memo,
handwritten, resume and advertisement. The source, licence and usage conditions
are recorded in `docs/dataset-card.md`.

## Why the split files matter

Every page carries a `doc_group_id` identifying the physical document it came
from, and splits are made over those groups rather than over individual pages.
Without that, two scans of the same document could land on opposite sides of
the train and test divide, and the test score would overstate real performance.

The three CSVs in `splits/` are the record of that assignment. Image paths are
stored relative to the project root, so the same files resolve on any machine.
**Do not regenerate them while results exist**: every recorded number was
measured on this split, and replacing it would make those numbers incomparable
while everything still appeared to work.
