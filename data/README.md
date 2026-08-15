# data/

Working area for the dataset. Image files are NOT committed — only the
manifests and split files, which are small, and which are what actually
make the experiment reproducible.

| Path | Contents | Committed? |
|------|----------|-----------|
| `raw/` | The dataset exactly as downloaded | No |
| `interim/` | Extracted and reorganised page images | No |
| `processed/` | The capped working set actually trained on | No |
| `splits/` | `manifest_clean.csv`, `splits.csv`, `cleaning_report.csv`, `split_summary.csv` | **Yes** |

## Dataset

RVL-CDIP, eight of its sixteen classes: invoice, budget, form, letter,
memo, handwritten, resume, advertisement. Source, licence and usage
conditions are recorded in `docs/dataset-card.md` and must be confirmed
before the data is used.

## The rule that matters

Every page carries a `doc_group_id` identifying the physical document it
came from. Splits are made over groups, never over individual pages.
`data/splits/splits.csv` is the record of that assignment, and
regenerating it from the recorded seed must reproduce it exactly.

Done when: a reader who has the repository and the download link can
recreate `processed/` and get split files identical to the committed
ones.
