# ITRI626 — Document image classification for client onboarding

**Project:** ITRI626 practical project · North-West University · 2026
**Group:** Colile Sibanda; [Member 2]; [Member 3]
**Title:** Comparative deep learning models for the automated classification of client onboarding document images

Classifies a scanned or photographed document page into one of eight
onboarding document types from its visual appearance alone, with no OCR,
so an intake queue can route the page automatically or flag it for human
review. Three models from three architectural families are compared on
one fixed, leakage-free split: a convolutional network trained from
scratch, a pretrained convolutional network fine-tuned by transfer
learning, and a pretrained vision transformer.

The claim this repository is built to defend is not "a model reached X
percent". It is that the number means what it says: the split is grouped
by source document so no page can leak between training and test, and the
metrics are computed and unit-tested against hand-worked values.

## Classes

invoice · budget · form · letter · memo · handwritten · resume · advertisement

Invoice and budget are the deliberate near-neighbour pair; letter and
memo are the second. Advertisement stands in for the irrelevant upload
that a real intake queue must reject.

## Layout

| Path | Purpose |
|------|---------|
| `docs/` | Dataset card, experiment log |
| `configs/` | One YAML per run; environment files are the only thing that differs per machine |
| `src/docclf/` | The library — see the module READMEs |
| `run_*.py` | Thin orchestrators, the only entry points |
| `notebooks/` | The six submitted notebooks |
| `tests/` | Unit tests, including the leakage and metric suites |
| `data/` | Manifests and split files are committed; images are not |
| `outputs/` | Metrics, figures, logs and checkpoints from runs |
| `report/` · `slides/` | The two assessed deliverables |

## Quick start

```bash
# Windows: create the environment OUTSIDE the repository. The repository path
# is deep enough that torch's own file names exceed the Windows path limit
# inside a repository-local .venv.
python -m venv C:\Users\<name>\venvs\itri626
C:\Users\<name>\venvs\itri626\Scripts\activate

pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

pytest tests/ -q          # 95 tests, about 20 seconds, no dataset needed

python run_prepare_data.py --config configs/smoke.yaml
python run_split.py        --config configs/smoke.yaml
python run_train.py        --config configs/smoke.yaml --model baseline_cnn
python run_evaluate.py     --config configs/smoke.yaml --model baseline_cnn
```

`smoke.yaml` is 25 images per class and two epochs — the whole pipeline
in under two minutes on a laptop CPU. Real runs swap the config and the
environment file.

## Dataset

RVL-CDIP, eight of its sixteen classes, capped per class to fit the
compute budget. Source, licence, usage conditions, class definitions and
the counts before and after cleaning are in `docs/dataset-card.md`. The
images are not committed; `docs/dataset-card.md` carries the download and
reproduction instructions.

## Where things run

The laptop authors code, prepares data and evaluates; Colab's T4 trains
all three models; the school GPU is escalation only. The hardware and
library versions behind every run are recorded in `docs/experiment-log.md`.

## Reproducibility contract

Every result is reproducible from three things: the config file, the
pinned `requirements.txt`, and the commit hash — all of which each run
records into `outputs/logs/`. One global seed drives every source of
randomness, and the split files in `data/splits/` are committed so that a
reader recreates the exact experiment rather than a similar one.
