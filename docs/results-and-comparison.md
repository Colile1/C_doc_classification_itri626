# Results, comparison and what each file is for

Where every number in the report comes from, and which command produced it.
Nothing in the report should be typed by hand: if a figure is needed and is not
listed here, add it to `run_compare.py` so it is reproducible.

## One command

```bash
python run_compare.py
```

Reads every finished run and writes `outputs/comparison/`. It needs no
checkpoints and no images — only the metric CSVs each run leaves behind — so it
works on any machine, and it combines runs that finished on different machines.

Models that have not finished are reported as a warning and left out, so this is
useful from the first completed model onward; the figures fill in as runs land.
Pass `--strict` to fail instead, which is what the final run before submission
should use.

## What it writes

| File | What it is for |
|------|----------------|
| `comparison.csv` | The headline table: one row per model, accuracy, macro-F1, per-class averages, latency, parameters, size. Every aggregate number in the report comes from here. |
| `per_class_all_models.csv` | Every model's per-class precision, recall and F1, stacked. |
| `class_strengths.csv` | Which model wins each class, the runner-up, and the margin. |
| `history_all_models.csv` | Every model's epoch history, stacked. |
| `errors_all_models.csv` | Every model's misclassified pages. |
| `figures/metric_comparison.png` | Accuracy and macro-F1 per model, bars coloured by family. |
| `figures/per_class_comparison.png` | Per-class F1, grouped by class, hardest first. |
| `figures/history_comparison.png` | Validation macro-F1 and loss per epoch, all models on shared axes. |
| `figures/accuracy_against_cost.png` | Macro-F1 against inference time, marker area is the parameter count. |

Each model also keeps its own detailed artefacts in its run directory:
confusion matrices (plain and normalised), the error-example grid, its learning
curves, its predictions, and its efficiency measurement.

## Which figure answers which question

The brief asks for specific things; these are the figures that answer them.

| The question | The figure or table |
|--------------|---------------------|
| Which model is best overall? | `comparison.csv`, `figures/metric_comparison.png` |
| Better *at what*? Where does each one win? | `class_strengths.csv`, `figures/per_class_comparison.png` |
| What did pretraining actually buy? | `figures/history_comparison.png` — where each model starts and how fast it climbs |
| Which classes are confused, and why? | Each model's `figures/confusion_*.png` and `errors_*.csv` |
| Which model to deploy? | `figures/accuracy_against_cost.png`, plus the recommendation line `run_compare.py` logs |
| Is the comparison fair? | All models share one committed split; `run_evaluate.py` re-checks group overlap before touching the held-out pages |

## Reading the per-class table

The aggregate score hides the thing the discussion needs. ViT-B/16 leads
overall, but it wins only four of the eight classes: form, budget, invoice and
resume. ResNet50 wins the other four: advertisement, handwritten, letter and
memo. ViT-B/16's lead comes from the three hardest classes, the confusable
business documents, where it gains most over the from-scratch baseline
(+0.106 F1 on invoice). On resume, advertisement and handwritten all three
models sit within about 0.03 F1 of each other. A report that only quotes
macro-F1 cannot say any of that.

## A caution on the latency numbers

`median_ms_per_image` is measured on whatever machine ran the evaluation, while
whatever else that machine was doing, so latencies from different machines or
different load are not comparable. The three figures in `comparison.csv` were
all measured on the same Colab Tesla T4 (1.5 ms, 3.3 ms and 13.4 ms per page),
so they compare directly.

To re-measure on another machine, evaluate all three back to back while it is
otherwise idle:

```bash
python run_evaluate.py --config configs/baseline_cnn_v4.yaml --env local --model baseline_cnn_v4
python run_evaluate.py --config configs/resnet50.yaml --env local --model resnet50
python run_evaluate.py --config configs/vit_b16.yaml --env local --model vit_b16
python run_compare.py
```

Accuracy and F1 are unaffected — they do not depend on machine load. Only the
timing column does.

## Rebuilding after a run finishes

```bash
git pull                 # collect what other machines finished
python run_compare.py    # rebuild every table and figure
python run_status.py     # confirm what is still outstanding
```

Notebook `00_run_control.ipynb` does the same from a notebook, and displays the
figures inline. Notebook `06_evaluation_and_comparison.ipynb` is the submitted
analysis, with the confusion matrices and error examples in full.
