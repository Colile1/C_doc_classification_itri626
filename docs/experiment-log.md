# Experiment log

One row per training run, written at the time, not reconstructed
afterwards. This is the second external memory: when the report asks why
a hyperparameter was chosen, the answer is here.

| # | Date | Model | Config | Commit | Environment | Epochs run | Best val macro-F1 | Test accuracy | Test macro-F1 | Wall-clock | What was learned |
|---|------|-------|--------|--------|-------------|-----------|-------------------|---------------|---------------|-----------|------------------|
| 1 | 2026-10-21 | baseline_cnn | `configs/baseline_cnn.yaml` | Local working tree (not committed) | Machine B (desktop), Linux, 24 logical cores, CPU-only; batch size 8 | 25 | 0.834627 (epoch 22) | 82.83% | 0.8274 | 7,983.688 s (2 h 13 m 4 s; 319.34 s/epoch) | Test result is close to validation; errors cluster among form, invoice, and budget. |
| 2 | 2026-10-22 | baseline_cnn_v2 | `configs/baseline_cnn_v2.yaml` | Local working tree (not committed) | Machine B (desktop), Linux, 24 logical cores, CPU-only; batch size 8; Python 3.14.6; torch 2.13.0+cpu | 12 | 0.838374 (epoch 12) | 83.37% | 0.8328 | 3,732.263 s (1 h 2 m 12 s; 311.01 s/epoch) | Warm-started from baseline epoch 22; held-out accuracy and macro-F1 each rose 0.54 percentage points versus baseline. Form remains the weakest class. |
| 3 | 2026-10-24 | baseline_cnn_v2 (timing benchmark) | `configs/bench_1epoch.yaml` | Local working tree (not committed) | Machine A (laptop), Windows 11, i5-12450H, 12 logical cores, CPU-only; batch size 8; Python 3.11.9; torch 2.13.0+cpu | 1 | 0.8158 (epoch 1) | not evaluated | not evaluated | 3 245.804 s (54 m 6 s for one epoch) | Hardware benchmark only, not a result run. Reproduced v2 epoch 1 to within 0.0013 macro-F1 while taking 9.05x the wall time, so the laptop is ~9-10x slower than the desktop and the remaining full runs belong on the desktop. |
| 4 | 2026-10-24 | vit_b16 | `configs/vit_b16.yaml` | Local working tree (not committed) | Machine C (Colab), Tesla T4, CUDA; batch size 32, AMP; Python 3.13.16; torch 2.11.0+cu130 | 15 | 0.88204 (epoch 14) | 88.29% | 0.8828 | 2 965.292 s (49 m 25 s; 188.47 s/epoch) | Best of the three families. Train loss fell to 0.0074 against a validation loss plateau near 0.58 from epoch 6, so it memorised the training set, yet validation macro-F1 kept rising to epoch 14. Capacity helps on the confusable classes and costs 327 MB. |
| 5 | 2026-10-24 | resnet50 | `configs/resnet50.yaml` | Local working tree (not committed) | Machine C (Colab), Tesla T4, CUDA; batch size 32, AMP; Python 3.13.16; torch 2.11.0+cu130 | 20 | 0.87382 (epoch 18) | 87.42% | 0.8736 | 2 139.073 s (35 m 39 s; 105.34 s/epoch) | Closes **D2**. `freeze_upto: layer2`, so 22 079 496 of 23 524 424 parameters were fine-tuned. Within 0.009 macro-F1 of ViT at a quarter of the inference cost and a third of the size. |
| 6 | 2026-10-25 | baseline_cnn_v4 | `configs/baseline_cnn_v4.yaml` | Local working tree (not committed) | Machine C (Colab), Tesla T4, CUDA; batch size 32, AMP; Python 3.13.16; torch 2.11.0+cu130 | 25 | — (epoch 24) | 83.13% | 0.8308 | — (best epoch 24 of 25) | The baseline retrained on the GPU platform so all three families share one device and batch size. Scores 0.8308 against the CPU run's 0.8274 at batch size 8; the architecture and schedule are identical, so the difference is batch size and run-to-run variation. |
| 7 | 2026-10-25 | swin_t | `configs/swin_t.yaml` | Local working tree (not committed) | Machine C (Colab), Tesla T4, CUDA; batch size 32, AMP; Python 3.13.16; torch 2.11.0+cu130 | 15 | 0.8680 (epoch 13) | 86.08% | 0.8603 | — | Secondary, not a fourth family. A hierarchical transformer at 27.5 M parameters against ViT-B/16's 85.8 M; it scores 0.0225 lower, so ViT-B/16 remains the transformer entry. |

## Environment, pinned 2026-08-15

The report has to state the library versions, so they are recorded once

here and repeated per run only when they change.

| Library | Version |

|---------|---------|

| Python | 3.11.9 on Machine A (laptop); 3.14.6 on Machine B (desktop), where both training runs were done |

| torch | 2.13.0+cpu on both machines |

| torchvision | 0.28.0 |

| timm | 1.0.28 |

| numpy | 2.4.6 |

| pandas | 3.0.5 |

| scikit-learn | 1.9.0 |

| pillow | 12.3.0 |

| imagehash | 4.3.2 |

| matplotlib | 3.11.1 |

| seaborn | 0.13.2 |

The full resolved environment, transitive dependencies included, is in

`requirements.lock.txt`. Colab supplies its own torch build; record the

version it reports at the top of each training notebook.

The baseline CNN training run completed on the prepared 16,000-example set:

11,200 train, 2,400 validation, and 2,400 held-out test examples. Validation

was measured at every epoch; the best validation macro-F1 was 0.834627 at

epoch 22. The best checkpoint was evaluated on the held-out test set with

82.83% accuracy and 0.8274 macro-F1. These test metrics are the baseline

reference for v2; do not use them to tune v2.

The training log and per-epoch CSV are `outputs/logs/train_baseline_cnn.log`

and `outputs/metrics/history_baseline_cnn.csv`. The result analysis and v2

checkpoint lineage are documented in `docs/model-v2.md`.

The v2 run completed all 12 epochs on the same split, initialized from the

saved baseline epoch 22 weights. Its best validation macro-F1 was 0.838374 at

epoch 12. The 2,400 held-out test examples were not read during training and

were evaluated after checkpoint selection: accuracy 0.8337 and macro-F1

0.8328, each 0.0054 above baseline. The v2 history, summary, test metrics,

per-class scores, predictions, error table, and figures are under

`outputs/v2/`.

### V2 work and evaluation record

- Built `baseline_cnn_v2` on the saved baseline epoch 22 weights. The existing

  baseline checkpoint, training log, history, and config were preserved.

- Added strict checkpoint initialization to the training command and registered

  the v2 model name for the same CNN architecture. Added proportional resize

  with white padding; v2 also uses no blur, crops of 95–100%, rotation up to

  3 degrees, brightness/contrast jitter of 0.10, learning rate 0.0002, AdamW,

  cosine scheduling, and patience 4. Training artifacts are isolated under

  `outputs/v2/`.

- Kept the grouped 70/15/15 split unchanged (11,200 train, 2,400 validation,

  2,400 test; 300 test pages per class). The split overlap check passed.

- Trained for all 12 configured epochs on CPU. Best validation checkpoint was

  epoch 12, with validation accuracy 83.96%, macro-F1 0.838374, and loss

  0.515219. The training log and epoch history are saved under `outputs/v2/`.

- Evaluated the selected checkpoint on the test split after training. The

  initial evaluation and two subsequent reruns all returned 83.37% accuracy,

  0.8328 macro-F1, and 0.8328 weighted F1 over 2,400 pages. Evaluation

  durations reported by the evaluator were approximately 33.2, 27.9, and

  31.3 seconds. No additional training occurred during these reruns.

- The test evaluation generated aggregate/per-class metrics, predictions,

  efficiency metrics, an error table, confusion matrices, and error plots in

  `outputs/v2/metrics/` and `outputs/v2/figures/`; the latest evaluator log is

  `outputs/v2/logs/evaluate_baseline_cnn_v2.log`.

- Compared with baseline, test accuracy and macro-F1 each increased by 0.54

  percentage points. V2's hardest class is form (F1 0.7014); the largest

  confusion pairs include invoice→form (31) and form→budget (30). The

  aggregate improvement is modest, and form remains the clearest area to

  investigate. No failure or interruption was recorded during the v2 run.

- Created `outputs/v2/figures/test_metrics_comparison.png` and linked it from

  `docs/model-v2.md`. The chart compares aggregate metrics only because a

  complete baseline per-class metrics file was not present.

The aggregate test comparison is visualized in

`outputs/v2/figures/test_metrics_comparison.png`. Accuracy and macro-F1 each

improved by 0.54 percentage points. V2 form F1 remains low at 0.7014, only

0.0035 above the baseline's recorded form F1 of 0.6979; the common v2

confusions also include form→budget. This points to form classification as

the clearest target for further investigation, while the aggregate gain is

modest and should be confirmed on another independent sample if available.

## Hardware: the three machines

Both completed training runs were done on **Machine B**. The report's
experimental-setup table must name the machine, because the two differ by
roughly an order of magnitude and the choice of machine decides what is
feasible before the deadline.

| | Machine A — laptop | Machine B — desktop | Machine C — Colab |
|---|---|---|---|
| CPU | Intel Core i5-12450H, 12th gen | not recorded directly; 24 logical cores | 2 logical cores (host) |
| Cores | 8 physical / 12 logical | 24 logical | 2 logical, plus the GPU |
| Base clock | 2.0 GHz | not recorded | not recorded |
| RAM | 15.75 GB total (~4.7 GB free during the run) | not recorded | not recorded |
| GPU | Intel UHD Graphics (unused; CPU-only build) | none used; CPU-only build | **NVIDIA Tesla T4** |
| OS | Windows 11 Home 10.0.26200 | Linux 7.0.0-30-generic, glibc 2.43 | Linux 6.6.122+ |
| Python | 3.11.9 | 3.14.6 | 3.13.16 |
| torch | 2.13.0+cpu | 2.13.0+cpu | 2.11.0+cu130 |
| pandas | 3.0.5 | 3.0.5 | 2.2.3 |
| CUDA | not available | not available | available |
| Project storage | internal disk, `D:\` | external Ventoy volume, `/run/media/student/Ventoy/` | Google Drive; images on local `/content` |

### Measured one-epoch comparison, 2026-10-24

One epoch was run on Machine A under settings identical to the v2 run — same
`baseline_cnn_v2` config and architecture, same starting checkpoint (baseline
epoch 22), same grouped split (11 200 train / 2 400 val, 1 400 batches of 8,
`num_workers: 0`), CPU-only, seed 626. Only the split file paths were
rewritten, because the committed CSVs hold Machine B's absolute Linux paths;
the image set is byte-identical, and all 16 000 images were verified present on
Machine A before the run. The comparison is against **v2 epoch 1**, the
matching cold-cache epoch.

| Measure | Machine A (laptop) | Machine B (desktop) | A / B |
|---------|--------------------|---------------------|-------|
| Epoch total | 3 245.8 s (54 m 06 s) | 358.7 s (5 m 59 s) | **9.05x slower** |
| Training phase | 3 017.2 s | 328.7 s | 9.18x |
| Validation phase | 228.5 s | 29.9 s | 7.63x |
| Training throughput | 3.7 images/s | 34.07 images/s | 9.2x |
| Validation throughput | 10.5 images/s | 80.13 images/s | 7.6x |
| Against B's steady-state epoch (306.5 s) | 3 245.8 s | 306.5 s | 10.59x |

The learning result was reproduced, which is what makes the timing a hardware
comparison rather than an accident of a different computation:

| Measure | Machine A epoch 1 | Machine B epoch 1 |
|---------|-------------------|-------------------|
| Train loss | 0.6061 | 0.605994 |
| Validation loss | 0.5855 | 0.585443 |
| Validation accuracy | 0.8167 | 0.815417 |
| Validation macro-F1 | 0.8158 | 0.814475 |

Residual differences sit at the fourth decimal and are consistent with
non-deterministic CPU kernel and thread-scheduling order across two different
CPUs and Python builds, not with a difference in the work performed.

Machine A sustained about 7.3 busy cores (73 CPU-seconds per 10 wall seconds)
with resident memory stable near 0.7–0.9 GB, so the gap is throughput, not
memory pressure or swapping. The ~9x figure is the honest like-for-like number;
against Machine B's steady-state epoch the laptop is ~10.6x slower, because
Machine B's own epoch 1 carried cold-cache cost that its later epochs did not.

**What this means for the remaining runs.** Reproducing the finished work on
Machine A would cost roughly 22.5 h for the 25-epoch baseline and 10.8 h for
the 12-epoch v2 fine-tune, against the 2.22 h and 1.04 h actually measured on
Machine B. Steps 14 and 15 — the pretrained CNN and the transformer, both
heavier than this 1.18 M-parameter baseline — belong on Machine B. Machine A is
suited to the test suite, smoke runs, evaluating a saved checkpoint, and
authoring.

To repeat the benchmark, copy `configs/baseline_cnn_v2.yaml` to a scratch
config with `train.epochs: 1` and `paths` pointing at a scratch directory
(create the checkpoints directory first), rewrite the split CSV paths for the
machine, and run `run_train.py --env local --model baseline_cnn_v2
--init-checkpoint outputs/checkpoints/baseline_cnn_best.pt`. The scratch config
used here was not kept, because its paths were session-specific.

This benchmark wrote nothing into `outputs/`: it used a scratch config and
scratch output directories, so the recorded baseline and v2 results are
untouched. The run stopped with a `RuntimeError` while saving the checkpoint
after the epoch, because the scratch checkpoint directory was not pre-created.
The epoch had already completed and been timed, and a timing measurement needs
no checkpoint, so it was not repeated.

## The three-family comparison

Built by `run_compare.py` from the recorded metrics, into `outputs/comparison/`.
All three were trained and evaluated on the same committed split, selected on
validation macro-F1, and scored once on the same 2 400 held-out pages, on the
same Tesla T4 at batch size 32.

| Family | Model | Accuracy | Macro-F1 | ms/image | Parameters | Size |
|--------|-------|----------|----------|----------|------------|------|
| Pretrained transformer | `vit_b16` | 0.8829 | **0.8828** | 13.37 | 85 804 808 | 327.3 MB |
| Pretrained convolutional | `resnet50` | 0.8742 | 0.8736 | 3.26 | 23 524 424 | 89.9 MB |
| CNN from scratch | `baseline_cnn_v4` | 0.8313 | 0.8308 | — | 1 175 272 | 4.5 MB |

Per-class F1, hardest class first:

| Class | baseline_cnn_v4 | resnet50 | vit_b16 |
|-------|-----------------|----------|---------|
| form | 0.7129 | 0.7801 | **0.7907** |
| budget | 0.7700 | 0.8228 | **0.8571** |
| invoice | 0.7621 | 0.8419 | **0.8677** |
| letter | 0.8131 | **0.8762** | 0.8720 |
| memo | 0.8307 | **0.8866** | 0.8750 |
| handwritten | 0.9047 | **0.9279** | 0.9201 |
| advertisement | 0.9146 | **0.9289** | 0.9269 |
| resume | 0.9386 | 0.9241 | **0.9532** |

### What the comparison shows

**Pretraining helps where discrimination is hard, not everywhere.** On the three
easiest classes — resume, advertisement, handwritten — the three models sit
within about 0.03 of each other, and the from-scratch baseline is already
competitive. The gap opens on the confusable business documents: invoice gains
0.106 from baseline to ViT, budget 0.087, form 0.078. A small convolutional
network learns a distinctive layout from 1 400 examples; it does not learn the
finer distinctions that separate an invoice from a budget sheet.

**The two pretrained models split the per-class wins.** ResNet50 takes letter,
memo, handwritten and advertisement; ViT-B/16 takes form, budget, invoice and
resume. ViT's advantage is concentrated in exactly the classes the error tables
identify as the hardest, which is consistent with self-attention relating
distant parts of a page rather than pooling local patches.

**Accuracy is not the whole recommendation.** ViT-B/16 leads by 0.009 macro-F1
while being 4.1x slower per page and 3.6x larger. For an intake queue measured
in pages per second, ResNet50 is the better engineering choice; ViT-B/16 is the
better choice where accuracy on form, budget and invoice matters most. The
recommendation therefore depends on the deployment constraint, and both are
defensible on this evidence.

**All three overfit the training split.** Final training losses were 0.0074
(ViT), 0.0172 (ResNet50) and low for the baseline, against validation losses
near 0.58, 0.63 and higher. Validation macro-F1 nevertheless kept improving to
the final epochs in each case, so the memorisation did not prevent
generalisation at this data size. With 11 200 training pages this is expected,
and it is the clearest argument for the per-class cap being the binding
limitation rather than the architectures.

### Secondary observation, within the transformer family

`swin_t` was also trained: 27 525 506 parameters, test accuracy 0.8608, macro-F1
0.8603, 4.88 ms per page. It is **not** a fourth family — the specification
counts architecture variants within one family as a single entry — so
`vit_b16` is the transformer in the comparison and Swin-T is reported here as
supporting evidence. It is worth noting that Swin-T reaches within 0.0225 of
ViT-B/16 at a third of the parameters and a third of the inference cost.

### Which baseline run belongs in the comparison

Three runs share the baseline architecture, and the specification counts them as
one family entry:

| Run | Platform | Batch | Test macro-F1 | Role |
|-----|----------|-------|---------------|------|
| `baseline_cnn` | Machine B, CPU | 8 | 0.8274 | training-cost evidence |
| `baseline_cnn_v2` | Machine B, CPU | 8 | 0.8328 | input and augmentation ablation |
| `baseline_cnn_v4` | Machine C, T4 | 32 | 0.8308 | **the comparison entry** |

v4 is the entry because it shares its platform and batch size with the other two
families, so the comparison needs no cross-machine caveat. v1 and v2 remain in
the record as the CPU cost measurement and the controlled ablation.

## What to record beyond the table

For each run, note in prose underneath:

- Anything that differed from the config on purpose, and why.

- Anything that went wrong: a disconnect, an out-of-memory, a restart.

- The single most useful thing the run revealed, even when the run failed.

A failed run with a recorded reason is worth more than a successful run

with no notes.

