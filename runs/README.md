# runs/

One directory per model in the three-family comparison, so no run can
overwrite another's results.

| Directory | Model | Family | Held-out macro-F1 |
|-----------|-------|--------|-------------------|
| `baseline_cnn_v4/` | Convolutional network trained from scratch | CNN from scratch | 0.8308 |
| `resnet50/` | ResNet50 with ImageNet weights, fine-tuned | Pretrained convolutional | 0.8736 |
| `vit_b16/` | ViT-B/16 with ImageNet weights, fine-tuned | Pretrained transformer | 0.8828 |

```
runs/<model>/
  metrics/      per-epoch history, held-out metrics, per-class scores,
                predictions, misclassified pages, inference cost
  figures/      confusion matrices (counts and normalised), learning curves,
                misclassified page examples
  logs/         the full training and evaluation log
  checkpoints/  weights, never committed (90 MB to 330 MB each)
```

All three were trained on one NVIDIA Tesla T4 at batch size 32, on the same
11 200 training pages, with the checkpoint chosen on the same 2 400 validation
pages and scored once on the same 2 400 held-out pages.

## Checkpoints

The weights are not in git. A finished run needs none: its metrics and figures
are committed, and that is everything the comparison, the report and the
dashboard's results pages read. The dashboard's live classifier needs one
`<model>_best.pt` copied into `runs/<model>/checkpoints/`.

To retrain a model, `./train_model.sh <model>` writes every epoch's checkpoint
here and resumes from the last one if interrupted.

## Training across several machines

`run_claim.py` records which machine is training which model in
`runs/claims/<model>.json`, with a heartbeat; a claim older than 90 minutes is
treated as stale. Claims exist only while models are being trained, so the
finished project has none. `./train_model.sh` claims and releases
automatically, Ctrl-C included.

```bash
python run_claim.py --model resnet50            # claim or refresh
python run_claim.py --model resnet50 --release  # give it up
python run_claim.py --model resnet50 --force    # take over a stale claim
```

The full procedure is in [`../docs/multi-machine.md`](../docs/multi-machine.md).
