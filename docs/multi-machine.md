# Running on several machines

Four models have to be trained, and no two machines may train the same one.
This file is the procedure. It assumes the repository is cloned on each machine
and that `data/raw/` holds the eight class folders.

## The rule

**One model per machine at a time. Claim it before you start.**

Every model writes only into its own directory, so two machines can train two
different models at the same time without touching each other's files:

| Model | Family | Everything it writes |
|-------|--------|----------------------|
| `baseline_cnn_v4` | CNN from scratch | `runs/baseline_cnn_v4/` |
| `resnet50` | Pretrained convolutional | `runs/resnet50/` |
| `vit_b16` | Pretrained transformer | `runs/vit_b16/` |

Checkpoints, epoch history, logs, figures and metrics all sit under that one
directory, named after the model. Nothing is shared except `data/splits/`,
which is read-only during training.

## Every session, in order

```bash
git pull                      # pick up other machines' claims and results
python run_status.py          # what is done, running, paused, or free
```

`run_status.py` prints the state of all four models and the exact next command
for this machine. It is read-only — it never changes a run.

States it reports:

| State | Meaning |
|-------|---------|
| `NOT STARTED` | No checkpoint. Free to claim. |
| `RUNNING` | Claimed within the last 90 minutes. Leave it alone. |
| `PAUSED` | Has a checkpoint, no live claim. Resume it, here or anywhere. |
| `STALE CLAIM` | Claimed over 90 minutes ago with no heartbeat. The machine probably died; take it over with `--force`. |
| `NEEDS EVAL` | All epochs done, held-out metrics missing. Run `run_evaluate.py`. |
| `COMPLETE` | Trained and evaluated. |

## Train one model

```bash
./train_model.sh resnet50
```

That one command claims the model, refreshes the claim every 30 minutes,
resumes automatically if a checkpoint exists, evaluates on the held-out split
when training finishes, releases the claim, and prints the new status. Ctrl-C
is safe: the claim is released and the next run continues from the last
completed epoch.

Doing it by hand is the same three steps:

```bash
python run_claim.py --model resnet50 --note "overnight"
python run_train.py --config configs/resnet50.yaml --env local --model resnet50
python run_evaluate.py --config configs/resnet50.yaml --env local --model resnet50
python run_claim.py --model resnet50 --release
```

## Pause here, finish somewhere else

A checkpoint is written **every epoch**, carrying the model weights, the
optimiser state, the scheduler position, the gradient scaler, the
early-stopping counter and the epoch history. Resuming therefore continues the
learning-rate schedule instead of restarting it.

On the machine that is stopping:

```bash
# Ctrl-C is enough. Then:
python run_claim.py --model resnet50 --release
git add runs/resnet50 && git commit -m "resnet50 through epoch 7" && git push
```

On the machine picking it up:

```bash
git pull
python run_claim.py --model resnet50
python run_train.py --config configs/resnet50.yaml --env local --model resnet50 --resume
```

`--resume` refuses to run if the checkpoint belongs to a different model, so a
wrong `--model` cannot quietly corrupt a result.

**The checkpoint files are large.** `runs/*/checkpoints/` is gitignored by
default. To move a paused run between machines, either commit that one
checkpoint deliberately (`git add -f runs/resnet50/checkpoints/resnet50_last.pt`,
~90 MB for ResNet50, ~330 MB for ViT) or copy it across on a USB drive. The
small files — history, metrics, logs, claims — are committed normally and are
what `run_status.py` reads.

## Splitting the work across machines

Two free models, two machines, no coordination needed beyond the claim:

```bash
# Machine 1
./train_model.sh resnet50

# Machine 2
./train_model.sh vit_b16
```

Both write into separate directories. When both finish, one `git pull` on
either machine gives you every model's metrics for the comparison.

Pick the faster machine for `vit_b16`: it is 85.8 M parameters against
ResNet50's 23.5 M, and measured roughly 3.9x slower per epoch on the same CPU.

## Measured epoch cost, batch size 8, CPU only

| Model | Params | Machine A (laptop, 12 cores) | Machine B (desktop, 24 cores) |
|-------|--------|------------------------------|-------------------------------|
| `baseline_cnn_v4` | 1.18 M | ~54 min/epoch | ~5.3 min/epoch |
| `resnet50` | 23.5 M (22.1 M trainable) | ~47 min/epoch | ~5 min/epoch (estimated) |
| `vit_b16` | 85.8 M | ~181 min/epoch | ~20 min/epoch (estimated) |

Machine A figures and the baseline's Machine B figure were measured. The other
Machine B figures are the laptop number divided by the measured 9.05x ratio.
These CPU costs are why all three final runs were trained on a Colab Tesla T4
instead (see [`colab.md`](colab.md)).

## Before you commit

```bash
pytest tests/ -q              # must be 122 passed
python run_status.py          # confirm the state you expect
```

Never commit a regenerated `data/splits/`. The committed split is the one all
recorded results were measured on; replacing it makes every earlier number
incomparable. `run_split.py` is only for rebuilding from scratch, and
`train_model.sh` refuses to start when the split files are missing rather than
silently regenerating them.
