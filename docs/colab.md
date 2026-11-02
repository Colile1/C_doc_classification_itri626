# Training on Google Colab

A free Colab T4 trains the heavy models in a fraction of the CPU time: ViT-B/16
measured roughly 181 minutes an epoch on the laptop, against an expected one to
three minutes on a T4. That turns a 45-hour run into one or two hours.

The project already supports it. `configs/env_colab.yaml` selects CUDA, AMP and
a batch size the GPU can hold, and `utils/env.py` detects Colab automatically,
so the same commands work with `--env colab`.

## What has to get there

| | Size | How |
|---|------|-----|
| The code | a few MB | `git clone` into Drive, or upload a zip |
| The split's images | ~2.8 GB | `make_colab_bundle.py`, then upload once to Drive |
| The split files | 1.8 MB | they come with the code |

`data/raw/` holds 200 000 images (~19 GB), but a run only ever reads the 16 000
the committed split names. Bundle only those.

## 1 · Build the image bundle, on your own machine

```bash
python make_colab_bundle.py
```

Writes `data/interim/colab_images.tar.gz`, about 2.8 GB, containing exactly the
images the split refers to, under the `data/raw/<class>/` layout the split files
expect. It refuses to build if any image the split names is missing, rather than
producing a bundle that fails mid-epoch.

Upload that file, once, to `MyDrive/itri626/`.

## 2 · Put the project on Drive

Everything a run writes — checkpoints, metrics, figures, logs — goes to the
project root, and on Colab that root is on Drive. A disconnect then costs the
current epoch, not the run.

In a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive
!git clone <your repo url> itri626-doc-classification
```

If the repository is private, use a personal access token in the URL, or upload
a zip of the project to Drive and unzip it there instead. The path must end up
as `/content/drive/MyDrive/itri626-doc-classification`, which is what
`env_colab.yaml` points `root` at.

## 3 · Set up the runtime

**Runtime → Change runtime type → T4 GPU** before running anything.

```bash
%cd /content/drive/MyDrive/itri626-doc-classification
!pip install -q timm imagehash
```

Colab already has torch, torchvision, pandas, scikit-learn, matplotlib and
pillow. Only `timm` (which provides ViT and Swin) and `imagehash` are missing.
Do not install from `requirements.lock.txt`: it pins the CPU build of torch and
would replace Colab's CUDA build.

## 4 · Unpack the images to local disk

```bash
!tar -xzf /content/drive/MyDrive/itri626/colab_images.tar.gz -C /content
```

This gives `/content/data/raw/<class>/...`, which is where `env_colab.yaml`
points `raw`. Keep the images on local disk, not Drive: reading 16 000 images
per epoch over the Drive mount is far slower than the GPU it feeds.

Takes a few minutes, and has to be repeated each time the runtime restarts.
The project and its outputs stay on Drive, so only this step repeats.

## 5 · Check, then train

```bash
!python run_check.py --config configs/vit_b16.yaml
!python run_status.py
```

`run_check.py` should report CUDA as the device and every split image readable.
If it says CPU, the runtime type was not changed to T4.

```bash
!python run_claim.py --model vit_b16 --note "colab T4"
!python run_train.py    --config configs/vit_b16.yaml --env colab --model vit_b16
!python run_evaluate.py --config configs/vit_b16.yaml --env colab --model vit_b16
!python run_claim.py --model vit_b16 --release
```

`--env colab` is what selects the GPU, AMP and the larger batch size. The
detection would choose it anyway, but stating it makes the run reproducible and
puts it in the log.

## 6 · Bring the results back

Everything is already on Drive. To get it into git, from the Colab cell:

```bash
!git add runs/vit_b16/metrics runs/vit_b16/figures runs/vit_b16/logs
!git -c user.email="<your no-reply address>" -c user.name="<your name>" \
    commit -m "vit_b16 trained on Colab T4"
!git push
```

Or download the `runs/vit_b16/` folder from Drive and commit it from your own
machine.

## When the runtime disconnects

Colab reclaims free runtimes after a period of inactivity, and caps sessions at
around twelve hours. A checkpoint is written every epoch to Drive, carrying the
optimiser, scheduler and early-stopping state, so a disconnect costs the epoch
in progress and nothing more.

To continue, repeat steps 3 and 4 — the runtime is new, so the packages and the
images have to be put back — then:

```bash
!python run_train.py --config configs/vit_b16.yaml --env colab \
    --model vit_b16 --resume
```

It reports the epoch it is resuming from. The learning-rate schedule continues
rather than restarting.

## Batch size and the comparison

`env_colab.yaml` uses batch size 32 where `env_local.yaml` uses 8, because the
T4 has the memory for it and a larger batch is substantially faster.

That is an honest difference to record, not a problem: the split, the seed, the
architecture and the number of epochs are unchanged, and the test set is the
same 2 400 pages for every model. State the batch size per run in the report's
experimental-setup table, as `docs/experiment-log.md` already does, and note
which runs were on a GPU.

If you would rather keep every hyperparameter identical across all models, pass
`--env local` on Colab instead: it holds batch size at 8 but still uses the GPU
if `train.device` is overridden. The simpler and more defensible choice is to
use the Colab settings and report them.

## Which models are worth running there

| Model | Laptop CPU | Colab T4 (expected) |
|-------|-----------|---------------------|
| `baseline_cnn_v4` | ~54 min/epoch | a few minutes |
| `resnet50` | ~47 min/epoch | a few minutes |
| `vit_b16` | ~181 min/epoch | one to three minutes |

All three models in the comparison were trained on a Colab Tesla T4 at batch
size 32, so they share one device, one batch size and one precision setting.

Claim the model before starting, exactly as on any other machine, so Colab and
your laptop cannot train the same one. `run_status.py` on either machine shows
what the other is doing once the claims are committed and pulled.
