# configs/

Every run is defined by a YAML file in this folder. Nothing is configured by
editing source, so any result in the report traces back to one committed file
and one commit.

## The files

| File | What it defines |
|------|-----------------|
| `base.yaml` | Settings shared by every run: the eight class names, image size, split ratios, the global seed, metric settings |
| `smoke.yaml` | Inherits `base`; 25 images per class and 2 epochs, so the whole pipeline can be proven on a laptop in under two minutes |
| `baseline_cnn.yaml` | The convolutional network trained from scratch |
| `baseline_cnn_v4.yaml` | The same architecture retrained on the GPU platform, so all three families share a device and batch size |
| `resnet50.yaml` | The transfer-learning entry, including which layers are frozen |
| `vit_b16.yaml` | The transformer entry |
| `env_local.yaml`, `env_colab.yaml`, `env_hpc.yaml` | Paths, batch size, worker count and device for one machine. These are the only files that differ between environments |

## How they are read

`docclf.config.loader` reads these files and validates them against the
`RunConfig` schema. A key the schema does not recognise is rejected by name, so
a misspelled learning-rate field fails immediately instead of silently falling
back to a default.

Switching machines means changing only the `--env` argument: the same model
config runs on a laptop or a Colab GPU without edits.
