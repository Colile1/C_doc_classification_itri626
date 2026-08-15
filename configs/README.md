# configs/

Every run is defined by a YAML file here. Nothing is configured by
editing source, so a result in the report can always be traced back to
one committed file plus one commit hash.

## Planned files

| File | Responsibility |
|------|----------------|
| `base.yaml` | Everything shared by all runs: the eight class names, image size, split ratios, the global seed, metric settings |
| `smoke.yaml` | Inherits `base`; 25 images per class, 2 epochs. Exists so the whole pipeline can be proven on the laptop in under two minutes |
| `baseline_cnn.yaml` | The from-scratch convolutional baseline |
| `resnet50.yaml` | The transfer-learning entry, including which layers are frozen |
| `vit_b16.yaml` | The transformer entry |
| `env_local.yaml` / `env_colab.yaml` / `env_hpc.yaml` | Paths, batch size, worker count and device per machine. The only files that differ between environments |

## Integration

Read by `docclf.config.loader`. A key not present in the `RunConfig`
schema is rejected, so a typo in a learning rate name fails loudly
instead of silently using the default.

Done when: `smoke.yaml` runs end-to-end on the laptop, and swapping only
the `env_*.yaml` file moves the identical run to Colab.
