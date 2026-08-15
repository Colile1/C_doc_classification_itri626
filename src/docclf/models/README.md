# src/docclf/models/

Builds the three architectures behind one interface, so the training loop
never contains an `if model_name ==` branch. Each builder returns a
module whose final layer already matches the project's class count.

## Planned files

| File | Responsibility |
|------|----------------|
| `registry.py` | `build_model(name: str, n_classes: int, cfg) -> nn.Module` — the only public entry point; an unknown name raises listing the valid ones |
| `baseline_cnn.py` | `build_baseline_cnn(n_classes, cfg)` — a small convolutional network trained from scratch; deliberately simple, and the bar the other two must clear |
| `pretrained_cnn.py` | `build_pretrained_cnn(n_classes, cfg)` — ResNet50 or MobileNetV3 with ImageNet weights; `freeze_backbone(model, upto)` records which layers are frozen so the report can justify the choice |
| `transformer.py` | `build_transformer(n_classes, cfg)` — ViT-B/16 or Swin-T with pretrained weights, via `timm` |
| `summary.py` | `describe_model(model) -> ModelSummary` — parameter count and on-disk size, needed for the deployment recommendation |

## Integration

Consumes `config` only. Knows nothing about datasets, splits or
training. Used by `training` and by `evaluation` (for the efficiency
comparison).

Done when: all three builders return a module that accepts a
`(batch, 3, H, W)` tensor and emits `(batch, n_classes)` logits, verified
by a shape test that needs no data.

Built at step 8. The registry holds five names across the three fixed
families: `baseline_cnn`; `resnet50` and `mobilenetv3` (one is chosen);
`vit_b16` and `swin_t` (one is chosen). A second variant of a family
already present is not a fourth model, and `model_family()` states which
family each name belongs to for the report. Every builder is exercised by
the shape test with `train.pretrained: false`, so the suite needs no
weight download and no network.
