# test_models.py — every builder must emit eight logits, with no dataset present.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from docclf.config.loader import load_config
from docclf.models.registry import MODEL_BUILDERS, build_model, model_family
from docclf.models.summary import describe_model
from docclf.training.class_weights import compute_class_weights
from docclf.training.history import EpochRecord, TrainingHistory
from docclf.training.optim import build_optimiser, build_scheduler

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
N_CLASSES = 8
BATCH = 2


def config():
    """Purpose: the smoke config, which builds every model without pretrained weights."""
    return load_config(CONFIGS / "smoke.yaml")


@pytest.mark.parametrize("name", sorted(MODEL_BUILDERS))
def test_every_builder_emits_eight_logits(name: str) -> None:
    """The one shape contract the training loop relies on, for all five variants."""
    cfg = config()
    model = build_model(name, N_CLASSES, cfg)
    batch = torch.randn(BATCH, 3, cfg.image.size, cfg.image.size)
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (BATCH, N_CLASSES)
    assert torch.isfinite(logits).all()


def test_three_families_are_present() -> None:
    """The brief fixes three families; a second variant of one is not a fourth model."""
    families = {model_family(name) for name in MODEL_BUILDERS}
    assert families == {
        "CNN from scratch",
        "Pretrained convolutional",
        "Pretrained transformer",
    }


def test_unknown_model_lists_the_valid_names() -> None:
    """A typo in train.model must not fall back to some default architecture."""
    with pytest.raises(ValueError, match="baseline_cnn"):
        build_model("resnet51", N_CLASSES, config())


def test_baseline_rejects_a_wrongly_shaped_batch() -> None:
    """A single-channel batch is a transform bug and must be caught at the model."""
    model = build_model("baseline_cnn", N_CLASSES, config())
    with pytest.raises(ValueError, match=r"\(batch, 3, H, W\)"):
        model(torch.randn(BATCH, 1, 64, 64))


def test_baseline_accepts_both_the_smoke_and_the_full_input_size() -> None:
    """The same network must run on 64-pixel and 224-pixel pages unchanged."""
    model = build_model("baseline_cnn", N_CLASSES, config())
    with torch.no_grad():
        for size in (64, 224):
            assert model(torch.randn(1, 3, size, size)).shape == (1, N_CLASSES)


def test_freezing_reduces_the_trainable_parameter_count() -> None:
    """The frozen-layer decision has to be visible in the numbers, not asserted."""
    from docclf.models.pretrained_cnn import build_pretrained_cnn, freeze_backbone

    cfg = config()
    model = build_pretrained_cnn(N_CLASSES, cfg, backbone="resnet50")
    before = describe_model(model).trainable_parameters
    frozen = freeze_backbone(model, "layer2")
    after = describe_model(model).trainable_parameters
    assert frozen == ["conv1", "bn1", "layer1", "layer2"]
    assert after < before


def test_describe_model_reports_size_and_counts() -> None:
    """The deployment recommendation needs a parameter count and a size in MB."""
    summary = describe_model(build_model("baseline_cnn", N_CLASSES, config()), "baseline_cnn")
    assert summary.total_parameters > 0
    assert summary.trainable_parameters + summary.frozen_parameters == summary.total_parameters
    assert summary.size_mb > 0


def test_optimiser_and_scheduler_are_built_from_the_config() -> None:
    """A per-architecture learning rate is a config value, never a code branch."""
    cfg = config()
    model = build_model("baseline_cnn", N_CLASSES, cfg)
    optimiser = build_optimiser(model, cfg)
    assert optimiser.param_groups[0]["lr"] == pytest.approx(cfg.train.learning_rate)
    assert build_scheduler(optimiser, cfg) is not None


def test_class_weights_are_inverse_to_frequency() -> None:
    """A rare class must receive a larger weight than a common one."""
    import pandas as pd

    train_split = pd.DataFrame({"class_id": [0] * 90 + [1] * 10 + list(range(2, 8))})
    weights = compute_class_weights(train_split, N_CLASSES)
    assert weights.shape == (N_CLASSES,)
    assert weights[1] > weights[0]
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)


def test_history_records_epochs_in_order_and_finds_the_best() -> None:
    """The checkpoint and the report both refer to the best epoch by the monitor."""
    history = TrainingHistory(model_name="baseline_cnn", config_name="smoke")
    for epoch, f1 in ((1, 0.40), (2, 0.65), (3, 0.55)):
        history.append(
            EpochRecord(epoch, 1.0, 1.0, 0.5, f1, 0.001, 1.0)
        )
    assert history.best("val_macro_f1").epoch == 2
    assert history.best("val_loss").epoch == 1
    assert len(history.to_frame()) == 3
    with pytest.raises(ValueError, match="recorded in order"):
        history.append(EpochRecord(2, 1.0, 1.0, 0.5, 0.9, 0.001, 1.0))
