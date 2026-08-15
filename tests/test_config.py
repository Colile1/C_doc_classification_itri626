# test_config.py — a typo in a config must fail by name, and a seed must reproduce.
# Author: Colile Sibanda

from __future__ import annotations

import random
from pathlib import Path

import pytest

from docclf.config.loader import load_config
from docclf.config.paths import resolve_paths
from docclf.config.schema import CLASS_NAMES, RunConfig
from docclf.config.seed import set_global_seed

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def write(tmp_path: Path, name: str, body: str) -> Path:
    """Purpose: write a throwaway YAML config for one test. Output: its path."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_committed_configs_all_load() -> None:
    """Every config in configs/ must load, or a run is one typo from failing late."""
    for path in sorted(CONFIGS.glob("*.yaml")):
        if path.name.startswith("env_"):
            continue
        assert isinstance(load_config(path), RunConfig), path.name


def test_smoke_config_is_small_enough_for_the_laptop() -> None:
    """smoke.yaml exists to prove the pipeline in under two minutes on a CPU."""
    cfg = load_config(CONFIGS / "smoke.yaml")
    assert cfg.data.n_per_class <= 25
    assert cfg.train.epochs <= 2
    assert cfg.seed == 626


def test_unknown_top_level_key_is_rejected_by_name(tmp_path: Path) -> None:
    """A misspelled key must name itself in the error, not be silently ignored."""
    path = write(tmp_path, "bad.yaml", "seed: 1\nlerning_rate: 0.1\n")
    with pytest.raises(ValueError, match="lerning_rate"):
        load_config(path)


def test_unknown_nested_key_is_rejected_by_name(tmp_path: Path) -> None:
    """The same must hold inside a section, where a typo is easier to miss."""
    path = write(tmp_path, "bad.yaml", "train:\n  learning_rat: 0.1\n")
    with pytest.raises(ValueError, match="learning_rat"):
        load_config(path)


def test_ratios_that_do_not_sum_to_one_are_rejected(tmp_path: Path) -> None:
    """Split ratios are the one place a silent rescale would be invisible."""
    path = write(tmp_path, "bad.yaml", "split:\n  ratios: [0.8, 0.15, 0.15]\n")
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(path)


def test_class_set_cannot_be_changed(tmp_path: Path) -> None:
    """The eight classes are fixed by the brief; a ninth is a design change."""
    body = "classes:\n" + "".join(f"  - {name}\n" for name in (*CLASS_NAMES, "cheque"))
    with pytest.raises(ValueError, match="classes"):
        load_config(write(tmp_path, "bad.yaml", body))


def test_inheritance_merges_and_overrides(tmp_path: Path) -> None:
    """A child config overrides the parent key by key, not section by section."""
    write(tmp_path, "parent.yaml", "seed: 1\ntrain:\n  epochs: 20\n  batch_size: 32\n")
    child = write(tmp_path, "child.yaml", "extends: parent.yaml\ntrain:\n  epochs: 2\n")
    cfg = load_config(child)
    assert cfg.train.epochs == 2
    assert cfg.train.batch_size == 32  # inherited, not reset to the schema default


def test_environment_overlay_changes_only_machine_settings() -> None:
    """Swapping env_*.yaml must move the same run to another machine."""
    plain = load_config(CONFIGS / "smoke.yaml")
    colab = load_config(CONFIGS / "smoke.yaml", env="colab")
    assert colab.train.device == "cuda"
    assert plain.seed == colab.seed
    assert plain.classes == colab.classes
    assert plain.split.ratios == colab.split.ratios


def test_unknown_environment_is_rejected() -> None:
    """A mistyped environment must not silently fall back to the laptop."""
    with pytest.raises(ValueError, match="Unknown environment"):
        load_config(CONFIGS / "smoke.yaml", env="labtop")


def test_missing_config_file_names_the_path(tmp_path: Path) -> None:
    """The error has to say which file was not found."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config(tmp_path / "nowhere.yaml")


def test_resolve_paths_are_absolute_and_under_the_root() -> None:
    """Every logical directory resolves to an absolute path for this machine."""
    cfg = load_config(CONFIGS / "smoke.yaml")
    paths = resolve_paths(cfg, "local")
    assert paths.root.is_absolute()
    for directory in (paths.raw, paths.splits, paths.metrics, paths.checkpoints):
        assert directory.is_absolute()
        assert paths.root in directory.parents


def test_same_seed_gives_the_same_random_draws() -> None:
    """One seed drives every source of randomness; two calls must agree."""
    set_global_seed(626)
    first = [random.random() for _ in range(5)]
    set_global_seed(626)
    second = [random.random() for _ in range(5)]
    assert first == second
    set_global_seed(627)
    assert [random.random() for _ in range(5)] != first


def test_seed_covers_numpy_and_torch() -> None:
    """The seed must reach the libraries that actually draw during training."""
    import numpy as np
    import torch

    set_global_seed(626)
    first = (np.random.rand(3).tolist(), torch.rand(3).tolist())
    set_global_seed(626)
    assert (np.random.rand(3).tolist(), torch.rand(3).tolist()) == first


def test_seed_must_be_an_integer() -> None:
    """A string seed would silently seed nothing useful."""
    with pytest.raises(TypeError, match="integer"):
        set_global_seed("626")  # type: ignore[arg-type]
