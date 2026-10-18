# test_pipeline.py — the four orchestrators must run end to end from a clean state.
# Author: Colile Sibanda
#
# End-to-end check. The real dataset is not available on the laptop
# and must never be required by the suite, so this runs the identical code path
# over a small synthetic dataset written into a temporary directory.

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageDraw

from docclf.config.loader import load_config
from docclf.config.paths import resolve_paths
from docclf.config.seed import set_global_seed

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CLASSES = (
    "invoice",
    "budget",
    "form",
    "letter",
    "memo",
    "handwritten",
    "resume",
    "advertisement",
)


def load_orchestrator(name: str):
    """Purpose: import a root `run_*.py` as a module, the way a test can call it.

    Input: the module name without its suffix. Output: the imported module.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def draw_page(path: Path, class_index: int, document: int, page: int) -> None:
    """Purpose: write one synthetic page whose look depends on its class.

    Inputs: the destination, the class index, the document and page numbers.
    Output: None. Each class gets a different layout so the models have a real,
    if easy, signal to learn, and each page carries its own scattered strokes so
    that the near-duplicate cleaner does not treat them as copies of each other.
    """
    import random

    image = Image.new("L", (160, 200), color=255)
    draw = ImageDraw.Draw(image)
    offset = 3 * document + page
    rng = random.Random(f"{class_index}:{document}:{page}")
    for _ in range(40):
        x, y = rng.randint(10, 150), rng.randint(10, 190)
        draw.rectangle((x, y, x + rng.randint(2, 9), y + rng.randint(2, 6)), fill=0)
    if class_index % 4 == 0:  # dense ruled table
        for y in range(20 + offset, 190, 12):
            draw.line((15, y, 145, y), fill=0, width=2)
    elif class_index % 4 == 1:  # two columns of text-like blocks
        for y in range(25 + offset, 180, 16):
            draw.rectangle((20, y, 70, y + 6), fill=0)
            draw.rectangle((90, y, 140, y + 6), fill=0)
    elif class_index % 4 == 2:  # a heading and sparse lines
        draw.rectangle((20, 20 + offset, 120, 40 + offset), fill=0)
        for y in range(60, 180, 30):
            draw.line((20, y, 120, y), fill=0, width=3)
    else:  # a large figure block
        draw.ellipse((30, 40 + offset, 130, 140 + offset), outline=0, width=4)
    if class_index >= 4:  # a second family of layouts, so all eight differ
        draw.rectangle((10, 10, 150, 190), outline=0, width=2)
    image.save(path)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Purpose: a synthetic eight-class dataset in the `raw` directory layout."""
    raw = tmp_path / "data" / "raw"
    for class_index, class_name in enumerate(CLASSES):
        class_dir = raw / class_name
        class_dir.mkdir(parents=True)
        for document in range(4):
            for page in range(2):
                draw_page(
                    class_dir / f"doc{document}_p{page}.png", class_index, document, page
                )
    return tmp_path


@pytest.fixture
def context(dataset: Path):
    """Purpose: a smoke config and paths rooted in the temporary dataset."""
    cfg = load_config(CONFIGS / "smoke.yaml", env="local")
    cfg = dataclasses.replace(
        cfg, paths=dataclasses.replace(cfg.paths, root=str(dataset), raw="data/raw")
    )
    set_global_seed(cfg.seed)
    return cfg, resolve_paths(cfg, "local").ensure()


def test_whole_pipeline_runs_end_to_end(context) -> None:
    """Prepare, split, train and evaluate, in order, with no logic in any main."""
    cfg, paths = context

    working_path = load_orchestrator("run_prepare_data").prepare_data(cfg, paths)
    working = pd.read_csv(working_path)
    assert len(working) == 8 * 8
    assert (paths.splits / "cleaning_report.csv").is_file()

    summary_path = load_orchestrator("run_split").build_split(cfg, paths)
    summary = pd.read_csv(summary_path)
    assert int(summary["count"].sum()) == len(working)
    for name in ("train", "val", "test"):
        assert (paths.splits / f"{name}.csv").is_file()

    history = load_orchestrator("run_train").train_model(cfg, paths)
    assert len(history.records) == cfg.train.epochs
    assert (paths.checkpoints / f"{cfg.train.model}_best.pt").is_file()
    assert (paths.metrics / f"history_{cfg.train.model}.csv").is_file()

    comparison = load_orchestrator("run_evaluate").evaluate_model(cfg, paths)
    assert len(comparison) == 1
    assert comparison.iloc[0]["model"] == cfg.train.model
    assert 0.0 <= comparison.iloc[0]["macro_f1"] <= 1.0
    assert (paths.metrics / "comparison.csv").is_file()
    assert (paths.figures / f"confusion_{cfg.train.model}.png").is_file()
    assert (paths.figures / f"curves_{cfg.train.model}.png").is_file()


def test_recapping_reuses_the_cleaned_manifest(context) -> None:
    """A new per-class cap must not re-read the images, and must cap the same pages."""
    cfg, paths = context
    prepare = load_orchestrator("run_prepare_data").prepare_data
    first = pd.read_csv(prepare(cfg, paths))
    smaller = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, n_per_class=4))
    for image in (paths.raw).rglob("*.png"):
        image.unlink()
    recapped = pd.read_csv(prepare(smaller, paths, reuse_cleaned=True))
    assert len(recapped) == 8 * 4
    assert set(recapped["path"]) <= set(first["path"])


def test_evaluation_uses_only_the_held_out_split(context) -> None:
    """The metric row must count exactly the pages in the held-out split file."""
    cfg, paths = context
    load_orchestrator("run_prepare_data").prepare_data(cfg, paths)
    load_orchestrator("run_split").build_split(cfg, paths)
    load_orchestrator("run_train").train_model(cfg, paths)
    comparison = load_orchestrator("run_evaluate").evaluate_model(cfg, paths)
    held_out = pd.read_csv(paths.splits / "test.csv")
    assert int(comparison.iloc[0]["n_samples"]) == len(held_out)


def test_split_is_reproducible_across_two_runs(context) -> None:
    """Anyone with the seed must recreate the identical split files."""
    cfg, paths = context
    load_orchestrator("run_prepare_data").prepare_data(cfg, paths)
    split_module = load_orchestrator("run_split")
    split_module.build_split(cfg, paths)
    first = (paths.splits / "train.csv").read_text(encoding="utf-8")
    set_global_seed(cfg.seed)
    split_module.build_split(cfg, paths)
    assert (paths.splits / "train.csv").read_text(encoding="utf-8") == first


def test_evaluating_without_a_checkpoint_is_actionable(context) -> None:
    """The error must say which command to run, not raise a bare file error."""
    cfg, paths = context
    load_orchestrator("run_prepare_data").prepare_data(cfg, paths)
    load_orchestrator("run_split").build_split(cfg, paths)
    with pytest.raises(FileNotFoundError, match="run_train.py"):
        load_orchestrator("run_evaluate").evaluate_model(cfg, paths)
