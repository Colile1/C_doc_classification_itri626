# test_no_test_leakage.py — the training module must never see the held-out split.
# Author: Colile Sibanda
#
# Training that peeks at the held-out set produces a number that cannot be
# defended, and nothing in the output would show it. This suite reads the source
# of the training package and proves the loader is not reachable from it.

from __future__ import annotations

import re
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1] / "src" / "docclf" / "training"
EVALUATION = Path(__file__).resolve().parents[1] / "src" / "docclf" / "evaluation"

HELD_OUT = "test"


def source_files(directory: Path) -> list[Path]:
    """Purpose: the Python files of one package. Output: the paths, sorted."""
    return sorted(path for path in directory.rglob("*.py"))


def code_lines(path: Path) -> list[tuple[int, str]]:
    """Purpose: the source lines with comments and docstring text excluded.

    Input: a file path. Output: `(line number, code)` for lines that are code.
    """
    lines: list[tuple[int, str]] = []
    in_docstring = False
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        fences = stripped.count('"""')
        if in_docstring:
            if fences:
                in_docstring = False
            continue
        if stripped.startswith('"""') and fences == 1:
            in_docstring = True
            continue
        if stripped.startswith("#") or not stripped:
            continue
        lines.append((number, stripped.split("#")[0]))
    return lines


def test_training_never_indexes_a_held_out_loader() -> None:
    """No `loaders["test"]` or equivalent subscript may exist in the training loop."""
    pattern = re.compile(rf"""\[\s*['"]{HELD_OUT}['"]\s*\]""")
    offenders = [
        f"{path.name}:{number}"
        for path in source_files(TRAINING)
        for number, line in code_lines(path)
        if pattern.search(line)
    ]
    assert not offenders, f"the held-out loader is indexed in: {', '.join(offenders)}"


def test_training_declares_only_the_two_loaders_it_uses() -> None:
    """The trainer reads `train` and `val`; the keys it accepts are stated once."""
    trainer = (TRAINING / "trainer.py").read_text(encoding="utf-8")
    assert 'TRAIN_KEY = "train"' in trainer
    assert 'VAL_KEY = "val"' in trainer
    assert 'HELD' not in trainer.upper().replace("HELD-OUT", "")


def test_training_does_not_import_the_evaluation_package() -> None:
    """The dependency direction is models -> training -> evaluation, never back."""
    offenders = [
        path.name
        for path in source_files(TRAINING)
        if re.search(r"^\s*(from|import)\s+docclf\.evaluation", path.read_text(encoding="utf-8"), re.M)
    ]
    assert not offenders, f"training imports evaluation in: {', '.join(offenders)}"


def test_evaluation_does_not_import_the_training_package() -> None:
    """Stated in the evaluation README: it never imports training."""
    offenders = [
        path.name
        for path in source_files(EVALUATION)
        if re.search(r"^\s*(from|import)\s+docclf\.training", path.read_text(encoding="utf-8"), re.M)
    ]
    assert not offenders, f"evaluation imports training in: {', '.join(offenders)}"


def test_splits_never_imports_models_or_training() -> None:
    """The split is fixed once and is not permitted to vary per architecture."""
    splits = Path(__file__).resolve().parents[1] / "src" / "docclf" / "splits"
    offenders = [
        path.name
        for path in source_files(splits)
        if re.search(r"^\s*(from|import)\s+docclf\.(models|training)", path.read_text(encoding="utf-8"), re.M)
    ]
    assert not offenders, f"splits imports a model package in: {', '.join(offenders)}"


def test_config_never_imports_the_heavy_packages() -> None:
    """A config load must work where torch weights are unavailable."""
    config = Path(__file__).resolve().parents[1] / "src" / "docclf" / "config"
    offenders = [
        path.name
        for path in source_files(config)
        if re.search(r"^\s*(from|import)\s+docclf\.(data|models|training|evaluation)", path.read_text(encoding="utf-8"), re.M)
    ]
    assert not offenders, f"config imports a downstream package in: {', '.join(offenders)}"


def test_the_package_has_no_import_cycles() -> None:
    """Stated in src/docclf/README.md as a done condition, so it is checked here."""
    package = Path(__file__).resolve().parents[1] / "src" / "docclf"
    edges: dict[str, set[str]] = {}
    for path in sorted(package.rglob("*.py")):
        module = path.parent.name if path.name == "__init__.py" else path.parent.name
        found = set(
            re.findall(r"^\s*from\s+docclf\.(\w+)", path.read_text(encoding="utf-8"), re.M)
        )
        edges.setdefault(module, set()).update(found - {module})
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str, trail: list[str]) -> None:
        """Purpose: depth-first search that raises on a back edge."""
        if node in done:
            return
        if node in visiting:
            raise AssertionError(f"Import cycle: {' -> '.join([*trail, node])}")
        visiting.add(node)
        for neighbour in sorted(edges.get(node, set())):
            visit(neighbour, [*trail, node])
        visiting.discard(node)
        done.add(node)

    for module in sorted(edges):
        visit(module, [])
