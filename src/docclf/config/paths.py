# paths.py — map the logical directory names onto real directories per machine.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from docclf.config.schema import RunConfig

VALID_ENVIRONMENTS = ("local", "colab", "hpc")


@dataclass(frozen=True)
class ProjectPaths:
    """Absolute directories for one run on one machine."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    splits: Path
    checkpoints: Path
    figures: Path
    metrics: Path
    logs: Path

    def ensure(self) -> "ProjectPaths":
        """Purpose: create every directory that a run writes into.

        Input: none. Output: self, so the call can be chained by an orchestrator.
        """
        for item in fields(self):
            if item.name == "root":
                continue
            getattr(self, item.name).mkdir(parents=True, exist_ok=True)
        return self


def resolve_paths(cfg: RunConfig, env: str) -> ProjectPaths:
    """Purpose: turn the configured directory names into absolute paths.

    Inputs: a RunConfig and an environment name (`local`, `colab` or `hpc`).
    Output: a ProjectPaths. Raises ValueError on an unknown environment name.
    """
    if env not in VALID_ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment: {env!r}. Valid values are "
            f"{', '.join(VALID_ENVIRONMENTS)}."
        )
    root = Path(cfg.paths.root).expanduser()
    if not root.is_absolute():
        root = (project_root() / root).resolve()
    resolved: dict[str, Path] = {"root": root}
    for item in fields(ProjectPaths):
        if item.name == "root":
            continue
        configured = Path(getattr(cfg.paths, item.name)).expanduser()
        resolved[item.name] = (
            configured if configured.is_absolute() else root / configured
        )
    return ProjectPaths(**resolved)


def project_root() -> Path:
    """Purpose: locate the repository root from this file's position.

    Input: none. Output: the directory holding `src/`, `configs/` and `tests/`.
    """
    return Path(__file__).resolve().parents[3]
