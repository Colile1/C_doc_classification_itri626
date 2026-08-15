# __init__.py — public surface of the configuration module.
# Author: Colile Sibanda

from docclf.config.loader import load_config
from docclf.config.paths import ProjectPaths, resolve_paths
from docclf.config.schema import (
    DataConfig,
    EvalConfig,
    ImageConfig,
    PathsConfig,
    RunConfig,
    SplitConfig,
    TrainConfig,
)
from docclf.config.seed import set_global_seed

__all__ = [
    "DataConfig",
    "EvalConfig",
    "ImageConfig",
    "PathsConfig",
    "ProjectPaths",
    "RunConfig",
    "SplitConfig",
    "TrainConfig",
    "load_config",
    "resolve_paths",
    "set_global_seed",
]
