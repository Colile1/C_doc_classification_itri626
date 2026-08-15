# logging.py — the one logger the package uses; no module here ever prints.
# Author: Colile Sibanda

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Mapping

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Purpose: return the package logger, configured once for the process.

    Input: the module name, conventionally `__name__`.
    Output: a `logging.Logger` writing to stderr in the project format.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root = logging.getLogger("docclf")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    if name == "docclf" or name.startswith("docclf."):
        return logging.getLogger(name)
    return logging.getLogger(f"docclf.{name}")


def add_file_handler(path: Path) -> None:
    """Purpose: mirror the log of a run into `outputs/logs/`.

    Input: the destination file path; its parent is created if missing.
    Output: None.
    """
    get_logger("docclf")
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.getLogger("docclf").addHandler(handler)


def log_values(logger: logging.Logger, values: Mapping[str, Any]) -> None:
    """Purpose: emit a group of results in the project's `Description: value` form.

    Inputs: a logger and a mapping of description to value.
    Output: None; one log line per entry.
    """
    for description, value in values.items():
        logger.info("%s: %s", description, value)
