# timing.py — wall-clock timing of the stages a run log has to report.
# Author: Colile Sibanda

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)


class Duration:
    """A mutable holder so a caller can read the elapsed time after the block."""

    def __init__(self) -> None:
        self.seconds: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"Duration(seconds={self.seconds:.3f})"


@contextmanager
def timed(label: str) -> Iterator[Duration]:
    """Purpose: record how long a stage took and log it in the project format.

    Input: a human-readable label for the stage.
    Output: a context manager yielding a Duration whose `seconds` is filled in
    when the block exits, including when the block raises.
    """
    duration = Duration()
    started = time.perf_counter()
    try:
        yield duration
    finally:
        duration.seconds = time.perf_counter() - started
        _LOGGER.info("%s duration seconds: %.3f", label, duration.seconds)
