# retry.py — ride out a drive that drops out briefly, instead of misreading it as corruption.
# Author: Colile Sibanda

from __future__ import annotations

import errno
import time
from pathlib import Path
from typing import Callable, TypeVar

from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)
T = TypeVar("T")

# Windows error codes raised when a volume disappears or an I/O fails in flight:
# 21 not ready, 55 and 59 network/device fault, 121 semaphore timeout,
# 433 device does not exist, 1117 I/O device error, 1167 device not connected.
_TRANSIENT_WINERRORS = {21, 55, 59, 121, 433, 1117, 1167}
_TRANSIENT_ERRNOS = {errno.EIO, errno.ENODEV}


def is_transient_io_error(error: BaseException, path: Path | str | None = None) -> bool:
    """Purpose: tell a storage fault from a genuinely bad file.

    Inputs: an exception; optionally the file being read. Output: True for
    drive-level faults (device gone, I/O error), False for anything about the
    file itself such as an unreadable image. When the path is given, any OSError
    raised while the file's own folder cannot be seen also counts: a removable
    drive that drops out surfaces as "path not found" or "invalid argument",
    which says nothing about the file.
    """
    if not isinstance(error, OSError):
        return False
    if getattr(error, "winerror", None) in _TRANSIENT_WINERRORS:
        return True
    if error.errno in _TRANSIENT_ERRNOS:
        return True
    return path is not None and not _folder_visible(Path(path))


def _folder_visible(path: Path) -> bool:
    """True when the folder holding `path` can be listed right now."""
    try:
        return path.parent.is_dir()
    except OSError:
        return False


def retry_io(
    operation: Callable[[], T],
    attempts: int = 6,
    wait_seconds: float = 10.0,
    path: Path | str | None = None,
) -> T:
    """Purpose: run a file operation, retrying while the storage is unavailable.

    Inputs: a no-argument callable; how many tries; seconds to wait between them;
    optionally the file the operation reads, used to recognise a vanished drive.
    Output: the callable's result. A fault that is not a storage fault is raised
    at once, and a storage fault that outlasts every attempt is raised with the
    original error, so a long run stops loudly rather than recording good files
    as corrupt.
    """
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OSError as error:
            if not is_transient_io_error(error, path) or attempt == attempts:
                raise
            _LOGGER.warning(
                "Storage fault (%s). Retry %d of %d in %.0fs.",
                error, attempt, attempts - 1, wait_seconds,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")
