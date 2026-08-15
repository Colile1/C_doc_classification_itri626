# acquire.py — locate the dataset archive, verify it, and extract it once.
# Author: Colile Sibanda

from __future__ import annotations

import hashlib
import tarfile
import zipfile
from pathlib import Path

from docclf.config.schema import RunConfig
from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)
_CHUNK = 1 << 20


def fetch_dataset(cfg: RunConfig, raw_dir: Path) -> Path:
    """Purpose: return the directory holding the extracted page images.

    Inputs: the RunConfig and the resolved `raw` directory.
    Output: the extracted dataset root. The archive is never re-downloaded and
    never re-extracted silently; a missing archive raises with the source URL.
    """
    raw_dir = Path(raw_dir)
    extracted = raw_dir / Path(cfg.data.archive_name).name.split(".")[0]
    if extracted.is_dir() and any(extracted.iterdir()):
        _LOGGER.info("Dataset already extracted: %s", extracted)
        return extracted
    if _looks_like_dataset(raw_dir):
        _LOGGER.info("Dataset found in place: %s", raw_dir)
        return raw_dir
    archive = raw_dir / cfg.data.archive_name
    if not archive.is_file():
        raise FileNotFoundError(
            f"Dataset archive not found: {archive}. Download RVL-CDIP from "
            f"{cfg.data.source_url or 'the source recorded in docs/dataset-card.md'} "
            f"and place the archive at that path, or extract the class folders "
            f"directly into {raw_dir}."
        )
    if cfg.data.archive_sha256:
        verify_checksum(archive, cfg.data.archive_sha256)
    _extract(archive, extracted)
    return extracted


def verify_checksum(path: Path, expected_sha256: str) -> str:
    """Purpose: prove the archive on disk is the one the dataset card describes.

    Inputs: the archive path and the expected hexadecimal SHA-256.
    Output: the computed digest. Raises ValueError on a mismatch.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}. "
            f"Delete the archive and download it again."
        )
    _LOGGER.info("Archive checksum verified: %s", actual)
    return actual


def _looks_like_dataset(directory: Path) -> bool:
    """Purpose: recognise an already-extracted class-per-directory layout.

    Input: a directory. Output: True when it holds at least one subdirectory.
    """
    return directory.is_dir() and any(item.is_dir() for item in directory.iterdir())


def _extract(archive: Path, destination: Path) -> None:
    """Purpose: unpack a tar or zip archive into the destination directory.

    Inputs: the archive path and the destination.
    Output: None. Raises ValueError on an unsupported archive type.
    """
    destination.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("Extracting archive: %s", archive)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            handle.extractall(destination, filter="data")
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
    else:
        raise ValueError(
            f"Unsupported archive format: {archive}. Expected a .tar, .tar.gz "
            f"or .zip file."
        )
    _LOGGER.info("Extraction destination: %s", destination)
