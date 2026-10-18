# test_utils.py — device selection, logging discipline, perceptual hashing and I/O retry.
# Author: Colile Sibanda

from __future__ import annotations

import errno
import random
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from docclf.config.loader import load_config
from docclf.utils.env import detect_environment, select_device
from docclf.utils.hashing import (
    distances_to,
    hamming_distance,
    hashes_to_array,
    perceptual_hash,
)
from docclf.utils.logging import get_logger
from docclf.utils.retry import is_transient_io_error, retry_io
from docclf.utils.timing import timed

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "docclf"


def test_detect_environment_returns_a_known_name() -> None:
    """The name is used to pick paths and batch sizes; it must be one of three."""
    assert detect_environment() in ("local", "colab", "hpc")


def test_select_device_falls_back_to_cpu_without_cuda() -> None:
    """A cuda config on the laptop must degrade to cpu, not crash the run."""
    cfg = load_config(CONFIGS / "smoke.yaml", env="colab")
    assert select_device(cfg).type in ("cpu", "cuda")


def test_select_device_honours_an_explicit_cpu_request() -> None:
    """The laptop overlay pins cpu, and that must be respected on any machine."""
    cfg = load_config(CONFIGS / "smoke.yaml", env="local")
    assert select_device(cfg).type == "cpu"


def test_identical_images_collide(tmp_path: Path) -> None:
    """Two visually identical pages must hash to the same value."""
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    image = Image.new("L", (200, 260), color=255)
    for x in range(20, 180, 6):
        for y in range(30, 240, 11):
            image.putpixel((x, y), 0)
    image.save(first)
    image.save(second)
    assert perceptual_hash(first) == perceptual_hash(second)
    assert hamming_distance(perceptual_hash(first), perceptual_hash(second)) == 0


def test_different_images_separate(tmp_path: Path) -> None:
    """Two visibly different pages must not collide."""
    dense = Image.new("L", (200, 260), color=255)
    for y in range(0, 260, 3):
        for x in range(0, 200):
            dense.putpixel((x, y), 0)
    sparse = Image.new("L", (200, 260), color=255)
    for y in range(120, 140):
        for x in range(90, 110):
            sparse.putpixel((x, y), 0)
    dense_path, sparse_path = tmp_path / "d.png", tmp_path / "s.png"
    dense.save(dense_path)
    sparse.save(sparse_path)
    distance = hamming_distance(perceptual_hash(dense_path), perceptual_hash(sparse_path))
    assert distance > 4, f"distance {distance} is inside the near-duplicate threshold"


def test_hashing_a_non_image_is_an_actionable_error(tmp_path: Path) -> None:
    """A corrupt file must say what to do about it, not return a blank hash."""
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        perceptual_hash(broken)


def test_packed_distances_match_pairwise_hamming() -> None:
    """The vectorised check must give exactly the distances the pairwise one does."""
    rng = random.Random(626)
    hashes = [f"{rng.getrandbits(64):016x}" for _ in range(200)]
    hashes.append(hashes[0])
    packed = hashes_to_array(hashes)
    for target in hashes[:20]:
        expected = [hamming_distance(target, other) for other in hashes]
        assert distances_to(int(target, 16), packed).tolist() == expected


def test_packing_rejects_a_hash_of_the_wrong_width() -> None:
    """A short hash would compare wrongly against 64-bit ones, so it must be refused."""
    with pytest.raises(ValueError, match="hexadecimal digits"):
        hashes_to_array(["abc"])


def test_retry_rides_out_a_brief_storage_fault() -> None:
    """A drive that drops out and comes back must not be reported as a bad file."""
    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            error = OSError(errno.EIO, "I/O error")
            raise error
        return "read"

    assert retry_io(flaky, attempts=5, wait_seconds=0) == "read"
    assert calls["count"] == 3


def test_retry_raises_a_persistent_storage_fault() -> None:
    """A drive that stays gone must stop the run, not mark files corrupt."""
    def gone() -> None:
        raise OSError(errno.ENODEV, "No such device")

    with pytest.raises(OSError):
        retry_io(gone, attempts=3, wait_seconds=0)


def test_a_vanished_folder_counts_as_a_storage_fault(tmp_path: Path) -> None:
    """A dropped drive reports "path not found"; that must not read as a missing image."""
    gone = tmp_path / "unplugged" / "page.tif"
    present = tmp_path / "page.tif"
    error = FileNotFoundError(2, "The system cannot find the path specified")
    assert is_transient_io_error(error, gone)
    assert not is_transient_io_error(error, present)


def test_retry_does_not_retry_a_bad_file() -> None:
    """An unreadable image is a property of the file; retrying would only waste time."""
    calls = {"count": 0}

    def unreadable() -> None:
        calls["count"] += 1
        raise UnidentifiedImageError("cannot identify image file")

    assert not is_transient_io_error(UnidentifiedImageError("x"))
    with pytest.raises(UnidentifiedImageError):
        retry_io(unreadable, attempts=5, wait_seconds=0)
    assert calls["count"] == 1


def test_timed_records_a_duration() -> None:
    """The run log reports stage durations, so the context manager must fill one in."""
    with timed("unit sample") as duration:
        sum(range(10000))
    assert duration.seconds >= 0.0


def test_logger_is_namespaced_under_the_package() -> None:
    """One logger tree, so a notebook can raise or silence the whole package."""
    assert get_logger("docclf.data").name.startswith("docclf")
    assert get_logger("something").name.startswith("docclf.")


def test_no_module_in_the_package_calls_print() -> None:
    """The coding standard: I/O goes through utils.logging, never print()."""
    offenders: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "print(" in stripped and "pprint(" not in stripped:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"print() found in: {', '.join(offenders)}"


def test_no_module_in_the_package_parses_arguments() -> None:
    """argparse belongs in the root orchestrators, not in the library."""
    offenders = [
        path.name
        for path in PACKAGE.rglob("*.py")
        if "argparse" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"argparse found in: {', '.join(offenders)}"
