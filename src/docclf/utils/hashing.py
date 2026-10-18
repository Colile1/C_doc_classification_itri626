# hashing.py — perceptual hashing, shared by data cleaning and the leakage check.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

from docclf.utils.retry import is_transient_io_error, retry_io

HASH_SIZE = 8


def perceptual_hash(path: str | Path) -> str:
    """Purpose: describe an image by appearance, so near-duplicates collide.

    Input: the path to an image file.
    Output: the hexadecimal perceptual hash. Raises ValueError naming the file
    when it cannot be opened as an image.
    """
    file_path = Path(path)

    def compute() -> str:
        with Image.open(file_path) as image:
            return str(imagehash.phash(image.convert("L"), hash_size=HASH_SIZE))

    try:
        return retry_io(compute, path=file_path)
    except (UnidentifiedImageError, OSError) as error:
        if is_transient_io_error(error, file_path):
            raise
        if isinstance(error, FileNotFoundError):
            raise FileNotFoundError(f"Image not found: {file_path}.") from None
        raise ValueError(
            f"Cannot hash {file_path}: the file is not a readable image ({error}). "
            f"Remove it in data.clean and record the removal in the cleaning report."
        ) from error


def hamming_distance(left: str, right: str) -> int:
    """Purpose: measure how far apart two perceptual hashes are.

    Inputs: two hexadecimal hashes of equal length.
    Output: the number of differing bits. Raises ValueError on a length mismatch.
    """
    if len(left) != len(right):
        raise ValueError(
            f"Cannot compare hashes of different lengths: {len(left)} and {len(right)}."
        )
    return imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right)


def hashes_to_array(hashes: Sequence[str]) -> np.ndarray:
    """Purpose: pack 64-bit hexadecimal hashes into integers for vectorised comparison.

    Input: hexadecimal hashes of HASH_SIZE * HASH_SIZE bits.
    Output: a uint64 array, one element per hash. Raises ValueError for any
    hash that is not 64 bits, since it would not fit the packed form.
    """
    width = HASH_SIZE * HASH_SIZE // 4
    for value in hashes:
        if len(value) != width:
            raise ValueError(
                f"Cannot pack hash {value!r}: expected {width} hexadecimal digits."
            )
    return np.array([int(value, 16) for value in hashes], dtype=np.uint64)


def distances_to(target: int | np.uint64, packed: np.ndarray) -> np.ndarray:
    """Purpose: Hamming distance from one packed hash to many, in one pass.

    Inputs: one hash as an integer; a uint64 array from `hashes_to_array`.
    Output: an integer array of differing-bit counts, aligned with `packed`.
    Equal to `hamming_distance` element by element.
    """
    return np.bitwise_count(np.bitwise_xor(packed, np.uint64(target)))
