# hashing.py — perceptual hashing, shared by data cleaning and the leakage check.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

HASH_SIZE = 8


def perceptual_hash(path: str | Path) -> str:
    """Purpose: describe an image by appearance, so near-duplicates collide.

    Input: the path to an image file.
    Output: the hexadecimal perceptual hash. Raises ValueError naming the file
    when it cannot be opened as an image.
    """
    file_path = Path(path)
    try:
        with Image.open(file_path) as image:
            return str(imagehash.phash(image.convert("L"), hash_size=HASH_SIZE))
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {file_path}.") from None
    except (UnidentifiedImageError, OSError) as error:
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
