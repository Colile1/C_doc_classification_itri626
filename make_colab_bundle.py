#!/usr/bin/env python
# make_colab_bundle.py — pack only the images the split uses, for Colab.
# Author: Colile Sibanda
#
# `data/raw/` holds 200 000 images (~19 GB), but a run only ever reads the
# 16 000 the committed split names (~2.8 GB). This writes an archive of just
# those, keeping the `class/filename` layout the split paths expect, so it
# unpacks on Colab and the split files resolve against it unchanged.

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

SPLIT_FILES = ("train.csv", "val.csv", "test.csv")


def collect_paths(splits_dir: Path) -> list[str]:
    """Purpose: every image path the committed split refers to.

    Input: the split directory. Output: the stored (relative) paths, de-duplicated.
    Raises FileNotFoundError when a split file is missing, because a partial
    bundle would fail a long way into a run rather than here.
    """
    stored: list[str] = []
    for name in SPLIT_FILES:
        path = splits_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {path}. The bundle must cover the whole "
                f"split, so all three files are required."
            )
        stored.extend(pd.read_csv(path)["path"].astype(str).tolist())
    return sorted(set(stored))


def build(destination: Path, splits_dir: Path, root: Path) -> Path:
    """Purpose: write the archive of exactly the split's images.

    Inputs: the archive to write, the split directory, and the project root the
    stored paths are relative to.
    Output: the archive written. Any image named by the split but missing on
    disk is reported and stops the build, because a bundle with holes in it
    fails mid-epoch on Colab.
    """
    stored = collect_paths(splits_dir)
    missing = [value for value in stored if not (root / value).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) named by the split are not on disk, e.g. "
            f"{missing[0]}. Fix the dataset before bundling."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(destination, "w:gz") as archive:
        for index, value in enumerate(stored, start=1):
            source = root / value
            total += source.stat().st_size
            # Store under the path the split files name, so they resolve as-is.
            archive.add(source, arcname=value)
            if index % 2000 == 0:
                print(f"  {index:,} of {len(stored):,} added", flush=True)

    size = destination.stat().st_size
    print(f"\n{len(stored):,} images, {total / 1024 ** 3:.2f} GB uncompressed")
    print(f"Archive: {destination}  ({size / 1024 ** 3:.2f} GB)")
    return destination


def main() -> int:
    """Purpose: the command-line entry point. Output: the exit code."""
    parser = argparse.ArgumentParser(
        description="Pack only the images the committed split uses, for Colab."
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "interim" / "colab_images.tar.gz"),
        help="Archive to write.",
    )
    parser.add_argument(
        "--splits",
        default=str(ROOT / "data" / "splits"),
        help="Directory holding train.csv, val.csv and test.csv.",
    )
    arguments = parser.parse_args()
    build(Path(arguments.out), Path(arguments.splits), ROOT)
    print("\nUpload this to Drive, then on Colab:")
    print("  !tar -xzf /content/drive/MyDrive/itri626/colab_images.tar.gz -C /content")
    print("\nThat yields /content/data/raw/<class>/..., which is what")
    print("env_colab.yaml's `raw: /content/data/raw` already points at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
