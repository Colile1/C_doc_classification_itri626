#!/usr/bin/env python3
"""Populate data/raw/<class>/ with the eight project classes from an extracted RVL-CDIP.

The full dataset (400 000 images, 16 classes) stays outside the repository.
This script reads the official train, val and test label files, keeps the
eight project classes, and places each image under data/raw/<class>/ as a
hard link. A hard link costs no extra disk space and is created in
milliseconds, but needs the source and the repository on the same NTFS
volume; on another volume or a file system without hard links (exFAT,
FAT32) the script falls back to copying, or use --copy.

Files are named <source folder>_<file name>. The source folder identifies
the scanned document a page came from, which is what the grouped split uses
to keep pages of one document in a single partition.

Usage:
    python extract_rvl_cdip.py --source D:\\data
    python extract_rvl_cdip.py --source /mnt/data --copy
    python extract_rvl_cdip.py --source D:\\data --per-class 200 --dest data/interim/smoke_raw

Safe to re-run: files already present are skipped.

Author: Colile Sibanda
"""
import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# RVL-CDIP label id to project class name.
LABEL_TO_CLASS = {
    11: "invoice",
    10: "budget",
    1: "form",
    0: "letter",
    15: "memo",
    3: "handwritten",
    14: "resume",
    4: "advertisement",
}


def read_labels(source):
    """Return {relative image path: class name} for the eight project classes."""
    mapping = {}
    for split in ("train", "val", "test"):
        path = source / "labels" / f"{split}.txt"
        if not path.exists():
            sys.exit(f"Label file not found: {path}")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                rel, _, label = line.strip().rpartition(" ")
                if label.isdigit() and int(label) in LABEL_TO_CLASS:
                    mapping[rel] = LABEL_TO_CLASS[int(label)]
    return mapping


def dest_filename(rel):
    """Prefix the file name with its parent folder so names never collide."""
    parts = rel.split("/")
    return f"{parts[-2]}_{parts[-1]}"


def sample_per_class(mapping, per_class, seed):
    """Keep a seeded random sample of per_class images in each class.

    Used to build the small smoke-test dataset: the pipeline cleans every raw
    image before capping, so a smoke run over the full raw set would hash
    200 000 pages instead of a few hundred.
    """
    rng = random.Random(seed)
    by_class = {}
    for rel, name in sorted(mapping.items()):
        by_class.setdefault(name, []).append(rel)
    sampled = {}
    for name, paths in sorted(by_class.items()):
        for rel in rng.sample(paths, min(per_class, len(paths))):
            sampled[rel] = name
    return sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path,
                        help="folder holding images/ and labels/ from the extracted archive")
    parser.add_argument("--dest", type=Path, default=PROJECT_ROOT / "data" / "raw",
                        help="output root (default: data/raw)")
    parser.add_argument("--copy", action="store_true",
                        help="copy instead of hard-linking")
    parser.add_argument("--per-class", type=int, default=None,
                        help="place only a seeded random sample of this many images per class")
    parser.add_argument("--seed", type=int, default=626,
                        help="seed for --per-class sampling (default: 626)")
    args = parser.parse_args()

    started = time.time()
    mapping = read_labels(args.source)
    if args.per_class is not None:
        mapping = sample_per_class(mapping, args.per_class, args.seed)
    print(f"{len(mapping)} labelled images across {len(LABEL_TO_CLASS)} classes")
    for name in LABEL_TO_CLASS.values():
        (args.dest / name).mkdir(parents=True, exist_ok=True)

    use_copy = args.copy
    counts = {name: 0 for name in LABEL_TO_CLASS.values()}
    created = skipped = missing = 0
    for index, (rel, name) in enumerate(mapping.items(), start=1):
        src = args.source / "images" / rel
        dst = args.dest / name / dest_filename(rel)
        if dst.exists():
            skipped += 1
            counts[name] += 1
        elif not src.exists():
            missing += 1
        else:
            if not use_copy:
                try:
                    os.link(src, dst)
                except OSError as err:
                    print(f"Hard link failed ({err}); copying from here on.")
                    use_copy = True
            if use_copy:
                shutil.copy2(src, dst)
            created += 1
            counts[name] += 1
        if index % 20000 == 0:
            print(f"...{index} processed ({time.time() - started:.0f}s)")

    print(f"Done in {time.time() - started:.0f}s. "
          f"Created {created}, already present {skipped}, missing in source {missing}.")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    if missing:
        sys.exit(f"{missing} labelled images were not found under {args.source / 'images'}.")


if __name__ == "__main__":
    main()
