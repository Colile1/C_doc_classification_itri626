# io.py — read and write the committed split files, so anyone can reproduce them.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import pandas as pd

from docclf.splits.grouped import SPLIT_NAMES
from docclf.splits.leakage import as_frame
from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)


def save_splits(splits: pd.DataFrame, directory: str | Path) -> dict[str, Path]:
    """Purpose: write one CSV per split into `data/splits/`.

    Inputs: the split assignment and the destination directory.
    Output: a mapping of split name to the file written.
    """
    frame = as_frame(splits)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        subset = frame[frame["split"] == name]
        if subset.empty:
            raise ValueError(
                f"Split {name!r} is empty; refusing to write split files. Check the "
                f"working set size and the ratios in the config."
            )
        path = destination / f"{name}.csv"
        subset.to_csv(path, index=False, encoding="utf-8")
        written[name] = path
        _LOGGER.info("Split file written: %s (%d rows)", path, len(subset))
    return written


def load_splits(directory: str | Path) -> pd.DataFrame:
    """Purpose: reload the committed split files as one labelled frame.

    Input: the directory holding `train.csv`, `val.csv` and `test.csv`.
    Output: the concatenated DataFrame with its `split` column.
    """
    source = Path(directory)
    frames: list[pd.DataFrame] = []
    for name in SPLIT_NAMES:
        path = source / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {path}. Run `python run_split.py --config "
                f"<config>` to create the split files."
            )
        frame = pd.read_csv(path)
        frame["split"] = name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    _LOGGER.info("Split rows loaded: %d", len(combined))
    return combined
