# io.py — read and write the committed split files, so anyone can reproduce them.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import pandas as pd

from docclf.splits.grouped import SPLIT_NAMES
from docclf.splits.leakage import as_frame
from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)


def to_relative_paths(frame: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Purpose: store image paths relative to the project root, so the split
    files read correctly on every machine.

    Inputs: a frame with a `path` column, and the project root those paths
    should be expressed against.
    Output: a copy whose `path` values are relative where possible. A path
    outside the root is left absolute, because making it relative would need
    `..` segments that break on another machine anyway.
    """
    if "path" not in frame.columns:
        return frame
    base = Path(root).resolve()
    out = frame.copy()

    def relate(value: str) -> str:
        candidate = Path(str(value))
        if not candidate.is_absolute():
            return candidate.as_posix()
        try:
            return candidate.resolve().relative_to(base).as_posix()
        except ValueError:
            return candidate.as_posix()

    out["path"] = [relate(value) for value in out["path"]]
    return out


def resolve_paths_against(
    frame: pd.DataFrame, root: str | Path, raw: str | Path | None = None
) -> pd.DataFrame:
    """Purpose: turn stored relative paths back into paths this machine can open.

    Inputs: a loaded split frame, the project root, and optionally where the
    image directory actually lives on this machine. Output: a copy whose `path`
    values are absolute.

    Paths are stored relative to the project root, as `data/raw/<class>/<file>`.
    That resolves against the root on a machine where the images sit inside the
    project, but not on one where they do not — on Colab the project is on
    Drive while the images are unpacked to local disk, because reading them over
    the Drive mount is far slower than the GPU they feed. When `raw` is given,
    the stored `data/raw` prefix is re-rooted onto it, so the same split files
    work in both layouts. Absolute values are kept as they are, so split files
    written before paths became relative still load.
    """
    if "path" not in frame.columns:
        return frame
    base = Path(root).resolve()
    raw_dir = Path(raw).resolve() if raw is not None else None
    # The prefix the paths were stored with, relative to the project root.
    prefix = None
    if raw_dir is not None and raw_dir != (base / "data" / "raw"):
        try:
            prefix = raw_dir.relative_to(base)
        except ValueError:
            prefix = Path("data") / "raw"

    def resolve(value: str) -> str:
        candidate = Path(str(value))
        if candidate.is_absolute():
            return str(candidate)
        if prefix is not None:
            parts = candidate.parts
            # Re-root `data/raw/...` onto wherever the images really are.
            if len(parts) > 2 and parts[0] == "data" and parts[1] == "raw":
                return str(raw_dir.joinpath(*parts[2:]))
        return str(base / candidate)

    out = frame.copy()
    out["path"] = [resolve(value) for value in out["path"]]
    return out


def save_splits(
    splits: pd.DataFrame, directory: str | Path, root: str | Path | None = None
) -> dict[str, Path]:
    """Purpose: write one CSV per split into `data/splits/`.

    Inputs: the split assignment, the destination directory, and the project
    root to store paths relative to. When `root` is None the paths are written
    exactly as they are, which is the old behaviour.
    Output: a mapping of split name to the file written.
    """
    frame = as_frame(splits)
    if root is not None:
        frame = to_relative_paths(frame, root)
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


def load_splits(
    directory: str | Path,
    root: str | Path | None = None,
    raw: str | Path | None = None,
) -> pd.DataFrame:
    """Purpose: reload the committed split files as one labelled frame.

    Inputs: the directory holding `train.csv`, `val.csv` and `test.csv`, the
    project root that relative image paths are stored against, and optionally
    where the image directory lives on this machine when it is not inside the
    project. When `root` is given, every path comes back absolute for this
    machine.
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
    if root is not None:
        combined = resolve_paths_against(combined, root, raw)
    _LOGGER.info("Split rows loaded: %d", len(combined))
    return combined
