# run_prepare_data.py — locate the dataset, build the manifest, clean it, cap it.
# Author: Colile Sibanda

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from docclf.config.loader import load_config  # noqa: E402
from docclf.config.paths import ProjectPaths, resolve_paths  # noqa: E402
from docclf.config.schema import RunConfig  # noqa: E402
from docclf.config.seed import set_global_seed  # noqa: E402
from docclf.data.acquire import fetch_dataset  # noqa: E402
from docclf.data.clean import clean_manifest  # noqa: E402
from docclf.data.manifest import build_manifest  # noqa: E402
from docclf.data.subsample import cap_per_class  # noqa: E402
from docclf.utils.env import detect_environment  # noqa: E402
from docclf.utils.logging import add_file_handler, get_logger, log_values  # noqa: E402
from docclf.utils.timing import timed  # noqa: E402

LOGGER = get_logger("run_prepare_data")
WORKING_SET_NAME = "working_set.csv"
CLEANING_REPORT_NAME = "cleaning_report.csv"
CLEANED_PREFIX = "cleaned_manifest_"


def prepare_data(cfg: RunConfig, paths: ProjectPaths, reuse_cleaned: bool = False) -> Path:
    """Purpose: produce the cleaned, capped working set the split is drawn from.

    Inputs: the RunConfig, the resolved paths, and whether to reuse the cleaned
    manifest from an earlier run instead of reading and hashing every image.
    Output: the path of the working-set manifest written to `data/processed/`.
    The cleaned (uncapped) manifest is kept in `data/interim/`, so changing
    `data.n_per_class` costs seconds rather than a full re-clean.
    """
    preparation_started = time.perf_counter()
    cleaned_path = paths.interim / f"{CLEANED_PREFIX}{cfg.name}.csv"
    report_path = paths.splits / CLEANING_REPORT_NAME
    if reuse_cleaned:
        cleaned = load_cleaned(cleaned_path, paths.raw)
        raw_count = len(pd.read_csv(report_path)) + len(cleaned)
    else:
        with timed("Dataset acquisition"):
            root = fetch_dataset(cfg, paths.raw)
        LOGGER.info("Dataset image root: %s", root)
        with timed("Manifest build"):
            manifest = build_manifest(root, cfg.classes, cfg.data.group_pattern)
        _log_class_counts(LOGGER, "Raw images", manifest)
        total_bytes = int(manifest["bytes"].sum())
        LOGGER.info("Raw image bytes: %d (%.2f GiB)", total_bytes, total_bytes / (1024 ** 3))
        with timed("Cleaning"):
            cleaned, report = clean_manifest(manifest, cfg)
        _log_class_counts(LOGGER, "Cleaned images", cleaned)
        for reason, count in report.summary().items():
            if reason not in ("raw_count", "kept_count"):
                LOGGER.info("Cleaning removal count (%s): %d", reason, count)
        cleaned.to_csv(cleaned_path, index=False, encoding="utf-8")
        report.to_frame().to_csv(report_path, index=False, encoding="utf-8")
        raw_count = report.raw_count
    with timed("Per-class cap"):
        working = cap_per_class(
            cleaned, cfg.data.n_per_class, cfg.seed, cfg.split.group_column
        )
    working_path = paths.processed / WORKING_SET_NAME
    working.to_csv(working_path, index=False, encoding="utf-8")
    _log_class_counts(LOGGER, "Working-set images", working)
    log_values(
        LOGGER,
        {
            "Raw pages": raw_count,
            "Pages after cleaning": len(cleaned),
            "Working-set pages": len(working),
            "Working-set documents": working[cfg.split.group_column].nunique(),
            "Cleaning report": report_path,
            "Cleaned manifest": cleaned_path,
            "Working set": working_path,
            "Preparation total seconds": round(time.perf_counter() - preparation_started, 3),
        },
    )
    return working_path


def _log_class_counts(logger, label: str, frame: pd.DataFrame) -> None:
    """Log the per-class distribution at each data preparation stage."""
    if "class_name" not in frame.columns:
        return
    for class_name, count in frame["class_name"].value_counts().sort_index().items():
        logger.info("%s by class (%s): %d", label, class_name, int(count))


def load_cleaned(cleaned_path: Path, raw_dir: Path) -> pd.DataFrame:
    """Purpose: reload a cleaned manifest so the cap can change without re-hashing.

    Inputs: the saved cleaned manifest and the configured raw directory.
    Output: the cleaned manifest. Raises when the file is missing or was built
    from a different raw directory, since capping it would describe other data.
    """
    if not cleaned_path.is_file():
        raise FileNotFoundError(
            f"No cleaned manifest at {cleaned_path}. Run without --reuse-cleaned first."
        )
    cleaned = pd.read_csv(
        cleaned_path, dtype={"path": str, "doc_group_id": str, "phash": str}
    )
    first = Path(cleaned["path"].iloc[0]).resolve()
    if Path(raw_dir).resolve() not in first.parents:
        raise ValueError(
            f"{cleaned_path} was built from another raw directory ({first.parent}). "
            f"Run without --reuse-cleaned to rebuild it from {raw_dir}."
        )
    LOGGER.info("Reusing cleaned manifest: %s (%d pages)", cleaned_path, len(cleaned))
    return cleaned


def main() -> int:
    """Purpose: parse the command line and run the data preparation stage.

    Input: none (reads `sys.argv`). Output: the process exit code.
    """
    parser = argparse.ArgumentParser(description="Prepare the document image working set.")
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument("--env", default=None, help="local, colab or hpc")
    parser.add_argument(
        "--reuse-cleaned",
        action="store_true",
        help="Re-cap the cleaned manifest from an earlier run instead of re-cleaning",
    )
    arguments = parser.parse_args()
    env = arguments.env or detect_environment()
    cfg = load_config(arguments.config, env=env)
    paths = resolve_paths(cfg, env).ensure()
    add_file_handler(paths.logs / f"prepare_data_{cfg.name}.log")
    set_global_seed(cfg.seed)
    log_values(LOGGER, {"Config": arguments.config, "Environment": env, "Seed": cfg.seed})
    prepare_data(cfg, paths, reuse_cleaned=arguments.reuse_cleaned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
