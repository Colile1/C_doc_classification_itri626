# run_prepare_data.py — locate the dataset, build the manifest, clean it, cap it.
# Author: Colile Sibanda

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def prepare_data(cfg: RunConfig, paths: ProjectPaths) -> Path:
    """Purpose: produce the cleaned, capped working set the split is drawn from.

    Inputs: the RunConfig and the resolved paths.
    Output: the path of the working-set manifest written to `data/processed/`.
    """
    with timed("Dataset acquisition"):
        root = fetch_dataset(cfg, paths.raw)
    with timed("Manifest build"):
        manifest = build_manifest(root, cfg.classes, cfg.data.group_pattern)
    with timed("Cleaning"):
        cleaned, report = clean_manifest(manifest, cfg)
    with timed("Per-class cap"):
        working = cap_per_class(
            cleaned, cfg.data.n_per_class, cfg.seed, cfg.split.group_column
        )
    report_path = paths.splits / CLEANING_REPORT_NAME
    report.to_frame().to_csv(report_path, index=False, encoding="utf-8")
    working_path = paths.processed / WORKING_SET_NAME
    working.to_csv(working_path, index=False, encoding="utf-8")
    log_values(
        LOGGER,
        {
            "Raw pages": report.raw_count,
            "Pages after cleaning": len(cleaned),
            "Working-set pages": len(working),
            "Working-set documents": working[cfg.split.group_column].nunique(),
            "Cleaning report": report_path,
            "Working set": working_path,
        },
    )
    return working_path


def main() -> int:
    """Purpose: parse the command line and run the data preparation stage.

    Input: none (reads `sys.argv`). Output: the process exit code.
    """
    parser = argparse.ArgumentParser(description="Prepare the document image working set.")
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument("--env", default=None, help="local, colab or hpc")
    arguments = parser.parse_args()
    env = arguments.env or detect_environment()
    cfg = load_config(arguments.config, env=env)
    paths = resolve_paths(cfg, env).ensure()
    add_file_handler(paths.logs / f"prepare_data_{cfg.name}.log")
    set_global_seed(cfg.seed)
    log_values(LOGGER, {"Config": arguments.config, "Environment": env, "Seed": cfg.seed})
    prepare_data(cfg, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
