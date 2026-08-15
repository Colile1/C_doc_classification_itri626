# run_split.py — draw the grouped split once, prove it is clean, and commit it.
# Author: Colile Sibanda

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from docclf.config.loader import load_config  # noqa: E402
from docclf.config.paths import ProjectPaths, resolve_paths  # noqa: E402
from docclf.config.schema import RunConfig  # noqa: E402
from docclf.config.seed import set_global_seed  # noqa: E402
from docclf.splits.grouped import make_grouped_split  # noqa: E402
from docclf.splits.io import save_splits  # noqa: E402
from docclf.splits.leakage import assert_no_group_overlap, check_near_duplicates  # noqa: E402
from docclf.splits.report import describe_splits, split_totals  # noqa: E402
from docclf.utils.env import detect_environment  # noqa: E402
from docclf.utils.logging import add_file_handler, get_logger, log_values  # noqa: E402
from docclf.utils.timing import timed  # noqa: E402

LOGGER = get_logger("run_split")
WORKING_SET_NAME = "working_set.csv"
SUMMARY_NAME = "split_summary.csv"


def build_split(cfg: RunConfig, paths: ProjectPaths) -> Path:
    """Purpose: create, verify and write the one split all three models reuse.

    Inputs: the RunConfig and the resolved paths.
    Output: the path of the split summary. The leakage assertion runs before the
    files are written, so a leaked split can never reach disk.
    """
    working_path = paths.processed / WORKING_SET_NAME
    if not working_path.is_file():
        raise FileNotFoundError(
            f"Working set not found: {working_path}. Run `python run_prepare_data.py "
            f"--config <config>` first."
        )
    working = pd.read_csv(working_path)
    with timed("Grouped split"):
        splits = make_grouped_split(
            working,
            cfg.split.ratios,
            cfg.seed,
            cfg.split.group_column,
            cfg.split.stratify_column,
        )
    assert_no_group_overlap(splits, cfg.split.group_column)
    violations = check_near_duplicates(splits, max_distance=0)
    if violations:
        for item in violations[:10]:
            LOGGER.warning("Near-duplicate across splits: %s", item)
    summary = describe_splits(splits, cfg.split.group_column)
    summary_path = paths.splits / SUMMARY_NAME
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    written = save_splits(splits, paths.splits)
    log_values(
        LOGGER,
        {
            "Seed": cfg.seed,
            "Working-set pages": len(working),
            "Split summary total": int(summary["count"].sum()),
            "Near-duplicate violations": len(violations),
            "Split summary": summary_path,
            **{f"Split file ({name})": path for name, path in written.items()},
        },
    )
    for row in split_totals(splits).to_dict("records"):
        LOGGER.info(
            "Split %s: %d pages, %d documents, share %.4f",
            row["split"],
            row["pages"],
            row["groups"],
            row["share"],
        )
    return summary_path


def main() -> int:
    """Purpose: parse the command line and run the split stage.

    Input: none (reads `sys.argv`). Output: the process exit code.
    """
    parser = argparse.ArgumentParser(description="Draw the grouped train/val/test split.")
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument("--env", default=None, help="local, colab or hpc")
    arguments = parser.parse_args()
    env = arguments.env or detect_environment()
    cfg = load_config(arguments.config, env=env)
    paths = resolve_paths(cfg, env).ensure()
    add_file_handler(paths.logs / f"split_{cfg.name}.log")
    set_global_seed(cfg.seed)
    log_values(LOGGER, {"Config": arguments.config, "Environment": env})
    build_split(cfg, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
