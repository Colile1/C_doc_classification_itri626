#!/usr/bin/env python
# run_check.py — is this machine ready to train? Run it before anything else.
# Author: Colile Sibanda
#
# Read-only. Catches the faults that otherwise appear hours into a run: a
# missing dataset, split files that point nowhere, a half-installed environment,
# or a model already being trained somewhere else.

from __future__ import annotations

import argparse
import importlib
import platform
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# How many split rows to open as a sample. Opening all 16 000 would be slow and
# would prove little more than a seeded sample does.
SAMPLE_SIZE = 40


class Report:
    """Collects check results and reports the worst outcome at the end."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        """Purpose: record one check. Inputs: status, name, detail."""
        self.rows.append((status, name, detail))

    def print(self) -> int:
        """Purpose: print every check. Output: the process exit code."""
        width = max(len(name) for _, name, _ in self.rows)
        for status, name, detail in self.rows:
            line = f"  [{status}] {name.ljust(width)}"
            if detail:
                line += f"  {detail}"
            print(line)
        failures = sum(1 for status, _, _ in self.rows if status == FAIL)
        warnings = sum(1 for status, _, _ in self.rows if status == WARN)
        print()
        if failures:
            print(f"{failures} check(s) failed. Fix those before training.")
            return 1
        if warnings:
            print(f"Ready to train, with {warnings} warning(s) above.")
            return 0
        print("Ready to train.")
        return 0


def check_environment(report: Report) -> None:
    """Purpose: the interpreter and the packages a run needs."""
    report.add(PASS, "python", f"{platform.python_version()} on {platform.system()}")
    for module, label in (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("timm", "timm"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("PIL", "pillow"),
        ("imagehash", "imagehash"),
    ):
        try:
            imported = importlib.import_module(module)
        except ImportError:
            report.add(
                FAIL, label,
                "not installed — pip install -r requirements.txt",
            )
            continue
        report.add(PASS, label, getattr(imported, "__version__", "installed"))

    try:
        import torch

        if torch.cuda.is_available():
            report.add(PASS, "device", f"CUDA: {torch.cuda.get_device_name(0)}")
        else:
            report.add(
                WARN, "device",
                f"CPU only, {torch.get_num_threads()} threads — expect hours per run",
            )
    except ImportError:
        pass


def check_dataset(report: Report, cfg, paths) -> None:
    """Purpose: the raw class folders this machine can actually read."""
    raw = paths.raw
    if not raw.is_dir():
        report.add(FAIL, "data/raw", f"{raw} does not exist")
        return
    missing = [name for name in cfg.classes if not (raw / name).is_dir()]
    if missing:
        report.add(FAIL, "data/raw", f"missing class folder(s): {', '.join(missing)}")
        return
    counts = {name: sum(1 for _ in (raw / name).iterdir()) for name in cfg.classes}
    smallest = min(counts.values())
    report.add(
        PASS if smallest else FAIL,
        "data/raw",
        f"{len(cfg.classes)} classes, {sum(counts.values()):,} images, "
        f"smallest class {smallest:,}",
    )


def check_splits(report: Report, cfg, paths) -> None:
    """Purpose: the committed split files, and whether their images open here.

    This is the check that matters most on a new machine: the split files are
    shared, but the images they name have to exist locally.
    """
    import pandas as pd

    from docclf.splits.io import load_splits

    splits_dir = paths.splits
    expected = ("train.csv", "val.csv", "test.csv")
    absent = [name for name in expected if not (splits_dir / name).is_file()]
    if absent:
        report.add(
            FAIL, "split files",
            f"missing {', '.join(absent)} — do NOT regenerate them if other "
            f"results exist; get them from git",
        )
        return

    frame = load_splits(splits_dir, root=paths.root, raw=paths.raw)
    counts = frame["split"].value_counts().to_dict()
    report.add(
        PASS, "split files",
        f"train {counts.get('train', 0):,} / val {counts.get('val', 0):,} / "
        f"test {counts.get('test', 0):,}",
    )

    stored = pd.read_csv(splits_dir / "train.csv")["path"].astype(str)
    absolute = stored.str.startswith("/").sum() + stored.str.contains(":").sum()
    if absolute:
        report.add(
            WARN, "split paths",
            f"{absolute:,} absolute path(s) — these break on other machines; "
            f"they should be relative to the project root",
        )
    else:
        report.add(PASS, "split paths", "relative to the project root")

    sample = frame.sample(min(SAMPLE_SIZE, len(frame)), random_state=cfg.seed)
    unreadable = [path for path in sample["path"] if not Path(path).is_file()]
    if unreadable:
        report.add(
            FAIL, "split images",
            f"{len(unreadable)} of {len(sample)} sampled images not found, "
            f"e.g. {unreadable[0]}",
        )
    else:
        report.add(PASS, "split images", f"{len(sample)} sampled, all readable")


def check_leakage(report: Report, cfg, paths) -> None:
    """Purpose: re-run the group overlap check on the committed split."""
    from docclf.splits.io import load_splits
    from docclf.splits.leakage import LeakageError, assert_no_group_overlap

    try:
        frame = load_splits(paths.splits, root=paths.root, raw=paths.raw)
        assert_no_group_overlap(frame, cfg.split.group_column)
    except LeakageError as error:
        report.add(FAIL, "leakage check", str(error)[:90])
        return
    except (FileNotFoundError, KeyError):
        return
    report.add(PASS, "leakage check", "no document group spans two splits")


def check_claims(report: Report) -> None:
    """Purpose: what other machines are training right now."""
    from run_status import MODELS, progress, state_of

    here = socket.gethostname()
    for name, spec in MODELS.items():
        row = progress(name, spec)
        state = state_of(row)
        claim = row["claim"] or {}
        if state == "RUNNING" and claim.get("machine") != here:
            report.add(
                WARN, f"model {name}",
                f"RUNNING on {claim.get('machine')} — do not start it here",
            )
        elif state == "STALE CLAIM":
            report.add(
                WARN, f"model {name}",
                f"claim by {claim.get('machine')} is "
                f"{claim.get('age_minutes', 0):.0f} min old; take over with --force",
            )
        else:
            report.add(PASS, f"model {name}", state)


def main() -> int:
    """Purpose: run every check and report. Output: the process exit code."""
    parser = argparse.ArgumentParser(
        description="Check this machine is ready to train before starting a run."
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Config to read the classes and paths from.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="local, colab or hpc. Detected from the machine when not given.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip the dataset and split checks (for a machine that only reports).",
    )
    arguments = parser.parse_args()

    print(f"\nITRI626 preflight — {socket.gethostname()}  ({ROOT})\n")
    report = Report()
    check_environment(report)

    try:
        from docclf.config.loader import load_config
        from docclf.config.paths import resolve_paths
        from docclf.utils.env import detect_environment

        env = arguments.env or detect_environment()
        cfg = load_config(ROOT / arguments.config, env=env)
        # Paths differ per environment — on Colab the images sit on local disk
        # while the project is on Drive — so the checks must look where this
        # environment actually puts them, not where `local` would.
        paths = resolve_paths(cfg, env)
        report.add(
            PASS, "config", f"{arguments.config} loads, env {env}, seed {cfg.seed}"
        )
    except Exception as error:  # the config is what every later check needs
        report.add(FAIL, "config", f"{type(error).__name__}: {error}")
        return report.print()

    if not arguments.skip_data:
        check_dataset(report, cfg, paths)
        check_splits(report, cfg, paths)
        check_leakage(report, cfg, paths)
    check_claims(report)

    code = report.print()
    if code == 0:
        print("Next: python run_status.py")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
