#!/usr/bin/env python
# run_status.py — what is done, what is running, and what to start next.
# Author: Colile Sibanda
#
# Read-only. Run this first on any machine, before starting a training job, so
# two machines do not train the same model twice.

from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

# Each model in the three-family comparison: the config that trains it, the run
# directory it writes into, and its planned number of epochs.
MODELS = {
    "baseline_cnn_v4": {
        "config": "configs/baseline_cnn_v4.yaml",
        "dir": "runs/baseline_cnn_v4",
        "epochs": 25,
        "family": "CNN from scratch",
    },
    "resnet50": {
        "config": "configs/resnet50.yaml",
        "dir": "runs/resnet50",
        "epochs": 20,
        "family": "Pretrained convolutional",
    },
    "vit_b16": {
        "config": "configs/vit_b16.yaml",
        "dir": "runs/vit_b16",
        "epochs": 15,
        "family": "Pretrained transformer",
    },
}

# The comparison needs one model per family.
PRIMARY_MODELS = ("baseline_cnn_v4", "resnet50", "vit_b16")

CLAIM_STALE_MINUTES = 90


def _claim_path(model: str) -> Path:
    """Purpose: where a machine records that it is training this model."""
    return ROOT / "runs" / "claims" / f"{model}.json"


def read_claim(model: str) -> dict | None:
    """Purpose: the current claim on this model, if any.

    Output: the claim, with `age_minutes` and `stale` added, or None.
    """
    path = _claim_path(model)
    if not path.is_file():
        return None
    try:
        claim = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        beat = datetime.fromisoformat(claim.get("heartbeat_utc", ""))
    except ValueError:
        return None
    age = (datetime.now(timezone.utc) - beat).total_seconds() / 60
    claim["age_minutes"] = round(age, 1)
    claim["stale"] = age > CLAIM_STALE_MINUTES
    return claim


def progress(model: str, spec: dict) -> dict:
    """Purpose: read one model's state from the files it leaves behind.

    Inputs: the model name and its entry in MODELS.
    Output: epochs completed, whether it is evaluated, and the claim on it.
    """
    base = ROOT / spec["dir"]
    history = base / "metrics" / f"history_{model}.csv"
    done = 0
    best_f1 = None
    if history.is_file():
        try:
            frame = pd.read_csv(history)
            done = int(frame["epoch"].max()) if len(frame) else 0
            if "val_macro_f1" in frame.columns and len(frame):
                best_f1 = float(frame["val_macro_f1"].max())
        except (pd.errors.EmptyDataError, KeyError, ValueError):
            done = 0
    metrics = base / "metrics" / f"metrics_{model}.csv"
    test_f1 = None
    if metrics.is_file():
        try:
            row = pd.read_csv(metrics).iloc[0]
            test_f1 = float(row["macro_f1"])
        except (pd.errors.EmptyDataError, IndexError, KeyError, ValueError):
            test_f1 = None
    return {
        "model": model,
        "family": spec["family"],
        "epochs_done": done,
        "epochs_target": spec["epochs"],
        "best_val_macro_f1": best_f1,
        "test_macro_f1": test_f1,
        "has_checkpoint": (base / "checkpoints" / f"{model}_last.pt").is_file(),
        "evaluated": test_f1 is not None,
        "claim": read_claim(model),
    }


def state_of(row: dict) -> str:
    """Purpose: one word for where this model stands. Output: the state."""
    if row["evaluated"]:
        return "COMPLETE"
    if row["epochs_done"] >= row["epochs_target"]:
        return "NEEDS EVAL"
    claim = row["claim"]
    if claim and not claim["stale"]:
        return "RUNNING"
    if row["epochs_done"] > 0:
        return "PAUSED" if not claim else "STALE CLAIM"
    return "NOT STARTED"


def main() -> int:
    """Purpose: print the status table and the recommended next command."""
    parser = argparse.ArgumentParser(
        description="Show every model's training state, so no work is duplicated."
    )
    parser.add_argument(
        "--json", action="store_true", help="Machine-readable output."
    )
    arguments = parser.parse_args()

    rows = [progress(name, spec) for name, spec in MODELS.items()]
    if arguments.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    print(f"\nITRI626 training status — this machine is {socket.gethostname()}")
    print(f"{'model':17} {'family':26} {'state':13} {'epochs':>8}  {'val F1':>7} {'test F1':>7}  who")
    print("-" * 104)
    for row in rows:
        claim = row["claim"]
        who = ""
        if claim:
            who = f"{claim.get('machine', '?')} ({claim['age_minutes']:.0f} min ago)"
            if claim["stale"]:
                who += " STALE"
        val = f"{row['best_val_macro_f1']:.4f}" if row["best_val_macro_f1"] else "-"
        test = f"{row['test_macro_f1']:.4f}" if row["test_macro_f1"] else "-"
        print(
            f"{row['model']:17} {row['family']:26} {state_of(row):13} "
            f"{row['epochs_done']:>3}/{row['epochs_target']:<4} {val:>7} {test:>7}  {who}"
        )

    print()
    families = {r["family"] for r in rows if r["evaluated"]}
    print(f"Families with a finished, evaluated model: {len(families)} of 3")

    # A family that is already being worked on needs nothing more, so a fallback
    # model is only suggested when its family has nothing running or finished.
    covered = {
        row["family"]
        for row in rows
        if row["evaluated"] or state_of(row) in ("RUNNING", "NEEDS EVAL", "PAUSED")
    }
    for row in rows:
        state = state_of(row)
        if state == "NEEDS EVAL":
            spec = MODELS[row["model"]]
            print(f"\nNext for this machine (evaluate {row['model']}):")
            print(
                f"  python run_evaluate.py --config {spec['config']} --env local "
                f"--model {row['model']}"
            )
            break
        if state in ("NOT STARTED", "PAUSED"):
            if state == "NOT STARTED" and row["model"] not in PRIMARY_MODELS:
                if row["family"] in covered:
                    continue  # a fallback whose family is already covered
            spec = MODELS[row["model"]]
            flag = " --resume" if row["epochs_done"] > 0 else ""
            print(f"\nNext for this machine ({state}: {row['model']}):")
            print(f"  ./train_model.sh {row['model']}")
            print("\nor by hand:")
            print(f"  python run_claim.py --model {row['model']}")
            print(
                f"  python run_train.py --config {spec['config']} --env local "
                f"--model {row['model']}{flag}"
            )
            break
    else:
        if len(families) >= 3:
            print("\nAll three families are covered. Next: python run_compare.py builds the comparison.")
        else:
            print(
                "\nNothing free to start here — every remaining model is running "
                "or finished elsewhere."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
