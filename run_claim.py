#!/usr/bin/env python
# run_claim.py — claim a model before training it, so two machines never duplicate work.
# Author: Colile Sibanda
#
# A claim is a small JSON file under runs/claims/. It is advisory, not a lock:
# it stops the ordinary mistake of starting the same model twice, and it tells
# you which machine to go and look at. Commit and pull claims to share them.

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from run_status import CLAIM_STALE_MINUTES, MODELS, _claim_path, read_claim  # noqa: E402


def write_claim(model: str, note: str | None) -> Path:
    """Purpose: record that this machine is training this model.

    Inputs: the model name and an optional note. Output: the claim path.
    """
    path = _claim_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    existing = read_claim(model)
    claim = {
        "model": model,
        "machine": socket.gethostname(),
        "pid": os.getpid(),
        "claimed_utc": existing["claimed_utc"] if existing else now,
        "heartbeat_utc": now,
        "note": note or "",
    }
    path.write_text(json.dumps(claim, indent=2), encoding="utf-8")
    return path


def main() -> int:
    """Purpose: claim, refresh or release a model's training slot."""
    parser = argparse.ArgumentParser(
        description="Claim a model before training it, so no two machines duplicate work."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument(
        "--release", action="store_true", help="Give up the claim (after pausing or finishing)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Take over a live claim held by another machine. Use only when you know it stopped.",
    )
    parser.add_argument("--note", default=None, help="Free text, e.g. 'overnight run'.")
    arguments = parser.parse_args()

    path = _claim_path(arguments.model)
    if arguments.release:
        if path.is_file():
            path.unlink()
            print(f"Released {arguments.model}.")
        else:
            print(f"No claim on {arguments.model} to release.")
        return 0

    existing = read_claim(arguments.model)
    mine = socket.gethostname()
    if existing and existing.get("machine") != mine and not existing["stale"]:
        if not arguments.force:
            print(
                f"REFUSED: {arguments.model} is claimed by {existing['machine']} "
                f"{existing['age_minutes']:.0f} min ago.\n"
                f"  If that machine really stopped, re-run with --force.\n"
                f"  Otherwise train a different model: python run_status.py"
            )
            return 1
        print(f"Taking over {arguments.model} from {existing['machine']}.")

    written = write_claim(arguments.model, arguments.note)
    print(f"Claimed {arguments.model} for {mine}.")
    print(f"  {written}")
    print(
        f"  Refresh it while training, or it goes stale after {CLAIM_STALE_MINUTES} min.\n"
        f"  Release when you pause or finish: python run_claim.py --model "
        f"{arguments.model} --release"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
