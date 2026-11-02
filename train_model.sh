#!/usr/bin/env bash
# train_model.sh — one command to train any model on any machine, resumably.
# Author: Colile Sibanda
#
#   ./train_model.sh resnet50          # start or resume, claim, heartbeat, release
#   ./train_model.sh vit_b16
#
# Safe to interrupt with Ctrl-C: the claim is released and the next run (here or
# on another machine) continues from the last completed epoch.

set -euo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "usage: ./train_model.sh <baseline_cnn_v4|resnet50|vit_b16>" >&2
    exit 2
fi

cd "$(dirname "$0")"
VENV_DIR="${ITRI626_VENV_DIR:-${HOME}/.venvs/itri626}"
PY="${VENV_DIR}/bin/python"
[[ -x "$PY" ]] || PY="${VENV_DIR}/Scripts/python.exe"
[[ -x "$PY" ]] || { echo "No interpreter at ${VENV_DIR}. Set ITRI626_VENV_DIR." >&2; exit 1; }

case "$MODEL" in
    baseline_cnn_v4) CONFIG=configs/baseline_cnn_v4.yaml ;;
    resnet50)        CONFIG=configs/resnet50.yaml ;;
    vit_b16)         CONFIG=configs/vit_b16.yaml ;;
    *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac

# Refuse to start when the split files are missing: training on a regenerated
# split would make the results incomparable with the runs already recorded.
for name in train val test; do
    [[ -f "data/splits/${name}.csv" ]] || {
        echo "Missing data/splits/${name}.csv. Run run_split.py before training." >&2
        exit 1
    }
done

"$PY" run_claim.py --model "$MODEL" --note "train_model.sh" || exit 1

# Keep the claim fresh while training, and release it however this exits.
( while true; do sleep 1800; "$PY" run_claim.py --model "$MODEL" >/dev/null 2>&1 || true; done ) &
HEARTBEAT=$!
cleanup() {
    kill "$HEARTBEAT" 2>/dev/null || true
    "$PY" run_claim.py --model "$MODEL" --release >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Resume when this model already has a checkpoint of its own.
RESUME=""
CKPT_DIR="$("$PY" -c "
import sys; sys.path.insert(0,'src')
from run_status import MODELS
print(MODELS['$MODEL']['dir'])
")"
if [[ -f "${CKPT_DIR}/checkpoints/${MODEL}_last.pt" ]]; then
    RESUME="--resume"
    echo "Found a checkpoint for ${MODEL}; resuming."
fi

echo "Training ${MODEL} from ${CONFIG} ${RESUME}"
"$PY" run_train.py --config "$CONFIG" --env local --model "$MODEL" $RESUME

echo "Evaluating ${MODEL} on the held-out split"
"$PY" run_evaluate.py --config "$CONFIG" --env local --model "$MODEL"

echo
"$PY" run_status.py
