#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./start_training.sh [model] [config]
# Defaults to baseline_cnn_v4 and configs/baseline_cnn_v4.yaml.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ITRI626_VENV_DIR:-${HOME}/.venvs/itri626}"
VENV_PYTHON="${VENV_DIR}/bin/python"
MODEL="${1:-baseline_cnn_v4}"
CONFIG="${2:-configs/${MODEL}.yaml}"
ENVIRONMENT="${ITRI626_ENV:-local}"

cd "${PROJECT_DIR}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "ERROR: Project environment not found at ${VENV_DIR}. Run ./setup_requirements.sh first." >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}" >&2
    echo "Usage: ./start_training.sh [baseline_cnn_v4|resnet50|vit_b16] [config path]" >&2
    exit 1
fi
if [[ ! -f data/processed/working_set.csv ]]; then
    echo "ERROR: Training data is missing: data/processed/working_set.csv" >&2
    echo "Prepare the dataset first with run_prepare_data.py." >&2
    exit 1
fi

if [[ ! -f data/splits/train.csv || ! -f data/splits/val.csv || ! -f data/splits/test.csv ]]; then
    echo "No complete split found. Building the grouped train/validation/test split..."
    "${VENV_PYTHON}" run_split.py --config "${CONFIG}" --env "${ENVIRONMENT}"
fi

echo "Starting ${MODEL} training with config ${CONFIG} (environment: ${ENVIRONMENT})..."
echo "Epoch progress and validation scores will appear below and be saved under outputs/logs."
echo "Press Ctrl+C to stop training. Completed epochs are checkpointed."
echo
"${VENV_PYTHON}" run_train.py --config "${CONFIG}" --env "${ENVIRONMENT}" --model "${MODEL}"

echo
echo "Training finished. Review outputs/metrics/history_${MODEL}.csv and outputs/logs/train_${MODEL}.log"
