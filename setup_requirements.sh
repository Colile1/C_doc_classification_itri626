#!/usr/bin/env bash
set -Eeuo pipefail

# Install pinned dependencies into a venv outside the repository, matching the
# project's documented laptop CPU setup. Override ITRI626_VENV_DIR if desired.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ITRI626_VENV_DIR:-${HOME}/.venvs/itri626}"

if command -v python3.11 >/dev/null 2>&1; then
    PYTHON="python3.11"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "ERROR: Python 3.11 or newer was not found. Install Python and venv support first." >&2
    exit 1
fi

"${PYTHON}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
    echo "ERROR: Python 3.11 or newer is required." >&2
    exit 1
}

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    mkdir -p "$(dirname -- "${VENV_DIR}")"
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
echo "Updating pip..."
"${VENV_PYTHON}" -m pip install --upgrade pip

echo "Installing CPU PyTorch as documented for a laptop without CUDA..."
"${VENV_PYTHON}" -m pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cpu

echo "Installing/checking the remaining pinned requirements..."
"${VENV_PYTHON}" -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo
echo "Requirements are ready."
echo "Virtual environment: ${VENV_DIR}"
echo "Activate with: source \"${VENV_DIR}/bin/activate\""
