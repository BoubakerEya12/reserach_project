#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
if [[ -n "${PYTHON_CMD:-}" ]]; then
  PYTHON_CMD="${PYTHON_CMD}"
elif [[ -x "${PROJECT_DIR}/venv_mtl/bin/python" ]]; then
  PYTHON_CMD="${PROJECT_DIR}/venv_mtl/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python3)"
else
  PYTHON_CMD="$(command -v python)"
fi
WEIGHTS_PATH="${WEIGHTS_PATH:-results/mtl_unsupervised.last.weights.h5}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "ERROR: PROJECT_DIR not found: ${PROJECT_DIR}"
  echo "Set PROJECT_DIR env var to your repo path."
  exit 1
fi

cd "${PROJECT_DIR}"

if [[ ! -f "${WEIGHTS_PATH}" ]]; then
  echo "ERROR: Weights not found: ${WEIGHTS_PATH}"
  echo "Set WEIGHTS_PATH or run training first."
  exit 1
fi

EVAL_CMD="${PYTHON_CMD} -m scripts.eval_mtl_unsupervised \
  --weights ${WEIGHTS_PATH} \
  --steps 50 --batch_size 64 --gamma_qos_db 5"

echo "Eval command:"
echo "${EVAL_CMD}"

${EVAL_CMD}
