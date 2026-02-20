#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/mtl_project/master1-main}"
PYTHON_CMD="${PYTHON_CMD:-python}"
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
