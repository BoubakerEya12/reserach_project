#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SESSION_NAME="${SESSION_NAME:-mtl_train}"
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

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "ERROR: PROJECT_DIR not found: ${PROJECT_DIR}"
  echo "Set PROJECT_DIR env var to your repo path."
  exit 1
fi

cd "${PROJECT_DIR}"

WEIGHTS_PATH="results/mtl_unsupervised.last.weights.h5"
INIT_ARG=""
if [[ -f "${WEIGHTS_PATH}" ]]; then
  INIT_ARG="--init_weights ${WEIGHTS_PATH}"
  echo "Resuming from checkpoint: ${WEIGHTS_PATH}"
else
  echo "No checkpoint found. Starting fresh."
fi

TRAIN_CMD="${PYTHON_CMD} -m scripts.train_mtl_unsupervised \
  ${INIT_ARG} \
  --skip_mismatch_init \
  --epochs 30 --steps_per_epoch 200 --batch_size 64 \
  --lr 3e-4 \
  --gamma_qos_db 5 \
  --w_power 2.0 --qos_weight 5.0 \
  --save_every 1 --save_every_steps 25 \
  --save_dir results"

echo "Training command:"
echo "${TRAIN_CMD}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found. Install tmux or run command manually."
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' exists. Attaching..."
  tmux attach -t "${SESSION_NAME}"
  exit 0
fi

echo "Creating tmux session '${SESSION_NAME}' and starting training..."
TMUX_TMPDIR="${TMUX_TMPDIR:-${PROJECT_DIR}/.tmux}"
mkdir -p "${TMUX_TMPDIR}"
export TMUX_TMPDIR
tmux new-session -d -s "${SESSION_NAME}" "cd '${PROJECT_DIR}' && ${TRAIN_CMD}"
tmux attach -t "${SESSION_NAME}"
