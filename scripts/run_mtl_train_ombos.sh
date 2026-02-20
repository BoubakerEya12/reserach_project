#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-mtl_train}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/mtl_project/master1-main}"
PYTHON_CMD="${PYTHON_CMD:-python}"

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
tmux new-session -d -s "${SESSION_NAME}" "cd '${PROJECT_DIR}' && ${TRAIN_CMD}"
tmux attach -t "${SESSION_NAME}"

