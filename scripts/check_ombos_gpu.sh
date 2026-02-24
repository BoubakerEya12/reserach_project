#!/usr/bin/env bash
set -euo pipefail

echo "== Environment Check =="
HOST="$(hostname)"
echo "Host: ${HOST}"
echo "PWD : $(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${PROJECT_DIR}/venv_mtl/bin/python" ]]; then
  PYTHON_CMD="${PROJECT_DIR}/venv_mtl/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python3)"
else
  PYTHON_CMD="$(command -v python)"
fi
echo "Python: ${PYTHON_CMD}"

if [[ "${HOST}" == narval* ]]; then
  echo "WARNING: You appear to be on Narval (${HOST}), not OMBOS."
  echo "Long training should run on OMBOS or scheduled compute nodes."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "== nvidia-smi =="
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found."
fi

echo
echo "== TensorFlow GPU devices =="
"${PYTHON_CMD}" - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
print("GPU count:", len(gpus))
PY

echo
echo "Check complete."
