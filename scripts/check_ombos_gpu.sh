#!/usr/bin/env bash
set -euo pipefail

echo "== Environment Check =="
HOST="$(hostname)"
echo "Host: ${HOST}"
echo "PWD : $(pwd)"

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
python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
print("GPU count:", len(gpus))
PY

echo
echo "Check complete."

