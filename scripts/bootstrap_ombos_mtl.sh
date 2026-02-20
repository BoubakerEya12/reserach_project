#!/usr/bin/env bash
set -euo pipefail

# One-shot bootstrap for OMBOS:
# 1) clone code
# 2) fetch resume checkpoint from Narval
# 3) create venv and install deps
# 4) run GPU checks

HOST_NOW="$(hostname)"
if [[ "${HOST_NOW}" == narval* ]]; then
  echo "ERROR: You are on Narval (${HOST_NOW}). Run this on OMBOS."
  exit 1
fi

USER_LOGIN="${USER_LOGIN:-$USER}"
REPO_URL="${REPO_URL:-https://github.com/BoubakerEya12/Master_V1.git}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/mtl_project}"
PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT/master1-main}"
NARVAL_HOST="${NARVAL_HOST:-narval.alliancecan.ca}"
NARVAL_CHECKPOINT="${NARVAL_CHECKPOINT:-/home/eyabou12/mtl_project/master1-main/results/mtl_unsupervised.last.weights.h5}"

echo "== OMBOS Bootstrap =="
echo "Host           : ${HOST_NOW}"
echo "Project root   : ${PROJECT_ROOT}"
echo "Project dir    : ${PROJECT_DIR}"
echo "Repo URL       : ${REPO_URL}"
echo "Narval source  : ${NARVAL_HOST}:${NARVAL_CHECKPOINT}"

mkdir -p "${PROJECT_ROOT}"

if [[ -d "${PROJECT_DIR}/.git" ]]; then
  echo "Repo already present at ${PROJECT_DIR}, skipping clone."
else
  echo "Cloning repo..."
  git clone "${REPO_URL}" "${PROJECT_DIR}"
fi

cd "${PROJECT_DIR}"
mkdir -p results

if [[ -f "results/mtl_unsupervised.last.weights.h5" ]]; then
  echo "Checkpoint already exists locally, skipping copy."
else
  echo "Copying checkpoint from Narval (you may be prompted for password)..."
  scp "${USER_LOGIN}@${NARVAL_HOST}:${NARVAL_CHECKPOINT}" "results/mtl_unsupervised.last.weights.h5" || {
    echo "WARNING: Checkpoint copy failed."
    echo "You can retry manually with:"
    echo "scp ${USER_LOGIN}@${NARVAL_HOST}:${NARVAL_CHECKPOINT} results/mtl_unsupervised.last.weights.h5"
  }
fi

if [[ ! -d "venv_mtl" ]]; then
  echo "Creating virtual environment: venv_mtl"
  python3 -m venv venv_mtl
else
  echo "Virtual environment already exists: venv_mtl"
fi

source venv_mtl/bin/activate
python -m pip install --upgrade pip

if [[ -f "requirements.txt" ]]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found; installing baseline dependencies..."
  pip install numpy scipy matplotlib pandas tensorflow sionna
fi

echo
echo "Running GPU checks..."
bash scripts/check_ombos_gpu.sh

echo
echo "Bootstrap complete."
echo "Next commands:"
echo "  source venv_mtl/bin/activate"
echo "  bash scripts/run_mtl_train_ombos.sh"
echo "  bash scripts/run_mtl_eval_ombos.sh"
