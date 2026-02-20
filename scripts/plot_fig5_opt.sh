#!/bin/bash
#SBATCH --job-name=p1_opt_s_l
#SBATCH --account=def-bselim
#SBATCH --time=144:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "=== JOB INFO ==="
echo "JobID: $SLURM_JOB_ID"
echo "Node : $(hostname)"
echo "Start: $(date)"
echo "==============="

# 1) Go to project
cd /home/eyabou12/mtl_project/master1-main

# 2) Activate python env
source ~/venv_mtl/bin/activate

# 3) Quick sanity check (avoids numpy issue)
which python
python -c "import numpy as np; import matplotlib; print('numpy OK:', np.__version__)"

# 4) Run SMALL
OUT_SMALL="figs/fig5_opt_small_${SLURM_JOB_ID}"
mkdir -p "$OUT_SMALL"

python -m scripts.plot_fig5_opt \
  --mode small \
  --samples 10000 \
  --save \
  --outdir "$OUT_SMALL" \
  --no-show \
  --verbose

echo "[DONE] small saved to: $OUT_SMALL"

# 5) Run LARGE
OUT_LARGE="figs/fig5_opt_large_${SLURM_JOB_ID}"
mkdir -p "$OUT_LARGE"

python -m scripts.plot_fig5_opt \
  --mode large \
  --samples 10000 \
  --save \
  --outdir "$OUT_LARGE" \
  --no-show \
  --verbose

echo "[DONE] large saved to: $OUT_LARGE"

echo "End: $(date)"
