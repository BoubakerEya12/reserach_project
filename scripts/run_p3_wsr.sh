#!/bin/bash
#SBATCH --job-name=p3_wsr
#SBATCH --account=def-bselim
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=144:00:00
#SBATCH --output=p3_%j.out
#SBATCH --error=p3_%j.err

module load python/3.10
source ~/venv_mtl/bin/activate

cd ~/mtl_project/master1-main
export PYTHONPATH=$PWD

OUTDIR=$SCRATCH/results_p3_wmmse
mkdir -p "$OUTDIR"

python -u -m scripts.plot_p3_wmmse_only \
  --snr_min 0 \
  --snr_max 30 \
  --snr_step 5 \
  --n_slots 10000 \
  --seed 1234 \
  --out_csv "$OUTDIR/p3_wmmse_vs_snr.csv" \
  --out_fig "$OUTDIR/p3_wmmse_vs_snr.png"
