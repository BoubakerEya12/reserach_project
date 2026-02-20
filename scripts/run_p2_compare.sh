#!/bin/bash
#SBATCH --job-name=p2_compare
#SBATCH --account=def-bselim
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=p2_%j.out
#SBATCH --error=p2_%j.err

module load python/3.10
source ~/venv_mtl/bin/activate

cd ~/mtl_project/master1-main
export PYTHONPATH=$PWD

OUTDIR=$SCRATCH/results_p2_compare
mkdir -p "$OUTDIR"

python -u -m scripts.plot_p2_optimal \
  --N_mc 2000 \
  --gammas "-5,0,5,10,15,20" \
  --max_iter 10000 \
  --tol 1e-5 \
  --outdir "$OUTDIR"