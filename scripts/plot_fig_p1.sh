#!/bin/bash 
#SBATCH --job-name=p1_fig5_10k 
#SBATCH --account=def-bselim_gpu 
#SBATCH --gpus=a100:1 
#SBATCH --cpus-per-task=64 
#SBATCH --mem=64G 
#SBATCH --time=2:00:00 
module load python/3.10 
source ~/venv_mtl/bin/activate 

cd ~/mtl_project/master1-main 
export PYTHONPATH=$PWD 

OUTDIR=figs/fig5_10k
mkdir -p $OUTDIR

python -u -m scripts.plot_fig5_2 \
  --mode both \
  --samples 100 \
  --snr-max 30 \
  --snr-step 6 \
  --outdir $OUTDIR \
  --save \
  --no-show \
  --verbose