#!/bin/bash
#SBATCH --job-name=p3_gpu
#SBATCH --account=def-bselim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=p3_gpu_%j.out
#SBATCH --error=p3_gpu_%j.err

module load python/3.10
source ~/venv_mtl/bin/activate
cd ~/mtl_project/master1-main
export PYTHONPATH=$PWD

python -m scripts.plot_p3 \
  --snr_min 0 --snr_max 30 --snr_step 5 \
  --n_slots 10000 --chunk_size 512 --seed 1234 \
  --out_csv $SCRATCH/results_p3/p3_wsr_vs_snr_gpu.csv \
  --out_fig $SCRATCH/results_p3/p3_wsr_vs_snr_gpu.png
