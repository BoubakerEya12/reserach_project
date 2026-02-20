#!/bin/bash
#SBATCH --job-name=check_p1_gpu_cpu
#SBATCH --account=def-bselim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=check_p1_gpu_cpu_%j.out
#SBATCH --error=check_p1_gpu_cpu_%j.err

module load python/3.10
source ~/venv_mtl/bin/activate

cd ~/mtl_project/master1-main
export PYTHONPATH=$PWD

python -u -m scripts.check_p1_gpu_cpu
