#!/bin/bash
#SBATCH --job-name=fig5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=fig5_%j.out

module load python/3.10
source ~/venv_mtl/bin/activate
cd ~/mtl_project/master1-main
export PYTHONPATH=$PWD

python -m scripts.plot_fig5 --samples 100000 --save --no-show --verbose
