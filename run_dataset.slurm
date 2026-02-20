#!/bin/bash
#SBATCH --job-name=mtl_dataset
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=log_%j.out

module load python/3.10
source ~/venv_mtl/bin/activate
cd ~/mtl_project/master1-main
export PYTHONPATH=$PWD
python -m scripts.plot_fig5 --samples 100000 --save --no-show --verbose

