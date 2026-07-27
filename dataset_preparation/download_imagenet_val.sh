#!/bin/bash

#SBATCH --job-name=imagenet
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=cpu
#SBATCH --time=01:00:00

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate jepa
python download_imagenet_val.py