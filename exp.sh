#!/bin/bash

#SBATCH --job-name=mc_l14
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=48:00:00

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate jepa
python jepa_score.py --config mc_l_14_last_layer_ForenSynths.yaml
