#!/bin/bash

#SBATCH --job-name=dino3spectral
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-12gb
#SBATCH --time=01:00:00

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate jepa

python jepa_score.py --config dino3_l_16_last_layer_spectral_ForenSynths.yaml
