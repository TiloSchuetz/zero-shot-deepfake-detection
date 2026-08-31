#!/bin/bash

#SBATCH --job-name=orbit_RAISE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=24:00:00

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate jepa

python jepa_score.py --config dino3_b_16_orbit_s032_RAISE1k.yaml
python analyze_orbit.py outputs/vit_base_patch16_dinov3_lvd1689m/vit_base_patch16_dinov3_lvd1689m_orbit_New-Generator_COCO17_unbiased.jsonl