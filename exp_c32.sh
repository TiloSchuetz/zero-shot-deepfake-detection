#!/bin/bash

#SBATCH --job-name=cb32all
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate jepa

python jepa_score.py --config clip_b32_openai_all_layers_ForenSynths_val.yaml
python jepa_score.py --config clip_b32_openai_last_layer_spectral_Imagenet.yaml
python jepa_score.py --config clip_b32_openai_last_layer_spectral_ForenSynths.yaml
python jepa_score.py --config clip_b32_openai_last_layer_spectral_NewGenerators.yaml
python jepa_score.py --config clip_b32_openai_last_layer_spectral_New-Generator_COCO17_unbiased.yaml
python jepa_score.py --config clip_b32_openai_last_layer_spectral_New-Generator_RAISE1k_unbiased.yaml