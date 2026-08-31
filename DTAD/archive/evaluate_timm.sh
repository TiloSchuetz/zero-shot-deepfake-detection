#!/bin/bash

#SBATCH --job-name=cB32FS
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate jepa

python evaluate_timm.py \
    --data_root /ceph/tischuet/replication_data \
    --dataset ForenSynths \
    --denoising_output_root /ceph/tischuet/DTAD_output_root \
    --model_name "vit_base_patch32_clip_224.openai"