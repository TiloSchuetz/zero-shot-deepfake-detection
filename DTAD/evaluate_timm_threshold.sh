#!/bin/bash

#SBATCH --job-name=d3_SG2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate jepa

python evaluate_timm_threshold.py \
    --data_root /ceph/tischuet/replication_data \
    --dataset 'ForenSynths_3' \
    --denoising_output_root /ceph/tischuet/DTAD_output_root \
    --threshold_dir ./thresholds \
    --model_name "vit_large_patch16_dinov3.lvd1689m" \
    --threshold_real_folder "Imagenet_val_5k" \
    --transform resize


# "vit_base_patch32_clip_quickgelu_224.openai"
# "vit_large_patch14_clip_quickgelu_224.openai"
# "vit_large_patch16_dinov3.lvd1689m"

# 'New-Generator_RAISE1k_unbiased'
# 'New-Generator_COCO2017_unbiased'