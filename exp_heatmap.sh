#!/bin/bash

#SBATCH --job-name=heatmap
#SBATCH --output=slurm-%x.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=04:00:00

# Per-pixel sensitivity-energy heatmaps, diag(J^T J), for a small balanced
# subsample of real and fake images, in fp32.
#
# Memory: one Jacobian at a time, [D, C, H, W] ~ 620 MB fp32 for ViT-L, plus the
# batched-VJP workspace. Layers are looped, not stacked, so adding layers costs
# time and not VRAM.
#
# Override the defaults positionally:
#   sbatch exp_heatmap.sh <model_name> <dataset> <n_per_class> "<layers>" "<hi lo>"
#
# Examples:
#   sbatch exp_heatmap.sh                                    # DINOv3-L/16, last layer
#   sbatch --job-name=hm_cL14_18-12 exp_heatmap.sh \
#       vit_large_patch14_clip_quickgelu_224.openai \
#       New-Generator_COCO17_unbiased 8 "12 18" "18 12"

set -euo pipefail

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate jepa

MODEL=${1:-vit_large_patch16_dinov3.lvd1689m}
DATASET=${2:-New-Generator_COCO17_unbiased}
N_PER_CLASS=${3:-8}
LAYERS=${4:--1}          # -1 is the post-norm output (paper-exact last layer)
LAYER_DIFF=${5:-}        # e.g. "18 12"; empty disables the difference maps

echo "model=$MODEL  dataset=$DATASET  n_per_class=$N_PER_CLASS"
echo "layers=$LAYERS  layer_diff=${LAYER_DIFF:-<none>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ARGS=(
    --model_name "$MODEL"
    --dataset "$DATASET"
    --n_per_class "$N_PER_CLASS"
    --layers $LAYERS
    --seed 0
    --patch_pool
    --score
    --out_dir ./outputs_heatmaps
)
if [[ -n "$LAYER_DIFF" ]]; then
    ARGS+=(--layer_diff $LAYER_DIFF)
fi

python sensitivity_energy.py "${ARGS[@]}"
