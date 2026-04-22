#!/bin/bash

#SBATCH --job-name=metaclip_b_16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=24:00:00

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate jepa
python -u jepa_score_fast.py --data_folder="ForenSynths" --csv_file_name="/ceph/tischuet/teja_results/vit_base_patch16_clip_224_metaclip_400m.csv" --batch_size 16 --model_name "vit_base_patch16_clip_224.metaclip_400m"