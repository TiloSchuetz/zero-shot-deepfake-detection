#!/bin/bash

#SBATCH --job-name=c32NG
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb

source /work/tischuet/miniconda3/etc/profile.d/conda.sh
conda activate DTAD

python evaluate.py \
    --data_root /ceph/tischuet/replication_data \
    --dataset ForenSynths \
    --denoising_output_root /ceph/tischuet/DTAD_output_root