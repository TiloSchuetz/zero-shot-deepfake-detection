#!/usr/bin/env python3

import os

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, Dataset
from torch.func import jacrev, vmap
import torch._inductor.config as inductor_config
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch._logging
from PIL import Image
import logging

import argparse
import timm
import pandas as pd
import pickle

from typing import Tuple
from pathlib import Path
from tqdm import tqdm
import time

from helpers import load_model, create_pd_data_paths, create_pd_data_paths_recursive, resume_or_create_dataset_dataloader

dataset_folder_path = "/ceph/tischuet/replication_data/"


def main(args):
    # ----- 1. check for CUDA -----

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    """ torch._logging.set_logs(
    recompiles=True,
    graph_breaks=True,
    cudagraphs=True,
    ) """


    # ----- 2. Load model & transforms -----

    print(torch.cuda.get_device_name(0))
    print(f"Model name: {args.model_name}")

    # config for model compilation
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True # just to be safe on older systems
    torch.backends.cudnn.allow_tf32 = True # optimizes for cuDNN convolutions #TODO makes it slower?
    inductor_config.compile_threads = 16 # more threads for model compilation

    # enable compilation logging

    model = load_model(device=device, model_name=args.model_name)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    model.requires_grad_(False)
    #model = torch.compile(model, mode="default")  # fuses the entire forward & backward pass into one CUDA kernel

    def f_single(x):
        return model(x.unsqueeze(0)).squeeze(0)

    jac_fn = jacrev(f_single, chunk_size=1)
    #jac_fn = torch.compile(jac_fn, mode="default")

    torch.set_grad_enabled(True)

    # ----- 3. Create dataset & dataloader -----

    filenames_labels = create_pd_data_paths_recursive(dataset_folder_path, args.data_folder)

    print(filenames_labels.label.value_counts())

    data_loader = resume_or_create_dataset_dataloader(
                args.csv_file_name,
                filenames_labels,
                dataset_folder_path,
                args.data_folder,
                transforms,
                args.batch_size)

    # ----- 4. Calculate JEPA-SCORE -----

    # Listing 1: JEPA-SCORE implementation in PyTorch. Our empirical ablations demonstrate that JEPA-SCORE is not sensitive to the choice of eps (we pick 1​e−6)
    eps = 1e-6

    print(f"JEPA-SCORE csv filename: {args.csv_file_name}")

    write_header = not os.path.exists(args.csv_file_name)

    #torch._logging.set_logs(recompiles=True)

    for i, (img, label, img_path) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Computing JEPA scores"):
        img = img.to(device).requires_grad_(True)    # shape: [batch size, C, H, W]
        #model.zero_grad(set_to_none=True)
        # label = label.item()    # shape: [1, batch size]; 0 = "real"; 1 = "fake"
        # img_path                # shape: (batch size)

        # Compute Jacobian for single image
        def f_single(x):
            return model(x.unsqueeze(0)).squeeze(0).float()

        with sdpa_kernel([SDPBackend.MATH]):
            J = jac_fn(img[0])  # match autocast dtype
    

        with torch.inference_mode():
            J = J.flatten(2).permute(1,0,2)
            svdvals = torch.linalg.svdvals(J)
            jepa_score = svdvals.clip_(eps).log_().sum(1) # one score per image in batch; clip with eps stabilizes calculations

        del J, svdvals, img

        # save labels, JEPA-SCOREs and image filenames in csv file
        batch_results = pd.DataFrame({
            "label": [label[0].item()],
            "score": [jepa_score[0].item()],
            "img_path": [img_path[0].removeprefix(dataset_folder_path + args.data_folder + "/")]
        })

        """ def f_single(x):
            return model(x.unsqueeze(0)).squeeze(0)

        # outer vmap batches over images, inner jacrev computes per-image Jacobian
        compute_jacobian_batched = vmap(jacrev(f_single, chunk_size=1))

        # In the loop — img has shape [batch_size, 3, 224, 224]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            J = compute_jacobian_batched(img)  # shape: [batch_size, 768, 3, 224, 224]

        with torch.inference_mode():
            J = J.flatten(2)  # [batch_size, 768, 150528]
            svdvals = torch.linalg.svdvals(J)  # [batch_size, 768]
            jepa_score = svdvals.clip_(eps).log_().sum(1)  # [batch_size]

        del J, svdvals, img

        batch_results = pd.DataFrame({
            "label": label.cpu().tolist(),
            "score": jepa_score.cpu().tolist(),
            "img_path": [x.removeprefix(dataset_folder_path + args.data_folder + "/") for x in img_path]
        }) """

        # adds the new results to the end of the csv file
        batch_results.to_csv(
            args.csv_file_name, 
            mode="a",
            header=write_header,
            index=False
        )
        write_header = False

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot generated image detection using "
            "JEPA-SCORE outlier detection"
        )
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="vit_base_patch16_clip_quickgelu_224.metaclip_400m",
        help="JEPA encoder model name",
    )

    parser.add_argument(
        "--data_folder",
        type=str,
        default="all",
        help="Folder name of image data. Default is a folder with all images",
    )

    parser.add_argument(
        "--csv_file_name",
        type=str,
        default="jepa_scores.csv",
        help="Name for csv file that stores labels, computed JEPA-SCOREs and file paths",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing images.",
    )

    args_ = parser.parse_args()
    main(args_)