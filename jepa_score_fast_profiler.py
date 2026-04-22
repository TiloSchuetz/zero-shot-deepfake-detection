#!/usr/bin/env python3

import os

import torch
from torch.autograd.functional import jacobian
import torch._inductor.config as inductor_config
import torch._logging
from torch.profiler import profile, record_function, ProfilerActivity


import argparse
import timm
import pandas as pd
import json

from tqdm import tqdm
from helpers import load_model, create_pd_data_paths, create_pd_data_paths_recursive, resume_or_create_dataset_dataloader

dataset_folder_path = "/ceph/tischuet/replication_data/"


def main(args):
    # ----- 1. check for CUDA -----

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----- 2. Load model & transforms -----

    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    print(gpu_name)
    """ safe_model = args.model_name.replace("/", "_").replace(".", "_")
    CACHE_FILE = Path(f"compile_cache_{safe_model}_{gpu_name}.bin")

    if CACHE_FILE.exists():
        artifact_bytes = CACHE_FILE.read_bytes()
        torch.compiler.load_cache_artifacts(artifact_bytes)
        print("Loaded compile cache from disk") """

    print(f"Model name: {args.model_name}")

    # config for model compilation
    torch.set_float32_matmul_precision('high') # TODO lower precision?
    torch.backends.cuda.matmul.allow_tf32 = True # just to be safe on older systems
    torch.backends.cudnn.allow_tf32 = True # optimizes for cuDNN convolutions #TODO makes it slower?
    inductor_config.compile_threads = 16 # more threads for model compilation

    model = load_model(device=device, model_name=args.model_name)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    model.requires_grad_(False)
    #model = torch.compile(model, mode="reduce-overhead", dynamic=False)
    torch.set_grad_enabled(True)

    # warmup
    """ print("Warming up...")
    for i in range(3):
        dummy_img = torch.randn(1, 3, 224, 224).to(device).requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            J = jacobian(lambda x: model(x).sum(0), inputs=dummy_img)
        del J, dummy_img
    print("Done.") """

    # save cache after first successful compilation
    """ if not CACHE_FILE.exists():
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is not None:
            artifact_bytes, cache_info = artifacts
            CACHE_FILE.write_bytes(artifact_bytes)
            print(f"Saved compile cache ({len(artifact_bytes) / 1e6:.1f} MB)") """

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

    """ metadata = {"GPU name": gpu_name, # save metadata
                "Model name": args.model_name,
                "Dataset": args.folder_name,
                "Batch size": args.batch_size} """

    """ if "val" in args.csv_file_name:
        folder_name = "val_jepa_score_csv/" # folder for validation JEPA-SCOREs
    elif "test" in args.csv_file_name:
        folder_name = "test_jepa_score_csv/" # folder for testing JEPA-SCOREs
    else:
        print("Warning: Wrong csv file naming scheme")
        folder_name = "" """
    
    write_header = not os.path.exists(args.csv_file_name)

    # checks if file exists and the metadata matches. Otherwise, it will create a new file
    """ if os.path.exists(args.csv_file_name):
        with open(args.csv_file_name, "r") as f:
            first_line = json.loads(f.readline())
        existing_metadata = first_line.get("_metadata", {})
        if existing_metadata != metadata:
            raise ValueError(
                f"Metadata mismatch! Existing file was created with different settings.\n"
                f"  Existing: {existing_metadata}\n"
                f"  Current:  {metadata}"
            )
        print(f"Resuming — metadata matches.")
    else:
        with open(args.csv_file_name, "w") as f:
            f.write(json.dumps({"_metadata": metadata}) + "\n")
        print(f"Created new output file: {args.csv_file_name}") """



    for i, (img, label, img_path) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Computing JEPA scores"):
        img = img.to(device).requires_grad_(True)    # shape: [batch size, C, H, W]
        #model.zero_grad(set_to_none=True)
        # label = label.item()    # shape: [1, batch size]; 0 = "real"; 1 = "fake"
        # img_path                # shape: (batch size)

        # Compute Jacobian for single image

        J = jacobian(lambda x: model(x).sum(0), inputs=img, vectorize=True)
        """ with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                J = jacobian(lambda x: model(x).sum(0), inputs=img) #, vectorize=True) # vectorization saves some time per image and makes a difference at the second decimal place
        print("Loop finished, printing profiler...", flush=True) # takes a long time to reach this after finishing training loop
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10), flush=True) """

        with torch.inference_mode():
            J = J.flatten(2).permute(1,0,2)
            svdvals = torch.linalg.svdvals(J)
            jepa_score = svdvals.clip_(eps).log_().sum(1) # one score per image in batch; clip with eps stabilizes calculations

        # save labels, JEPA-SCOREs and image filenames in csv file
        batch_results = pd.DataFrame({
            "label": label.cpu().tolist(),
            "score": jepa_score.cpu().tolist(),
            "img_path": [x.removeprefix(dataset_folder_path + args.data_folder + "/") for x in img_path] # remove prefix to only save filename
        })

        # adds the new results to the end of the csv file
        batch_results.to_csv(
            args.csv_file_name, 
            mode="a",
            header=write_header,
            index=False
        )
        write_header = False

        if i >= 10:
            break

    
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