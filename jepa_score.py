#!/usr/bin/env python3

import torch
import argparse
import yaml
import timm
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
import pandas as pd
import json
from datetime import datetime, timezone

from helpers import load_model, create_pd_data_paths, create_pd_data_paths_recursive, resume_or_create_dataset_dataloader_json, score_last_layer, calculate_metrics

DATASET_PATHS = {
    "ForenSynths": "/ceph/tischuet/replication_data/",
    "New-Generator": "/ceph/tischuet/replication_data/",
    "Imagenet_val": "/ceph/tischuet/",
    }

CONFIG_FOLDER_PATH = Path("./configs")

OUTPUT_FOLDER_PATH = Path("./outputs")

MODELS = {
    "vit_base_patch16_clip_quickgelu_224.metaclip_400m",
    "vit_base_patch32_clip_quickgelu_224.metaclip_400m",
    "vit_large_patch14_clip_quickgelu_224.metaclip_400m",
    "vit_huge_patch14_clip_quickgelu_224.metaclip_2pt5b",
    "vit_base_patch16_dinov3.lvd1689m",
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_large_patch14_dinov2.lvd142m",
    }

REQUIRED_KEYS = {"model_name", "fast_settings", "dataset", "batch_size", "profiler",
                 "vectorize", "score_type",}

SCORERS = {
    "last_layer": score_last_layer,
    "layerwise": None, #TODO implement drop in layer-wise JEPA-SCORE calcs with singular value spectrum
    "local_last_layer": None, #TODO
}

def main(args):

    # ----- 0. Load config yaml -----

    with open(CONFIG_FOLDER_PATH / args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # check if all necessary keys exist
    missing = REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    
    # ----- 1. check for CUDA -----

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"Using device: {device}")

        gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
        print(gpu_name)
    else: print("Execution on CPU. Might be slow.")

    # ----- 2. Load model & transforms -----
    model_name = config["model_name"]

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model name '{model_name}'.\n"
            f"  Available: {set(MODELS)}"
        )
    
    print(f"Model name: {model_name}")

    model = load_model(device=device, model_name=model_name)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    model.requires_grad_(False)

    if config["fast_settings"]:
        torch.set_float32_matmul_precision('high') # TODO lower precision?
        torch.backends.cuda.matmul.allow_tf32 = True # just to be safe on older systems
        torch.backends.cudnn.allow_tf32 = True # optimizes for cuDNN convolutions #TODO makes it slower?
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        torch.set_grad_enabled(True)
        print("Fast mode activated.")
    else:
        print("Standard mode activated.")

    # ----- 3. Create dataset & dataloader -----

    # 3.1 Checks if data folder is valid

    dataset = config["dataset"]
    dataset_path = DATASET_PATHS[dataset]
    
    if dataset == "Imagenet_val":
        filenames_labels = create_pd_data_paths(dataset_path, dataset)
    elif dataset in set(DATASET_PATHS.keys()) - {"Imagenet_val"}:
        filenames_labels = create_pd_data_paths_recursive(dataset_path, dataset)
    else:
        raise ValueError(
            f"Unknown data folder '{dataset}'.\n"
            f"  Available: {set(DATASET_PATHS.keys())}"
        )

    print(filenames_labels.label.value_counts())

    output_file = f"{model_name.replace('.', '_')}_{config['score_type']}_{dataset}.jsonl"
    full_output_path = OUTPUT_FOLDER_PATH / output_file

    # add metadata to json file
    if not full_output_path.exists():
        metadata = {
            "_meta": True,
            "config_file": args.config,
            "config": config,
            "model_name": model_name,
            "score_type": config["score_type"],
            "dataset": dataset,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        full_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_output_path, "w") as f:
            f.write(json.dumps(metadata) + "\n")


    data_loader = resume_or_create_dataset_dataloader_json(
                full_output_path,
                filenames_labels,
                dataset_path,
                dataset,
                transforms,
                config["batch_size"])

    # Build the profiler once, outside the loop
    if config["profiler"]:
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=5, active=3, repeat=1), # records batch 6,7,8
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs"),
            record_shapes=True,
        )
    else:
        profiler_ctx = nullcontext()

    # Listing 1: JEPA-SCORE implementation in PyTorch. Our empirical ablations demonstrate that JEPA-SCORE is not sensitive to the choice of eps (we pick 1​e−6)
    eps = 1e-6

    score_fn = SCORERS[config["score_type"]]

    with profiler_ctx as prof:
        for i, (img, label, img_path) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Computing JEPA scores"):
            img = img.to(device).requires_grad_(True)

            batch_results = score_fn(
                model,
                img,
                config["vectorize"],
                eps,
                config["fast_settings"],
                label,
                dataset_path,
                dataset,
                img_path,
            )

            batch_results.to_json(
                full_output_path,
                orient="records",
                lines=True,
                mode="a",
            )

            if config["profiler"]:
                prof.step()
    # TODO make it work for layer-wise
    calculate_metrics(full_output_path, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot generated image detection using "
            "JEPA-SCORE outlier detection"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        #default="",
        help="yaml config to specify experimentation settings",
    )

    args_ = parser.parse_args()
    main(args_)