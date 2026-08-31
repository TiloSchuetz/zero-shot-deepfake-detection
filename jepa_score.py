#!/usr/bin/env python3

import torch
import argparse
import yaml
import timm
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
import json
from datetime import datetime, timezone

from helpers import (
    load_model,
    create_pd_data_paths,
    create_pd_data_paths_recursive,
    create_pd_data_paths_New_Generators,
    resume_or_create_dataset_dataloader_json,
    score_last_layer, calculate_metrics,
    load_calibration_set,
    conditional_score_last_layer,
    score_all_layers,
    score_all_layers_fast,
    score_last_layer_spectral,
    score_orbit,
    resume_or_create_orbit_dataloader_json)

from orbit_aug import build_orbit_transforms

#TODO simplify
DATASET_PATHS = {
    "ForenSynths": "/ceph/tischuet/replication_data/",
    "New-Generator": "/ceph/tischuet/replication_data/",
    "Imagenet_val_5k": "/ceph/tischuet/replication_data/",
    "ForenSynths_val": "/ceph/tischuet/replication_data/",
    "ForenSynths_rest": "/ceph/tischuet/replication_data/",
    "New-Generator_COCO17_unbiased": "/ceph/tischuet/replication_data/",
    "New-Generator_RAISE1k_unbiased": "/ceph/tischuet/replication_data/",
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
    "vit_base_patch16_clip_quickgelu_224.openai", # vit_base_patch16_clip_224.openai would use clip with GELU, which is not how it was trained by OpenAI
    "vit_base_patch32_clip_quickgelu_224.openai",
    "vit_large_patch14_clip_quickgelu_224.openai",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    }

REQUIRED_KEYS = {"model_name", "fast_settings", "dataset", "batch_size", "profiler",
                 "vectorize", "score_type", "cal_set_path", "k_neighbors"}

ORBIT_KEYS = {"m_views", "orbit_seed", "global_crops_scale"}

SCORERS = {
    "last_layer": score_last_layer,
    "last_layer_spectral": score_last_layer_spectral,
    "layerwise": score_all_layers, #TODO implement drop in layer-wise JEPA-SCORE calcs with singular value spectrum
    "layerwise_fast": score_all_layers_fast,
    "local_last_layer": conditional_score_last_layer,
    "orbit": score_orbit,
}

def main(args):

    # ----- 0. Load config yaml -----

    with open(CONFIG_FOLDER_PATH / args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # check if all necessary keys exist TODO make dependend on SCORERS
    if config["score_type"] in {"last_layer", "layerwise", "layerwise_fast", "last_layer_spectral"}:
        missing = REQUIRED_KEYS - {"cal_set_path", "k_neighbors"} - config.keys()
        if missing:
            raise ValueError(f"Config is missing required keys: {missing}")
    elif config["score_type"] == "local_last_layer":
        missing = REQUIRED_KEYS - config.keys()
        if missing:
            raise ValueError(f"Config is missing required keys: {missing}")
    elif config["score_type"] == "orbit":
        missing = (REQUIRED_KEYS - {"cal_set_path", "k_neighbors"} | ORBIT_KEYS) - config.keys()
        if missing:
            raise ValueError(f"Config is missing required keys: {missing}")
    else: raise ValueError(f"This score_type doesn't exist")
    
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

    # ----- 2.1 Pre-bake RoPE embedding to avoid cudagraph break (DINOv3) -----
    if "dinov3" in model_name and device == "cuda":
        img_size = data_config["input_size"][-1]
        patch_size = model.patch_embed.patch_size[0]
        H = W = img_size // patch_size

        with torch.no_grad():
            cached_embed = model.rope.get_embed(shape=(H, W)).to(device)

        # Register as a buffer so it moves with the module and Dynamo treats it as static
        model.rope.register_buffer("_cached_embed", cached_embed, persistent=False)

        def _patched_get_embed(self, shape=None):
            return self._cached_embed

        # Bind as a method on the instance, no closure over external state
        import types
        model.rope.get_embed = types.MethodType(_patched_get_embed, model.rope)


    if config["fast_settings"]:
        torch.set_float32_matmul_precision('high') # TODO lower precision?
        torch.backends.cuda.matmul.allow_tf32 = True # just to be safe on older systems
        torch.backends.cudnn.allow_tf32 = True # optimizes for cuDNN convolutions #TODO makes it slower?
        if config["score_type"] != "layerwise":
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        torch.set_grad_enabled(True)
        print("Fast mode activated.")
    else:
        print("Standard mode activated.")

    # ----- 3. Create dataset & dataloader -----

    # load calibration dataset, if needed
    cal_emb_mat, cal_scores = None, None
    if config["score_type"] == "local_last_layer":
        cal_path = Path(config["cal_set_path"])
        if not cal_path.exists():
            raise FileNotFoundError(
                f"Calibration set not found at {cal_path}. "
                f"Run the pipeline with score_type='last_layer' on your calibration "
                f"dataset (e.g. Imagenet_val) first to produce it."
            )
        cal_emb_mat, cal_scores = load_calibration_set(cal_path, device=device)


    # 3.1 Checks if data folder is valid

    dataset = config["dataset"]
    dataset_path = DATASET_PATHS[dataset]
    
    # TODO this is ugly
    if dataset == "Imagenet_val_5k":
        filenames_labels = create_pd_data_paths(dataset_path, dataset)
    elif dataset in set(DATASET_PATHS.keys()) - {"Imagenet_val", "New-Generator", "New-Generator_COCO17_unbiased", "New-Generator_RAISE1k_unbiased"}:
        filenames_labels = create_pd_data_paths_recursive(dataset_path, dataset)
    elif dataset in {"New-Generator", "New-Generator_COCO17_unbiased", "New-Generator_RAISE1k_unbiased"}:
        filenames_labels = create_pd_data_paths_New_Generators(dataset_path, dataset)
    else:
        raise ValueError(
            f"Unknown data folder '{dataset}'.\n"
            f"  Available: {set(DATASET_PATHS.keys())}"
        )

    # Optional seeded per-generator subsample. Orbit scoring costs (1+m) Jacobians
    # per image, so first-pass AUC estimates are taken on a subset.
    if config.get("subset_per_class"):
        k = config["subset_per_class"]
        seed = config.get("orbit_seed", 0)
        # Group on (generator, label), not generator alone. For New-Generator the
        # top-level folder already determines the label, but ForenSynths nests both
        # classes under one generator (stylegan/car/{0_real,1_fake}) -- grouping on
        # the folder only would draw k images of mixed label per generator.
        generator = filenames_labels["img_path"].str.split("/").str[0]
        filenames_labels = (
            filenames_labels
            .groupby([generator, filenames_labels["label"]], group_keys=False)
            .apply(lambda g: g.sample(min(k, len(g)), random_state=seed))
        )
        print(f"Subsampled to {k} images per (generator, label) (seed={seed}).")

    print(filenames_labels.label.value_counts())

    model_folder = OUTPUT_FOLDER_PATH / model_name.replace('.', '_')
    model_folder.mkdir(parents=True, exist_ok=True)

    output_file = f"{model_name.replace('.', '_')}_{config['score_type']}_{dataset}.jsonl"
    full_output_path = model_folder / output_file

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


    if config["score_type"] == "orbit":
        t0, t1, t2 = build_orbit_transforms(
            img_size=data_config["input_size"][-1],
            mean=data_config["mean"],
            std=data_config["std"],
            global_crops_scale=config["global_crops_scale"],
        )
        print(f"Orbit augmentations: DINOv3 global_transfo1/2, "
              f"scale={tuple(config['global_crops_scale'])}, m={config['m_views']}")
        data_loader = resume_or_create_orbit_dataloader_json(
                    full_output_path,
                    filenames_labels,
                    dataset_path,
                    dataset,
                    t0, t1, t2,
                    m=config["m_views"],
                    seed=config["orbit_seed"])
    else:
        data_loader = resume_or_create_dataset_dataloader_json(
                    full_output_path,
                    filenames_labels,
                    dataset_path,
                    dataset,
                    transforms,
                    config["batch_size"])

    # Build the profiler once, outside the loop
    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        #torch.profiler.tensorboard_trace_handler("./tb_logs")(prof)

    if config["profiler"]:
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=5, active=3, repeat=1), # records batch 6,7,8
            on_trace_ready=trace_handler,
            record_shapes=False, # if True OOM for profiler
        )
    else:
        profiler_ctx = nullcontext()

    # Listing 1: JEPA-SCORE implementation in PyTorch. Our empirical ablations demonstrate that JEPA-SCORE is not sensitive to the choice of eps (we pick 1​e−6)
    eps = 1e-6

    score_fn = SCORERS[config["score_type"]]

    with profiler_ctx as prof:
        for i, (img, label, img_path) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Computing JEPA scores"):
            if config["score_type"] == "orbit":
                # [1, 1+m, C, H, W] -> [1+m, C, H, W]: the orbit is the batch, and
                # this shape is constant for every step (one CUDA graph capture).
                img = img.squeeze(0)
            img = img.to(device).requires_grad_(True)

            if config["score_type"] == "local_last_layer":
                batch_results = score_fn(
                    model, img, config["vectorize"], eps, config["fast_settings"],
                    label, dataset_path, dataset, img_path,
                    cal_emb_mat=cal_emb_mat,
                    cal_scores=cal_scores,
                    k=config["k_neighbors"],
                )
            else:
                batch_results = score_fn(
                    model, img, device, eps,
                    label, dataset_path, dataset, img_path,
                    config,
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
    if dataset != "Imagenet_val_5k" and config["score_type"] != "layerwise":
        calculate_metrics(full_output_path, dataset)
    else: print("No metrics calculated. Calibration set has no fakes. / Layerwise aggregation not yet implemented for final evaluation.")

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