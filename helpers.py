#!/usr/bin/env python3
from torch.utils.data import DataLoader, Dataset
from torch.autograd.functional import jacobian
from torch.func import jacrev
from PIL import Image
import timm
import os
import pandas as pd
import torch
from typing import Tuple
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from statistics import mean


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file: pd.DataFrame, dataset_folder_path: str, img_dir: str, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.dataset_folder_path = dataset_folder_path
        self.img_dir = dataset_folder_path + img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

def load_model(device: str = "cuda", model_name: str = "") -> Tuple[torch.nn.Module, callable]:
    """Loads JEPA encoder model"""
    model = timm.create_model(model_name, pretrained=True, num_classes=0) # num_classes is important, as we want the input Jacobian w.r.t. to the embeddings and not output logits
    model.eval()

    model = model.to(device)

    return model

def create_pd_data_paths_New_Generators(dataset_folder_path: str, folder_name: str) -> pd.DataFrame:

    base_dir = os.path.join(dataset_folder_path, folder_name)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    paths = []
    labels = []
    
    for generator in os.listdir(base_dir):
        generator_path = os.path.join(base_dir, generator)
        if not os.path.isdir(generator_path):
            continue  # skip any stray files in base_dir
        label = 0.0 if generator == "real" else 1.0
        for file in os.listdir(generator_path):
            if Path(file).suffix.lower() in image_extensions:
                rel_path = os.path.join(generator, file)
                paths.append(rel_path)
                labels.append(label)
    
    paths_labels = pd.DataFrame({
        "img_path": paths,
        "label": labels
    })
    
    return paths_labels

def create_pd_data_paths_recursive(dataset_folder_path: str, folder_name: str) -> pd.DataFrame:
    """
    Recursively loads image filenames and labels from specified folder.
    Labels are derived from parent folder names: 0_real -> 0.0, 1_fake -> 1.0
    """
    base_dir = os.path.join(dataset_folder_path, folder_name)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    paths = []
    labels = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in sorted(files):
            if Path(file).suffix.lower() in image_extensions:
                # Get path relative to base_dir
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                
                # Walk up parent folders to find label folder
                label = None
                current = root
                while current != base_dir:
                    folder_name_part = os.path.basename(current)
                    if folder_name_part.startswith("0_real"):
                        label = 0.0
                        break
                    elif folder_name_part.startswith("1_fake"):
                        label = 1.0
                        break
                    current = os.path.dirname(current)
                
                paths.append(rel_path)
                labels.append(label)
    
    paths_labels = pd.DataFrame({
        "img_path": paths,
        "label": labels
    })
    
    return paths_labels

def create_pd_data_paths(dataset_folder_path: str, folder_name: str) -> pd.DataFrame:
    """
    Loads image filenames and labels from specified folder.

    Args:
        path: string with image folder path.

    Returns:
        dataloader: dataloader used for JEPA-SCORE calculation.
    """

    dir_list_test = os.listdir(dataset_folder_path+folder_name)

    paths_labels = pd.DataFrame({
        "img_path": dir_list_test,
        "label": [float(0) for _ in dir_list_test]
    })

    return paths_labels

def resume_or_create_dataset_dataloader(
        output_file: str,
        filenames_labels: pd.DataFrame,
        dataset_folder_path: str,
        data_folder: str,
        transforms,
        batch_size: int
        ) -> DataLoader[CustomImageDataset]:
    
    """
    Creates either a new or a resumed dataset and dataloader instance.

    Args:
        csv_file_name: string of filename of csv with computed JEPA-SCOREs.
        filenames_labels: pd.DataFrame with filenames and corresponding labels (0 or 1).
        dataset_folder_path: folder path of dataset root directory.
        data_folder: name of folder that contains relevant data.
        transforms: image transforms w.r.t. the pretrained encoder model.
        batch_size: batch size.

    Returns:
        CustomImageDataset: returns pandas dataframes with image filenames and labels.
    """

    file_path = Path(output_file)

    if file_path.exists():
        print("Load existing csv file with JEPA-SCOREs.")

        calculated = pd.read_csv(output_file)
        len_calculated = len(calculated)
        print(f"Already calculated {len_calculated} JEPA-SCORES")
        resumed_df = filenames_labels[~filenames_labels['img_path'].isin(calculated['img_path'])]

        test_dataset = CustomImageDataset(resumed_df, dataset_folder_path, data_folder, transform=transforms)

        data_loader = DataLoader(dataset = test_dataset,
                                batch_size = batch_size,
                                shuffle = False,
                                pin_memory = True, # pin_memory & num_workers seem to have an influence with higer batch numbers
                                num_workers = 4
                                )
        print(f"Batch size of dataloader: {batch_size}")
    else:
        print("No existing csv file found. Creating new one...")

        test_dataset = CustomImageDataset(filenames_labels, dataset_folder_path, data_folder, transform=transforms)

        data_loader = DataLoader(dataset = test_dataset,
                                batch_size = batch_size,
                                shuffle = False,
                                pin_memory = True,
                                num_workers = 4
                                )
        print(f"Batch size of dataloader: {batch_size}")

    return data_loader

def resume_or_create_dataset_dataloader_json(
        output_path: Path,
        filenames_labels: pd.DataFrame,
        dataset_folder_path: str,
        data_folder: str,
        transforms,
        batch_size: int
        ) -> DataLoader[CustomImageDataset]:
    """
    Creates either a new or a resumed dataset and dataloader instance.
    Assumes output_path exists and has a metadata line as line 1, followed by
    one JSON record per scored image.
    Args:
        csv_file_name: string of filename of jsonl with computed JEPA-SCOREs.
        filenames_labels: pd.DataFrame with filenames and corresponding labels (0 or 1).
        dataset_folder_path: folder path of dataset root directory.
        data_folder: name of folder that contains relevant data.
        transforms: image transforms w.r.t. the pretrained encoder model.
        batch_size: batch size.
    Returns:
        DataLoader: returns a dataloader for the (remaining) images.
    """

    assert output_path.exists(), f"Expected {output_path} to exist with metadata header."

    with open(output_path, "r") as f:
        next(f)  # skip metadata line. Assumes line 1 of an existing output file is the metadata header.
        already_done = {json.loads(line)["img_path"] for line in f if line.strip()}
    
    if already_done:
        print(f"Resuming: {len(already_done)} JEPA-SCORES already calculated.")
    else:
        print("Starting fresh run.")

    resumed_df = filenames_labels[~filenames_labels['img_path'].isin(already_done)]
    test_dataset = CustomImageDataset(resumed_df, dataset_folder_path, data_folder, transform=transforms)
    
    data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    print(f"Batch size of dataloader: {batch_size}")
    return data_loader
                
def score_last_layer(model, img, device, eps, label, dataset_path, dataset, img_path, config) -> pd.DataFrame:
    """
    Calculates singular value spectrum, JEPA-SCORE, AND embedding based on ∂f(x) / ∂x.
    The embedding is saved so the resulting JSONL can serve as a calibration set
    for conditional (local z-score) scoring at test time.
    Args:
        model: pretrained backbone.
        img: batch of input images [B, C, H, W].
        vectorize: flag if computation should be vectorized.
        eps: constant for JEPA-SCORE calculation stability.
        enabled: if bfloat16 should be enabled.
        label: labels of images in batch.
        dataset_path: directory path to dataset.
        dataset: dataset name.
        img_path: full directory path of image.
    Returns:
        batch results: pandas DataFrame with label, score, img_path, singular values spectrum and embedding for each image in batch.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["fast_settings"]):
        with torch.no_grad():
            emb = model(img)  # [B, D]
        J = jacobian(lambda x: model(x).sum(0), inputs=img, vectorize=config["vectorize"])
 
    with torch.inference_mode():
        J = J.flatten(2).permute(1, 0, 2).float()
        svdvals = torch.linalg.svdvals(J)
        log_sv = svdvals.clamp(min=eps).log()
        jepa_score = log_sv.sum(1) # [B]
 
    batch_results = pd.DataFrame({
        "label": label.cpu().tolist(),
        "score": jepa_score.cpu().tolist(),
        "img_path": [x.removeprefix(dataset_path + dataset + "/") for x in img_path],
        "svdvals": svdvals.cpu().tolist(),
        "embedding": emb.float().cpu().tolist(),
    })
    return batch_results

def load_calibration_set(cal_set_path: Path, device: str = "cuda"):
    """
    Loads a calibration JSONL produced by score_last_layer and returns:
      - emb_mat:  [N_cal, D] tensor of L2-normalized embeddings (for cosine sim)
      - scores:   [N_cal]    tensor of raw scalar scores (JEPA-SCOREs)
    Called once at startup; result is held in memory and passed to the scorer.
    """
    with open(cal_set_path, "r") as f:
        next(f)  # skip metadata header line
        df = pd.read_json(f, lines=True)
 
    emb_mat = torch.tensor(df["embedding"].tolist(), dtype=torch.float32, device=device)
    emb_mat = torch.nn.functional.normalize(emb_mat, dim=1)  # pre-normalize once
    scores = torch.tensor(df["score"].tolist(), dtype=torch.float32, device=device)
 
    print(f"Loaded calibration set: {emb_mat.shape[0]} images, embedding dim {emb_mat.shape[1]}")
    return emb_mat, scores

def conditional_score_last_layer(
    model, img, vectorize, eps, enabled, label, dataset_path, dataset, img_path,
    cal_emb_mat: torch.Tensor,   # [N_cal, D], pre-normalized
    cal_scores: torch.Tensor,    # [N_cal]
    k: int = 100,
) -> pd.DataFrame:
    """
    Computes the raw scalar score s(x) = sum(log svdvals) AND a conditional
    z-score against the k nearest calibration images (cosine sim in embedding space).
 
    Conditional z-score:
        z(x) = (s(x) - mean_kNN) / std_kNN
 
    Lower z (negative, large magnitude) means the encoder responds with less
    spectral mass than usual for similar real images -- a candidate fake signature.
    Sign convention here is "raw minus mean", same as a standard z-score; flip if you
    have a directional hypothesis.
    """

    #TODO implement hook to skip "duplicate" forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
        with torch.no_grad():
            emb = model(img)  # [B, D]
        J = jacobian(lambda x: model(x).sum(0), inputs=img, vectorize=vectorize)
 
    with torch.inference_mode():
        J = J.flatten(2).permute(1, 0, 2).float()
        svdvals = torch.linalg.svdvals(J)
        log_sv = svdvals.clamp(min=eps).log()
        jepa_score = log_sv.sum(1) # [B]
 
        # --- Conditional z-score against kNN in calibration set ---
        emb_norm = torch.nn.functional.normalize(emb.float(), dim=1)  # [B, D]
        sims = emb_norm @ cal_emb_mat.T                                # [B, N_cal]
        topk = torch.topk(sims, k=k, dim=1, largest=True).indices      # [B, k]
        neighbor_scores = cal_scores[topk]                             # [B, k]
 
        local_mean = neighbor_scores.mean(dim=1)                       # [B]
        local_std = neighbor_scores.std(dim=1).clamp_min(1e-8)         # [B], guard against zero
        z_score = (jepa_score - local_mean) / local_std                 # [B]
 
    batch_results = pd.DataFrame({
        "label": label.cpu().tolist(),
        "score": z_score.cpu().tolist(),              # primary score: conditional z
        "raw_score": jepa_score.cpu().tolist(),        # unconditional, for comparison
        "local_mean": local_mean.cpu().tolist(),      # diagnostic
        "local_std": local_std.cpu().tolist(),        # diagnostic
        "img_path": [x.removeprefix(dataset_path + dataset + "/") for x in img_path],
        "svdvals": svdvals.cpu().tolist(),
    })
    return batch_results

""" def score_all_layers(model, img, vectorize, device, eps, enabled, label, dataset_path, dataset, img_path) -> pd.DataFrame:
    # vectorize/enabled is never used. only included to make score calculation less cluttered
    B, C, H, W = img.shape
    num_layers = len(model.blocks)
    D = model.embed_dim
    I_D = torch.eye(D, device=device).unsqueeze(1)   # [D, 1, D]
    num_outputs = num_layers + 1                     # blocks + final post-norm feature
    layer_scores = {l: [] for l in range(num_outputs)}

    #print(model.num_prefix_tokens)  # should be 1 for standard CLS
    #print(model.cls_token)          # should not be None
    
    # ── Forward: capture per-block CLS + final model output ──
    cls_outs = []  # one [B, D] tensor per block
    
    def cls_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        cls_outs.append(output[:, 0])  # raw block output, no norm. output of each layer

    handles = [b.register_forward_hook(cls_hook) for b in model.blocks]
    try:
        feat = model(img)  # post-final-norm, equals model.norm(last_block_out[:,0])
    finally:
        for h in handles:
            h.remove()

    cls_outs.append(feat)  # final entry: paper-exact JEPA-SCORE
    #embeddings = torch.stack(cls_outs, dim=0).squeeze(1)

    # ── Backward: per-layer Jacobian via batched VJPs ─────────
    J = torch.empty(num_outputs, D, C, H, W, device=device)
    
    # only works for batch size of 1
    for l in range(num_outputs):
        grads = torch.autograd.grad(
            outputs=cls_outs[l],
            inputs=img,
            grad_outputs=I_D,                        # [D, 1, D] one-hots
            is_grads_batched=True,
            retain_graph=(l < num_outputs - 1),
        )[0]                                         # [D, 1, C, H, W]
        J[l] = grads.squeeze(1)
        
    # ── SVD-based score per layer ─────────────────────────────
    J_flat = J.reshape(num_outputs, D, -1).float()   # [L+1, D, C*H*W]
    svdvals = torch.linalg.svdvals(J_flat)           # [L+1, D]
    scores = svdvals.clamp_min(eps).log().sum(dim=1) # [L+1]    
    
    for l in range(num_outputs):
        layer_scores[l].append(scores[l])
    layer_scores = {l: torch.stack(v) for l, v in layer_scores.items()}

    # save labels, JEPA-SCOREs and image filenames in csv file
    batch_results = pd.DataFrame({
    "label": label.cpu().tolist(),
    **{f"layer_{l}": layer_scores[l].cpu().tolist() for l in range(num_outputs - 1)},
    "layer_norm": layer_scores[num_outputs - 1].cpu().tolist(),
    "img_path": [x.removeprefix(dataset_path + dataset + "/") for x in img_path]
    })

    return batch_results """

def score_all_layers(model, img, device, eps, label, dataset_path, dataset, img_path, config) -> pd.DataFrame:
    # vectorize/enabled is never used. only included to make score calculation less cluttered

    # ── Define layers to compute scores for ───────────────────────────────────
    num_layers  = len(model.blocks)
    num_outputs = num_layers + 1      # num_outputs - 1 is the post-norm layer
    layers = config["layers"] # all layers; edit to e.g. [0, 4, 8, num_layers]

    B, C, H, W = img.shape
    D          = model.embed_dim
    I_D        = torch.eye(D, device=device).unsqueeze(1).bfloat16()   # [D, 1, D]

    assert all(0 <= l < num_outputs for l in layers), \
        f"Layer indices must be in [0, {num_outputs - 1}]. {num_outputs - 1} is the post-norm layer."

    # ── Forward: capture per-block CLS + final model output ───────────────────
    cls_outs = {}  # keyed by layer index

    def cls_hook(module, input, output):
        idx = list(model.blocks).index(module)
        if idx in layers:
            if isinstance(output, tuple):
                output = output[0]
            cls_outs[idx] = output[:, 0]  # raw block output, no norm

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["fast_settings"]):
        handles = [b.register_forward_hook(cls_hook) for b in model.blocks]
        try:
            feat = model(img)  # post-final-norm, equals model.norm(last_block_out[:,0])
        finally:
            for h in handles:
                h.remove()

        if (num_outputs - 1) in layers:
            cls_outs[num_outputs - 1] = feat  # final entry: paper-exact JEPA-SCORE

        # ── Backward: per-layer Jacobian via batched VJPs ─────────────────────────
        J = torch.empty(len(layers), D, C, H, W, device=device, dtype=torch.bfloat16)

        for i, l in enumerate(layers):
            grads = torch.autograd.grad(
                outputs=cls_outs[l],
                inputs=img,
                grad_outputs=I_D,                        # [D, 1, D] one-hots
                is_grads_batched=True,
                retain_graph=(i < len(layers) - 1),
            )[0]                                         # [D, 1, C, H, W]
            J[i] = grads.squeeze(1)

    # ── SVD-based score per layer ──────────────────────────────────────────────
    J_flat  = J.reshape(len(layers), D, -1).float() # [len(layers), D, C*H*W]
    svdvals = torch.linalg.svdvals(J_flat)           # [len(layers), D]
    scores  = svdvals.clamp_min(eps).log().sum(dim=1) # [len(layers)]

    # ── DataFrame ─────────────────────────────────────────────────────────────
    layer_cols = {}
    for i, l in enumerate(layers):
        col = "layer_norm" if l == num_outputs - 1 else f"layer_{l}"
        layer_cols[col] = [scores[i].item()]

    batch_results = pd.DataFrame({
        "label"   : label.cpu().tolist(),
        **layer_cols,
        "img_path": [x.removeprefix(dataset_path + dataset + "/") for x in img_path],
    })
    return batch_results

# ── Module-level cache ─────────────────────────────────────────────────────────
# Prevents recompilation on every call to score_all_layers.
# Key: (model_id, layers_set, chunk_size, fast_settings)
_J_FN_CACHE: dict = {}


def _make_forward_fn(model, layers_set: frozenset, num_outputs: int):
    """
    Factory for a single-image forward function.

    Defined at module level (not inside score_all_layers) so torch.compile
    always sees the same stable callable — if it were a local function,
    Python would create a new object each call and trigger recompilation.

    Returns stacked CLS tokens [len(layers), D] for selected layers.
    Intermediate layers are pre-norm (raw block output).
    Final layer (num_outputs-1) is post-norm (paper-exact JEPA-SCORE).
    """
    # Unwrap compiled model if jepa_score.py already compiled it.
    # We need access to submodules directly for the manual forward.
    base_model = getattr(model, '_orig_mod', model)

    def forward_single(img_single: torch.Tensor) -> torch.Tensor:
        """
        img_single : [C, H, W]   — single image, no batch dim
        returns    : [L, D]      — stacked CLS tokens for selected layers
        """
        x = base_model.patch_embed(img_single.unsqueeze(0))
        x = base_model._pos_embed(x)   # handles CLS token, registers, pos embed
        x = base_model.patch_drop(x)   # no-op at eval
        x = base_model.norm_pre(x)     # usually Identity, but needed for correctness

        cls_list = []
        for i, block in enumerate(base_model.blocks):
            x = block(x)               # RoPE (DINOv3) is applied inside block
            if i in layers_set:
                cls_list.append(x[0, 0])   # CLS token, pre-norm

        x = base_model.norm(x)
        if (num_outputs - 1) in layers_set:
            cls_list.append(x[0, 0])   # post-norm, paper-exact JEPA-SCORE layer

        return torch.stack(cls_list)   # [L, D]

    return forward_single


def score_all_layers_fast(
    model, img, device, eps,
    label, dataset_path, dataset, img_path,
    config,
) -> pd.DataFrame:
    """
    Computes JEPA-SCORE for each selected layer using jacrev + CUDA graphs.

    Replaces the autograd.grad loop with torch.func.jacrev, which:
    - Expresses the Jacobian as a pure function torch.compile can trace
    - Enables CUDA graphs via mode="reduce-overhead"
    - Uses chunk_size to control memory vs speed tradeoff

    chunk_size tuning:
        chunk_size = D (e.g. 768 for ViT-B):
            Same number of backward passes as the old autograd.grad approach.
            Good default — compilable with no regression.
        chunk_size < D:
            More passes, less peak VRAM. Use if you OOM.
        chunk_size > D (up to L*D):
            Fewer passes, more peak VRAM. Use if you have headroom.
        chunk_size = None:
            Single giant vmap — fastest but almost certainly OOMs.

    First call is slow (compilation + CUDA graph recording).
    With your profiler schedule (wait=0, warmup=5, active=3), compilation
    happens during warmup batches, so the profiler window captures the fast path.

    Config keys used:
        layers        : list of layer indices to score
        fast_settings : bool — enables compile + bfloat16
        chunk_size    : int  — jacrev chunk size (default: model.embed_dim)
    """

    num_layers  = len(model.blocks)
    num_outputs = num_layers + 1
    layers      = config["layers"]
    layers_set  = frozenset(layers)
    fast        = config["fast_settings"]

    # Default chunk_size = D: same pass count as old autograd.grad approach
    # but now compilable. Override in config to tune memory.
    D          = model.embed_dim
    chunk_size = config.get("chunk_size", D)

    B, C, H, W = img.shape
    L          = len(layers)

    assert all(0 <= l < num_outputs for l in layers), \
        f"Layer indices must be in [0, {num_outputs - 1}]. " \
        f"{num_outputs - 1} is the post-norm layer."

    # ── Build / retrieve compiled Jacobian function ────────────────────────────
    base_model = getattr(model, '_orig_mod', model)
    cache_key  = (id(base_model), layers_set, chunk_size, fast)

    if cache_key not in _J_FN_CACHE:
        if fast:
            base_model.bfloat16()  # convolution_backward requires matching dtypes for input and weight

        fwd_fn = _make_forward_fn(model, layers_set, num_outputs)

        # Apply jacrev BEFORE torch.compile so Dynamo traces through both
        # the jacrev machinery AND the forward in one shot.
        # This lets CUDA graphs capture the full Jacobian computation,
        # not just the forward pass.
        J_fn = jacrev(fwd_fn, chunk_size=chunk_size)

        if fast:
            # reduce-overhead records a CUDA graph on the first call
            # and replays it on all subsequent calls.
            # dynamic=False: fixed shapes required (same img size every call).
            J_fn = torch.compile(J_fn, mode="reduce-overhead", dynamic=False)

        _J_FN_CACHE[cache_key] = J_fn

    J_fn = _J_FN_CACHE[cache_key]

    # ── Compute Jacobian ───────────────────────────────────────────────────────
    # jacrev manages its own gradient tracking internally.
    # Detach from the outer autograd graph to avoid interference.
    #
    # bfloat16 input: halves Jacobian memory during computation.
    # The dtype propagates through ops (weights are float32, so mixed precision
    # applies where supported). SVD is always done in float32 after casting.
    img_input = img.detach().bfloat16() if fast else img.detach()
    out_dtype  = torch.bfloat16 if fast else torch.float32

    J = torch.empty(B, L, D, C, H, W, device=device, dtype=out_dtype)

    for b in range(B):
        # J_fn: [C, H, W] → [L, D, C, H, W]
        # First call: slow (compilation + CUDA graph recording ~30-60s)
        # All subsequent calls: fast (CUDA graph replay)
        J[b] = J_fn(img_input[b])

    # ── SVD ────────────────────────────────────────────────────────────────────
    # Reshape to [B*L, D, C*H*W], cast to float32, force contiguous layout.
    # .contiguous() pays one copy upfront instead of triggering hidden copies
    # inside svdvals (which requires contiguous memory).
    J_flat  = J.reshape(B * L, D, C * H * W).float().contiguous()  # [B*L, D, C*H*W]
    svdvals = torch.linalg.svdvals(J_flat)                          # [B*L, D]
    scores  = svdvals.clamp_min(eps).log().sum(dim=1).reshape(B, L) # [B, L]

    # ── DataFrame ──────────────────────────────────────────────────────────────
    layer_cols = {}
    for i, l in enumerate(layers):
        col = "layer_norm" if l == num_outputs - 1 else f"layer_{l}"
        layer_cols[col] = scores[:, i].cpu().tolist()

    return pd.DataFrame({
        "label"   : label.cpu().tolist(),
        **layer_cols,
        "img_path": [p.removeprefix(dataset_path + dataset + "/") for p in img_path],
    })

def calculate_metrics(full_output_path, dataset) -> None:
    """
    Calculates AUC and AP for each class, overall (pooled), macro-average and weighted-average AUC/AP.
    Prints all metrics in a table.
    Args:
        full_output_path: path to output file
        dataset: dataset name
    Returns:
        None
    """

    CLASSES = {
        "ForenSynths": ["biggan", "crn", "cyclegan", "deepfake", "gaugan", "imle", "progan", "san", "seeingdark", "stargan", "stylegan", "stylegan2", "whichfaceisreal"],
        "ForenSynths_val": ["ForenSynths_val"],
        "NewGenerators": None, #TODO
        "GenImage": None, #TODO
    }

    with open(full_output_path, "r") as f:
        next(f)  # skip metadata line
        df = pd.read_json(f, lines=True)
    print(len(df))
    classes = CLASSES[dataset]
    AUCs, APs, counts = [], [], []
    for cls in classes:
        subset = df[df["img_path"].str.split("/").str[0] == cls].copy()
        AUCs.append(roc_auc_score(subset.label, subset.score))
        APs.append(average_precision_score(subset.label, subset.score))
        counts.append(len(subset))

    total = sum(counts)
    overall_auc = roc_auc_score(df.label, df.score)
    overall_ap = average_precision_score(df.label, df.score)
    macro_auc, macro_ap = mean(AUCs), mean(APs)
    #TODO implement for GANs only
    weighted_auc = sum(a * n for a, n in zip(AUCs, counts)) / total
    weighted_ap  = sum(a * n for a, n in zip(APs,  counts)) / total

    results = pd.DataFrame({
        "class": classes + ["Overall (pooled)", "Macro-average", "Weighted-average"],
        "N":     counts  + [total, None, total],
        "AUC":   AUCs    + [overall_auc, macro_auc, weighted_auc],
        "AP":    APs     + [overall_ap,  macro_ap,  weighted_ap],
    })

    print(f"\nResults on {dataset}")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))