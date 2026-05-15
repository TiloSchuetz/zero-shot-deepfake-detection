#!/usr/bin/env python3
from torch.utils.data import DataLoader, Dataset
from torch.autograd.functional import jacobian
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
        "label": [
            float(0) if name.startswith("0_real") # only works for "all" & "MS_COCO_val2017"
            else float(1) if name.startswith("1_fake")
            else None
            for name in dir_list_test
        ]
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

def score_last_layer(model, img, vectorize, eps, enabled, label, dataset_path, dataset, img_path) -> pd.DataFrame:
    """
    Calculates singular value spectrum and final JEPA-SCORE based ∂f(x) / ∂x.
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
        batch results: pandas DataFrame with label, score, img_path and singular values spectrum for each image in batch.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
        J = jacobian(lambda x: model(x).sum(0), inputs=img, vectorize=vectorize)
    with torch.inference_mode():
        J = J.flatten(2).permute(1, 0, 2).float() # .float() decouples SVD precision from autocast
        svdvals = torch.linalg.svdvals(J) # D: size of embedding dimension
        jepa_score = svdvals.clip_(eps).log_().sum(1)
    
    batch_results = pd.DataFrame({
                "label": label.cpu().tolist(),
                "score": jepa_score.cpu().tolist(),
                "img_path": [x.removeprefix(dataset_path + dataset + "/") for x in img_path],
                "svdvals": svdvals.cpu().tolist(),
            })
    return batch_results

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