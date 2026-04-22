#!/usr/bin/env python3
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
import os
import pandas as pd
import torch
from typing import Tuple
from pathlib import Path
import json

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
    #TODO: implement recursive
    #TODO: sort files? tests show that there is no problem with reordering

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
        csv_file_name: str,
        filenames_labels: pd.DataFrame,
        dataset_folder_path: str,
        data_folder: str,
        transforms,
        batch_size: int
        ) -> DataLoader[CustomImageDataset]:
    """
    Creates either a new or a resumed dataset and dataloader instance.
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
    file_path = Path(csv_file_name)
    if file_path.exists():
        print("Load existing JSONL file with JEPA-SCOREs.")
        with open(csv_file_name, "r") as f:
            next(f)  # skip metadata line
            already_done = {json.loads(line)["img_path"] for line in f if line.strip()}
        print(f"Already calculated {len(already_done)} JEPA-SCORES")
        resumed_df = filenames_labels[~filenames_labels['img_path'].isin(already_done)]
        test_dataset = CustomImageDataset(resumed_df, dataset_folder_path, data_folder, transform=transforms)
    else:
        print("No existing JSONL file found. Creating new one...")
        test_dataset = CustomImageDataset(filenames_labels, dataset_folder_path, data_folder, transform=transforms)

    data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    print(f"Batch size of dataloader: {batch_size}")
    return data_loader