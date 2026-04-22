#!/usr/bin/env python3

import torch
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, Dataset
from torch.func import jacrev, vmap
import torch._inductor.config as inductor_config
from PIL import Image

import argparse
import timm
import pandas as pd

from typing import Tuple
from pathlib import Path
import os
from tqdm import tqdm
import time

dataset_folder_path = "../../../work/tischuet/replication_datasets/"

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file: pd.DataFrame, img_dir: str, transform=None, target_transform=None):
        self.img_labels = annotations_file
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

def create_pd_data_paths(folder_name: str) -> pd.DataFrame:
    """
    Loads image filenames and labels from specified folder.

    Args:
        path: string with image folder path.

    Returns:
        paths_labels: returns pandas dataframes with image filenames and labels.
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



def main(args):
    # ----- 1. check for CUDA -----

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    print(gpu_name)

    # ----- 2. Load model & transforms -----

    print(f"Model name: {args.model_name}")

    model = load_model(device=device, model_name=args.model_name)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    print(f"# of model embedding dimensions: {model.num_features}")


    # ----- 3. Create dataset & dataloader -----

    filenames_labels = create_pd_data_paths(folder_name=args.data_folder)

    print(filenames_labels.label.value_counts())

    test_dataset = CustomImageDataset(filenames_labels, args.data_folder, transform=transforms)

    file_path = Path(args.csv_file_name)

    # loads existing csv file and checks for already calculated JEPA-SCOREs or creates new file
    if file_path.exists():
        print("Load existing csv file with JEPA-SCOREs.")

        calculated = pd.read_csv(args.csv_file_name)
        len_calculated = len(calculated)
        print(f"Already calculated {len_calculated} JEPA-SCORES")
        resumed_df = filenames_labels[~filenames_labels['img_path'].isin(calculated['img_path'])]

        resumed_ds = CustomImageDataset(resumed_df, args.data_folder, transform=transforms)
        data_loader = DataLoader(dataset=resumed_ds, batch_size=args.batch_size, shuffle=False)
        print(f"Batch size of dataloader: {args.batch_size}")
    else:
        print("No existing csv file found. Creating new one...")

        data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False) #TODO there seem to be a differences in JEPA-SCOREs when using 1 vs 4 images per batch
        print(f"Batch size of dataloader: {args.batch_size}")

    # ----- 4. Calculate JEPA-SCORE -----

    # Listing 1: JEPA-SCORE implementation in PyTorch. Our empirical ablations demonstrate that JEPA-SCORE is not sensitive to the choice of eps (we pick 1​e−6)
    eps = 1e-6

    #torch.set_grad_enabled(True)
    #model.requires_grad_(False)


    print(f"JEPA-SCORE csv filename: {args.csv_file_name}")

    """ if "val" in args.csv_file_name:
        folder_name = "val_jepa_score_csv/" # folder for validation JEPA-SCOREs
    elif "test" in args.csv_file_name:
        folder_name = "test_jepa_score_csv/" # folder for testing JEPA-SCOREs
    else:
        print("Warning: Wrong csv file naming scheme")
        folder_name = "" """

    write_header = not os.path.exists(args.csv_file_name)

    for i, (img, label, img_path) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Computing JEPA scores"):
        img = img.to(device).requires_grad_(True)    # shape: [batch size, C, H, W]
        # label = label.item()    # shape: [1, batch size]; 0 = "real"; 1 = "fake"
        # img_path                # shape: (batch size)

        # Compute Jacobian for single image
        start = time.time()
        J = jacobian(lambda x: model(x).sum(0), inputs=img) #, vectorize=True) # vectorization saves some time per image and makes a difference at the second decimal place
        torch.cuda.synchronize()  # needed for accurate CUDA timing
        print(f"Jacobian took {time.time() - start:.2f}s", flush=True)

        start = time.time()
        with torch.inference_mode():
            J = J.flatten(2).permute(1,0,2)
            svdvals = torch.linalg.svdvals(J)
            jepa_score = svdvals.clip_(eps).log_().sum(1) # one score per image in batch; clip with eps stabilizes calculations
        torch.cuda.synchronize()
        print(f"SVD took {time.time() - start:.2f}s", flush=True)    

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