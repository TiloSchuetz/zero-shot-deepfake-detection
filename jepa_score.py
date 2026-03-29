#!/usr/bin/env python3

import torch 
from torch.autograd.functional import jacobian
from torchvision.datasets import ImageFolder

import argparse
import timm

from typing import List, Tuple, Sequence
from pathlib import Path


def load_model(device: str = "cuda", model_name: str = "vit_small_patch14_dinov2.lvd142m") -> Tuple[torch.nn.Module, callable]:
    """Loads JEPA encoder model"""
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0) # num_classes is important, as we want the input Jacobian w.r.t. to the embeddings and not output logits
    model.eval()

    model = model.to(device)

    return model




def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(device=device, model_name=args.model_name)


    
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
        default="vit_small_patch14_dinov2.lvd142m",
        help="JEPA encoder model name",
    )

    args_ = parser.parse_args()
    main(args_)