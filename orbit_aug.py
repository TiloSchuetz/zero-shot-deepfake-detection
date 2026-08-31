#!/usr/bin/env python3
"""
DINOv3 augmentation orbit.

Replicates the resolution-preserving subset of `DataAugmentationDINO`
(dinov3/data/augmentations.py). Two global sub-recipes are reproduced verbatim:

    global_transfo1: crop+flip -> color jitter -> GaussianBlur(p=1.0)
    global_transfo2: crop+flip -> color jitter -> GaussianBlur(p=0.1) -> Solarize(p=0.2)

Deviations from pretraining (documented on purpose):
  * Local crops (96px) are dropped: a second resolution breaks the static input
    shape required for CUDA graph capture.
  * Crops are taken from the resize+center-crop image at the model input size,
    not from the original, so every image sees the same resampling regime.
"""

import torch
from torchvision.transforms import v2

BIC = v2.InterpolationMode.BICUBIC

# dinov3/data/augmentations.py ~L132-142
COLOR_JITTER = dict(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
COLOR_JITTER_P = 0.8
GRAYSCALE_P = 0.2
BLUR_KERNEL = 9                 # dinov3/data/transforms.py ~L19-28
BLUR_SIGMA = (0.1, 2.0)
SOLARIZE_THRESHOLD = 128        # applied pre-normalization, i.e. uint8 space
SOLARIZE_P = 0.2

# global_crops_scale is a training-stage config value, not a constant:
#   pretrain        (dinov3_vit7b16_pretrain.yaml)        -> (0.14, 1.0) @ 224
#   high-res adapt  (dinov3_vit7b16_high_res_adapt.yaml)  -> (0.32, 1.0) @ 518
# Both are exposed via config; (0.32, 1.0) is the pre-registered primary.
DEFAULT_GLOBAL_CROPS_SCALE = (0.32, 1.0)


def build_orbit_transforms(img_size, mean, std, global_crops_scale=DEFAULT_GLOBAL_CROPS_SCALE):
    """
    Returns (t0, t1, t2):
        t0 -- deterministic reference view. Identical to the timm eval transform,
              so scores from view 0 stay comparable with existing `last_layer` runs.
        t1 -- DINOv3 global_transfo1
        t2 -- DINOv3 global_transfo2
    """
    scale = tuple(global_crops_scale)

    # resize + center crop to the model input size; all augmentation happens on top
    base = v2.Compose([
        v2.Resize(img_size, interpolation=BIC),
        v2.CenterCrop(img_size),
    ])

    normalize = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    geometric = v2.Compose([
        v2.RandomResizedCrop(img_size, scale=scale, interpolation=BIC),
        v2.RandomHorizontalFlip(p=0.5),
    ])

    color_jittering = v2.Compose([
        v2.RandomApply([v2.ColorJitter(**COLOR_JITTER)], p=COLOR_JITTER_P),
        v2.RandomGrayscale(p=GRAYSCALE_P),
    ])

    def blur(p):
        # v2.GaussianBlur, NOT PIL ImageFilter: the fixed 9px kernel truncates at
        # +-4px, so it is weaker than DINOv2's PIL blur at the top of the sigma range.
        return v2.RandomApply([v2.GaussianBlur(BLUR_KERNEL, sigma=BLUR_SIGMA)], p=p)

    t0 = v2.Compose([base, normalize])
    t1 = v2.Compose([base, geometric, color_jittering, blur(1.0), normalize])
    t2 = v2.Compose([base, geometric, color_jittering, blur(0.1),
                     v2.RandomSolarize(threshold=SOLARIZE_THRESHOLD, p=SOLARIZE_P),
                     normalize])
    return t0, t1, t2
