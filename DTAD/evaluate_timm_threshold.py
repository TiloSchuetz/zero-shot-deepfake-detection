import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import random
import re
from PIL import Image
import argparse
from glob import glob
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score, f1_score
from dataset_setting import TestDatasets
import timm
from torchvision import transforms
import json


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


class TimmModel(nn.Module):
    """Wraps a timm ViT-based model to return intermediate layer representations
    in the same format as clip_models.Model: [token, batch, layer, embedding_dim].
    Features are taken from each block's pre-MLP LayerNorm (norm2), the timm
    analogue of CLIP's ln_2. Works for MetaCLIP, DINOv2, DINOv3, and any other
    timm model with .blocks."""

    def __init__(self, model_name, device):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval().to(device)
        self.model.requires_grad_(False)

        # Hook the pre-MLP LayerNorm (norm2) of every transformer block, matching
        # evaluate.py, which hooks CLIP's ln_2 (the same position in the block)
        # rather than the raw block output.
        self.hooks = [
            Hook(name, module)
            for name, module in self.model.named_modules()
            if re.fullmatch(r'blocks\.\d+\.norm2', name)
        ]
        assert len(self.hooks) > 0, (
            f"No transformer blocks found in '{model_name}'. "
            "Expected submodules named 'blocks.0.norm2', 'blocks.1.norm2', ..."
        )

    def forward(self, x):
        with torch.no_grad():
            self.model(x)
            # each hook output: [batch, seq_len, dim]
            g = torch.stack([h.output for h in self.hooks], dim=0)  # [num_layers, batch, seq_len, dim]
            return g.permute(2, 1, 0, 3)  # [token, batch, layer, dim] — matches clip_models.Model



def get_denoise_sim_results(img_path, denoising_dir, model, processor, device, save_path=None):
    denoising_output_list = sorted(glob(osp.join(denoising_dir, '*')))

    with torch.inference_mode():
        ori_img = Image.open(img_path).convert('RGB')
        ori_img = processor(ori_img).unsqueeze(0).to(device)
        ori_fea = model(ori_img).detach().cpu()

        sims = []
        for denoising_path in denoising_output_list:
            denoise_img = Image.open(denoising_path).convert('RGB')
            denoise_img = processor(denoise_img).unsqueeze(0).to(device)
            denoise_fea = model(denoise_img).detach().cpu()

            sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(ori_fea.squeeze(), denoise_fea.squeeze())
            sims.append(sim[0])

        sims = torch.stack(sims, dim=0)  # [num_steps, num_layers]

        if save_path is not None:
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            torch.save(sims, save_path)

        return sims.mean()


def get_threshold(args, model, processor):
    """Return the decision threshold for args.model_name calibrated on
    args.threshold_real_folder with args.transform preprocessing.

    Thresholds are cached in one JSON file per backbone,
    <threshold_dir>/<model_name>.json, structured as
    {calibration_set: {transform: threshold}}. An existing cached value is
    always returned as-is; calibration only runs when the
    (calibration set, transform) pair has no entry yet: threshold =
    mean + threshold_std_mult * std of the real-image scores
    (default mean + 1 std), saved back to the JSON file. The write merges
    with the latest on-disk contents and replaces the file atomically, so
    concurrent jobs for the same backbone don't clobber each other."""
    assert args.threshold_real_folder is not None, (
        '--threshold_real_folder is required: it selects which calibration '
        "set's threshold to use (and to compute if not cached yet)."
    )

    print(f"Threshold calculation based on {args.threshold_real_folder} ({args.transform})")

    # one file per backbone; sanitize since hf-hub model names can contain : and /
    safe_model_name = re.sub(r'[^\w.\-]+', '_', args.model_name)
    threshold_file = osp.join(args.threshold_dir, safe_model_name + '.json')

    def load_thresholds():
        if osp.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                return json.load(f)
        return {}

    cal_set = args.threshold_real_folder
    threshold = load_thresholds().get(cal_set, {}).get(args.transform)
    if threshold is not None:
        print(f"Loaded cached threshold {threshold:.4f} for calibration set "
              f"'{cal_set}' / transform '{args.transform}' from {threshold_file}")
        return threshold

    # no cached value for this (calibration set, transform) pair -> calibrate
    cal_dir = osp.join(args.data_root, args.threshold_real_folder)
    real_image_list = sorted(glob(osp.join(cal_dir, '*')))
    assert len(real_image_list) > 0, f'No images found in {cal_dir}'

    scores = []
    for img_path in tqdm(real_image_list, desc='calibrating threshold'):
        denoising_dir = img_path.replace(args.data_root, args.denoising_output_root).split('.')[0]
        score = get_denoise_sim_results(img_path, denoising_dir, model, processor, args.device)
        scores.append(score.item())

    scores = np.asarray(scores)
    threshold = float(scores.mean() + args.threshold_std_mult * scores.std())

    # re-read to pick up entries written by concurrent jobs, then replace atomically
    thresholds = load_thresholds()
    thresholds.setdefault(cal_set, {})[args.transform] = threshold
    os.makedirs(args.threshold_dir, exist_ok=True)
    tmp_file = f'{threshold_file}.tmp.{os.getpid()}'
    with open(tmp_file, 'w') as f:
        json.dump(thresholds, f, indent=4)
    os.replace(tmp_file, threshold_file)
    print(f"Computed threshold {threshold:.4f} for calibration set '{cal_set}' "
          f"/ transform '{args.transform}' "
          f'(mean + {args.threshold_std_mult} * std over {len(scores)} real images), '
          f'saved to {threshold_file}')
    return threshold


def main(args):
    print(args.model_name)
    model = TimmModel(args.model_name, args.device)
    print(f"  -> {len(model.hooks)} transformer blocks hooked")

    data_config = timm.data.resolve_model_data_config(model.model)
    print(data_config)
    # both branches use the full frame like the original CLIP preprocessing,
    # not timm's default crop_pct 0.9 context crop
    if args.transform == 'crop':
        # no resize: center-crop the original image at the model's native input size
        processor = transforms.Compose([
            transforms.CenterCrop(data_config['input_size'][1:]),  # (H, W)
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
    elif args.transform == 'resize':
        # shortest side -> input size, then center crop: same as clip.load's preprocess
        processor = timm.data.create_transform(**{**data_config, 'crop_pct': 1.0}, is_training=False)
    else:
        raise ValueError(f"unknown --transform: {args.transform} (use 'crop' or 'resize')")
    print(processor)

    threshold = get_threshold(args, model, processor)
    print(f"Threshold: {threshold}")

    dataset_name = TestDatasets[args.dataset]['dataset_name']
    classes = TestDatasets[args.dataset]['classes']

    for cls in classes:
        print(cls)

        if dataset_name == 'ForenSynths':
            # ForenSynths-style layout: <cls>/0_real and <cls>/1_fake
            real_image_list = sorted(glob(osp.join(args.data_root, dataset_name, cls, '0_real/*')))
            fake_image_list = sorted(glob(osp.join(args.data_root, dataset_name, cls, '1_fake/*')))
        elif dataset_name in {'New-Generator', 'New-Generator_COCO17_unbiased', 'New-Generator_RAISE1k_unbiased'}:
            # shared real folder for all fakes
            real_image_list = sorted(glob(osp.join(args.data_root, dataset_name, 'real/*')))
            fake_image_list = sorted(glob(osp.join(args.data_root, dataset_name, cls, '*')))
        else:
            raise ValueError(f"{dataset_name} is misspelled or doesn't exist")

        assert len(real_image_list) > 0 and len(fake_image_list) > 0, \
            f"No images found for {dataset_name}/{cls} under {args.data_root}"
        
        real_scores = []
        for img_path in tqdm(real_image_list):
            denoising_dir = img_path.replace(args.data_root, args.denoising_output_root).split('.')[0]
            save_path = osp.join(args.sim_save_root, osp.splitext(osp.relpath(img_path, args.data_root))[0] + '.pt') if args.sim_save_root else None
            score = get_denoise_sim_results(img_path, denoising_dir, model, processor, args.device, save_path=save_path)
            real_scores.append(score)
        real_scores = torch.stack(real_scores, dim=0)

        fake_scores = []
        for img_path in tqdm(fake_image_list):
            denoising_dir = img_path.replace(args.data_root, args.denoising_output_root).split('.')[0]
            save_path = osp.join(args.sim_save_root, osp.splitext(osp.relpath(img_path, args.data_root))[0] + '.pt') if args.sim_save_root else None
            score = get_denoise_sim_results(img_path, denoising_dir, model, processor, args.device, save_path=save_path)
            fake_scores.append(score)
        fake_scores = torch.stack(fake_scores, dim=0)

        # compute metrics
        scores = np.concatenate((real_scores.cpu().numpy(), fake_scores.cpu().numpy()), axis=0)
        labels = np.asarray([0] * len(real_scores) + [1] * len(fake_scores))
        preds = (scores > threshold).astype(int)

        acc = accuracy_score(labels, preds)
        ap = average_precision_score(labels, scores)
        auc = roc_auc_score(labels, scores)
        f1 = f1_score(labels, preds)

        print(cls)
        print(f'Acc: {acc:.4f}, AP: {ap:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=' ', type=str,
                        help='the root directory of datasets')
    parser.add_argument('--dataset', type=str, default=' ', help='dataset name')
    parser.add_argument('--denoising_output_root', default=' ', help='the directory for saving the denoising outputs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--sim_save_root', default=None, help='directory to save per-step per-layer cosine similarities as .pt files')
    parser.add_argument('--threshold_dir', default='./thresholds',
                        help='directory for cached thresholds, one JSON file per backbone, '
                             'keyed by calibration set and transform')
    parser.add_argument('--threshold_real_folder', default=None,
                        help='folder of real images for threshold calibration, relative to both data_root and '
                             'denoising_output_root (only needed when no cached value exists)')
    parser.add_argument('--threshold_std_mult', type=float, default=1.0,
                        help='threshold = mean + this many std deviations of the real-image scores')
    parser.add_argument('--transform', type=str, default='resize')
    args = parser.parse_args()

    main(args)
