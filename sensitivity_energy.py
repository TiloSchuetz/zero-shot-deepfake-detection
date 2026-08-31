#!/usr/bin/env python3
"""
Per-pixel sensitivity energy of the input Jacobian the JEPA-SCORE is built on,
per layer, with real-vs-fake and layer-difference comparisons.

The map is diag(J^T J) reshaped to the input grid:

    E[p] = sum_d sum_c (d f_d / d x_{c,p})^2 = sum_i sigma_i^2 v_i[p]^2

i.e. the sigma^2-weighted mixture of all right singular vectors, summed over the
colour channels. It is the spatial counterpart of jacobian_freq_profile() in
helpers.py -- same quantity, keeping the pixel index instead of binning the
rfft2 radially -- and it needs no SVD.

Deliberate choices:
  * fp32 only, no autocast. bf16 costs ~38 nats of logdet; the sigma^2 energy
    itself survives it, but the cross-check against sum_i sigma_i^2 would not,
    and the cost here is a handful of images.
  * The energy is reported twice: in normalised-input units (directly comparable
    to the `jac_energy` column of score_last_layer_spectral) and in raw-pixel
    units, dividing channel c by std_c, which is what gets plotted -- otherwise
    the channels are scaled against each other for no reason.
  * Reduction is a sum of squares, not a signed sum. Embedding coordinates have
    no canonical sign, so a signed sum over d cancels to sqrt(D) noise; squares
    are also the only simple reduction invariant under J -> QJ, the same
    symmetry that leaves sum_i log sigma_i unchanged.

Layer difference
----------------
The layer-difference score is s(l_hi) - s(l_lo) = log det ratio, so its spatial
analogue is not unique. Three maps are written and the last two are the ones to
trust:

  raw       E_hi - E_lo          dominated by whichever layer has more total
                                 energy; useful only as a magnitude check.
  dist      E_hi/sum - E_lo/sum  where the deeper layer *relocates* sensitivity,
                                 with the shared image-generic scale divided out.
  logratio  log E_hi - log E_lo  pointwise, mirrors the logdet-difference form.

Examples
--------
    # last layer, balanced real/fake subsample
    python sensitivity_energy.py --model_name vit_large_patch16_dinov3.lvd1689m \
        --dataset New-Generator_COCO17_unbiased --n_per_class 8

    # the 18-12 layer difference used in the CLIP-L/14 results
    python sensitivity_energy.py --model_name vit_large_patch14_clip_quickgelu_224.openai \
        --dataset New-Generator_COCO17_unbiased --n_per_class 8 \
        --layers 12 18 --layer_diff 18 12
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from helpers import (
    load_model,
    create_pd_data_paths,
    create_pd_data_paths_recursive,
    create_pd_data_paths_New_Generators,
)
from jepa_score import DATASET_PATHS, MODELS

FLOOR = 1e-8  # relative floor before any log, as a fraction of the map maximum


# ----------------------------------------------------------------------------
# Jacobians
# ----------------------------------------------------------------------------

def layer_jacobians(model, img, layers, device):
    """
    Input Jacobian of the CLS token at several depths, in fp32, one layer at a
    time so peak memory stays at one [D, C, H, W] block (~620 MB for ViT-L).

    Mirrors score_all_layers() in helpers.py: raw block output for intermediate
    layers, post-final-norm model output for index len(model.blocks).

    Args:
        img: [1, C, H, W] with requires_grad_(True). Batch size must be 1 --
             grad_outputs carries the batched one-hots.
    Yields:
        (layer_index, J_l) with J_l of shape [D, C, H, W], detached fp32.
    """
    assert img.shape[0] == 1, "layer_jacobians expects batch size 1"
    final = len(model.blocks)
    block_index = {id(b): i for i, b in enumerate(model.blocks)}
    B, C, H, W = img.shape
    D = model.embed_dim
    I_D = torch.eye(D, device=device).unsqueeze(1)  # [D, 1, D] one-hots

    cls_outs = {}

    def cls_hook(module, inputs, output):
        idx = block_index[id(module)]
        if idx in layers:
            if isinstance(output, tuple):
                output = output[0]
            cls_outs[idx] = output[:, 0]  # raw block output, no norm

    handles = [b.register_forward_hook(cls_hook) for b in model.blocks]
    try:
        feat = model(img)  # post-final-norm
    finally:
        for h in handles:
            h.remove()
    if final in layers:
        cls_outs[final] = feat

    ordered = sorted(layers)
    for i, l in enumerate(ordered):
        grads = torch.autograd.grad(
            outputs=cls_outs[l],
            inputs=img,
            grad_outputs=I_D,
            is_grads_batched=True,
            retain_graph=(i < len(ordered) - 1),
        )[0]                                    # [D, 1, C, H, W]
        yield l, grads.squeeze(1).detach().float()


def sensitivity_energy(J_b, std=None):
    """
    Args:
        J_b: [D, C, H, W] Jacobian rows for one image, w.r.t. the *normalised*
             input tensor.
        std: [C] normalisation std. If given, the Jacobian is converted to
             raw-pixel units first (d f / d x_raw = (d f / d x_norm) / std_c).
    Returns:
        heat [H, W], per_channel [C, H, W]
    """
    if std is not None:
        J_b = J_b / std.view(1, -1, 1, 1)
    per_channel = (J_b ** 2).sum(dim=0)
    return per_channel.sum(dim=0), per_channel


def patch_pool(heat, patch):
    """Sum-pool a [H, W] map onto the ViT patch grid, so energy is conserved."""
    pooled = F.avg_pool2d(heat[None, None], kernel_size=patch, stride=patch)
    return pooled[0, 0] * patch * patch


def as_distribution(heat):
    """Normalise a map to sum 1, so images/layers of different overall
    sensitivity scale become comparable as spatial distributions."""
    return heat / heat.sum().clamp_min(1e-30)


def layer_diff_maps(E_hi, E_lo):
    """raw, distribution-difference and pointwise log-ratio maps."""
    raw = E_hi - E_lo
    dist = as_distribution(E_hi) - as_distribution(E_lo)
    lo = E_lo.clamp_min(E_lo.max() * FLOOR)
    hi = E_hi.clamp_min(E_hi.max() * FLOOR)
    return raw, dist, hi.log() - lo.log()


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------

def denormalise(img, mean, std):
    """[C, H, W] normalised tensor -> [H, W, C] in [0, 1] for display."""
    x = img * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    return x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _seq_limits(h, clip_pct):
    return h.min(), np.percentile(h, clip_pct)


def _div_limit(h, clip_pct):
    """Symmetric limit, so zero sits at the centre of the diverging colormap."""
    v = np.percentile(np.abs(h), clip_pct)
    return (-v, v) if v > 0 else (-1.0, 1.0)


def plot_triptych(rgb, heat, out_path, title, clip_pct=99.0, log_scale=False):
    """Image | heatmap | overlay. The raw map is heavy-tailed, so the colour
    scale is percentile-clipped or a hot pixel flattens everything else."""
    h = heat.copy()
    if log_scale and (h > 0).any():
        h = np.log10(h + h[h > 0].min() * 1e-3)
    vmin, vmax = _seq_limits(h, clip_pct)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].imshow(rgb)
    axes[0].set_title("input crop")
    im = axes[1].imshow(h, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("sensitivity energy" + (" (log10)" if log_scale else ""))
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(rgb)
    axes[2].imshow(h, cmap="inferno", alpha=0.6, vmin=vmin, vmax=vmax)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_layer_diff(rgb, maps, out_path, title, clip_pct=99.0):
    """Input | E_lo | E_hi | dist-difference | log-ratio."""
    E_lo, E_hi, dist, logratio = maps
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.4))
    axes[0].imshow(rgb)
    axes[0].set_title("input crop")

    for ax, h, name in ((axes[1], E_lo, "E(lo)"), (axes[2], E_hi, "E(hi)")):
        vmin, vmax = _seq_limits(h, clip_pct)
        im = ax.imshow(h, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax, h, name in ((axes[3], dist, "dist diff (hi - lo)"),
                        (axes[4], logratio, "log E(hi) - log E(lo)")):
        vmin, vmax = _div_limit(h, clip_pct)
        im = ax.imshow(h, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_class_means(real, fake, out_path, title, clip_pct=99.5, diverging=False):
    """Mean map for label 0 | label 1 | fake - real."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    cmap = "coolwarm" if diverging else "inferno"
    lim = _div_limit if diverging else (lambda h, p: _seq_limits(h, p))
    both = np.concatenate([real.ravel(), fake.ravel()])
    vmin, vmax = lim(both, clip_pct)

    for ax, h, name in ((axes[0], real, "mean real (label 0)"),
                        (axes[1], fake, "mean fake (label 1)")):
        im = ax.imshow(h, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)

    d = fake - real
    dv, dvmax = _div_limit(d, clip_pct)
    im = axes[2].imshow(d, cmap="coolwarm", vmin=dv, vmax=dvmax)
    axes[2].set_title("fake - real")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# image selection
# ----------------------------------------------------------------------------

def select_images(args):
    """Returns a frame with absolute `path` and (possibly NaN) `label`."""
    if args.images:
        labels = args.labels if args.labels else [float("nan")] * len(args.images)
        if len(labels) != len(args.images):
            raise ValueError("--labels must have one entry per --images path")
        return pd.DataFrame({"path": [str(Path(p)) for p in args.images],
                             "label": [float(l) for l in labels]})

    dataset = args.dataset
    root = DATASET_PATHS[dataset]
    if dataset == "Imagenet_val_5k":
        fl = create_pd_data_paths(root, dataset)
    elif dataset in {"New-Generator", "New-Generator_COCO17_unbiased",
                     "New-Generator_RAISE1k_unbiased"}:
        fl = create_pd_data_paths_New_Generators(root, dataset)
    else:
        fl = create_pd_data_paths_recursive(root, dataset)

    # Group on (generator, label) for the same reason jepa_score.py does:
    # ForenSynths nests both classes under one generator folder. Grouping this
    # way is also what guarantees real images are sampled, not just fakes.
    generator = fl["img_path"].str.split("/").str[0]
    fl = (fl.groupby([generator, fl["label"]], group_keys=False)
            .apply(lambda g: g.sample(min(args.n_per_class, len(g)),
                                      random_state=args.seed)))
    fl["path"] = root + dataset + "/" + fl["img_path"]
    return fl[["path", "label"]].reset_index(drop=True)


# ----------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.model_name not in MODELS:
        print(f"Warning: '{args.model_name}' is not in the MODELS whitelist.")

    model = load_model(device=device, model_name=args.model_name)
    model.requires_grad_(False)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    mean = torch.tensor(data_config["mean"], device=device)
    std = torch.tensor(data_config["std"], device=device)
    patch = model.patch_embed.patch_size[0]
    final = len(model.blocks)  # index of the post-norm output

    # -1 is shorthand for the post-norm output; default is that layer alone,
    # which reproduces the paper-exact last-layer JEPA-SCORE.
    layers = sorted({final if l == -1 else l for l in (args.layers or [-1])})
    if not all(0 <= l <= final for l in layers):
        raise ValueError(f"Layer indices must lie in [0, {final}] (or -1).")

    diff = None
    if args.layer_diff:
        hi, lo = args.layer_diff
        hi, lo = (final if hi == -1 else hi), (final if lo == -1 else lo)
        missing = {hi, lo} - set(layers)
        if missing:
            raise ValueError(f"--layer_diff needs {sorted(missing)} in --layers")
        diff = (hi, lo)

    out_dir = Path(args.out_dir) / args.model_name.replace(".", "_")
    (out_dir / "npy").mkdir(parents=True, exist_ok=True)
    (out_dir / "png").mkdir(parents=True, exist_ok=True)

    images = select_images(args)
    print(f"{len(images)} images, layers={layers}"
          + (f", diff={diff[0]}-{diff[1]}" if diff else "") + f" -> {out_dir}")
    print("Class balance:\n", images["label"].value_counts(dropna=False))

    rows = []
    # Accumulators for the class-mean panels. Maps are accumulated as
    # distributions so a single high-energy image cannot dominate the mean.
    acc = {}

    for i, row in images.iterrows():
        img = transforms(Image.open(row.path).convert("RGB"))
        img = img.unsqueeze(0).to(device).requires_grad_(True)
        stem = f"{i:04d}_" + Path(row.path).stem

        record = {"img_path": row.path, "label": row.label}
        heats, dists = {}, {}

        # No autocast: fp32 throughout (see module docstring).
        for l, J_l in layer_jacobians(model, img, layers, device):
            with torch.no_grad():
                # Normalised units: comparable to the `jac_energy` column of
                # score_last_layer_spectral, and equal to ||J||_F^2 = sum sigma^2.
                heat_norm, _ = sensitivity_energy(J_l)
                energy_norm = heat_norm.sum().item()
                # Raw-pixel units: what gets plotted and saved.
                heat, per_channel = sensitivity_energy(J_l, std=std)

                tag = "norm" if l == final else str(l)
                record[f"energy_norm_l{tag}"] = energy_norm
                record[f"energy_raw_l{tag}"] = heat.sum().item()
                record[f"channel_energy_l{tag}"] = per_channel.sum(dim=(1, 2)).cpu().tolist()

                if args.score:
                    # Cross-check: sum_i sigma_i^2 must equal energy_norm, and
                    # the logdet is the score these maps are meant to explain.
                    sv = torch.linalg.svdvals(J_l.flatten(1))
                    record[f"score_l{tag}"] = sv.clamp_min(1e-6).log().sum().item()
                    rel = abs((sv ** 2).sum().item() - energy_norm) / max(energy_norm, 1e-30)
                    record[f"energy_rel_err_l{tag}"] = rel
                    if rel > 1e-3:
                        print(f"  warning: layer {tag} energy/sigma^2 mismatch {rel:.2e}")

                heats[l] = heat
                dists[l] = as_distribution(heat)
                np.save(out_dir / "npy" / f"{stem}_l{tag}.npy",
                        heat.cpu().numpy().astype(np.float32))
                if args.patch_pool:
                    np.save(out_dir / "npy" / f"{stem}_l{tag}_p{patch}.npy",
                            patch_pool(heat, patch).cpu().numpy().astype(np.float32))
            del J_l
            if device == "cuda":
                torch.cuda.empty_cache()

        with torch.no_grad():
            rgb = denormalise(img[0].detach(), mean, std)
            lbl = "" if pd.isna(row.label) else f"  label={row.label:.0f}"

            for l in layers:
                tag = "norm" if l == final else str(l)
                sc = record.get(f"score_l{tag}")
                plot_triptych(
                    rgb, heats[l].cpu().numpy(),
                    out_dir / "png" / f"{stem}_l{tag}.png",
                    f"{Path(row.path).name}{lbl}   layer {tag}   "
                    f"energy={record[f'energy_norm_l{tag}']:.3e}"
                    + (f"   score={sc:.1f}" if sc is not None else ""),
                    clip_pct=args.clip_pct, log_scale=args.log_scale)
                acc.setdefault(("layer", l, row.label), []).append(dists[l].cpu().numpy())

            if diff:
                hi, lo = diff
                raw, dist, logratio = layer_diff_maps(heats[hi], heats[lo])
                for name, m in (("diff_raw", raw), ("diff_dist", dist),
                                ("diff_logratio", logratio)):
                    np.save(out_dir / "npy" / f"{stem}_{name}_{hi}-{lo}.npy",
                            m.cpu().numpy().astype(np.float32))
                shi = record.get(f"score_l{'norm' if hi == final else hi}")
                slo = record.get(f"score_l{'norm' if lo == final else lo}")
                if shi is not None and slo is not None:
                    record[f"score_diff_{hi}-{lo}"] = shi - slo
                    ds = f"   ds={shi - slo:.1f}"
                else:
                    ds = ""
                plot_layer_diff(
                    rgb,
                    (heats[lo].cpu().numpy(), heats[hi].cpu().numpy(),
                     dist.cpu().numpy(), logratio.cpu().numpy()),
                    out_dir / "png" / f"{stem}_diff_{hi}-{lo}.png",
                    f"{Path(row.path).name}{lbl}   layers {hi} vs {lo}{ds}",
                    clip_pct=args.clip_pct)
                acc.setdefault(("diff", (hi, lo), row.label), []).append(dist.cpu().numpy())

        rows.append(record)
        del img, heats, dists
        if device == "cuda":
            torch.cuda.empty_cache()
        tags = ["norm" if l == final else str(l) for l in layers]
        head = "  ".join(f"l{t}={record['energy_norm_l' + t]:.3e}" for t in tags)
        print(f"[{i + 1}/{len(images)}] {Path(row.path).name}  {head}")

    df = pd.DataFrame(rows)
    df.to_json(out_dir / "sensitivity_energy.jsonl", orient="records", lines=True)
    print(f"\nWrote {out_dir / 'sensitivity_energy.jsonl'}")

    # ── real vs fake class means ──────────────────────────────────────────────
    def class_mean(kind, key):
        r = acc.get((kind, key, 0.0))
        f = acc.get((kind, key, 1.0))
        if not r or not f:
            return None
        return np.mean(r, axis=0), np.mean(f, axis=0), len(r), len(f)

    for l in layers:
        tag = "norm" if l == final else str(l)
        cm = class_mean("layer", l)
        if cm is None:
            continue
        r, f, nr, nf = cm
        plot_class_means(r, f, out_dir / "png" / f"class_mean_l{tag}.png",
                         f"{args.model_name}   layer {tag}   mean energy "
                         f"distribution   (n_real={nr}, n_fake={nf})")
        np.save(out_dir / "npy" / f"class_mean_real_l{tag}.npy", r.astype(np.float32))
        np.save(out_dir / "npy" / f"class_mean_fake_l{tag}.npy", f.astype(np.float32))

    if diff:
        cm = class_mean("diff", diff)
        if cm is not None:
            r, f, nr, nf = cm
            plot_class_means(r, f, out_dir / "png" /
                             f"class_mean_diff_{diff[0]}-{diff[1]}.png",
                             f"{args.model_name}   layers {diff[0]} vs {diff[1]}   "
                             f"mean dist difference   (n_real={nr}, n_fake={nf})",
                             diverging=True)

    if df["label"].notna().any() and df["label"].nunique() > 1:
        cols = [c for c in df.columns if c.startswith(("energy_norm", "score"))]
        print("\nMeans by label:")
        print(df.groupby("label")[cols].mean().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", type=str, required=True)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=str, nargs="+", help="explicit image paths")
    src.add_argument("--dataset", type=str, choices=sorted(DATASET_PATHS))

    p.add_argument("--labels", type=float, nargs="+",
                   help="labels for --images (0 real, 1 fake); enables class means")
    p.add_argument("--n_per_class", type=int, default=4,
                   help="images per (generator, label) when --dataset is used")
    p.add_argument("--layers", type=int, nargs="+",
                   help="block indices; -1 (default) is the post-norm output")
    p.add_argument("--layer_diff", type=int, nargs=2, metavar=("HI", "LO"),
                   help="difference maps between two layers, both listed in --layers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="./outputs_heatmaps")
    p.add_argument("--patch_pool", action="store_true",
                   help="also save the maps sum-pooled to the ViT patch grid")
    p.add_argument("--score", action="store_true", default=True,
                   help="also compute the SVD, for the logdet and the energy check")
    p.add_argument("--no_score", dest="score", action="store_false")
    p.add_argument("--clip_pct", type=float, default=99.0,
                   help="upper percentile for the colour scale")
    p.add_argument("--log_scale", action="store_true",
                   help="plot log10(energy); useful when a few pixels dominate")

    main(p.parse_args())
