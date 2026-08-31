#!/usr/bin/env python3
"""
Is JEPA-SCORE doing anything a JPEG encoder cannot?

Runs JPEG bits-per-pixel ALONE as a deepfake detector over every
encoder x dataset cell for which a `*_last_layer_spectral_*.jsonl` exists, and
compares it cell-by-cell with the stored JEPA-SCORE (`logdet_score`).

Why this is the right control
-----------------------------
The per-layer analysis showed the JEPA score tracks image complexity at
Spearman ~ -0.7, and that the sign of its real/fake separation follows whether
fakes happen to be more or less complex than reals in that dataset. If bpp alone
reproduces the logdet AUROC pattern across cells, the Jacobian machinery is not
adding anything over an entropy coder.

Protocol
--------
* bpp is measured AFTER each encoder's own eval geometry (Resize + CenterCrop
  resolved from timm), on the uint8 RGB the model actually sees. Normalization is
  deliberately not applied -- it is meaningless for an entropy coder. bpp
  normalizes by pixel count, so different input sizes stay comparable.
* bpp depends only on (geometry, dataset), never on encoder weights, so each
  (geometry, dataset) pair is computed once and shared by every cell that uses it.
* Each image is decoded once and every geometry needed for its dataset is applied
  to the decoded image -- decode, not resize, is the bottleneck.
* AUROC is per-generator (fakes of one generator vs the pooled real set of the
  same dataset), macro-averaged, and reported RAW with no sign flipping, so
  direction stays visible: <50 means fakes have the lower value.

Usage
-----
    python bpp_vs_jepa.py                    # all cells (slow the first time)
    python bpp_vs_jepa.py --limit 2000       # quick smoke test
    python bpp_vs_jepa.py --workers 24
    python bpp_vs_jepa.py --datasets ForenSynths New-Generator_COCO17_unbiased

Per-(dataset, geometry) bpp is cached as parquet, so re-runs cost seconds.
"""

import argparse
import io
import json
import os
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

DEFAULT_OUTPUTS = Path(__file__).resolve().parent / "outputs"
DEFAULT_ROOT = Path("/ceph/tischuet/replication_data")
DEFAULT_CACHE = Path(__file__).resolve().parent / ".bpp_cache"

# datasets with no fake (or no real) images cannot yield an AUROC
SKIP_DATASETS = {"Imagenet_val_5k"}


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def geometry(model_name):
    """
    The exact (Resize, CenterCrop) timm builds for this model at eval time.
    Returns (ops, size, key). `key` identifies the geometry so encoders sharing a
    preprocessing pipeline share one bpp computation.
    """
    import timm

    model = timm.create_model(model_name, pretrained=False)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    ops = [o for o in transform.transforms
           if o.__class__.__name__ in ("Resize", "CenterCrop")]
    size = cfg["input_size"][-1]
    key = f"s{size}_c{cfg['crop_pct']}_{cfg['interpolation']}"
    return ops, size, key


# --------------------------------------------------------------------------- #
# reading the stored scores
# --------------------------------------------------------------------------- #
def parse_head(line):
    """
    Parse a record WITHOUT touching `svdvals` / `freq_profile`.

    Those two arrays are ~1024 and 32 floats per image and dominate parse time,
    while every field we need (label, logdet_score, img_path) precedes them.
    Cutting the line at the `svdvals` key and closing the object is ~20x faster
    than json.loads on the full line.
    """
    i = line.find('"svdvals"')
    if i == -1:
        return json.loads(line)
    return json.loads(line[:i].rstrip().rstrip(",") + "}")


def read_scores(path):
    """-> (model_name, dataset, DataFrame[label, img_path, generator, logdet])"""
    with open(path) as fh:
        meta = json.loads(next(fh))
        rows = []
        for line in fh:
            d = parse_head(line)
            rows.append((float(d["label"]),
                         d["img_path"],
                         d.get("logdet_score", d.get("score"))))
    df = pd.DataFrame(rows, columns=["label", "img_path", "logdet"])
    df["generator"] = df.img_path.str.split("/").str[0]
    return meta["model_name"], meta["dataset"], df


# --------------------------------------------------------------------------- #
# bits per pixel
# --------------------------------------------------------------------------- #
_GEOM = {}       # worker-local: geometry key -> (ops, size)
_CFG = {}


def _init_worker(specs, root, dataset, quality):
    for key, model_name in specs.items():
        ops, size, _ = geometry(model_name)
        _GEOM[key] = (ops, size)
    _CFG["dir"] = Path(root) / dataset
    _CFG["q"] = quality


def _bpp_one(rel):
    """Decode once, apply every geometry, JPEG-encode each, return bits/pixel."""
    try:
        img = Image.open(_CFG["dir"] / rel).convert("RGB")
    except Exception as exc:                       # missing/corrupt file
        return {"img_path": rel, "_error": str(exc)}
    out = {"img_path": rel}
    for key, (ops, size) in _GEOM.items():
        im = img
        for op in ops:
            im = op(im)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=_CFG["q"])
        out[f"bpp_{key}"] = len(buf.getvalue()) * 8 / (size * size)
    return out


def compute_bpp(dataset, img_paths, specs, args):
    """specs: {geometry_key: representative model_name}. Cached per dataset."""
    tag = "-".join(sorted(specs))
    cache = Path(args.cache) / f"bpp_{dataset}_{tag}_q{args.quality}.parquet"
    if cache.exists() and not args.refresh:
        cached = pd.read_parquet(cache)
        if set(img_paths).issubset(set(cached.img_path)):
            return cached
    print(f"  computing bpp: {len(img_paths)} images x {len(specs)} geometries "
          f"({', '.join(sorted(specs))})", flush=True)
    with Pool(args.workers, initializer=_init_worker,
              initargs=(specs, args.root, dataset, args.quality)) as pool:
        recs = pool.map(_bpp_one, list(img_paths), chunksize=16)
    df = pd.DataFrame(recs)
    if "_error" in df.columns:
        bad = df._error.notna().sum()
        if bad:
            print(f"  WARNING: {bad} images failed to load and are dropped",
                  file=sys.stderr)
        df = df[df._error.isna()].drop(columns=["_error"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def macro_auc(df, col):
    """Per-generator AUROC vs the pooled real set, macro-averaged. Raw direction."""
    real = df[df.label == 0.0]
    if real.empty:
        return None
    vals = []
    for g in sorted(df.loc[df.label == 1.0, "generator"].unique()):
        sub = pd.concat([df[(df.label == 1.0) & (df.generator == g)], real])
        if sub.label.nunique() == 2:
            vals.append(roc_auc_score(sub.label, sub[col]) * 100)
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default=str(DEFAULT_OUTPUTS),
                    help="directory holding <encoder>/<run>.jsonl")
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="image root; images live at <root>/<dataset>/<img_path>")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N images per cell (smoke test)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these dataset names")
    ap.add_argument("--refresh", action="store_true", help="ignore bpp caches")
    ap.add_argument("--csv", default="bpp_vs_jepa.csv")
    args = ap.parse_args()

    files = sorted(Path(args.outputs).glob("*/*last_layer_spectral_*.jsonl"))
    if not files:
        sys.exit(f"no spectral jsonl found under {args.outputs}")

    # ---- 1. read every cell's stored scores -------------------------------- #
    print(f"reading {len(files)} spectral runs ...", flush=True)
    cells = []
    for path in files:
        model_name, dataset, df = read_scores(path)
        if dataset in SKIP_DATASETS or df.label.nunique() < 2:
            print(f"  skip {path.name} (no real/fake contrast)")
            continue
        if args.datasets and dataset not in args.datasets:
            continue
        if args.limit:
            # stratify by (generator, label): in ForenSynths the reals sit INSIDE
            # each generator folder, so slicing by generator alone can yield a
            # subsample with no real images and silently drop the cell.
            per = max(1, args.limit // max(len(df.groupby(["generator", "label"])), 1))
            df = df.groupby(["generator", "label"], group_keys=False).head(per)
        _, _, gkey = geometry(model_name)
        cells.append(dict(model=model_name, dataset=dataset, geom=gkey, df=df))
        print(f"  {model_name:52s} {dataset:32s} n={len(df):6d} geom={gkey}")

    # ---- 2. bpp once per (dataset, geometry) ------------------------------- #
    print("\ncomputing bits-per-pixel ...", flush=True)
    bpp_by_dataset = {}
    for dataset in sorted({c["dataset"] for c in cells}):
        group = [c for c in cells if c["dataset"] == dataset]
        specs = {c["geom"]: c["model"] for c in group}
        paths = pd.unique(pd.concat([c["df"].img_path for c in group]))
        print(f"[{dataset}]", flush=True)
        bpp_by_dataset[dataset] = compute_bpp(dataset, paths, specs, args)

    # ---- 3. compare -------------------------------------------------------- #
    rows = []
    for c in cells:
        df = c["df"].merge(bpp_by_dataset[c["dataset"]], on="img_path", how="inner")
        col = f"bpp_{c['geom']}"
        bpp_auc, log_auc = macro_auc(df, col), macro_auc(df, "logdet")
        if bpp_auc is None or log_auc is None:
            continue
        real, fake = df[df.label == 0], df[df.label == 1]
        rows.append({
            "encoder": c["model"].split(".")[0].replace("vit_", ""),
            "tag": c["model"].split(".")[-1],
            "dataset": c["dataset"],
            "geom": c["geom"],
            "n": len(df),
            "bpp_real": round(real[col].mean(), 3),
            "bpp_fake": round(fake[col].mean(), 3),
            "bpp_AUROC": round(bpp_auc, 1),
            "logdet_AUROC": round(log_auc, 1),
        })

    t = pd.DataFrame(rows).sort_values(["dataset", "encoder"])
    t["bpp_dev"] = (t.bpp_AUROC - 50).round(1)
    t["logdet_dev"] = (t.logdet_AUROC - 50).round(1)
    t["same_sign"] = np.where(t.bpp_dev * t.logdet_dev > 0, "yes", "NO")

    pd.set_option("display.width", 200)
    print("\n=== JPEG bpp alone vs JEPA-SCORE, per cell ===")
    print("    AUROC raw; <50 means fakes have the LOWER value\n")
    print(t.to_string(index=False))

    if len(t) > 2:
        from scipy.stats import pearsonr, spearmanr
        print(f"\nPearson (bpp_dev, logdet_dev)  = "
              f"{pearsonr(t.bpp_dev, t.logdet_dev).statistic:+.3f}")
        print(f"Spearman(bpp_dev, logdet_dev)  = "
              f"{spearmanr(t.bpp_dev, t.logdet_dev).statistic:+.3f}")
        print(f"same-sign cells                = "
              f"{(t.same_sign == 'yes').sum()}/{len(t)}")
        print(f"mean |AUROC-50|: bpp {t.bpp_dev.abs().mean():.1f}  "
              f"logdet {t.logdet_dev.abs().mean():.1f}")
        print("\nA high correlation here means the Jacobian spectrum is largely "
              "re-measuring\nwhat a JPEG encoder already reports.")

    t.to_csv(args.csv, index=False)
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
