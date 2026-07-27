from datasets import load_dataset
from pathlib import Path

out_dir = Path("/ceph/tischuet/Imagetnet_val_5k")
out_dir.mkdir(parents=True, exist_ok=True)

# Stream the validation split instead of reading remote parquet with Dask
ds = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    streaming=True
)

# Shuffle the stream first, then take 2000 examples
sample = ds.shuffle(seed=42, buffer_size=50_000).take(5000)

for i, example in enumerate(sample):
    img = example["image"]          # already a PIL image for Image columns
    label = example.get("label", -1)
    img.convert("RGB").save(out_dir / f"{i:04d}_class{label}.png")