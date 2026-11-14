from pathlib import Path
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from prepare_field_test import target_for, CLASSES, EXT

TRAIN = Path("data/raw/plantdoc_clean/train")
TEST = Path("data/raw/plantdoc_clean/test")
ph = lambda p: imagehash.phash(Image.open(p).convert("RGB"))

test_h = [ph(f) for f in TEST.rglob("*") if f.suffix.lower() in EXT]
rows, dup = [], 0
for d in sorted(p for p in TRAIN.iterdir() if p.is_dir()):
    t = target_for(d.name)
    if t in (None, "_unknown") or t not in CLASSES:
        continue
    for f in d.iterdir():
        if f.suffix.lower() not in EXT:
            continue
        h = ph(f)
        if any(h - x <= 4 for x in test_h):
            dup += 1
            continue
        rows.append({"path": str(f), "label": t})
df = pd.DataFrame(rows)
rng = np.random.RandomState(42)
df["split"] = np.where(rng.rand(len(df)) < 0.2, "val", "train")
df.to_csv("data/processed/field_train.csv", index=False)
print(len(df), "kept;", dup, "dropped as near-duplicates of test images")
print(df.groupby(["label", "split"]).size())