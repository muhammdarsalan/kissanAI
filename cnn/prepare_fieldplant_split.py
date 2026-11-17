import argparse, re, shutil
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/kaggle/input/datasets/bloox2/fieldplant/train")
ap.add_argument("--out", default="/kaggle/working/fp2")
ap.add_argument("--max-unknown", type=int, default=200)
a = ap.parse_args()
out = Path(a.out)


def target(c):
    c = c.lower()
    if "cassava" in c: return "_unknown"
    if "corn" in c and "gray" in c: return "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    if "corn" in c and "rust" in c: return "Corn_(maize)___Common_rust_"
    if "corn" in c and "healthy" in c: return "Corn_(maize)___healthy"
    if "tomato" in c and "mosaic" in c: return "Tomato___Tomato_mosaic_virus"
    if "tomato" in c and "healthy" in c: return "Tomato___healthy"
    return None


def photo_group(fname):
    stem = re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", fname, flags=re.I)
    m = re.match(r"^(.*?)(\d+)\D*$", stem)
    return f"{m.group(1)}|{int(m.group(2)) // 20}" if m else stem


df = pd.read_csv(f"{a.src}/_annotations.csv")
df["target"] = df["class"].map(target)
df = df[df.target.notna()]
df = df[((df.xmax - df.xmin) >= 100) & ((df.ymax - df.ymin) >= 100)].copy()
df["group"] = df.filename.map(photo_group)

known = df[df.target != "_unknown"].reset_index(drop=True)
unk = df[df.target == "_unknown"]
unk = unk.sample(min(a.max_unknown, len(unk)), random_state=0).assign(split="test")

sg = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
_, test_idx = next(iter(sg.split(known, known.target, known.group)))
known["split"] = "train"
known.loc[test_idx, "split"] = "test"
rest = known[known.split == "train"]
sg2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42)
_, val_idx = next(iter(sg2.split(rest, rest.target, rest.group)))
known.loc[rest.index[val_idx], "split"] = "val"

for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
    shared = set(known[known.split == x].group) & set(known[known.split == y].group)
    print(f"photo groups shared between {x} and {y}: {len(shared)}")

use = pd.concat([known, unk], ignore_index=True)
print(pd.crosstab(use.target, use.split))

rows = []
for n, (_, r) in enumerate(use.iterrows()):
    d = out / r.split / r.target
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"fp_{n}.jpg"
    im = Image.open(f"{a.src}/{r.filename}").convert("RGB").crop((r.xmin, r.ymin, r.xmax, r.ymax))
    im.thumbnail((640, 640))
    im.save(p, quality=92)
    if r.split in ("train", "val"):
        rows.append(dict(path=str(p), label=r.target, split=r.split))

Path("data/processed").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("data/processed/fieldplant_train.csv", index=False)
shutil.make_archive("/kaggle/working/fp2_test", "zip", out / "test")
print("train/val crops:", len(rows), "| test zip saved")