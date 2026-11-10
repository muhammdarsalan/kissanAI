from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path("data/raw/plantvillage_hf")
KEEP_CROPS = {"Tomato", "Potato", "Corn_(maize)"}   # edit as you decide

# 1. find the color folder and list images
color_dir = next(p for p in (BASE / "images").rglob("color") if p.is_dir())
print("color dir:", color_dir)

rows = []
for class_dir in sorted(color_dir.iterdir()):
    if not class_dir.is_dir():
        continue
    crop, _, condition = class_dir.name.partition("___")
    for img in class_dir.iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            rows.append({"path": str(img), "name": img.name.lower(),
                         "label": class_dir.name, "crop": crop,
                         "condition": condition})
df = pd.DataFrame(rows)
print("all color images:", len(df))
print("sample names:", df["name"].head(3).tolist())

# 2. leaf groups from leaf-map.json
leaf_map = json.loads((BASE / "leaf_grouping" / "leaf-map.json").read_text())
print("sample leaf-map keys:", list(leaf_map)[:3])

def key_of(name):
    stem = name.rsplit(".", 1)[0]
    return stem.split("___", 1)[-1]          # part after the uuid

df["key"] = df["name"].map(key_of)
df["group"] = df["key"].map(lambda k: leaf_map[k][0] if k in leaf_map else None)
print("images matched to a leaf group:", df["group"].notna().sum(), "/", len(df))

# 3. official split lists
def read_split(fname):
    lines = (BASE / "splits" / fname).read_text().splitlines()
    print(fname, "sample line:", repr(lines[0]))
    return {Path(l.strip()).name.lower() for l in lines if l.strip()}

train_names = read_split("color_train.txt")
test_names = read_split("color_test.txt")
df["official"] = df["name"].map(
    lambda n: "test" if n in test_names else ("train" if n in train_names else None))
print(df["official"].value_counts(dropna=False))

# 4. keep only our crops, drop anything unmatched
df = df[df["crop"].isin(KEEP_CROPS)].copy()
bad = df["group"].isna() | df["official"].isna()
print("dropped (no group or no split):", int(bad.sum()))
df = df[~bad].copy()

# 5. carve validation out of train, grouped by leaf
df["split"] = df["official"]
tr = df[df["official"] == "train"]
sgkf = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42)
_, val_idx = next(sgkf.split(tr, tr["label"], tr["group"]))
df.loc[tr.index[val_idx], "split"] = "val"

# 6. leakage check
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    shared = set(df[df.split == a]["group"]) & set(df[df.split == b]["group"])
    print(f"leaf groups shared between {a} and {b}: {len(shared)}")

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.drop(columns=["name", "official"]).to_csv("data/processed/splits.csv", index=False)
print(pd.crosstab(df["label"], df["split"]))