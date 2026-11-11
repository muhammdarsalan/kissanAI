import re, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path("data/raw/plantvillage_hf")
COLOR = BASE / "images" / "raw" / "color"
KEEP = {"Tomato", "Potato", "Corn_(maize)"}
BUCKET = 50

leaf_map = json.loads((BASE / "leaf_grouping" / "leaf-map.json").read_text())

def parse(key):
    m = re.match(r"^(.*?)\s*(\d+)(?:\.\d+)?$", key)
    return (m.group(1), int(m.group(2))) if m else (key, None)

rows = []
for d in sorted(COLOR.iterdir()):
    if not d.is_dir():
        continue
    crop, _, cond = d.name.partition("___")
    if crop not in KEEP:
        continue
    for f in d.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            key = f.name.lower().rsplit(".", 1)[0].split("___", 1)[-1]
            prefix, num = parse(key)
            rows.append(dict(path=str(f), label=d.name, crop=crop,
                             condition=cond, key=key, prefix=prefix, num=num))
df = pd.DataFrame(rows)
print("images:", len(df))

def bucket_id(frame, size):
    return pd.Series(
        [f"p:{l}|{p}|{int(n) // size if pd.notna(n) else k}"
         for l, p, n, k in zip(frame.label, frame.prefix, frame.num, frame.key)],
        index=frame.index)

cov = df[df["key"].isin(leaf_map)].copy()
cov["mgroup"] = cov["key"].map(lambda k: leaf_map[k][0])
print("images with a map entry:", len(cov), "/", len(df))
for size in (20, 50, 100):
    cov["b"] = bucket_id(cov, size)
    inside = (cov.groupby("mgroup")["b"].nunique() == 1).mean()
    print(f"bucket={size}: {inside:.1%} of map groups inside one bucket, "
          f"avg {cov.groupby('b').size().mean():.1f} images per bucket")

parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

proxy = bucket_id(df, BUCKET)
nodes = []
for i, label, key in zip(df.index, df["label"], df["key"]):
    if key in leaf_map:
        gs = ["m:" + g for g in leaf_map[key]]
        for g in gs[1:]:
            union(gs[0], g)
        node = gs[0]
    else:
        node = proxy[i]
    union(node, f"k:{label}|{key}")
    nodes.append(node)
df["group"] = [find(n) for n in nodes]
print(f"BUCKET={BUCKET} | groups: {df['group'].nunique()} | "
      f"largest group: {df['group'].value_counts().iloc[0]} images")
print(df.groupby("label")["group"].nunique().rename("groups per class"))

sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
folds = [idx for _, idx in sgkf.split(df, df["label"], df["group"])]
df["split"] = "train"
df.loc[df.index[folds[0]], "split"] = "test"
df.loc[df.index[folds[1]], "split"] = "val"

for split in ("val", "test"):
    for label in df["label"].unique():
        cls = df[df.label == label]
        if (cls.split == split).any():
            continue
        sizes = cls[cls.split == "train"].groupby("group").size()
        target = max(1, int(0.1 * len(cls)))
        pick = (sizes - target).abs().idxmin()
        df.loc[df.group == pick, "split"] = split
        print(f"moved group {pick} ({sizes[pick]} images) to {split} for {label}")

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    shared = set(df[df.split == a].group) & set(df[df.split == b].group)
    print(f"groups shared between {a} and {b}: {len(shared)}")

ct = pd.crosstab(df["label"], df["split"])
print(ct)
missing = [(l, s) for l in ct.index for s in ("val", "test") if ct.loc[l, s] == 0]
print("classes missing from val/test:", missing)

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.drop(columns=["key", "prefix", "num"]).to_csv("data/processed/splits.csv", index=False)