import re, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path("data/raw/plantvillage_hf")
COLOR = BASE / "images" / "raw" / "color"
KEEP = {"Tomato", "Potato", "Corn_(maize)"}
GAP = 2   # edit after reading the validation lines below

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

def gap_cluster_ids(frame, gap):
    ids = pd.Series([f"p:{l}|{k}" for l, k in zip(frame.label, frame.key)],
                    index=frame.index, dtype=object)
    ok = frame.dropna(subset=["num"])
    for (label, prefix), g in ok.groupby(["label", "prefix"]):
        g = g.sort_values("num")
        c, prev = 0, None
        for idx, n in zip(g.index, g["num"]):
            if prev is not None and n - prev > gap:
                c += 1
            ids[idx] = f"p:{label}|{prefix}|{c}"
            prev = n
    return ids

# --- validate on images that DO have a leaf-map entry ---
cov = df[df["key"].isin(leaf_map)].copy()
cov["mgroup"] = cov["key"].map(lambda k: leaf_map[k][0])
print("images with a map entry:", len(cov), "/", len(df))
for g in (0, 1, 2, 3, 5):
    cov["c"] = gap_cluster_ids(cov, g)
    inside = (cov.groupby("mgroup")["c"].nunique() == 1).mean()
    sz = cov.groupby("c").size()
    print(f"gap={g}: {inside:.1%} of map groups inside one cluster | "
          f"avg {sz.mean():.1f}, max {sz.max()} images per cluster")

# --- union-find: final groups ---
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

proxy = gap_cluster_ids(df, GAP)
nodes = []
for i, (label, key) in zip(df.index, zip(df["label"], df["key"])):
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
print(f"GAP={GAP} | groups: {df['group'].nunique()} | "
      f"largest group: {df['group'].value_counts().iloc[0]} images")
print(df.groupby("label")["group"].nunique().rename("groups per class"))

# --- our own leaf-grouped 80/10/10 split ---
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
folds = [idx for _, idx in sgkf.split(df, df["label"], df["group"])]
df["split"] = "train"
df.loc[df.index[folds[0]], "split"] = "test"
df.loc[df.index[folds[1]], "split"] = "val"

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    shared = set(df[df.split == a].group) & set(df[df.split == b].group)
    print(f"groups shared between {a} and {b}: {len(shared)}")

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.drop(columns=["key", "prefix", "num"]).to_csv("data/processed/splits.csv", index=False)
print(pd.crosstab(df["label"], df["split"]))