from pathlib import Path
import json
import pandas as pd

BASE = Path("data/raw/plantvillage_hf")
color_dir = BASE / "images" / "raw" / "color"
leaf_map = json.loads((BASE / "leaf_grouping" / "leaf-map.json").read_text())

rows = []
for d in sorted(color_dir.iterdir()):
    if d.is_dir():
        for f in d.iterdir():
            rows.append({"name": f.name.lower(), "label": d.name})
df = pd.DataFrame(rows)
df["key"] = df["name"].map(lambda n: n.rsplit(".", 1)[0].split("___", 1)[-1])
df["in_map"] = df["key"].isin(leaf_map)

# A: which classes are missing from the leaf map?
a = df.groupby("label")["in_map"].agg(["sum", "count"])
a["missing"] = a["count"] - a["sum"]
print(a.sort_values("missing", ascending=False).head(15))

# B: example unmatched names
miss = df[~df["in_map"]]
for prefix in ["Corn", "Tomato", "Potato"]:
    s = miss[miss.label.str.startswith(prefix)].head(4)
    print(prefix, s[["name", "key"]].values.tolist())

# C: key collisions and multi-entry leaf-map values
print("images sharing a key with another image:", int(df["key"].duplicated(keep=False).sum()))
print("leaf-map keys with more than one entry:", sum(len(v) > 1 for v in leaf_map.values()))

# D: groups spanning train and test
def names(f):
    return {Path(l.strip()).name.lower()
            for l in (BASE / "splits" / f).read_text().splitlines() if l.strip()}
tr, te = names("color_train.txt"), names("color_test.txt")
df["official"] = df["name"].map(lambda n: "test" if n in te else ("train" if n in tr else None))
df["group"] = df["key"].map(lambda k: leaf_map[k][0] if k in leaf_map else None)
m = df.dropna(subset=["group"])
g = m.groupby("group")["official"].nunique()
bad = list(g[g > 1].index)
print("groups spanning train and test:", len(bad), "of", m["group"].nunique())
for grp in bad[:3]:
    print(grp)
    print(m[m["group"] == grp][["name", "official"]].head(6).to_string())