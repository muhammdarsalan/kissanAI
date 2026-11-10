from pathlib import Path
import json

base = Path("data/raw/plantvillage_hf")

# 1. top-level layout of the extracted images
for p in sorted((base / "images").iterdir())[:10]:
    print("images/", p.name)

# 2. first lines of the split files
for f in ["color_train.txt", "color_test.txt"]:
    lines = (base / "splits" / f).read_text().splitlines()
    print(f, len(lines), "lines")
    print(lines[:3])

# 3. leaf map structure
m = json.loads((base / "leaf_grouping" / "leaf-map.json").read_text())
print(type(m), len(m))
items = list(m.items())[:3] if isinstance(m, dict) else m[:3]
print(items)