from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile

REPO = "mohanty/PlantVillage"
OUT = Path("data/raw/plantvillage_hf")
OUT.mkdir(parents=True, exist_ok=True)

for name in ["leaf_grouping/leaf-map.json",
             "splits/color_train.txt",
             "splits/color_test.txt",
             "data.zip"]:
    p = hf_hub_download(REPO, name, repo_type="dataset", local_dir=OUT)
    print("downloaded:", p)

with zipfile.ZipFile(OUT / "data.zip") as z:
    names = z.namelist()
    print(len(names), "entries in zip")
    z.extractall(OUT / "images")

print("done")