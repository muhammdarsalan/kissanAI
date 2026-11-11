from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile

REPO = "mohanty/PlantVillage"
OUT = Path("data/raw/plantvillage_hf")
KEEP = ("Tomato", "Potato", "Corn_(maize)")

OUT.mkdir(parents=True, exist_ok=True)
zpath = hf_hub_download(REPO, "data.zip", repo_type="dataset", local_dir=OUT)
with zipfile.ZipFile(zpath) as z:
    members = [n for n in z.namelist()
               if n.startswith("raw/color/") and not n.endswith("/")
               and n.split("/")[2].split("___")[0] in KEEP]
    print(len(members), "images to extract")
    z.extractall(OUT / "images", members=members)
print("done")