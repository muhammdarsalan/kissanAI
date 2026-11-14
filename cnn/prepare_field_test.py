import argparse, json, shutil
from pathlib import Path

SRC = Path("data/raw/plantdoc_clean/test")
DST = Path("data/field_test")
EXT = {".jpg", ".jpeg", ".png", ".webp"}
CLASSES = json.loads(Path("configs/classes.json").read_text())

RULES = [
    (("corn", "gray"), "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"),
    (("corn", "blight"), "Corn_(maize)___Northern_Leaf_Blight"),
    (("corn", "rust"), "Corn_(maize)___Common_rust_"),
    (("potato", "early"), "Potato___Early_blight"),
    (("potato", "late"), "Potato___Late_blight"),
    (("tomato", "bacterial"), "Tomato___Bacterial_spot"),
    (("tomato", "early"), "Tomato___Early_blight"),
    (("tomato", "late"), "Tomato___Late_blight"),
    (("tomato", "mold"), "Tomato___Leaf_Mold"),
    (("tomato", "septoria"), "Tomato___Septoria_leaf_spot"),
    (("tomato", "spider"), "Tomato___Spider_mites Two-spotted_spider_mite"),
    (("tomato", "yellow"), "Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
    (("tomato", "mosaic"), "Tomato___Tomato_mosaic_virus"),
]
EXACT_HEALTHY = {"tomato leaf": "Tomato___healthy",
                 "potato leaf": "Potato___healthy",
                 "corn leaf": "Corn_(maize)___healthy"}
OUR_CROPS = ("tomato", "potato", "corn")


def norm(name):
    return " ".join(name.lower().replace("_", " ").split())


def target_for(folder):
    n = norm(folder)
    if n in EXACT_HEALTHY:
        return EXACT_HEALTHY[n]
    for words, cls in RULES:
        if all(w in n for w in words):
            return cls
    if any(c in n for c in OUR_CROPS):
        return None            # our crop, but no matching class: skip
    return "_unknown"          # another crop: used to test unknown-leaf rejection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Clone PlantDoc first (Step 1).")

    plan = []
    for d in sorted(p for p in SRC.iterdir() if p.is_dir()):
        imgs = [f for f in d.iterdir() if f.suffix.lower() in EXT]
        t = target_for(d.name)
        if t not in (None, "_unknown") and t not in CLASSES:
            t = None
        plan.append((d.name, t, imgs))
        print(f"{d.name:45s} -> {str(t):55s} {len(imgs):4d} images")

    if not a.apply:
        print("\nDry run only. Check the mapping above, then rerun with --apply")
        return
    n = 0
    for folder, t, imgs in plan:
        if t is None:
            continue
        out = DST / t
        out.mkdir(parents=True, exist_ok=True)
        for f in imgs:
            shutil.copy2(f, out / f"plantdoc_{folder.replace(' ', '_')}_{f.name}")
            n += 1
    print("copied", n, "images into", DST)


if __name__ == "__main__":
    main()