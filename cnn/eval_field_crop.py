import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from dataset import make_transforms

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def crop_of(c):
    return c.split("___")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/mobilenetv3_large_100_field_best.pt")
    ap.add_argument("--dir", default="data/field_test")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    classes = ck["classes"]
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["model"])
    model.eval()
    _, tf = make_transforms()
    crops = np.array([crop_of(c) for c in classes])

    rows = []
    for d in sorted(Path(a.dir).iterdir()):
        if not d.is_dir() or d.name not in classes:
            continue
        mask = torch.tensor(crops == crop_of(d.name))
        true_idx = classes.index(d.name)
        for f in d.iterdir():
            if f.suffix.lower() not in EXT:
                continue
            x = tf(Image.open(f).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)[0]
            full = torch.softmax(logits, 0)
            masked = torch.softmax(logits.masked_fill(~mask, -1e9), 0)
            top3 = masked.topk(min(3, int(mask.sum()))).indices.tolist()
            rows.append(dict(true=d.name,
                             pred_all=classes[int(full.argmax())],
                             pred_crop=classes[int(masked.argmax())],
                             conf_crop=float(masked.max()),
                             in_top3=true_idx in top3))
    df = pd.DataFrame(rows)
    Path("eval").mkdir(exist_ok=True)
    df.to_csv("eval/field_crop_predictions.csv", index=False)

    print(f"images: {len(df)}")
    print(f"crop unknown (any class)      : {(df.true == df.pred_all).mean():.3f}")
    print(f"crop selected by the farmer   : {(df.true == df.pred_crop).mean():.3f}")
    print(f"crop selected, correct in top3: {df.in_top3.mean():.3f}")
    per = df.assign(ok=df.true == df.pred_crop).groupby("true")["ok"].agg(["mean", "count"])
    print(per.round(2).to_string())
    print("\nthreshold  coverage  accuracy_on_kept (crop selected)")
    for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        keep = df[df.conf_crop >= t]
        acc = (keep.true == keep.pred_crop).mean() if len(keep) else float("nan")
        print(f"{t:>9}  {len(keep)/len(df):>8.3f}  {acc:>16.3f}")


if __name__ == "__main__":
    main()