import argparse
from pathlib import Path
import pandas as pd
import timm
import torch
from PIL import Image
from dataset import make_transforms

EXT = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/mobilenetv3_large_100_best.pt")
    ap.add_argument("--dir", default="data/field_test")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu")
    classes = ck["classes"]
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    _, tf = make_transforms()

    rows = []
    for d in sorted(Path(a.dir).iterdir()):
        if not d.is_dir():
            continue
        if d.name != "_unknown" and d.name not in classes:
            print("skipping folder (not a class name):", d.name)
            continue
        for f in d.iterdir():
            if f.suffix.lower() not in EXT:
                continue
            x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.softmax(model(x), dim=1)[0].cpu()
            rows.append(dict(path=str(f), true=d.name,
                             pred=classes[int(p.argmax())], conf=float(p.max())))
    df = pd.DataFrame(rows)
    Path("eval").mkdir(exist_ok=True)
    df.to_csv("eval/field_predictions.csv", index=False)

    known = df[df.true != "_unknown"]
    if len(known):
        print(f"FIELD accuracy: {(known.true == known.pred).mean():.3f} on {len(known)} images")
        per = known.assign(ok=known.true == known.pred).groupby("true")["ok"].agg(["mean", "count"])
        print(per.round(2).to_string())
        wrong = known[known.true != known.pred].groupby(["true", "pred"]).size()
        print("most common mistakes:\n", wrong.sort_values(ascending=False).head(10).to_string())
        print("\nthreshold  coverage  accuracy_on_kept")
        for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
            keep = known[known.conf >= t]
            acc = (keep.true == keep.pred).mean() if len(keep) else float("nan")
            print(f"{t:>9}  {len(keep)/len(known):>8.3f}  {acc:>16.3f}")
    unk = df[df.true == "_unknown"]
    if len(unk):
        print(f"\nUNKNOWN leaves: {len(unk)} images")
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            print(f"  rejected at threshold {t}: {(unk.conf < t).mean():.3f}")

if __name__ == "__main__":
    main()