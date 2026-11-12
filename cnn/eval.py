import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import timm
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from dataset import get_loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/mobilenetv3_large_100_best.pt")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu")
    classes = ck["classes"]
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["model"])
    model.eval().to(device)

    loaders, data_classes, _ = get_loaders(batch_size=a.batch_size, num_workers=0)
    assert data_classes == classes, "class list differs from the checkpoint"
    loader = loaders[a.split]

    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1).cpu()
            probs.append(p)
            targets += y.tolist()
    probs = torch.cat(probs).numpy()
    targets = np.array(targets)
    preds = probs.argmax(1)
    conf = probs.max(1)

    tag = f"{Path(a.ckpt).stem}_{a.split}"
    Path("eval").mkdir(exist_ok=True)
    acc = (preds == targets).mean()
    macro = f1_score(targets, preds, average="macro")
    print(f"{a.split}: accuracy {acc:.4f} | macro-F1 {macro:.4f} | n={len(targets)}")

    rep = classification_report(targets, preds, labels=range(len(classes)),
                                target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(f"eval/{tag}_per_class.csv")
    print(classification_report(targets, preds, labels=range(len(classes)),
                                target_names=classes, zero_division=0))

    cm = confusion_matrix(targets, preds, labels=range(len(classes)))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    short = [c.replace("___", ": ")[:28] for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), short, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"eval/{tag}_confusion.png", dpi=110)

    rows = []
    for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        keep = conf >= t
        rows.append(dict(threshold=t, coverage=keep.mean(),
                         accuracy_on_kept=(preds[keep] == targets[keep]).mean() if keep.any() else np.nan))
    cov = pd.DataFrame(rows)
    print(cov.round(3).to_string(index=False))
    cov.to_csv(f"eval/{tag}_confidence.csv", index=False)

    out = pd.DataFrame({"path": loader.dataset.paths,
                        "true": [classes[i] for i in targets],
                        "pred": [classes[i] for i in preds],
                        "confidence": conf})
    out.to_csv(f"eval/{tag}_predictions.csv", index=False)


if __name__ == "__main__":
    main()