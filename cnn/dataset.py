import json
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

SPLITS_CSV = "data/processed/splits.csv"
CLASSES_JSON = "configs/classes.json"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def make_transforms(img_size=224):
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.RandomApply([T.GaussianBlur(5)], p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf

class LeafDataset(Dataset):
    def __init__(self, df, label2idx, transform):
        self.paths = df["path"].str.replace("\\", "/", regex=False).tolist()
        self.targets = [label2idx[l] for l in df["label"]]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.targets[i]

def get_loaders(batch_size=32, img_size=224, num_workers=2, csv=SPLITS_CSV):
    df = pd.read_csv(csv)
    classes = sorted(df["label"].unique())
    label2idx = {c: i for i, c in enumerate(classes)}
    Path("configs").mkdir(exist_ok=True)
    Path(CLASSES_JSON).write_text(json.dumps(classes, indent=2))
    train_tf, eval_tf = make_transforms(img_size)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    tfs = {"train": train_tf, "val": eval_tf, "test": eval_tf}
    sets = {s: LeafDataset(parts[s], label2idx, tfs[s]) for s in parts}
    loaders = {
        s: DataLoader(sets[s], batch_size=batch_size, shuffle=(s == "train"),
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
        for s in sets
    }
    counts = parts["train"]["label"].value_counts()
    weights = torch.tensor(
        [len(parts["train"]) / (len(classes) * counts[c]) for c in classes],
        dtype=torch.float)
    return loaders, classes, weights

if __name__ == "__main__":
    loaders, classes, weights = get_loaders(batch_size=16, num_workers=0)
    print(len(classes), "classes")
    for s, l in loaders.items():
        print(s, len(l.dataset), "images")
    x, y = next(iter(loaders["train"]))
    print("batch:", x.shape, y[:8].tolist())
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for ax, img, t in zip(axs.flat, x, y):
        ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0))
        ax.set_title(classes[t][:24], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    Path("eval").mkdir(exist_ok=True)
    plt.savefig("eval/aug_check.png", dpi=100)
    print(dict(zip(classes, weights.round(decimals=2).tolist())))