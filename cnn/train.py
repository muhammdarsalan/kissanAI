import argparse, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from dataset import get_loaders


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None,
              use_amp=False, freeze_bn=False):
    train = optimizer is not None
    model.train(train)
    if train and freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    total, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * x.size(0)
            preds += out.argmax(1).cpu().tolist()
            targets += y.cpu().tolist()
    n = len(targets)
    acc = float(np.mean(np.array(preds) == np.array(targets)))
    return total / n, acc, f1_score(targets, preds, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv3_large_100")
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--ft-epochs", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU-friendly test run")
    a = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    bs, workers = (16, 0) if a.smoke else (a.batch_size, a.workers)
    loaders, classes, weights = get_loaders(batch_size=bs, num_workers=workers)

    if a.smoke:
        for split, n in (("train", 256), ("val", 128)):
            ds = loaders[split].dataset
            idx = torch.randperm(len(ds))[:n].tolist()
            loaders[split] = DataLoader(Subset(ds, idx), batch_size=bs,
                                        shuffle=(split == "train"), num_workers=0)
        a.head_epochs, a.ft_epochs = 1, 1

    model = timm.create_model(a.model, pretrained=True, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    Path("models").mkdir(exist_ok=True)
    Path("eval").mkdir(exist_ok=True)
    tag = a.model + ("_smoke" if a.smoke else "")
    state = {"best": -1.0}
    log = []

    def phase(name, epochs, lr, train_all):
        for p in model.parameters():
            p.requires_grad = train_all
        for p in model.get_classifier().parameters():
            p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tl, ta, tf = run_epoch(model, loaders["train"], device, criterion,
                                   opt, scaler, use_amp, freeze_bn=not train_all)
            vl, va, vf = run_epoch(model, loaders["val"], device, criterion, use_amp=use_amp)
            sched.step()
            print(f"[{name} {ep}/{epochs}] train loss {tl:.3f} acc {ta:.3f} | "
                  f"val loss {vl:.3f} acc {va:.3f} macroF1 {vf:.3f} | {time.time()-t0:.0f}s")
            log.append(dict(phase=name, epoch=ep, train_loss=tl, train_acc=ta, train_f1=tf,
                            val_loss=vl, val_acc=va, val_f1=vf))
            if vf > state["best"]:
                state["best"] = vf
                torch.save({"model": model.state_dict(), "classes": classes,
                            "arch": a.model}, f"models/{tag}_best.pt")

    phase("head", a.head_epochs, 1e-3, train_all=False)
    phase("finetune", a.ft_epochs, 1e-4, train_all=True)
    pd.DataFrame(log).to_csv(f"eval/{tag}_log.csv", index=False)
    print("best val macro-F1:", round(state["best"], 4))


if __name__ == "__main__":
    main()