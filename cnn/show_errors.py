import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("eval/field_predictions.csv")
bad = df[(df.true != "_unknown") & (df.true != df.pred)]
bad = bad.sort_values("conf", ascending=False).head(12)
fig, axs = plt.subplots(3, 4, figsize=(14, 11))
for ax, (_, r) in zip(axs.flat, bad.iterrows()):
    ax.imshow(Image.open(r.path).convert("RGB"))
    ax.set_title(f"true: {r.true.split('___')[-1][:18]}\npred: {r.pred.split('___')[-1][:18]} ({r.conf:.2f})", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("eval/field_confident_errors.png", dpi=100)
print("saved eval/field_confident_errors.png")
