import argparse, io, subprocess
from math import comb
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base-commit", default="e513d05")
ap.add_argument("--after", default="eval/field_predictions.csv")
a = ap.parse_args()

txt = subprocess.run(["git", "show", f"{a.base_commit}:eval/field_predictions.csv"],
                     capture_output=True, check=True).stdout.decode("utf-8")
before = pd.read_csv(io.StringIO(txt))
after = pd.read_csv(a.after)
m = before.merge(after, on="path", suffixes=("_b", "_a"))
m = m[m.true_b != "_unknown"]
cb, ca = (m.true_b == m.pred_b).values, (m.true_a == m.pred_a).values
fixed, broken = int((~cb & ca).sum()), int((cb & ~ca).sum())
n, k = fixed + broken, min(fixed, broken)
p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
rng = np.random.RandomState(0)
diffs = []
for _ in range(10000):
    s = rng.randint(0, len(m), len(m))
    diffs.append(ca[s].mean() - cb[s].mean())
print(f"images: {len(m)} | before {cb.mean():.3f} | after {ca.mean():.3f}")
print(f"fixed {fixed}, broken {broken}, McNemar p = {p:.4f}")
print("difference 95% CI:", np.percentile(diffs, [2.5, 97.5]).round(3))