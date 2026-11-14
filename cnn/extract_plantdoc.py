import argparse, re, subprocess
from pathlib import Path
from PIL import Image

REPO = Path("data/raw/plantdoc")
OUT = Path("data/raw/plantdoc_clean")
EXT = {".jpg", ".jpeg", ".png", ".webp"}


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, check=True).stdout


def clean(name):
    name = re.sub(r'[<>:"/\\|?*%\x00-\x1f]', "_", name).strip(" .")
    return name or "file"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test"])
    a = ap.parse_args()

    raw = git("ls-tree", "-r", "-z", "--name-only", "HEAD")
    paths = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
    print(len(paths), "files in the repo")

    counts, bad = {}, 0
    for i, path in enumerate(paths):
        parts = path.split("/")
        if len(parts) != 3 or parts[0] not in a.splits:
            continue
        split, folder, fname = parts
        suffix = Path(fname).suffix.lower()
        if suffix not in EXT:
            continue
        stem = clean(Path(fname).stem)[:90]
        dest = OUT / split / clean(folder) / f"{i:05d}_{stem}{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(git("show", f"HEAD:{path}"))
            Image.open(dest).verify()
        except Exception:
            dest.unlink(missing_ok=True)
            bad += 1
            continue
        counts[(split, folder)] = counts.get((split, folder), 0) + 1

    for (split, folder), n in sorted(counts.items()):
        print(f"{split:6s} {folder:45s} {n:4d}")
    print("extracted:", sum(counts.values()), "| unreadable/skipped:", bad)


if __name__ == "__main__":
    main()