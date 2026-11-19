"""Split one class's images into data/<class>/train and data/<class>/test.

Files are copied, never moved, so the source folders stay untouched.
Near-duplicate images (same perceptual hash within a small distance) are kept
together in the same split so a photo can't appear in both train and test.

Usage:
    python scripts/split_dataset.py sunflower --src raw_data/sunflower
    python scripts/split_dataset.py pug --src raw_data/pug --src new_data/pug
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def phash_int(path):
    with Image.open(path) as im:
        im.load()
        return int(str(imagehash.phash(im.convert("RGB"))), 16)


def group_near_duplicates(hashes, max_distance):
    """Union-find over all pairs whose 64-bit pHash differs by <= max_distance bits."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    arr = np.array(hashes, dtype=np.uint64)
    for i in range(n - 1):
        dist = np.bitwise_count(arr[i + 1:] ^ arr[i])
        for j in np.nonzero(dist <= max_distance)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("class_name")
    ap.add_argument("--src", action="append", required=True, help="source folder (repeatable)")
    ap.add_argument("--test-ratio", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-distance", type=int, default=4, help="pHash bits for near-duplicates")
    args = ap.parse_args()

    dest = ROOT / "data" / args.class_name
    for sub in ("train", "test"):
        d = dest / sub
        if d.exists() and any(d.iterdir()):
            sys.exit(f"{d} already has files; refusing to mix splits. Remove it first.")

    files = []
    for src in args.src:
        src_dir = (ROOT / src).resolve()
        files += sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    good, hashes, bad = [], [], []
    for p in files:
        try:
            hashes.append(phash_int(p))
            good.append(p)
        except Exception as e:  # unreadable or corrupt image
            bad.append((p, repr(e)))

    groups = group_near_duplicates(hashes, args.max_distance)
    dup_groups = [g for g in groups if len(g) > 1]

    rng = random.Random(args.seed)
    rng.shuffle(groups)
    target_test = round(len(good) * args.test_ratio)
    assignment, n_test = {}, 0
    for g in groups:
        split = "test" if n_test < target_test else "train"
        if split == "test":
            n_test += len(g)
        for i in g:
            assignment[i] = split

    names_used = set()
    rows = []
    for i, p in enumerate(good):
        split = assignment[i]
        name = p.name
        if name in names_used:  # same filename coming from two source folders
            name = f"{p.parent.name}_{name}"
        names_used.add(name)
        out = dest / split / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        rows.append([str(out.relative_to(ROOT)), split, str(p.relative_to(ROOT)), f"{hashes[i]:016x}"])

    man_dir = ROOT / "manifests"
    man_dir.mkdir(exist_ok=True)
    with open(man_dir / f"{args.class_name}_split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "source_file", "phash"])
        w.writerows(rows)

    n_train = sum(1 for r in rows if r[1] == "train")
    print(f"{args.class_name}: {len(files)} files, {len(bad)} unreadable, "
          f"{len(dup_groups)} near-duplicate groups ({sum(len(g) for g in dup_groups)} images)")
    print(f"  train={n_train} test={len(rows) - n_train}  (seed {args.seed})")
    for p, err in bad:
        print(f"  skipped unreadable: {p} {err}")


if __name__ == "__main__":
    main()
