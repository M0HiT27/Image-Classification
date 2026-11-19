"""Copy our four breeds out of the Tsinghua Dogs low-resolution archive into new_data/<breed>/.

Reads the zip directly (no full extraction). A random sample (fixed seed) of up to
--cap images per breed is saved as new_data/<breed>/thu_<original name>.
Writes manifests/new_data_tsinghua.csv.

Dataset page: https://cg.cs.tsinghua.edu.cn/ThuDogs/ (academic use)
"""

import argparse
import csv
import io
import random
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BREEDS = {  # Tsinghua folder id -> our class name
    "n000126": "golden_retriever",
    "n000122": "labrador_retriever",
    "n000003": "siberian_husky",
    "n000130": "pug",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def breed_of(member):
    for part in PurePosixPath(member).parts[:-1]:
        for folder_id, cls in BREEDS.items():
            if folder_id in part:
                return cls
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--out", default="new_data")
    ap.add_argument("--cap", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip_path)
    members = {cls: [] for cls in BREEDS.values()}
    for name in zf.namelist():
        if name.endswith("/") or PurePosixPath(name).suffix.lower() not in IMAGE_EXTS:
            continue
        cls = breed_of(name)
        if cls:
            members[cls].append(name)

    rows = []
    for cls, names in members.items():
        names.sort()
        random.Random(args.seed).shuffle(names)
        chosen = names[: args.cap]
        out_dir = ROOT / args.out / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = bad = 0
        for name in chosen:
            data = zf.read(name)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
            except Exception:
                bad += 1
                continue
            # our own file name, never the archive path, so nothing can land outside out_dir
            out = out_dir / f"thu_{PurePosixPath(name).name}"
            out.write_bytes(data)
            saved += 1
            rows.append([str(out.relative_to(ROOT)), cls, "tsinghua_dogs", name])
        print(f"{cls:20s} in archive={len(names):5d} saved={saved:5d} unreadable={bad}")

    man = ROOT / "manifests"
    man.mkdir(exist_ok=True)
    with open(man / "new_data_tsinghua.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "source", "archive_path"])
        w.writerows(sorted(rows))


if __name__ == "__main__":
    main()
