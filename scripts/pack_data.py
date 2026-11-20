"""Pack data/ into one zip to upload to Google Colab (Google Drive) or Kaggle (a dataset).

    python scripts/pack_data.py                                # -> dist/imgcls_data.zip
    python scripts/pack_data.py --out D:/uploads/imgcls_data.zip

Only reads data/; nothing there is changed. JPEGs are already compressed, so files are stored as they
are: fast, and the zip is about the size of data/. Inside, the layout is data/<class>/{train,test}/...,
which imgcls.setup_environment() finds on Colab and Kaggle. See README > Colab and Kaggle.
"""

import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tqdm import tqdm  # noqa: E402

from imgcls.config import DATA_DIR, EXPECTED_CLASSES  # noqa: E402
from imgcls.data import list_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "imgcls_data.zip")
    args = ap.parse_args()

    files, counts = [], Counter()
    for cls in sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir()):
        for split in ("train", "test"):
            found = list_images(DATA_DIR / cls / split)
            files += found
            counts[cls, split] = len(found)
    missing = [c for c in EXPECTED_CLASSES if not counts[c, "train"]]
    if missing:
        sys.exit(f"No training images for {missing} in {DATA_DIR}; nothing packed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in tqdm(files, desc="packing", unit="img"):
            z.write(f, "data/" + f.relative_to(DATA_DIR).as_posix())
    with zipfile.ZipFile(partial) as z:
        packed = len(z.namelist())
        bad = z.testzip()
    if packed != len(files) or bad:
        sys.exit(f"Zip check failed ({packed} of {len(files)} files, first bad entry: {bad}); "
                 f"left {partial} for inspection.")
    partial.replace(args.out)

    print(f"\n{'class':20s} {'train':>6s} {'test':>5s}")
    for cls in sorted({c for c, _ in counts}):
        print(f"{cls:20s} {counts[cls, 'train']:6d} {counts[cls, 'test']:5d}")
    size_mb = args.out.stat().st_size / 2**20
    print(f"{len(files)} images -> {args.out} ({size_mb:,.0f} MB). Every entry checked.")


if __name__ == "__main__":
    main()
