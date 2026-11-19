"""Download candidate images for our classes from Open Images V7 into new_data/<class>/.

Input: the Open Images label CSVs already filtered down to our classes' label IDs
(bbox_<split>.csv and lbl_<split>.csv, header rows removed), in --labels-dir.

Selection rules:
  * image-level labels must be human-verified positives (Confidence = 1)
  * an image that is positive for two or more of our classes is skipped (ambiguous)
  * for classes with bounding boxes (banana, orange, mushroom): drawings/cartoons
    (IsDepiction = 1) are ignored and the largest box must cover >= --min-area
    of the image, so the object is the main subject rather than a speck in a scene

Images come from the public Open Images S3 bucket and are saved as
new_data/<class>/oi_<ImageID>.jpg. A manifest is written to
manifests/new_data_openimages.csv. Re-running skips files already downloaded.
"""

import argparse
import csv
import io
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"train": "train", "validation": "validation", "test": "test"}
S3 = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"

# class -> (Open Images label IDs, uses bounding boxes?, default cap)
CLASSES = {
    "banana":             (["/m/09qck"], True, 1200),
    "orange":             (["/m/0cyhj_"], True, 1200),
    "mushroom":           (["/m/052sf"], True, 1200),
    "pine_cone":          (["/m/02_wsd"], False, 1200),
    "sunflower":          (["/m/03h54r", "/m/0ftb8"], False, 500),
    "golden_retriever":   (["/m/01t032"], False, 300),
    "labrador_retriever": (["/m/0km3f"], False, 300),
    "siberian_husky":     (["/m/071jj"], False, 300),
    "pug":                (["/m/016wkx"], False, 300),
}
# butterfly is only used to detect images that also show another of our classes
EXTRA_MIDS = {"/m/0cyf8": "butterfly"}

MID_TO_CLASS = {m: c for c, (mids, _, _) in CLASSES.items() for m in mids}
MID_TO_CLASS.update(EXTRA_MIDS)


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def collect_candidates(labels_dir, min_area):
    positives = defaultdict(set)          # (split, id) -> our classes it is positive for
    max_area = defaultdict(float)         # (split, id, class) -> largest non-depiction box area
    has_box = set()                       # (split, id, class) with any box at all
    for split in SPLITS:
        for row in read_rows(labels_dir / f"lbl_{split}.csv"):
            image_id, _, mid, conf = row[:4]
            if mid in MID_TO_CLASS and conf.strip() in ("1", "1.0"):
                positives[(split, image_id)].add(MID_TO_CLASS[mid])
        for row in read_rows(labels_dir / f"bbox_{split}.csv"):
            image_id, _, mid = row[:3]
            if mid not in MID_TO_CLASS:
                continue
            xmin, xmax, ymin, ymax = map(float, row[4:8])
            is_depiction = row[11].strip() == "1"
            cls = MID_TO_CLASS[mid]
            positives[(split, image_id)].add(cls)
            has_box.add((split, image_id, cls))
            if not is_depiction:
                key = (split, image_id, cls)
                max_area[key] = max(max_area[key], (xmax - xmin) * (ymax - ymin))

    candidates = defaultdict(list)
    unboxed = defaultdict(list)           # box classes: verified positive but never boxed (size unknown)
    stats = defaultdict(lambda: defaultdict(int))
    for (split, image_id), classes in positives.items():
        for cls in classes:
            if cls not in CLASSES:
                continue
            stats[cls]["positive"] += 1
            if len(classes) > 1:
                stats[cls]["skipped_multi_class"] += 1
                continue
            uses_boxes = CLASSES[cls][1]
            area = max_area.get((split, image_id, cls), 0.0)
            if uses_boxes and area < min_area:
                stats[cls]["skipped_small_or_drawing"] += 1
                if (split, image_id, cls) not in has_box:
                    unboxed[cls].append((split, image_id, 0.0))
                continue
            candidates[cls].append((split, image_id, area))
    return candidates, unboxed, stats


def load_metadata(path):
    """ImageID -> row from the Open Images image metadata (Flickr URLs, license, author)."""
    meta = {}
    if not path:
        return meta
    cols = ["ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License", "AuthorProfileURL",
            "Author", "Title", "OriginalSize", "OriginalMD5", "Thumbnail300KURL", "Rotation"]
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= len(cols):
                meta[row[0]] = dict(zip(cols, row))
    return meta


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        final_url = r.geturl()
        data = r.read()
    if "unavailable" in final_url:  # Flickr serves a placeholder image for deleted photos
        raise ValueError("photo no longer available on Flickr")
    with Image.open(io.BytesIO(data)) as im:
        im.verify()
    return data


def download(split, image_id, out_path, meta):
    if out_path.exists():
        return out_path, S3.format(split=split, id=image_id), "exists"
    urls = [S3.format(split=split, id=image_id)]
    m = meta.get(image_id)
    if m:  # images without boxes are not on S3; fall back to their Flickr copies
        urls += [u for u in (m["Thumbnail300KURL"], m["OriginalURL"]) if u]
    err = None
    for url in urls:
        try:
            out_path.write_bytes(fetch(url))
            return out_path, url, "ok"
        except Exception as e:
            err = e
    return out_path, urls[-1], f"failed: {err!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out", default="new_data")
    ap.add_argument("--min-area", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", help="limit to these classes")
    ap.add_argument("--dry-run", action="store_true", help="print counts, download nothing")
    ap.add_argument("--include-unboxed", type=int, default=0,
                    help="box classes: also take up to N verified-positive images that were never boxed")
    ap.add_argument("--metadata", help="filtered Open Images image metadata CSV (Flickr fallback URLs)")
    ap.add_argument("--write-train-ids", help="write the selected train ImageIDs to this file and stop")
    args = ap.parse_args()

    candidates, unboxed, stats = collect_candidates(Path(args.labels_dir), args.min_area)
    meta = load_metadata(args.metadata)
    classes = args.only or list(CLASSES)

    jobs = []
    for cls in classes:
        cap = CLASSES[cls][2]
        pool = sorted(candidates[cls])
        random.Random(args.seed).shuffle(pool)
        chosen = pool[:cap]
        extra = []
        if CLASSES[cls][1] and args.include_unboxed:
            extra = sorted(unboxed[cls])
            random.Random(args.seed).shuffle(extra)
            extra = extra[:args.include_unboxed]
        s = stats[cls]
        print(f"{cls:20s} positive={s['positive']:6d} multi-class={s['skipped_multi_class']:5d} "
              f"small/drawing={s['skipped_small_or_drawing']:5d} usable={len(pool):6d} -> taking {len(chosen)}"
              + (f" + {len(extra)} unboxed (of {len(unboxed[cls])})" if extra else ""))
        out_dir = ROOT / args.out / cls
        if not args.dry_run and not args.write_train_ids:
            out_dir.mkdir(parents=True, exist_ok=True)
        for split, image_id, area in chosen + extra:
            jobs.append((cls, split, image_id, area, out_dir / f"oi_{image_id}.jpg"))

    if args.write_train_ids:
        ids = sorted({iid for _, split, iid, _, _ in jobs if split == "train"})
        Path(args.write_train_ids).write_text("".join(f"{i},\n" for i in ids), encoding="utf-8")
        print(f"wrote {len(ids)} train ids to {args.write_train_ids}")
        return
    if args.dry_run:
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, split, iid, path, meta): (cls, split, iid, area)
                for cls, split, iid, area, path in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            cls, split, iid, area = futs[fut]
            path, url, status = fut.result()
            results.append((cls, split, iid, area, path, url, status))
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)} done")

    # merge with rows from earlier runs for classes not fetched this time
    man = ROOT / "manifests"
    man.mkdir(exist_ok=True)
    man_path = man / "new_data_openimages.csv"
    header = ["file", "class", "source", "source_id", "source_split", "largest_box_area",
              "url", "license", "author", "status"]
    kept = []
    if man_path.exists():
        with open(man_path, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if r["class"] not in classes]
    new_rows = []
    for cls, split, iid, area, path, url, status in sorted(results):
        m = meta.get(iid, {})
        new_rows.append(dict(zip(header, [str(path.relative_to(ROOT)), cls, "openimages_v7", iid, split,
                                          f"{area:.3f}" if area else "", url, m.get("License", ""),
                                          m.get("Author", ""), status])))
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(kept + new_rows, key=lambda r: (r["class"], r["file"])))

    by_class = defaultdict(lambda: defaultdict(int))
    for cls, *_, status in results:
        by_class[cls]["ok" if status in ("ok", "exists") else "failed"] += 1
    for cls in classes:
        print(f"{cls:20s} saved={by_class[cls]['ok']:5d} failed={by_class[cls]['failed']:4d}")


if __name__ == "__main__":
    main()
