"""Build data_object/: a detection dataset (images + boxes, YOLO format) made from our classification photos.

Our photos only say WHAT they show; a detector also needs WHERE. The boxes come from two sources:
  * human boxes from Open Images V7, for photos that came from Open Images (oi_<ImageID>.jpg: all banana,
    orange and mushroom photos, part of pine cone, sunflower and the dogs). Google's box files are streamed once
    and only the rows of our photos are kept (_cache/openimages_boxes.csv);
  * OWLv2 (imgcls/detect.py) for every other photo: its boxes of the photo's own class are kept (any "dog" box
    for the four breeds; the breed comes from the folder). OWLv2 results are cached per photo
    (_cache/owlv2_<model>.jsonl), so an interrupted or bigger build only runs it on new photos.
Photos where neither source has a box of their own class are left out: as images without boxes they would
teach the detector that the object is background. OWLv2 boxes are pseudo-labels: good, not perfect. The
contact sheets (contact_sheet) are there to spot-check them.

Written layout (readable by ultralytics and by imgcls.objdet.dataset):
  data.yaml                               classes and split folders
  images/{train,val,test}/<class>__<name>.jpg   longer side <= max_side pixels
  labels/{train,val,test}/<class>__<name>.txt   one line per box: class cx cy w h (0-1, of the image size)
  boxes.csv                               every box with its source (openimages / owlv2) and score
  info.json                               settings and counts
  _cache/                                 Open Images rows and OWLv2 results, reused by later builds
val is carved out of data/<class>/train (stratified, like the classifiers' validation split); test comes from
data/<class>/test, so a detector's test score is measured on photos no model was trained on.
"""

import csv
import json
import random
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageOps

from .. import config
from ..data import list_images, load_rgb, split_train_val
from .paths import CLASSES

SPLITS = ("train", "val", "test")
DOG_CLASSES = ("golden_retriever", "labrador_retriever", "siberian_husky", "pug")
OPENIMAGES_BOX_CSV = {  # Open Images V7 boxes (the same as V6): train 2.26 GB, validation 25 MB, test 77 MB
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
}
_DOG_MID = "/m/0bt9lr"  # Open Images "Dog": boxes exist for it, not for the breeds
OPENIMAGES_LABELS = {  # Open Images label ids whose boxes count for each class (scripts/fetch_openimages.py)
    "banana": {"/m/09qck"}, "orange": {"/m/0cyhj_"}, "mushroom": {"/m/052sf"}, "pine_cone": {"/m/02_wsd"},
    "sunflower": {"/m/03h54r", "/m/0ftb8"}, "butterfly": {"/m/0cyf8"},
    "golden_retriever": {_DOG_MID, "/m/01t032"}, "labrador_retriever": {_DOG_MID, "/m/0km3f"},
    "siberian_husky": {_DOG_MID, "/m/071jj"}, "pug": {_DOG_MID, "/m/016wkx"},
}
SOURCE_COLORS = {"openimages": (40, 200, 60), "owlv2": (255, 140, 0)}


@dataclass
class Photo:
    path: Path
    cls: str
    split: str                   # train / val / test in data_object
    oi_id: Optional[str] = None  # Open Images ImageID, when the photo came from Open Images

    @property
    def key(self):
        """File name (without extension) in data_object, also the cache key."""
        return f"{self.cls}__{self.path.stem}"


def select_photos(max_train_per_class=700, test_per_class=150, val_fraction=0.10, limit_per_class=None,
                  seed=42, data_dir=None):
    """Pick the photos: up to max_train_per_class from data/<class>/train (val carved out of them) and up to
    test_per_class from data/<class>/test. limit_per_class (a smoke test) overrides both caps."""
    data_dir = Path(data_dir or config.DATA_DIR)
    if limit_per_class:
        max_train_per_class, test_per_class = limit_per_class, max(1, limit_per_class // 3)
    photos = []
    for idx, cls in enumerate(CLASSES):
        train = list_images(data_dir / cls / "train")
        test = list_images(data_dir / cls / "test")
        rng = random.Random(f"{seed}-{cls}")
        rng.shuffle(train)
        rng.shuffle(test)
        train = train[:max_train_per_class] if max_train_per_class else train
        test = test[:test_per_class] if test_per_class else test
        tr, va = split_train_val([(str(f), idx) for f in train], val_fraction, seed)
        photos += [Photo(Path(f), cls, "train") for f, _ in tr] + [Photo(Path(f), cls, "val") for f, _ in va]
        photos += [Photo(f, cls, "test") for f in sorted(test)]
    for p in photos:
        if p.path.stem.startswith("oi_"):
            p.oi_id = p.path.stem[3:]
    return photos


def photo_table(photos):
    """Photos per class and split, and how many came from Open Images."""
    import pandas as pd

    rows = {cls: Counter() for cls in CLASSES}
    for p in photos:
        rows[p.cls][p.split] += 1
        rows[p.cls]["from Open Images"] += p.oi_id is not None
    return pd.DataFrame(rows).T.reindex(columns=[*SPLITS, "from Open Images"]).fillna(0).astype(int)


# ---------------------------------------------------------------- human boxes from Open Images

def _manifest_splits():
    """Open Images split of every photo we downloaded from there: {ImageID: train / validation / test}."""
    path = config.ROOT / "manifests" / "new_data_openimages.csv"
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["source_id"]: r["source_split"] for r in csv.DictReader(f) if r.get("source_id")}


def _stream_filter(url, want, log, attempts=5):
    """Rows of the Open Images box file at url whose ImageID is in want. The file is read as a stream (never
    saved); a dropped connection resumes where it stopped."""
    want = {i.encode() for i in want}
    columns = ("ImageID", "LabelName", "XMin", "YMin", "XMax", "YMax", "IsGroupOf", "IsDepiction")
    kept, done, col, total, t0 = [], 0, None, 0, time.time()
    for attempt in range(1, attempts + 1):
        headers = {"User-Agent": "imgcls"} if done == 0 else {"User-Agent": "imgcls", "Range": f"bytes={done}-"}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                if done == 0:
                    total = int(r.headers.get("Content-Length") or 0)
                    log(f"  streaming {url.rsplit('/', 1)[-1]} ({total / 1e9:.2f} GB; only our photos' rows are kept)")
                    header_line = r.readline()
                    done += len(header_line)
                    header = header_line.decode().strip().split(",")
                    col = [header.index(c) for c in columns]
                next_report = (done // 250_000_000 + 1) * 250_000_000
                for line in r:
                    done += len(line)
                    if line[:16] in want:  # ImageIDs are 16 characters, first column
                        f = line.decode().rstrip("\r\n").split(",")
                        kept.append([f[i] for i in col])
                    if done >= next_report:
                        rate = done / max(time.time() - t0, 1e-6)
                        log(f"    {done / 1e9:.2f} / {total / 1e9:.2f} GB, {len(kept)} rows kept, "
                            f"about {(total - done) / max(rate, 1) / 60:.0f} min left")
                        next_report += 250_000_000
            return kept
        except (OSError, ValueError) as e:  # URLError, connection reset, timeout
            if attempt == attempts:
                raise
            log(f"    connection problem ({e!r}); resuming at {done / 1e9:.2f} GB")
            time.sleep(5 * attempt)
    return kept


def openimages_boxes(cache_dir, data_dir=None, log=print):
    """Human boxes for every photo in data/ that came from Open Images (all of them, not only the selected
    ones, so one download serves every later build). Returns {ImageID: [(label, x1, y1, x2, y2, is_group,
    is_depiction), ...]} with coordinates 0-1."""
    data_dir, cache_dir = Path(data_dir or config.DATA_DIR), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows_file, searched_file = cache_dir / "openimages_boxes.csv", cache_dir / "openimages_searched.json"
    ids = {p.stem[3:] for cls in CLASSES for split in ("train", "test")
           for p in list_images(data_dir / cls / split) if p.stem.startswith("oi_")}
    searched = set(json.loads(searched_file.read_text())) if searched_file.exists() else set()
    rows = []
    if rows_file.exists():
        with open(rows_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]
    todo = ids - searched
    if todo:
        split_of = _manifest_splits()
        by_split = defaultdict(set)
        for image_id in todo:
            by_split[split_of.get(image_id)].add(image_id)
        unknown = by_split.pop(None, set())  # not in the manifest: look in every file
        log(f"Open Images boxes for {len(todo)} photos ({len(searched)} looked up before)...")
        for split, url in OPENIMAGES_BOX_CSV.items():
            want = by_split.get(split, set()) | unknown
            if want:
                rows += _stream_filter(url, want, log)
        searched |= todo
        with open(rows_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ImageID", "LabelName", "XMin", "YMin", "XMax", "YMax", "IsGroupOf", "IsDepiction"])
            w.writerows(rows)
        searched_file.write_text(json.dumps(sorted(searched)))
    boxes = defaultdict(list)
    for image_id, label, x1, y1, x2, y2, group, depiction in rows:
        boxes[image_id].append((label, float(x1), float(y1), float(x2), float(y2), group == "1", depiction == "1"))
    log(f"Open Images: human boxes for {len(boxes)} of our {len(ids)} Open Images photos")
    return dict(boxes)


def human_boxes(photo, human):
    """[((x1, y1, x2, y2) 0-1, "openimages", 1.0)] boxes of the photo's own class drawn by people."""
    if not photo.oi_id or photo.oi_id not in human:
        return []
    labels = OPENIMAGES_LABELS[photo.cls]
    return [((x1, y1, x2, y2), "openimages", 1.0) for label, x1, y1, x2, y2, _, _ in human[photo.oi_id]
            if label in labels]


# ---------------------------------------------------------------- pseudo-labels from OWLv2

def owlv2_boxes(photos, cache_dir, device="cpu", model_id=None, keep_threshold=0.1, log=print, every=25):
    """Run OWLv2 once per photo. Every box scoring >= keep_threshold (any class) is cached, so the threshold
    used for labels can change later without running OWLv2 again. Returns {photo key: record}."""
    from .. import detect

    model_id = model_id or detect.DEFAULT_DETECTOR
    cache = Path(cache_dir) / f"owlv2_{model_id.rsplit('/', 1)[-1]}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    results[rec["key"]] = rec
    todo = [p for p in photos if p.key not in results]
    log(f"OWLv2: {len(todo)} photos to do, {len(photos) - len(todo)} done in an earlier run (cache: {cache.name})")
    if not todo:
        return results
    det = detect.Detector(model_id, device=str(device))
    t0 = time.time()
    with open(cache, "a", encoding="utf-8") as f:
        for n, p in enumerate(todo, 1):
            img = load_rgb(p.path)
            found = det.detect(img, threshold=keep_threshold)
            rec = {"key": p.key, "w": img.width, "h": img.height,
                   "boxes": [[*d["box"], d["detector_score"], d["detector_label"]] for d in found]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            results[p.key] = rec
            if n % every == 0 or n == len(todo):
                per = (time.time() - t0) / n
                log(f"  {n}/{len(todo)} photos, {per:.2f} s each, about {per * (len(todo) - n) / 60:.0f} min left")
    return results


def pseudo_boxes(photo, owl, threshold):
    """[((x1, y1, x2, y2) 0-1, "owlv2", score)] OWLv2 boxes of the photo's own class scoring >= threshold."""
    rec = owl.get(photo.key)
    if not rec:
        return []
    want = "Dog" if photo.cls in DOG_CLASSES else config.DISPLAY_NAMES[photo.cls]
    w, h = rec["w"], rec["h"]
    return [((x1 / w, y1 / h, x2 / w, y2 / h), "owlv2", score)
            for x1, y1, x2, y2, score, label in rec["boxes"] if label == want and score >= threshold]


# ---------------------------------------------------------------- writing the dataset

def write_dataset(photos, human, owl, out_dir, threshold=0.3, max_side=640, settings=None, log=print):
    """Write images, labels, data.yaml, boxes.csv and info.json (images/ and labels/ are rebuilt from scratch;
    _cache/ is kept). Returns a table of photos and boxes per class and split."""
    import pandas as pd
    import yaml

    out_dir = Path(out_dir)
    for sub in ("images", "labels"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
        for split in SPLITS:
            (out_dir / sub / split).mkdir(parents=True, exist_ok=True)
    counts = {cls: Counter() for cls in CLASSES}
    box_rows = []
    t0 = time.time()
    for n, p in enumerate(photos, 1):
        boxes = human_boxes(p, human) or pseudo_boxes(p, owl, threshold)
        clean = []
        for (x1, y1, x2, y2), source, score in boxes:
            x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
            if x2 - x1 > 0.005 and y2 - y1 > 0.005:
                clean.append(((x1, y1, x2, y2), source, score))
        if not clean:
            counts[p.cls][f"{p.split} no box"] += 1
            continue
        img = load_rgb(p.path)
        if max(img.size) > max_side:
            img = ImageOps.contain(img, (max_side, max_side), Image.Resampling.BICUBIC)
        img.save(out_dir / "images" / p.split / f"{p.key}.jpg", quality=90)
        cls_id = CLASSES.index(p.cls)
        lines = [f"{cls_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                 for (x1, y1, x2, y2), _, _ in clean]
        (out_dir / "labels" / p.split / f"{p.key}.txt").write_text("\n".join(lines) + "\n")
        counts[p.cls][p.split] += 1
        counts[p.cls][f"{p.split} boxes"] += len(clean)
        counts[p.cls][clean[0][1]] += 1
        box_rows += [[p.split, f"{p.key}.jpg", p.cls, source, round(score, 4), *(round(v, 6) for v in box)]
                     for box, source, score in clean]
        if n % 1000 == 0:
            log(f"  {n}/{len(photos)} photos written ({time.time() - t0:.0f} s)")

    with open(out_dir / "boxes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "class", "source", "score", "x1", "y1", "x2", "y2"])
        w.writerows(box_rows)
    data_yaml = {"train": "images/train", "val": "images/val", "test": "images/test",
                 "nc": len(CLASSES), "names": dict(enumerate(CLASSES))}
    (out_dir / "data.yaml").write_text(
        "# made by imgcls.objdet.build (object_detection_notebooks/01_build_data_object.ipynb)\n"
        "# no 'path' key: Ultralytics then resolves the folders relative to this file\n"
        + yaml.safe_dump(data_yaml, sort_keys=False))
    table = pd.DataFrame(counts).T.fillna(0).astype(int)
    columns = [c for c in ["train", "train boxes", "val", "val boxes", "test", "test boxes", "openimages", "owlv2",
                           "train no box", "val no box", "test no box"] if c in table.columns]
    table = table.reindex(columns=columns)
    info = {"created": datetime.now().isoformat(timespec="seconds"), "classes": CLASSES,
            "settings": settings or {}, "pseudo_threshold": threshold, "max_side": max_side,
            "photos": {s: int(table[s].sum()) if s in table else 0 for s in SPLITS},
            "boxes": {s: int(table[f"{s} boxes"].sum()) if f"{s} boxes" in table else 0 for s in SPLITS},
            "per_class": {cls: dict(c) for cls, c in counts.items()}}
    (out_dir / "info.json").write_text(json.dumps(info, indent=1))
    log(f"data_object written to {out_dir}: " + ", ".join(f"{s} {info['photos'][s]} photos / {info['boxes'][s]} boxes"
                                                          for s in SPLITS))
    return table


def contact_sheet(object_dir, split="train", per_class=6, tile=180, seed=0, path=None):
    """One row per class with per_class random photos and their boxes (green = Open Images, drawn by people;
    orange = OWLv2). Returns the PNG path."""
    object_dir = Path(object_dir)
    by_image = defaultdict(list)
    with open(object_dir / "boxes.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                by_image[(r["class"], r["image"])].append(r)
    rng = random.Random(seed)
    label_w = 150
    sheet = Image.new("RGB", (label_w + per_class * (tile + 4), len(CLASSES) * (tile + 4)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, cls in enumerate(CLASSES):
        images = sorted(k for k in by_image if k[0] == cls)
        rng.shuffle(images)
        y = row * (tile + 4)
        draw.text((5, y + tile // 2 - 6), f"{config.DISPLAY_NAMES[cls]}\n({len(images)} photos)", fill="black")
        for col, key in enumerate(images[:per_class]):
            img = Image.open(object_dir / "images" / split / key[1]).convert("RGB")
            d = ImageDraw.Draw(img)
            for r in by_image[key]:
                box = [float(r["x1"]) * img.width, float(r["y1"]) * img.height,
                       float(r["x2"]) * img.width, float(r["y2"]) * img.height]
                d.rectangle(box, outline=SOURCE_COLORS.get(r["source"], (255, 0, 0)), width=max(2, img.width // 120))
            img = ImageOps.contain(img, (tile, tile))
            sheet.paste(img, (label_w + col * (tile + 4), y + (tile - img.height) // 2))
    path = Path(path or object_dir / f"contact_sheet_{split}.png")
    sheet.save(path)
    return path


def make_zip(object_dir, zip_path):
    """data_object.zip for uploading as a Kaggle dataset (everything except _cache/)."""
    object_dir, zip_path = Path(object_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # JPEGs don't compress further
        for f in sorted(object_dir.rglob("*")):
            rel = f.relative_to(object_dir)
            if f.is_file() and rel.parts[0] != "_cache" and f.suffix != ".zip":
                z.write(f, Path(object_dir.name) / rel)
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 2**20:.0f} MB)")
    return zip_path
