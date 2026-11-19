"""Turn CLIP scores into keep/reject decisions, remove duplicates and build data/<class>/train|test.

Stages:
  python scripts/apply_screening.py --report SHEET_DIR   # decisions summary + contact sheets, changes nothing
  python scripts/apply_screening.py --apply              # writes the cleaned dataset

--apply does, per class:
  * raw_data/<class> + new_data/<class>: rejected images are copied to rejected/<class>/<reason>/
    (sources are left untouched); kept images are de-duplicated (near-identical photos keep the
    largest copy) and split 70/30 by a fixed seed into data/<class>/train and data/<class>/test
  * butterfly (already split by the user): rejected images are *moved* from data/butterfly/<split>
    to rejected/butterfly/<reason>/<split>/, and test images that are near-copies of a train image
    are moved the same way, so the existing split is kept but cleaned
Every decision is logged in manifests/screening_decisions.csv and the split in manifests/<class>_split.csv.
"""

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "manifests" / "clip_scores.csv"

OUR = ["golden_retriever", "labrador_retriever", "siberian_husky", "pug", "sunflower",
       "banana", "orange", "mushroom", "butterfly", "pine_cone"]
DOGS = {"golden_retriever", "labrador_retriever", "siberian_husky", "pug"}
DOG_LIKE = DOGS | {"x_other_dog", "x_malamute"}

# thresholds (tuned by looking at contact sheets of what each rule removes)
MIN_SIDE = 96             # smaller images are too low-resolution to be useful
NOT_PHOTO = 0.85          # 1 - P(photo) above this -> drawing / clip art / packaging
DOG_PRESENT = 0.30        # dog classes: total dog probability below this -> no dog in the picture
DOG_OWN_MIN = 0.10        # dog classes: own breed below this -> wrong breed / not usable
                          # (spot check on golden retriever: <0.10 was ~2/3 other breeds or scenes,
                          #  0.10-0.30 was mostly real goldens in odd poses, so those are kept)
OWN_OK = 0.25             # other classes: own class at least this likely -> keep
OTHER_SURE = 0.50         # ... otherwise reject if something else is at least this likely
# banana / orange: Open Images photos tagged without a bounding box are very noisy ("orange" often
# means the colour, "banana" is often a market stall), so these classes need a confident CLIP match
# pine_cone: the Commons category also holds pine-cone motifs in art (reliefs, coins, fountains)
STRICT_MIN = {"banana": 0.40, "orange": 0.40, "pine_cone": 0.70}  # pine cone 0.4-0.7 was ~half sculpture
UNBOXED_MIN = 0.70        # ... and Open Images photos that have no box need this much


def _unboxed_files():
    path = ROOT / "manifests" / "new_data_openimages.csv"
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["file"] for r in csv.DictReader(f)
                if r["class"] in STRICT_MIN and not r["largest_box_area"]}


UNBOXED = _unboxed_files()

# pine cone: sculptures / ornaments shaped like cones (scored by screen_pinecone_art.py).
# Spot check: >= 0.9 was almost all artwork (Vatican Pigna, finials, castings); 0.4-0.9 mostly real
# cones photographed on plain backgrounds, so only the top band is rejected.
ART_MAX = 0.90


def _art_scores():
    path = ROOT / "manifests" / "pine_cone_art_scores.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["file"]: float(r["p_art"]) for r in csv.DictReader(f)}


ART = _art_scores()


def decide(r):
    cls = r["class"]
    if r["readable"] != "True":
        return "unreadable", ""
    if min(int(r["width"]), int(r["height"])) < MIN_SIDE:
        return "too_small", ""
    # butterfly: white-background specimen photos look like "clip art" to CLIP, so no style check
    if cls != "butterfly" and 1 - float(r["s_photo"]) >= NOT_PHOTO:
        style = max(("s_drawing", "s_clipart", "s_text"), key=lambda k: float(r[k]))
        return "not_a_photo", style[2:]
    content = {k: float(v) for k, v in r.items() if k in OUR or k.startswith("x_")}
    top = max(content, key=content.get)
    own = content[cls]
    if cls in DOGS:
        if sum(content[k] for k in DOG_LIKE) < DOG_PRESENT:
            return "no_dog", top
        if cls == "siberian_husky":  # CLIP splits huskies between "husky" and "malamute"
            own += content["x_malamute"]
        if own < DOG_OWN_MIN:
            return "wrong_breed", top
        return "keep", ""
    if cls == "pine_cone" and ART.get(r["file"], 0.0) >= ART_MAX:
        return "artwork", f"p_art {ART[r['file']]:.2f}"
    if cls in STRICT_MIN:
        needed = UNBOXED_MIN if r["file"] in UNBOXED else STRICT_MIN[cls]
        if own >= needed:
            return "keep", ""
        return ("wrong_subject" if top != cls else "low_confidence"), top
    if cls == "butterfly":  # skippers, leafwings etc. are real butterflies that CLIP calls "moth"
        own += content["x_moth"]
        if top == "x_moth":
            top = cls
    if own >= OWN_OK:
        return "keep", ""
    if top != cls and content[top] >= OTHER_SURE:
        return "wrong_subject", top
    return "keep", ""


def load_scores(skip_classes=()):
    with open(SCORES, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["class"] not in skip_classes]
    for r in rows:
        r["decision"], r["detail"] = decide(r)
    return rows


def contact_sheet(rows, path, title, n=48, seed=0):
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    if not rows:
        return
    cols, cell = 8, 150
    sheet = Image.new("RGB", (cols * cell, 24 + ((len(rows) + cols - 1) // cols) * (cell + 14)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), title, fill="black")
    for i, r in enumerate(rows):
        x, y = (i % cols) * cell, 24 + (i // cols) * (cell + 14)
        try:
            with Image.open(ROOT / r["file"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((cell - 4, cell - 4))
                sheet.paste(im, (x + 2, y + 2))
        except Exception:
            pass
        d.text((x + 2, y + cell - 2), f"{float(r[r['class']]):.2f} {r['detail'][:14]}", fill="red")
    sheet.save(path, quality=85)


def report(sheet_dir):
    rows = load_scores()
    sheet_dir = Path(sheet_dir)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    by = defaultdict(Counter)
    for r in rows:
        by[r["class"]][r["decision"]] += 1
    reasons = sorted({r["decision"] for r in rows})
    print(f"{'class':20s} {'total':>6s} " + " ".join(f"{x:>13s}" for x in reasons))
    for cls in OUR:
        if by[cls]:
            print(f"{cls:20s} {sum(by[cls].values()):6d} " + " ".join(f"{by[cls][x]:13d}" for x in reasons))
    groups = defaultdict(list)
    for r in rows:
        if r["decision"] != "keep":
            groups[(r["class"], r["decision"])].append(r)
        elif float(r[r["class"]]) < 0.4:
            content = {k: float(v) for k, v in r.items() if k in OUR or k.startswith("x_")}
            groups[(r["class"], "kept_borderline")].append({**r, "detail": max(content, key=content.get)})
    for (cls, reason), rs in sorted(groups.items()):
        contact_sheet(rs, sheet_dir / f"{cls}__{reason}.jpg", f"{cls} / {reason}: {len(rs)} images (sample)")
    print(f"contact sheets in {sheet_dir}")


def phash(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return int(str(imagehash.phash(im)), 16), im.size[0] * im.size[1]


def near_groups(hashes, max_distance=4):
    arr = np.array(hashes, dtype=np.uint64)
    parent = list(range(len(arr)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(arr) - 1):
        for j in np.nonzero(np.bitwise_count(arr[i + 1:] ^ arr[i]) <= max_distance)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a
    groups = defaultdict(list)
    for i in range(len(arr)):
        groups[find(i)].append(i)
    return list(groups.values())


MAX_MISSING = 0.20  # refuse to rebuild a class if more than this share of its sources is gone


def apply(skip_classes, test_ratio, seed, allow_missing=False):
    rows = load_scores(skip_classes)
    log = []
    rejected_root = ROOT / "rejected"

    for cls in [c for c in OUR if c not in skip_classes]:
        crow = [r for r in rows if r["class"] == cls]
        if not crow:
            continue
        if cls != "butterfly":
            # files deleted from raw_data/new_data since scoring were removed by hand: respect that
            gone = [r for r in crow if not (ROOT / r["file"]).exists()]
            if len(gone) > MAX_MISSING * len(crow) and not allow_missing:
                # the source folders were cleaned up after data/ was built; rebuilding now would
                # throw away most of data/<class>, so leave it alone
                print(f"{cls:20s} SKIPPED: {len(gone)}/{len(crow)} source files are gone; "
                      f"data/{cls} left untouched (use --allow-missing to rebuild anyway)")
                continue
            for r in gone:
                log.append([r["file"], cls, "removed_by_user", "", ""])
            crow = [r for r in crow if (ROOT / r["file"]).exists()]
            # rejected/<class> only holds copies for non-butterfly classes, so it is safe to rebuild
            if (rejected_root / cls).exists():
                shutil.rmtree(rejected_root / cls)
        rej = [r for r in crow if r["decision"] != "keep"]
        keep = [r for r in crow if r["decision"] == "keep"]

        if cls == "butterfly":
            for r in rej:
                src = ROOT / r["file"]
                if not src.exists():  # already moved by an earlier run
                    continue
                split = src.parent.name
                dest = rejected_root / cls / r["decision"] / split / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dest)
                log.append([r["file"], cls, r["decision"], r["detail"], str(dest.relative_to(ROOT))])
            train = [ROOT / r["file"] for r in keep if Path(r["file"]).parent.name == "train"]
            test = [ROOT / r["file"] for r in keep
                    if Path(r["file"]).parent.name == "test" and (ROOT / r["file"]).exists()]
            th = np.array([phash(p)[0] for p in train], dtype=np.uint64)
            moved = 0
            for p in test:
                if (np.bitwise_count(th ^ np.uint64(phash(p)[0])) <= 4).any():
                    dest = rejected_root / cls / "duplicate_of_train" / "test" / p.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), dest)
                    log.append([str(p.relative_to(ROOT)), cls, "duplicate_of_train", "", str(dest.relative_to(ROOT))])
                    moved += 1
            print(f"{cls:20s} rejected={len(rej)} test-dups-moved={moved} "
                  f"train={len(list((ROOT / 'data' / cls / 'train').iterdir()))} "
                  f"test={len(list((ROOT / 'data' / cls / 'test').iterdir()))}")
            continue

        for r in rej:
            src = ROOT / r["file"]
            origin = src.parent.parent.name  # raw_data or new_data
            dest = rejected_root / cls / r["decision"] / f"{origin}__{src.name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            log.append([r["file"], cls, r["decision"], r["detail"], str(dest.relative_to(ROOT))])

        paths = [ROOT / r["file"] for r in keep]
        info = [phash(p) for p in paths]
        groups = near_groups([h for h, _ in info])
        chosen = []
        for g in groups:
            best = max(g, key=lambda i: (info[i][1], str(paths[i])))
            chosen.append(best)
            for i in g:
                if i != best:
                    log.append([str(paths[i].relative_to(ROOT)), cls, "duplicate", "",
                                str(paths[best].relative_to(ROOT))])

        dest_root = ROOT / "data" / cls
        if dest_root.exists():
            shutil.rmtree(dest_root)  # rebuilt from sources every time
        chosen.sort(key=lambda i: str(paths[i]))
        random.Random(seed).shuffle(chosen)
        n_test = round(len(chosen) * test_ratio)
        split_rows = []
        for k, i in enumerate(chosen):
            split = "test" if k < n_test else "train"
            src = paths[i]
            out = dest_root / split / src.name  # raw names and new_data prefixes (thu_, oi_, wm_) don't clash
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            split_rows.append([str(out.relative_to(ROOT)), split, str(src.relative_to(ROOT))])
        with open(ROOT / "manifests" / f"{cls}_split.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "split", "source_file"])
            w.writerows(sorted(split_rows))
        n_dup = sum(len(g) - 1 for g in groups)
        n_gone = sum(1 for row in log if row[1] == cls and row[2] == "removed_by_user")
        print(f"{cls:20s} available={len(crow)} removed_by_you={n_gone} rejected={len(rej)} "
              f"duplicates={n_dup} train={len(chosen) - n_test} test={n_test}")

    # rewrite the decision log: rows for classes processed now replace their earlier rows
    log_path = ROOT / "manifests" / "screening_decisions.csv"
    header = ["file", "class", "decision", "detail", "moved_or_copied_to"]
    processed = {row[1] for row in log}
    earlier = []
    if log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as f:
            earlier = [row for row in list(csv.reader(f))[1:] if row and row[1] not in processed]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(earlier + log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", metavar="SHEET_DIR")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--skip", nargs="*", default=["pine_cone"])
    ap.add_argument("--test-ratio", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-missing", action="store_true",
                    help="rebuild a class even if most of its source files no longer exist")
    args = ap.parse_args()
    if args.report:
        report(args.report)
    if args.apply:
        apply(set(args.skip), args.test_ratio, args.seed, args.allow_missing)


if __name__ == "__main__":
    main()
