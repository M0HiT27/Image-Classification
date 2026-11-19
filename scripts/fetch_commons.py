"""Download a random sample of images from a Wikimedia Commons category tree into new_data/<class>/.

Example (what was used for pine cones):
    python scripts/fetch_commons.py pine_cone --category "Category:Pinus cones" --n 700 \
        --exclude pollen,male,drawing,illustration,fossil,herbarium,seed

Subcategories whose names contain an --exclude word are not walked. Images are
fetched as 500px-wide thumbnails and saved as new_data/<class>/wm_<id>.<ext>.
License and author for each file go to manifests/new_data_commons.csv
(most Commons files require attribution if redistributed).
"""

import argparse
import csv
import hashlib
import io
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
API = "https://commons.wikimedia.org/w/api.php"
UA = "ImageClassificationDatasetBuilder/1.0 (educational project; python-urllib)"


def get(url, pause=1.0):
    for attempt in range(6):
        try:
            time.sleep(pause)
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(10 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(5 * (attempt + 1))  # network hiccup: wait and retry
    raise RuntimeError(f"gave up after retries: {url}")


def api(params):
    params = {**params, "format": "json", "maxlag": 5}
    return json.loads(get(API + "?" + urllib.parse.urlencode(params)))


def walk(root, exclude, max_depth):
    seen, files, frontier = set(), set(), [(root, 0)]
    while frontier:
        cat, depth = frontier.pop()
        if cat in seen:
            continue
        seen.add(cat)
        cont = {}
        while True:
            d = api({"action": "query", "list": "categorymembers", "cmtitle": cat,
                     "cmtype": "file|subcat", "cmlimit": 500, **cont})
            for m in d["query"]["categorymembers"]:
                title = m["title"]
                if m["ns"] == 14:
                    if depth < max_depth and not any(w in title.lower() for w in exclude):
                        frontier.append((title, depth + 1))
                elif m["ns"] == 6 and title.lower().endswith((".jpg", ".jpeg", ".png")):
                    files.add(title)
            if "continue" not in d:
                break
            cont = d["continue"]
    return sorted(files)


def image_info(titles):
    info = {}
    for i in range(0, len(titles), 50):
        d = api({"action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "imageinfo",
                 # 500 is one of Wikimedia's standard thumbnail widths; other widths get throttled
                 "iiprop": "url|extmetadata", "iiurlwidth": 500})
        for page in d["query"]["pages"].values():
            ii = (page.get("imageinfo") or [None])[0]
            if ii:
                meta = ii.get("extmetadata", {})
                info[page["title"]] = {
                    "thumb": ii.get("thumburl") or ii.get("url"),
                    "page": ii.get("descriptionurl", ""),
                    "license": meta.get("LicenseShortName", {}).get("value", ""),
                    "artist": meta.get("Artist", {}).get("value", ""),
                }
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("class_name")
    ap.add_argument("--category", required=True)
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--exclude", default="", help="comma-separated words; matching subcategories are skipped")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="new_data")
    args = ap.parse_args()

    exclude = [w.strip().lower() for w in args.exclude.split(",") if w.strip()]
    cache = ROOT / "manifests" / f"commons_{args.class_name}_titles.json"
    if cache.exists():
        titles = json.loads(cache.read_text(encoding="utf-8"))
    else:
        titles = walk(args.category, exclude, args.max_depth)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(titles, indent=0), encoding="utf-8")
    random.Random(args.seed).shuffle(titles)
    chosen = titles[: args.n]
    print(f"{len(titles)} image files under {args.category}; taking {len(chosen)}")

    info = image_info(chosen)
    out_dir = ROOT / args.out / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, ok, failed = [], 0, 0
    for title in chosen:
        meta = info.get(title)
        if not meta or not meta["thumb"]:
            failed += 1
            continue
        ext = Path(urllib.parse.urlparse(meta["thumb"]).path).suffix.lower() or ".jpg"
        out = out_dir / f"wm_{hashlib.md5(title.encode()).hexdigest()[:12]}{ext}"
        status = "exists"
        if not out.exists():
            try:
                data = get(meta["thumb"], pause=0.3)
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
                out.write_bytes(data)
                status = "ok"
            except Exception as e:
                status = f"failed: {e!r}"
        ok += status in ("ok", "exists")
        failed += status.startswith("failed")
        rows.append([str(out.relative_to(ROOT)), args.class_name, "wikimedia_commons", title,
                     meta["page"], meta["license"], meta["artist"], status])

    man = ROOT / "manifests" / "new_data_commons.csv"
    man.parent.mkdir(exist_ok=True)
    kept = []
    if man.exists():
        with open(man, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.reader(f)][1:]
            kept = [r for r in kept if r[1] != args.class_name]
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "source", "title", "page_url", "license", "artist", "status"])
        w.writerows(sorted(kept + rows))
    print(f"{args.class_name}: saved={ok} failed={failed}")


if __name__ == "__main__":
    main()
