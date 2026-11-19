"""Score images with a pretrained CLIP model so obvious junk can be pruned.

For every image two zero-shot predictions are stored (nothing is moved here):
  * content: probabilities over our 10 classes plus common distractors
    (other dog breeds, cooked food, other flowers, text, people, ...)
  * style:   photo vs drawing/cartoon vs clip-art/logo vs packaging/text

Output: manifests/clip_scores.csv (appends; already-scored files are skipped,
so an interrupted run can be resumed). Decisions are made by apply_screening.py.

Usage:
    python scripts/screen_images.py                       # default folders, all classes but pine cone
    python scripts/screen_images.py --only pine_cone      # later, once pine cone is complete
"""

import argparse
import csv
from pathlib import Path

import open_clip
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = {
    "golden_retriever": "golden retriever, a type of dog",
    "labrador_retriever": "labrador retriever, a type of dog",
    "siberian_husky": "siberian husky, a type of dog",
    "pug": "pug, a type of dog",
    "sunflower": "sunflower",
    "banana": "banana",
    "orange": "orange, the citrus fruit",
    "mushroom": "mushroom",
    "butterfly": "butterfly",
    "pine_cone": "pine cone",
}
DISTRACTORS = {
    "x_other_dog": "dog of some other breed",
    "x_malamute": "alaskan malamute, a type of dog",
    "x_cat": "cat",
    "x_person": "person",
    "x_cooked_food": "plate of cooked food",
    "x_lemon": "lemon or lime",
    "x_other_flower": "flower that is not a sunflower",
    "x_moth": "moth",
    "x_tree": "tree",
    "x_text": "page of text",
    "x_building": "building or street",
}
TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a blurry photo of a {}."]
STYLES = {
    "s_photo": ["a photograph.", "a real photo taken with a camera."],
    "s_drawing": ["a drawing, painting or cartoon.", "an illustration."],
    "s_clipart": ["clip art, an icon or a logo.", "a vector graphic on a plain white background."],
    "s_text": ["a product package, poster or page with text.", "a screenshot."],
}


def default_sources():
    src = []
    for cls in CLASSES:
        if cls == "pine_cone":
            continue
        for folder in (ROOT / "raw_data" / cls, ROOT / "new_data" / cls):
            src.append((cls, folder))
    src += [("butterfly", ROOT / "data" / "butterfly" / "train"),
            ("butterfly", ROOT / "data" / "butterfly" / "test")]
    return src


class Images(Dataset):
    def __init__(self, items, preprocess):
        self.items, self.preprocess = items, preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path = self.items[i][1]
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, "white")
                    im = Image.alpha_composite(bg, im)
                im = im.convert("RGB")
                size = im.size
                return self.preprocess(im), i, size[0], size[1], True
        except Exception:
            return torch.zeros(3, 224, 224), i, 0, 0, False


@torch.no_grad()
def text_matrix(model, tokenizer, prompt_lists):
    feats = []
    for prompts in prompt_lists:
        t = model.encode_text(tokenizer(prompts))
        t = t / t.norm(dim=-1, keepdim=True)
        t = t.mean(0)
        feats.append(t / t.norm())
    return torch.stack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="limit to these classes (default sources)")
    # ViT-B-32 runs ~7x faster than ViT-B-16 on this CPU-only machine
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="manifests/clip_scores.csv")
    args = ap.parse_args()

    sources = [(c, f) for c, f in default_sources() if not args.only or c in args.only]
    if args.only and "pine_cone" in args.only:
        sources.append(("pine_cone", ROOT / "new_data" / "pine_cone"))

    out_path = ROOT / args.out
    done = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            done = {r["file"] for r in csv.DictReader(f)}

    items = []
    for cls, folder in sources:
        if folder.exists():
            for p in sorted(folder.iterdir()):
                rel = str(p.relative_to(ROOT))
                if p.suffix.lower() in IMAGE_EXTS and rel not in done:
                    items.append((cls, p))
    print(f"{len(items)} images to score ({len(done)} already scored)", flush=True)
    if not items:
        return

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    content_keys = list(CLASSES) + list(DISTRACTORS)
    content_text = text_matrix(model, tokenizer,
                               [[t.format(d) for t in TEMPLATES] for d in list(CLASSES.values()) + list(DISTRACTORS.values())])
    style_keys = list(STYLES)
    style_text = text_matrix(model, tokenizer, list(STYLES.values()))
    scale = model.logit_scale.exp()

    header = ["file", "class", "width", "height", "readable"] + content_keys + style_keys
    new_file = not out_path.exists()
    out_path.parent.mkdir(exist_ok=True)
    loader = DataLoader(Images(items, preprocess), batch_size=args.batch, num_workers=args.workers)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        n = 0
        with torch.no_grad():
            for x, idx, wd, ht, ok in loader:
                img = model.encode_image(x)
                img = img / img.norm(dim=-1, keepdim=True)
                pc = (scale * img @ content_text.T).softmax(-1)
                ps = (scale * img @ style_text.T).softmax(-1)
                for k in range(len(idx)):
                    cls, p = items[int(idx[k])]
                    w.writerow([str(p.relative_to(ROOT)), cls, int(wd[k]), int(ht[k]), bool(ok[k])]
                               + [f"{v:.4f}" for v in pc[k].tolist()] + [f"{v:.4f}" for v in ps[k].tolist()])
                n += len(idx)
                f.flush()
                if n % 1024 < args.batch:
                    print(f"  {n}/{len(items)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
