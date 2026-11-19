"""Extra check for pine cones from Wikimedia Commons: real cone vs artwork of a cone.

The Commons "Pinus cones" category also contains sculptures, carvings and ornaments shaped like
pine cones, which the main CLIP screen scores as genuine pine cones. This asks CLIP a narrower
question for the Commons images only and writes manifests/pine_cone_art_scores.csv (file, p_art).
apply_screening.py rejects pine cone images with p_art >= its ART_MAX threshold.
"""

import csv
from pathlib import Path

import open_clip
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = {
    "real": ["a photo of a real pine cone in nature.", "a photo of a real pine cone on a tree branch.",
             "a photo of a pine cone held in a hand.", "a close-up photo of a pine cone on the ground."],
    "art": ["a photo of a sculpture or statue shaped like a pine cone.", "a photo of a stone carving of a pine cone.",
            "a photo of a metal or ceramic ornament shaped like a pine cone.", "a photo of a fountain or monument."],
}


@torch.no_grad()
def main():
    files = sorted((ROOT / "new_data" / "pine_cone").glob("wm_*"))
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="datacomp_xl_s13b_b90k")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    text = []
    for prompts in PROMPTS.values():
        t = model.encode_text(tok(prompts))
        t = (t / t.norm(dim=-1, keepdim=True)).mean(0)
        text.append(t / t.norm())
    text = torch.stack(text)
    rows = []
    for i in range(0, len(files), 64):
        batch = files[i:i + 64]
        x = torch.stack([preprocess(ImageOps.exif_transpose(Image.open(f)).convert("RGB")) for f in batch])
        img = model.encode_image(x)
        img = img / img.norm(dim=-1, keepdim=True)
        p = (model.logit_scale.exp() * img @ text.T).softmax(-1)[:, 1]
        rows += [[str(f.relative_to(ROOT)), f"{v:.4f}"] for f, v in zip(batch, p.tolist())]
    out = ROOT / "manifests" / "pine_cone_art_scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "p_art"])
        w.writerows(rows)
    print(f"scored {len(rows)} Commons pine cone images -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
