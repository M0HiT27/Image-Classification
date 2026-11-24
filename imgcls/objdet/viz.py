"""Pictures and videos with boxes, for any detector given as predict(PIL image) -> [(class, score, box px)]."""

import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .. import config
from .paths import CLASSES

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 140, 0), (0, 130, 200), (145, 30, 180),
           (70, 200, 200), (240, 50, 230), (128, 128, 0), (0, 0, 128), (170, 110, 40)]


def color(cls):
    return PALETTE[CLASSES.index(cls) % len(PALETTE)] if cls in CLASSES else (255, 0, 0)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw(image, detections, truth=None):
    """Copy of image with predicted boxes (coloured, with class and score) and, optionally, the true boxes
    (white, thin) given as [(class, (x1, y1, x2, y2) pixels)]."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    width = max(2, round(max(img.size) / 250))
    font = _font(max(12, round(max(img.size) / 45)))
    for cls, box in truth or []:
        d.rectangle(box, outline=(255, 255, 255), width=max(1, width // 2))
    for cls, score, box in detections:
        c = color(cls)
        d.rectangle(box, outline=c, width=width)
        text = f"{config.DISPLAY_NAMES.get(cls, cls)} {score:.0%}"
        x, y = box[0], max(0, box[1] - font.size - 4)
        d.rectangle(d.textbbox((x, y), text, font=font), fill=c)
        d.text((x, y), text, fill="white", font=font)
    return img


def predictions_grid(items, predict, path, n=12, cols=4, tile=320, seed=0):
    """Predictions on n random data_object images (white = true boxes). Returns the PNG path."""
    picks = random.Random(seed).sample(items, min(n, len(items)))
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tile + 6), rows * (tile + 6)), "white")
    for i, (img_path, boxes, labels) in enumerate(picks):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        truth = [(CLASSES[int(c)], tuple(b * np.array([w, h, w, h]))) for b, c in zip(boxes, labels)]
        shown = ImageOps.contain(draw(img, predict(img), truth), (tile, tile))
        sheet.paste(shown, ((i % cols) * (tile + 6), (i // cols) * (tile + 6)))
    sheet.save(path)
    return Path(path)


def annotate_video(video_path, predict, out_path, fps=2.0, max_frames=600, sheet_frames=8):
    """Run the detector on fps frames per second of a video. Writes an annotated .mp4 at out_path and a sheet of
    sheet_frames evenly spaced frames next to it. Returns (video path, sheet path, frames per class)."""
    import cv2

    from ..detect import read_frames, video_info

    info = video_info(video_path)
    if info is None:
        raise ValueError(f"Can't read {video_path} as a video")
    out_path = Path(out_path)
    writer, drawn, per_class = None, [], Counter()
    for n, (t, frame) in enumerate(read_frames(video_path, 1 / fps)):
        if n >= max_frames:
            break
        dets = predict(frame)
        per_class.update({cls for cls, _, _ in dets})
        img = draw(frame, dets)
        ImageDraw.Draw(img).text((8, 8), f"t = {t:.1f} s", fill="yellow", font=_font(20))
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.size)
        writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        drawn.append(img)
    if writer is not None:
        writer.release()
    sheet_path = out_path.with_name(out_path.stem + "_frames.png")
    if drawn:
        picks = [drawn[round(i)] for i in np.linspace(0, len(drawn) - 1, min(sheet_frames, len(drawn)))]
        tile = 480
        cols = 2
        rows = (len(picks) + cols - 1) // cols
        thumbs = [ImageOps.contain(p, (tile, tile)) for p in picks]
        th = max(t.height for t in thumbs)
        sheet = Image.new("RGB", (cols * (tile + 6), rows * (th + 6)), "white")
        for i, t in enumerate(thumbs):
            sheet.paste(t, ((i % cols) * (tile + 6), (i // cols) * (th + 6)))
        sheet.save(sheet_path)
    print(f"{len(drawn)} frames analysed; frames with each class: "
          + (", ".join(f"{config.DISPLAY_NAMES[c]} {k}" for c, k in per_class.most_common()) or "none"))
    return out_path, sheet_path, dict(per_class)
