"""Predict a folder of images (e.g. the hidden test set) with a saved checkpoint.

Writes the deliverable format, one row per image: filename, predicted category.
"""

import csv
from pathlib import Path

import numpy as np

from .config import DISPLAY_NAMES
from .data import build_transforms, list_images
from .device import get_device
from .evaluate import predict_probs
from .model import load_checkpoint


def predict_folder(checkpoint, folder, out_csv=None, device=None, batch_size=64, recursive=False):
    device = device or get_device("auto")
    model, ckpt = load_checkpoint(checkpoint, device)
    classes = ckpt["classes"]
    folder = Path(folder)
    files = list_images(folder, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"no images found in {folder}")
    tf = build_transforms(ckpt["img_size"], ckpt["mean"], ckpt["std"], train=False)
    probs = predict_probs(model, [(str(f), -1) for f in files], tf, device, batch_size)
    top3 = np.argsort(-probs, axis=1)[:, :3]

    out_csv = Path(out_csv or Path(checkpoint).parent / "predictions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_category"])
        for path, idx in zip(files, top3[:, 0]):
            w.writerow([path.relative_to(folder).as_posix(), DISPLAY_NAMES.get(classes[idx], classes[idx])])
    detailed = out_csv.with_name(out_csv.stem + "_detailed.csv")
    with open(detailed, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_category", "confidence", "second", "second_conf", "third", "third_conf"])
        for path, p, t in zip(files, probs, top3):
            w.writerow([path.relative_to(folder).as_posix()]
                       + [x for i in t for x in (classes[i], f"{p[i]:.4f}")])
    print(f"{len(files)} predictions written to {out_csv} (details: {detailed.name})")
    return out_csv
