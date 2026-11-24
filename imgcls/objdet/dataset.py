"""Read data_object/ (YOLO format) for our own detector, with flips and mosaic augmentation.

Our photos nearly always show one big object in the middle. Mosaic stitches four photos into one image, so the
detector also sees small objects, several objects at once and objects near the edges, as in a video frame.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from ..data import list_images

FILL = (114, 114, 114)


def read_split(object_dir, split):
    """[(image path, boxes (n, 4) x1 y1 x2 y2 in 0-1, class ids (n,))] for one split of data_object/."""
    object_dir = Path(object_dir)
    items = []
    for img in list_images(object_dir / "images" / split):
        label = object_dir / "labels" / split / f"{img.stem}.txt"
        rows = np.loadtxt(label, ndmin=2) if label.exists() and label.stat().st_size else np.zeros((0, 5))
        cls, cx, cy, w, h = rows.T if len(rows) else (np.zeros(0),) * 5
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).clip(0, 1) if len(rows) else np.zeros((0, 4))
        items.append((img, boxes.astype(np.float32), cls.astype(np.int64)))
    return items


def ground_truth(items):
    """{image path: (boxes 0-1, class ids)} for imgcls.objdet.metrics. Boxes stay in 0-1 units for both
    detectors: IoU doesn't change when both axes are rescaled, so no image needs opening."""
    return {str(path): (boxes, labels) for path, boxes, labels in items}


class DetectionDataset(Dataset):
    """Images resized so the longer side is img_size; targets in torchvision's format (boxes in pixels,
    labels 1..10 because 0 is the background class)."""

    def __init__(self, items, img_size=512, train=False, mosaic=0.5, flip=0.5):
        self.items, self.img_size, self.train = items, img_size, train
        self.mosaic, self.flip = (mosaic, flip) if train else (0.0, 0.0)

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        path, boxes, labels = self.items[i]
        img = Image.open(path).convert("RGB")
        img = ImageOps.contain(img, (self.img_size, self.img_size), Image.Resampling.BILINEAR)
        w, h = img.size
        return img, boxes * np.array([w, h, w, h], np.float32), labels.copy()

    def _mosaic(self, i):
        s = self.img_size
        canvas = Image.new("RGB", (s, s), FILL)
        xc, yc = (int(random.uniform(0.3, 0.7) * s) for _ in range(2))
        quads = [(0, 0, xc, yc), (xc, 0, s, yc), (0, yc, xc, s), (xc, yc, s, s)]
        all_boxes, all_labels = [], []
        for q, j in enumerate([i] + random.sample(range(len(self.items)), 3)):
            img, boxes, labels = self._load(j)
            qx1, qy1, qx2, qy2 = quads[q]
            scale = min((qx2 - qx1) / img.width, (qy2 - qy1) / img.height)
            nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            img = img.resize((nw, nh), Image.Resampling.BILINEAR)
            # push each photo against the mosaic's centre point
            ox = qx2 - nw if q in (0, 2) else qx1
            oy = qy2 - nh if q in (0, 1) else qy1
            canvas.paste(img, (ox, oy))
            all_boxes.append(boxes * scale + np.array([ox, oy, ox, oy], np.float32))
            all_labels.append(labels)
        boxes, labels = np.concatenate(all_boxes), np.concatenate(all_labels)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 4) & ((boxes[:, 3] - boxes[:, 1]) >= 4)
        return canvas, boxes[keep], labels[keep]

    def __getitem__(self, i):
        if self.train and len(self.items) >= 4 and random.random() < self.mosaic:
            img, boxes, labels = self._mosaic(i)
        else:
            img, boxes, labels = self._load(i)
        if random.random() < self.flip:
            img = ImageOps.mirror(img)
            boxes = boxes.copy()
            boxes[:, [0, 2]] = img.width - boxes[:, [2, 0]]
        image = pil_to_tensor(img).float().div(255)
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(labels, dtype=torch.int64) + 1}
        return image, target


def collate(batch):
    return tuple(zip(*batch))
