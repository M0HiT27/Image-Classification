"""Dataset discovery, train/val/test splits, augmentation and data loaders.

Layout on disk: data/<class>/train/*.jpg and data/<class>/test/*.jpg.
The validation set is carved out of train (stratified, fixed seed) and is what picks the
best epoch, so the test set is only touched once, after training.
"""

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from .config import DATA_DIR, EXPECTED_CLASSES

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


def list_images(folder, recursive=False):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_rgb(path):
    """Open any image as RGB: EXIF rotation applied, transparency flattened onto white."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return im.convert("RGB")


def discover_classes(data_dir=DATA_DIR):
    classes = sorted(d.name for d in Path(data_dir).iterdir()
                     if d.is_dir() and list_images(d / "train"))
    missing = [c for c in EXPECTED_CLASSES if c not in classes]
    if missing:
        print(f"WARNING: no training images yet for {missing}; the model will only know "
              f"{len(classes)} classes. Re-run once data/<class>/train exists.")
    return classes


def scan(split, classes, data_dir=DATA_DIR, limit_per_class=None, seed=42):
    samples = []
    for idx, cls in enumerate(classes):
        files = list_images(Path(data_dir) / cls / split)
        if limit_per_class:
            random.Random(seed).shuffle(files)
            files = sorted(files[:limit_per_class])
        samples += [(str(f), idx) for f in files]
    return samples


def split_train_val(samples, fraction, seed=42):
    """Stratified split: the same share of every class goes to validation."""
    if not fraction:
        return samples, []
    by_class = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    train, val = [], []
    rng = random.Random(seed)
    for _, items in sorted(by_class.items()):
        items = sorted(items)
        rng.shuffle(items)
        n_val = max(1, round(len(items) * fraction))
        val += items[:n_val]
        train += items[n_val:]
    return sorted(train), sorted(val)


def build_transforms(img_size, mean, std, train):
    if train:
        # random crops/flips/rotations/colour and lighting changes: the briefing asks for robustness
        # to size, colour, angle, lighting and background
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.35, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.25),
        ])
    return T.Compose([
        T.Resize(int(round(img_size / 0.875)), antialias=True),
        T.CenterCrop(img_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std),
    ])


class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = load_rgb(path)
        except Exception as e:  # keep going on a broken file; it is reported, not fatal
            print(f"WARNING: could not read {path}: {e!r}")
            img = Image.new("RGB", (256, 256), (127, 127, 127))
        return self.transform(img), label


@dataclass
class Data:
    classes: list
    train_samples: list
    val_samples: list
    test_samples: list
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    mean: tuple
    std: tuple
    img_size: int

    def counts(self):
        rows = {}
        for name, samples in (("train", self.train_samples), ("val", self.val_samples),
                              ("test", self.test_samples)):
            c = Counter(y for _, y in samples)
            for i, cls in enumerate(self.classes):
                rows.setdefault(cls, {})[name] = c.get(i, 0)
        return rows


def build_data(cfg, classes, data_cfg, data_dir=DATA_DIR):
    """cfg: the notebook's config dict. data_cfg: timm's normalisation settings for the model."""
    seed = cfg.get("SEED", 42)
    mean, std = tuple(data_cfg["mean"]), tuple(data_cfg["std"])
    img_size = cfg["IMG_SIZE"]
    limit = cfg.get("LIMIT_PER_CLASS")

    train_all = scan("train", classes, data_dir, limit, seed)
    test = scan("test", classes, data_dir, limit, seed)
    train, val = split_train_val(train_all, cfg.get("VAL_FRACTION", 0.1), seed)
    if not val:
        print("WARNING: VAL_FRACTION=0, so the best epoch is chosen on the test set; "
              "test accuracy will be optimistic.")
        val = test

    train_ds = ImageDataset(train, build_transforms(img_size, mean, std, train=True))
    eval_tf = build_transforms(img_size, mean, std, train=False)
    val_ds, test_ds = ImageDataset(val, eval_tf), ImageDataset(test, eval_tf)

    workers = cfg.get("NUM_WORKERS", 4)
    common = dict(batch_size=cfg["BATCH_SIZE"], num_workers=workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    per_class = cfg.get("SAMPLES_PER_CLASS")
    if per_class:
        # class-balanced sampling: butterfly has ~10x more images than the other classes
        counts = Counter(y for _, y in train)
        weights = [1.0 / counts[y] for _, y in train]
        g = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, num_samples=per_class * len(classes),
                                        replacement=True, generator=g)
        train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True, **common)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    data = Data(classes, train, val, test, train_loader, val_loader, test_loader, mean, std, img_size)
    print(f"{'class':20s} {'train':>6s} {'val':>5s} {'test':>5s}")
    for cls, c in data.counts().items():
        print(f"{cls:20s} {c['train']:6d} {c['val']:5d} {c['test']:5d}")
    per_epoch = per_class * len(classes) if per_class else len(train)
    print(f"{len(train)} train / {len(val)} val / {len(test)} test images; "
          f"{per_epoch} training samples per epoch ({len(train_loader)} batches)")
    return data
