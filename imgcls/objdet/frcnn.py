"""Our own detector: a Faster R-CNN whose backbone is our fine-tuned MobileNetV3 classifier (notebook 03).

A detector has three parts:
  backbone  turns the image into feature maps. Here: the MobileNetV3 from our classifier's best.pth (its
            layers up to the classification head; the head itself is dropped), via timm;
  neck      a feature pyramid (FPN) that mixes the backbone's maps at strides 8, 16 and 32, so small and big
            objects are both seen at a useful resolution;
  heads     a region proposal network suggests boxes that may hold an object, and the box head names each
            proposal (10 classes + background) and refines its edges (torchvision's Faster R-CNN).
The classifier already knows WHAT our objects look like; the neck and heads start from scratch and learn WHERE
from the boxes in data_object/. BatchNorm in the backbone is frozen: batches of a few images are too small to
estimate its statistics.
"""

import csv
import json
import math
import random
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader

from .. import config
from ..engine import new_run_dir
from ..environment import KAGGLE_INPUT
from .dataset import DetectionDataset, collate, ground_truth, read_split
from .metrics import mean_average_precision
from .paths import CLASSES

DEFAULT_ARCH = "mobilenetv3_large_100.ra_in1k"
KIND = "fasterrcnn_timm_fpn"


class TimmFPN(nn.Module):
    """timm feature extractor + feature pyramid, in the form torchvision's detectors expect."""

    def __init__(self, body, channels, out_channels=256):
        super().__init__()
        from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

        self.body = body
        self.fpn = FeaturePyramidNetwork(channels, out_channels, extra_blocks=LastLevelMaxPool())
        self.out_channels = out_channels

    def forward(self, x):
        return self.fpn(OrderedDict((str(i), f) for i, f in enumerate(self.body(x))))


def _arch_of_run(run_dir):
    """Architecture of a classifier run folder; None for detector runs and unknown folders."""
    for name in ("summary.json", "config.json"):
        path = Path(run_dir) / name
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except ValueError:
                continue
            if data.get("kind"):  # a detector run (this module), not a classifier
                return None
            arch = data.get("arch") or data.get("config", {}).get("ARCH")
            if arch:
                return arch
    return None


def find_backbone_checkpoint(arch=DEFAULT_ARCH):
    """Newest classifier best.pth trained with this architecture: under output/ and, on Kaggle, any attached
    dataset (attach the run folder holding best.pth and summary.json). None when there is none."""
    base = arch.split(".")[0]
    roots = [Path(config.OUTPUT_DIR)] + ([Path(KAGGLE_INPUT)] if Path(KAGGLE_INPUT).is_dir() else [])
    found = []
    for root in roots:
        if root.is_dir():
            for ckpt in root.rglob("best.pth"):
                run_arch = _arch_of_run(ckpt.parent)
                if run_arch and run_arch.split(".")[0] == base:
                    found.append(ckpt)
    return max(found, key=lambda p: (p.parent.name, str(p))) if found else None


def build_model(num_classes, arch=DEFAULT_ARCH, img_size=512, backbone_checkpoint=None, pretrained=True,
                out_indices=(2, 3, 4), log=print):
    """Faster R-CNN with a timm backbone. backbone_checkpoint: our classifier's best.pth (its weights replace
    the ImageNet ones). Returns (model, meta) where meta is saved with the detector."""
    import timm
    from timm.layers import freeze_batch_norm_2d
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    body = timm.create_model(arch, features_only=True, out_indices=tuple(out_indices),
                             pretrained=pretrained and backbone_checkpoint is None)
    try:
        data_cfg = timm.data.resolve_model_data_config(body)
        mean, std = list(data_cfg["mean"]), list(data_cfg["std"])
    except Exception:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    source = "ImageNet weights (timm)" if pretrained else "random weights"
    if backbone_checkpoint is not None:
        ckpt = torch.load(backbone_checkpoint, map_location="cpu", weights_only=True)
        if ckpt["arch"].split(".")[0] != arch.split(".")[0]:
            raise ValueError(f"{backbone_checkpoint} was trained with {ckpt['arch']}, not {arch}")
        missing, unexpected = body.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            raise RuntimeError(f"{len(missing)} backbone weights missing in {backbone_checkpoint}: {missing[:5]}")
        mean, std = list(ckpt["mean"]), list(ckpt["std"])
        source = (f"our classifier {backbone_checkpoint} (epoch {ckpt.get('epoch')}, val acc {ckpt.get('val_acc')}; "
                  f"{len(unexpected)} classifier-head tensors not used)")
    channels = body.feature_info.channels()
    body = freeze_batch_norm_2d(body)
    sizes = tuple((s, int(s * 1.41), 2 * s) for s in (32, 64, 128, 256))  # one size group per pyramid level
    anchors = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))
    roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2"], output_size=7, sampling_ratio=2)
    model = FasterRCNN(TimmFPN(body, channels), num_classes=num_classes + 1, rpn_anchor_generator=anchors,
                       box_roi_pool=roi_pool, min_size=img_size, max_size=img_size, image_mean=mean, image_std=std,
                       box_score_thresh=0.001, box_detections_per_img=100)
    meta = {"kind": KIND, "arch": arch, "img_size": img_size, "mean": mean, "std": std,
            "out_indices": list(out_indices), "backbone_source": source}
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Faster R-CNN, {n_params:.1f}M parameters; backbone {arch} from {source}")
    return model, meta


@torch.inference_mode()
def predict_items(model, items, device, img_size, batch_size=8):
    """{image path: (boxes 0-1, scores, class ids)} for data_object items (metrics format)."""
    model.eval()
    ds = DetectionDataset(items, img_size, train=False)
    preds = {}
    for start in range(0, len(ds), batch_size):
        images = [ds[i][0] for i in range(start, min(start + batch_size, len(ds)))]
        outs = model([im.to(device) for im in images])
        for (path, _, _), image, out in zip(items[start:start + batch_size], images, outs):
            h, w = image.shape[-2:]
            boxes = out["boxes"].float().cpu().numpy() / np.array([w, h, w, h], np.float32)
            preds[str(path)] = (boxes, out["scores"].float().cpu().numpy(), out["labels"].cpu().numpy() - 1)
    return preds


def _plot(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(epochs, [h["loss"] for h in history], marker="o")
    a.set(title="training loss", xlabel="epoch")
    b.plot(epochs, [h["val_map50"] for h in history], marker="o", label="val mAP50")
    b.plot(epochs, [h["val_map50_95"] for h in history], marker="o", label="val mAP50-95")
    b.set(title="validation", xlabel="epoch", ylim=(0, 1))
    b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train(cfg, model, meta, object_dir, device, log=print):
    """Train on data_object/train, keep the epoch with the best val mAP50-95 as best.pth. Interrupting the
    cell stops early and keeps the best checkpoint. Returns the run folder."""
    run_dir = new_run_dir(cfg["MODEL_NAME"])
    img_size, batch_size, epochs = cfg["IMG_SIZE"], cfg["BATCH_SIZE"], cfg["EPOCHS"]
    train_items, val_items = read_split(object_dir, "train"), read_split(object_dir, "val")
    if cfg.get("VAL_LIMIT") and len(val_items) > cfg["VAL_LIMIT"]:
        val_items = random.Random(cfg.get("SEED", 42)).sample(val_items, cfg["VAL_LIMIT"])
    val_gt = ground_truth(val_items)
    workers = cfg.get("NUM_WORKERS", 0)
    loader = DataLoader(DetectionDataset(train_items, img_size, train=True, mosaic=cfg.get("MOSAIC", 0.5)),
                        batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate,
                        drop_last=len(train_items) > batch_size, persistent_workers=workers > 0,
                        pin_memory=device.type == "cuda")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["LR"], weight_decay=cfg.get("WEIGHT_DECAY", 1e-4))
    total, warm = epochs * len(loader), len(loader)  # 1 warm-up epoch, then cosine down to 1%
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warm if s < warm else
                                                  0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    (run_dir / "config.json").write_text(json.dumps({
        "config": cfg, **meta, "classes": CLASSES, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if amp else None, "data_object": str(object_dir),
        "images": {"train": len(train_items), "val": len(val_items)},
        "versions": {"torch": torch.__version__}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    log(f"Training on {len(train_items)} images ({len(loader)} batches of {batch_size}), validating on "
        f"{len(val_items)}; run folder {run_dir}")
    history, best, started = [], None, time.time()
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            t0, sums, batches, skipped = time.time(), Counter(), 0, 0
            for images, targets in loader:
                images = [im.to(device, non_blocking=True) for im in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    losses = model(images, targets)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, 10.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                for k, v in losses.items():
                    sums[k] += v.detach().item()
                batches += 1
            val = mean_average_precision(predict_items(model, val_items, device, img_size), val_gt, len(CLASSES))
            row = {"epoch": epoch, "loss": round(sum(sums.values()) / max(batches, 1), 4),
                   **{k: round(v / max(batches, 1), 4) for k, v in sums.items()},
                   "val_map50": round(val["map50"], 4), "val_map50_95": round(val["map50_95"], 4),
                   "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
            history.append(row)
            improved = best is None or row["val_map50_95"] > best["val_map50_95"]
            if improved:
                best = row
                torch.save({"model_state": model.state_dict(), **meta, "classes": CLASSES, "epoch": epoch,
                            "val_map50": row["val_map50"], "val_map50_95": row["val_map50_95"], "config": cfg},
                           run_dir / "best.pth")
            log(f"epoch {epoch:3d}/{epochs}  loss {row['loss']:.3f}  val mAP50 {row['val_map50']:.3f}  "
                f"mAP50-95 {row['val_map50_95']:.3f}  {row['seconds']:.0f} s" + ("  * saved" if improved else "")
                + (f"  ({skipped} batches skipped: loss not finite)" if skipped else ""))
            with open(run_dir / "history.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(history[0]))
                w.writeheader()
                w.writerows(history)
    except KeyboardInterrupt:
        log("Interrupted: keeping the best checkpoint so far.")
    if history:
        _plot(history, run_dir / "training_curves.png")
    summary = {"model_name": cfg["MODEL_NAME"], "kind": KIND, "arch": meta["arch"],
               "backbone_source": meta["backbone_source"], "best_epoch": best and best["epoch"],
               "best_val_map50": best and best["val_map50"], "best_val_map50_95": best and best["val_map50_95"],
               "epochs_run": len(history), "minutes": round((time.time() - started) / 60, 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir


def load_checkpoint(path, device="cpu"):
    """Rebuild a detector saved by train(). Returns (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = build_model(len(ckpt["classes"]), ckpt["arch"], ckpt["img_size"], pretrained=False,
                           out_indices=tuple(ckpt["out_indices"]), log=lambda *a: None)
    model.load_state_dict(ckpt["model_state"])
    model.transform.image_mean, model.transform.image_std = list(ckpt["mean"]), list(ckpt["std"])
    model.to(device).eval()
    return model, ckpt


def evaluate_run(run_dir, object_dir, device, split="test"):
    """Score best.pth once on a split of data_object/; writes <split>_metrics.json and adds the numbers to
    summary.json. Returns a per-class table."""
    import pandas as pd

    run_dir = Path(run_dir)
    model, ckpt = load_checkpoint(run_dir / "best.pth", device)
    items = read_split(object_dir, split)
    m = mean_average_precision(predict_items(model, items, device, ckpt["img_size"]), ground_truth(items),
                               len(ckpt["classes"]))
    table = pd.DataFrame(m["per_class"], index=[config.DISPLAY_NAMES.get(c, c) for c in ckpt["classes"]])
    table = table.rename(columns={"gt": "boxes"})[["boxes", "ap50", "ap50_95"]].round(3)
    result = {"split": split, "checkpoint_epoch": ckpt["epoch"], "images": len(items),
              "map50": round(m["map50"], 4), "map50_95": round(m["map50_95"], 4),
              "per_class": table.to_dict(orient="index")}
    (run_dir / f"{split}_metrics.json").write_text(json.dumps(result, indent=2))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({f"{split}_map50": result["map50"], f"{split}_map50_95": result["map50_95"]})
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"{split}: mAP50 {result['map50']:.3f}, mAP50-95 {result['map50_95']:.3f} on {len(items)} images")
    return table


def predictor(model, img_size, device, threshold=0.5):
    """predict(PIL image) -> [(class folder, score, (x1, y1, x2, y2) pixels)] for imgcls.objdet.viz."""
    @torch.inference_mode()
    def predict(image):
        small = ImageOps.contain(image.convert("RGB"), (img_size, img_size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(small).copy()).permute(2, 0, 1).float().div(255)
        out = model.eval()([tensor.to(device)])[0]
        sx, sy = image.width / small.width, image.height / small.height
        return [(CLASSES[int(c) - 1], float(s), (x1 * sx, y1 * sy, x2 * sx, y2 * sy))
                for (x1, y1, x2, y2), s, c in zip(out["boxes"].float().cpu().tolist(), out["scores"].float().cpu().tolist(),
                                                  out["labels"].cpu().tolist()) if s >= threshold]
    return predict
