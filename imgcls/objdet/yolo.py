"""Fine-tune an Ultralytics YOLO model on data_object/ (notebook 02).

Ultralytics (pip install ultralytics; licence AGPL-3.0) does the training: mosaic and colour augmentation, the
loss, and picking the best epoch on the val split. The test split is then scored twice: by Ultralytics, and by
imgcls.objdet.metrics, the same code that scores the Faster R-CNN of notebook 03, so the two detectors compare
on equal terms. YOLO starts from weights pretrained on COCO (80 everyday classes, including dog, banana and
orange); its 80-class head is replaced by one for our 10.
"""

import contextlib
import csv
import json
import os
import time
from pathlib import Path

from .. import config
from ..engine import new_run_dir
from ..environment import detect
from .dataset import ground_truth, read_split
from .metrics import mean_average_precision
from .paths import CLASSES


def device_arg(device):
    kind = getattr(device, "type", str(device))
    return 0 if kind == "cuda" else ("mps" if kind == "mps" else "cpu")


def weights_dir():
    """Where pretrained YOLO weights are downloaded: model_cache/ultralytics in the project (on D:, git-ignored);
    on Kaggle /tmp, so they don't clutter the notebook's output."""
    return Path("/tmp/ultralytics") if detect() == "kaggle" else config.ROOT / "model_cache" / "ultralytics"


@contextlib.contextmanager
def _inside(folder):
    """Ultralytics downloads weights given by name (and a small model for its GPU check) into the current
    folder; run it from weights_dir() so they land there."""
    old = os.getcwd()
    Path(folder).mkdir(parents=True, exist_ok=True)
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(old)


def write_data_yaml(object_dir, path):
    """A copy of data_object/data.yaml with its absolute path (the folder may be read-only on Kaggle)."""
    import yaml

    data = yaml.safe_load((Path(object_dir) / "data.yaml").read_text())
    data["path"] = str(Path(object_dir).resolve())
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))
    return Path(path)


def _best_epoch(run_dir):
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0) or 0))
    return {"best_epoch": int(float(best["epoch"])), "best_val_map50": round(float(best["metrics/mAP50(B)"]), 4),
            "best_val_map50_95": round(float(best["metrics/mAP50-95(B)"]), 4), "epochs_run": len(rows)}


def train(cfg, object_dir, device, log=print):
    """Fine-tune cfg["YOLO_MODEL"] (e.g. yolo26n.pt) on data_object/. Returns the run folder: Ultralytics'
    usual files (weights/best.pt, results.csv, results.png, ...) plus config.json and summary.json."""
    from ultralytics import YOLO

    run_dir = new_run_dir(cfg["MODEL_NAME"])
    data_yaml = write_data_yaml(object_dir, run_dir / "data.yaml")
    (run_dir / "config.json").write_text(json.dumps(
        {"config": cfg, "kind": "ultralytics_yolo", "classes": CLASSES, "device": str(device),
         "data_object": str(object_dir), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    started = time.time()
    with _inside(weights_dir()):
        model = YOLO(cfg["YOLO_MODEL"])  # downloaded into weights_dir() on first use (yolo26n.pt: about 5 MB)
        try:
            model.train(data=str(data_yaml), epochs=cfg["EPOCHS"], imgsz=cfg["IMG_SIZE"], batch=cfg["BATCH_SIZE"],
                        device=device_arg(device), workers=cfg.get("NUM_WORKERS", 2), seed=cfg.get("SEED", 42),
                        patience=cfg.get("PATIENCE") or cfg["EPOCHS"], fraction=cfg.get("FRACTION", 1.0),
                        project=str(run_dir.parent), name=run_dir.name, exist_ok=True, plots=True)
        except KeyboardInterrupt:
            log("Interrupted: weights/best.pt keeps the best epoch so far.")
    summary = {"model_name": cfg["MODEL_NAME"], "kind": "ultralytics_yolo", "arch": cfg["YOLO_MODEL"],
               **_best_epoch(run_dir), "minutes": round((time.time() - started) / 60, 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir


def evaluate_run(run_dir, object_dir, device, split="test", img_size=640, batch=16):
    """Score weights/best.pt once on a split: Ultralytics' own validation (plots in <run>/<split>/) and
    imgcls.objdet.metrics on the same predictions style as notebook 03. Writes <split>_metrics.json, adds the
    numbers to summary.json and returns a per-class table."""
    import pandas as pd
    from ultralytics import YOLO

    run_dir = Path(run_dir)
    model = YOLO(str(run_dir / "weights" / "best.pt"))
    data_yaml = write_data_yaml(object_dir, run_dir / "data.yaml")
    with _inside(weights_dir()):
        m = model.val(data=str(data_yaml), split=split, imgsz=img_size, batch=batch, device=device_arg(device),
                      project=str(run_dir), name=split, exist_ok=True, plots=True, verbose=False)
    items = read_split(object_dir, split)
    preds = {}
    step = batch if batch and batch > 0 else 16
    # Ultralytics loads a list of paths into memory and runs it as ONE batch (all 1,463 test photos = 9 GB of
    # GPU memory), so hand it `step` photos at a time. rect=False pads every photo to the full square; with
    # predict's default (rect=True) a chunk whose photos share one shape gets less padding, so the score would
    # depend on the batch size.
    for start in range(0, len(items), step):
        chunk = items[start:start + step]
        results = model.predict(source=[str(p) for p, _, _ in chunk], imgsz=img_size, conf=0.001, iou=0.7,
                                max_det=100, rect=False, device=device_arg(device), verbose=False)
        for (path, _, _), r in zip(chunk, results):
            b = r.boxes
            preds[str(path)] = (b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
    ours = mean_average_precision(preds, ground_truth(items), len(CLASSES))
    table = pd.DataFrame(ours["per_class"], index=[config.DISPLAY_NAMES[c] for c in CLASSES])
    table = table.rename(columns={"gt": "boxes"})[["boxes", "ap50", "ap50_95"]].round(3)
    result = {"split": split, "images": len(items),
              "map50": round(ours["map50"], 4), "map50_95": round(ours["map50_95"], 4),
              "ultralytics_map50": round(float(m.box.map50), 4), "ultralytics_map50_95": round(float(m.box.map), 4),
              "per_class": table.to_dict(orient="index")}
    (run_dir / f"{split}_metrics.json").write_text(json.dumps(result, indent=2))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({f"{split}_map50": result["map50"], f"{split}_map50_95": result["map50_95"]})
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"{split}: mAP50 {result['map50']:.3f}, mAP50-95 {result['map50_95']:.3f} on {len(items)} images "
          f"(Ultralytics' own numbers: {result['ultralytics_map50']:.3f} / {result['ultralytics_map50_95']:.3f})")
    return table


def predictor(run_dir, device, img_size=640, threshold=0.25):
    """predict(PIL image) -> [(class folder, score, (x1, y1, x2, y2) pixels)] for imgcls.objdet.viz."""
    from ultralytics import YOLO

    model = YOLO(str(Path(run_dir) / "weights" / "best.pt"))

    def predict(image):
        r = model.predict(source=image.convert("RGB"), imgsz=img_size, conf=threshold, device=device_arg(device),
                          verbose=False)[0]
        return [(CLASSES[int(c)], float(s), tuple(box)) for box, s, c in
                zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist())]
    return predict
