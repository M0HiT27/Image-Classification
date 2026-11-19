"""Evaluate a run's best checkpoint on the test split and save the report material.

Files added to the run folder:
  test_metrics.json     accuracy, balanced accuracy, macro F1, per-class precision/recall/F1
  test_predictions.csv  every test image: true class, predicted class, confidence
  top_confusions.csv    most frequent (true -> predicted) mistakes: the report's failure cases
  confusion_matrix.png  row-normalised confusion matrix
  misclassified.png     the most confident mistakes, with true and predicted labels
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ImageDataset, build_transforms, load_rgb
from .model import load_checkpoint


@torch.no_grad()
def predict_probs(model, samples, transform, device, batch_size=64, num_workers=0):
    loader = DataLoader(ImageDataset(samples, transform), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    out = []
    for x, _ in tqdm(loader, desc="predict", leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        out.append(logits.float().softmax(1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0))


def _confusion_png(cm, classes, path):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(classes)
    fig = Figure(figsize=(1.0 + 0.75 * n, 0.8 + 0.7 * n))
    ax = fig.subplots()
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=45, ha="right")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title("test confusion matrix (counts; colour = share of true class)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def _misclassified_png(errors, path, max_images=24):
    errors = errors[:max_images]
    if not errors:
        return
    cols = 6
    rows = (len(errors) + cols - 1) // cols
    fig = Figure(figsize=(cols * 2.4, rows * 2.7))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, e in zip(axes, errors):
        try:
            img = load_rgb(e["file"])
            img.thumbnail((256, 256))
            ax.imshow(img)
        except Exception:
            pass
        ax.set_title(f"true: {e['true']}\npred: {e['predicted']} ({e['confidence']:.2f})", fontsize=8)
    fig.suptitle("most confident mistakes on the test split")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def evaluate_run(run_dir, data, device, batch_size=None):
    run_dir = Path(run_dir)
    model, ckpt = load_checkpoint(run_dir / "best.pth", device)
    classes = ckpt["classes"]
    if classes != data.classes:
        raise ValueError(f"checkpoint classes {classes} differ from the data's {data.classes}")
    tf = build_transforms(ckpt["img_size"], ckpt["mean"], ckpt["std"], train=False)
    probs = predict_probs(model, data.test_samples, tf, device, batch_size or data.test_loader.batch_size)
    y_true = np.array([y for _, y in data.test_samples])
    y_pred = probs.argmax(1)
    conf = probs.max(1)

    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)),
                                                         zero_division=0)
    metrics = {
        "checkpoint_epoch": ckpt["epoch"],
        "val_acc": round(float(ckpt["val_acc"]), 4),
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "test_macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": {c: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                          "f1": round(float(f1[i]), 4), "support": int(sup[i])}
                      for i, c in enumerate(classes)},
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows, errors = [], []
    for (path, t), p, c in zip(data.test_samples, y_pred, conf):
        row = {"file": path, "true": classes[t], "predicted": classes[p], "confidence": float(c)}
        rows.append(row)
        if t != p:
            errors.append(row)
    with open(run_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "true", "predicted", "confidence"])
        w.writeheader()
        w.writerows({**r, "confidence": f"{r['confidence']:.4f}"} for r in rows)

    pairs = Counter((e["true"], e["predicted"]) for e in errors).most_common()
    with open(run_dir / "top_confusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true", "predicted", "count"])
        w.writerows([t, p, n] for (t, p), n in pairs)

    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    _confusion_png(cm, classes, run_dir / "confusion_matrix.png")
    _misclassified_png(sorted(errors, key=lambda e: -e["confidence"]), run_dir / "misclassified.png")

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update({k: metrics[k] for k in ("test_accuracy", "test_balanced_accuracy", "test_macro_f1")})
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"test accuracy {metrics['test_accuracy']:.4f} | balanced {metrics['test_balanced_accuracy']:.4f} "
          f"| macro F1 {metrics['test_macro_f1']:.4f}  ({len(errors)} mistakes / {len(rows)})")
    print("most common mistakes:")
    for (t, p), n in pairs[:5]:
        print(f"  {t:20s} -> {p:20s} {n}")
    return metrics
