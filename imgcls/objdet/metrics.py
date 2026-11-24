"""COCO-style mean average precision, so YOLO and Faster R-CNN are scored by the same code.

For each class, every predicted box (strongest first) is matched to an unmatched ground-truth box of that class
in the same image with IoU >= t; precision/recall over the ranked list gives the average precision (101-point
interpolation, as in COCO). mAP50 uses t = 0.5; mAP50-95 averages t = 0.50, 0.55, ..., 0.95 (tighter boxes
score higher). Classes with no ground-truth box are left out of the mean.
"""

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def box_iou(a, b):
    """IoU matrix between boxes a (n, 4) and b (m, 4), x1 y1 x2 y2."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clip(0).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _average_precision(tp, n_gt):
    """101-point interpolated AP from the match flags of the ranked predictions."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp = np.asarray(tp, float)
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # precision envelope
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def mean_average_precision(predictions, ground_truth, num_classes, iou_thresholds=IOU_THRESHOLDS):
    """predictions:  {image_key: (boxes (n, 4), scores (n,), labels (n,))}
    ground_truth: {image_key: (boxes (m, 4), labels (m,))}   boxes x1 y1 x2 y2, labels 0..num_classes-1.
    Returns {"map50", "map50_95", "per_class": [{"ap50", "ap50_95", "gt"} per class]}."""
    per_class = []
    for c in range(num_classes):
        gts, n_gt = {}, 0
        for key, (boxes, labels) in ground_truth.items():
            g = np.asarray(boxes, float).reshape(-1, 4)[np.asarray(labels) == c]
            if len(g):
                gts[key] = g
                n_gt += len(g)
        preds = []
        for key, (boxes, scores, labels) in predictions.items():
            keep = np.asarray(labels) == c
            boxes = np.asarray(boxes, float).reshape(-1, 4)[keep]
            for box, score in zip(boxes, np.asarray(scores, float)[keep]):
                preds.append((score, key, box))
        preds.sort(key=lambda p: -p[0])
        ious = [box_iou(box[None], gts[key])[0] if key in gts else np.zeros(0) for _, key, box in preds]
        aps = []
        for t in iou_thresholds:
            used = {key: np.zeros(len(g), bool) for key, g in gts.items()}
            tp = []
            for (_, key, _), iou in zip(preds, ious):
                hit = False
                if len(iou):
                    candidates = np.where((iou >= t) & ~used[key])[0]
                    if len(candidates):
                        used[key][candidates[np.argmax(iou[candidates])]] = True
                        hit = True
                tp.append(hit)
            aps.append(_average_precision(tp, n_gt))
        per_class.append({"ap50": aps[0], "ap50_95": float(np.mean(aps)) if n_gt else float("nan"), "gt": n_gt})
    present = [p for p in per_class if p["gt"]]
    return {"map50": float(np.mean([p["ap50"] for p in present])) if present else float("nan"),
            "map50_95": float(np.mean([p["ap50_95"] for p in present])) if present else float("nan"),
            "per_class": per_class}
