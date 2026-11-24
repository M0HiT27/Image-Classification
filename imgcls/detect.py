"""Find our categories in images and videos with a ready-made open-vocabulary detector (no training).

An open-vocabulary detector finds objects described in words ("a banana", "a dog") and draws a box around each
one. We use OWLv2 / OWL-ViT from Hugging Face transformers, only to find WHERE our objects are: each box is
cropped and named by our own fine-tuned classifier (`classify`), and the detector's label is kept alongside for
comparison. Without our classifier the detector's labels are used, and dog breeds can still come from a
breed-only callback (`classify_dog`), because the detector can't reliably tell the four breeds apart.

Two extra packages, imported only when this module is used (the rest of imgcls works without them):
    pip install transformers opencv-python-headless
Model files are kept in <project>/model_cache (override with IMGCLS_DETECTOR_CACHE), not in the user-wide
Hugging Face cache on C:.
"""

import os
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from . import config

DEFAULT_DETECTOR = "google/owlv2-base-patch16-ensemble"
# score a box needs before it counts; chosen per model on 60 train/ images (README > Video detection)
DEFAULT_THRESHOLDS = {"owlv2": 0.3, "owlvit": 0.2}
DOG_CLASSES = ("golden_retriever", "labrador_retriever", "siberian_husky", "pug")
_FOLDER = {name: folder for folder, name in config.DISPLAY_NAMES.items()}  # "Pine cone" -> "pine_cone"
UNCLEAR_DOG = "Dog (breed unclear)"
BREED_MIN_CONFIDENCE = 0.5  # below this the breed from classify_dog isn't trusted

# what the detector is asked to find: our class (folder name, or "dog" for the 4 breeds) -> phrasings
QUERIES = {
    "dog": ["a dog", "a puppy"],
    "banana": ["a banana", "a bunch of bananas"],
    "orange": ["an orange fruit", "an orange"],
    "mushroom": ["a mushroom", "a toadstool"],
    "butterfly": ["a butterfly"],
    "sunflower": ["a sunflower"],
    "pine_cone": ["a pine cone"],
}
# look-alikes: a box that matches one of these better than any of our phrasings is dropped
DISTRACTORS = ["a cat", "a person", "a lemon", "a moth", "a daisy", "an acorn"]


def _cache_dir():
    return Path(os.environ.get("IMGCLS_DETECTOR_CACHE", "").strip() or config.ROOT / "model_cache")


class Detector:
    """Open-vocabulary detector (OWLv2 or OWL-ViT) set up to look for our categories."""

    def __init__(self, model_id=DEFAULT_DETECTOR, device="cpu", use_distractors=True):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        cache = _cache_dir()
        self.model_id, self.device = model_id, torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=cache)
        self.model.to(self.device).eval()
        self.kind = self.model.config.model_type
        if self.kind not in DEFAULT_THRESHOLDS:
            raise ValueError(f"{model_id} is a {self.kind} model; only OWLv2 and OWL-ViT models are supported")
        self.default_threshold = DEFAULT_THRESHOLDS[self.kind]
        self.queries = [q for qs in QUERIES.values() for q in qs]
        self.query_class = [c for c, qs in QUERIES.items() for _ in qs]
        if use_distractors:
            self.queries += DISTRACTORS
            self.query_class += [None] * len(DISTRACTORS)
        text = self.processor(text=[self.queries], return_tensors="pt")
        self._text = {k: v.to(self.device) for k, v in text.items() if k in ("input_ids", "attention_mask")}

    @torch.inference_mode()
    def raw(self, image):
        """Scores (patches x queries, 0-1) and boxes (patches x 4, x1 y1 x2 y2 in pixels) for one PIL image."""
        pixels = self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
        out = self.model(pixel_values=pixels.to(self.device), **self._text)
        scores = out.logits[0].float().sigmoid().cpu()
        cx, cy, bw, bh = out.pred_boxes[0].float().cpu().unbind(-1)
        w, h = image.size
        # OWLv2 pads the image to a square (bottom/right) before resizing; OWL-ViT stretches it
        sx, sy = (max(w, h), max(w, h)) if self.kind == "owlv2" else (w, h)
        boxes = torch.stack([(cx - bw / 2) * sx, (cy - bh / 2) * sy, (cx + bw / 2) * sx, (cy + bh / 2) * sy], -1)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        return scores, boxes

    def detect(self, image, threshold=None, classify=None, classify_dog=None, nms_iou=0.3):
        """Boxes of our categories in one PIL image, strongest detector score first.

        The detector decides WHERE something of ours is. Who decides WHAT it is:
          classify(crops) -> [(class, confidence), ...]   our classifier names every box ("category",
              "confidence"); the detector's own label stays in "detector_label", and "agrees" says whether
              the two match (any breed matches the detector's "Dog");
          otherwise the detector's label is used, and classify_dog(crop) -> (breed, confidence) names the
              breed of dog boxes (without it, or when it is unsure, they are "Dog (breed unclear)").
        """
        from torchvision.ops import batched_nms, nms

        threshold = self.default_threshold if threshold is None else threshold
        scores, boxes = self.raw(image)
        best, query = scores.max(1)  # each patch gets the phrasing it matches best
        ours = torch.tensor([c is not None for c in self.query_class])[query]
        size_ok = ((boxes[:, 2] - boxes[:, 0]) >= 2) & ((boxes[:, 3] - boxes[:, 1]) >= 2)
        keep = (best >= threshold) & ours & size_ok
        if not keep.any():
            return []
        idx = keep.nonzero().flatten()
        groups = torch.tensor([list(QUERIES).index(self.query_class[q]) for q in query[idx].tolist()])
        idx = idx[batched_nms(boxes[idx], best[idx], groups, nms_iou)]  # one box per object and class
        idx = idx[nms(boxes[idx], best[idx], 0.7)]                       # and one class per object
        idx = _drop_parts(idx.tolist(), boxes, groups_of=lambda i: self.query_class[query[i]])

        found, kinds = [], []
        for i in idx:
            kind = self.query_class[query[i]]  # "dog" or one of our class folders
            kinds.append(kind)
            found.append({"category": None, "confidence": None, "box": [round(v) for v in boxes[i].tolist()],
                          "detector_label": "Dog" if kind == "dog" else config.DISPLAY_NAMES[kind],
                          "detector_score": round(float(best[i]), 4), "matched_text": self.queries[query[i]]})
        if not found:
            return found

        if classify is not None:  # our classifier names every box
            names = classify([_box_crop(image, d["box"]) for d in found])
            for d, kind, (name, conf) in zip(found, kinds, names):
                folder = _FOLDER.get(name, name)
                d["category"] = config.DISPLAY_NAMES.get(folder, name)
                d["confidence"] = round(float(conf), 4)
                d["agrees"] = folder == kind or (kind == "dog" and folder in DOG_CLASSES)
            return found

        for d, kind in zip(found, kinds):  # the detector names the boxes
            d["category"], d["confidence"] = d["detector_label"], d["detector_score"]
            if kind == "dog":
                d["category"] = UNCLEAR_DOG
                if classify_dog is not None:
                    breed, conf = classify_dog(_box_crop(image, d["box"]))
                    d["breed_confidence"] = round(float(conf), 4)
                    if conf >= BREED_MIN_CONFIDENCE:
                        d["category"] = config.DISPLAY_NAMES.get(breed, breed)
        return found


def _drop_parts(order, boxes, groups_of, inside=0.8):
    """Drop a box that lies mostly (`inside` of its area) within a stronger box of the same class, e.g. a
    second "mushroom" box around just the stem. `order` is sorted by score, strongest first."""
    kept = []
    for i in order:
        x1, y1, x2, y2 = boxes[i].tolist()
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        part = False
        for k in kept:
            if groups_of(k) != groups_of(i):
                continue
            a1, b1, a2, b2 = boxes[k].tolist()
            overlap = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
            if overlap / area >= inside:
                part = True
                break
        if not part:
            kept.append(i)
    return kept


def _box_crop(image, box, margin=0.05, fill=(124, 116, 104)):
    """The box plus `margin` on each side, padded to a square with grey (the ImageNet / CLIP mean colour).

    The classifiers resize and centre-crop to a square, so a long thin box (a banana) cropped as-is would lose its
    ends; padding keeps the whole object. An earlier version grew the box into a square of the image instead,
    which pulled neighbours into the crop: a butterfly sitting on a wide, flat cut orange filled most of the
    orange's crop, and every classifier called it a butterfly. On the 64 boxes of that video, this crop names
    49 (CLIP linear probe), 57 (zero-shot CLIP) and 43 (MobileNetV3) right, against 29 / 36 / 32 before."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    crop = image.crop((max(0, x1 - mx), max(0, y1 - my), min(image.width, x2 + mx), min(image.height, y2 + my)))
    side = max(crop.size)
    square = Image.new("RGB", (side, side), fill)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square


def read_frames(path, every_s=0.5):
    """Yield (time in seconds, PIL image) every `every_s` seconds of a video file. An unreadable file yields
    nothing; check it with video_info() first."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        index, next_t = 0, 0.0
        while cap.grab():
            t = index / fps if fps > 0 else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            index += 1
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if ok:
                yield t, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            while next_t <= t + 1e-6:
                next_t += every_s
    finally:
        cap.release()


def video_info(path):
    """fps, frame count, duration and size, or None if the file isn't a video OpenCV can read."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or not cap.grab():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"fps": round(fps, 3), "frame_count": frames,
                "duration_s": round(frames / fps, 2) if fps > 0 and frames > 0 else None,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def summarize(frames, step):
    """Per category: when it was seen (time ranges), how often, and the most boxes in one frame."""
    hits = {}
    for f in frames:
        counts = Counter(d["category"] for d in f["detections"])
        for cat, n in counts.items():
            conf = max(d["confidence"] for d in f["detections"] if d["category"] == cat)
            hits.setdefault(cat, []).append((f["time_s"], n, conf))
    summary = []
    for cat, seen in hits.items():
        segments = []
        for t, _, _ in seen:  # neighbouring sampled frames join into one time range
            if segments and t - segments[-1]["end_s"] <= 1.5 * step + 1e-6:
                segments[-1]["end_s"], segments[-1]["frames"] = t, segments[-1]["frames"] + 1
            else:
                segments.append({"start_s": t, "end_s": t, "frames": 1})
        summary.append({"category": cat, "frames": len(seen), "first_seen_s": seen[0][0],
                        "last_seen_s": seen[-1][0], "max_count": max(n for _, n, _ in seen),
                        "max_confidence": max(c for _, _, c in seen), "segments": segments})
    summary.sort(key=lambda s: s["first_seen_s"])
    return summary


def detect_video(detector, path, fps=1.0, threshold=None, classify=None, classify_dog=None, max_frames=300,
                 progress=None):
    """Run the detector on `fps` frames per second of a video; returns frames with boxes plus a summary.

    classify / classify_dog: who names the boxes, see Detector.detect. With classify, "disagreements" counts
    the boxes where our classifier and the detector's own label differ.
    At most `max_frames` frames are analysed; "truncated" is True when the video was longer than that.
    """
    step = 1.0 / fps
    threshold = detector.default_threshold if threshold is None else threshold
    frames, truncated = [], False
    for t, image in read_frames(path, step):
        if len(frames) >= max_frames:
            truncated = True
            break
        found = detector.detect(image, threshold, classify=classify, classify_dog=classify_dog)
        frames.append({"time_s": round(t, 2), "detections": found})
        if progress:
            progress(len(frames), t)
    boxes = [d for f in frames for d in f["detections"]]
    return {"sampled_fps": fps, "threshold": threshold, "frames_analyzed": len(frames), "truncated": truncated,
            "boxes": len(boxes),
            "disagreements": sum(not d["agrees"] for d in boxes) if classify is not None else None,
            "summary": summarize(frames, step), "frames": frames}
