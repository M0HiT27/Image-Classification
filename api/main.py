"""Prediction API: POST /predict (an image -> its category) and POST /detect-video (a video -> which of our
categories appear, when, and where in the frame).

Configured with environment variables (see api/.env.example):

  IMGCLS_CHECKPOINT  path to a best.pth from output/<model>/<timestamp>/ (absolute, or relative to the project
                     root). When set, the fine-tuned model predicts our 10 categories.
                     When empty, IMGCLS_MODEL is used without training:
                       - a CLIP backbone listed in imgcls/zeroshot.py runs zero-shot: it names our 10 categories
                         by comparing the image with text prompts ("a photo of a pug.");
                       - any other model must be ImageNet-trained: it returns ImageNet labels, and "category" is
                         filled only when the ImageNet label matches one of our categories.
  IMGCLS_MODEL       timm architecture. Used when there is no checkpoint (default: mobilenetv3_large_100.ra_in1k).
                     Optional with a checkpoint, but if set it must match the checkpoint's architecture.
  IMGCLS_DEVICE      auto (GPU if available, otherwise CPU; default) | cuda | cpu
  IMGCLS_TOP_K       how many guesses to return (default 3)
  IMGCLS_DETECTOR    ready-made detector for /detect-video: an OWLv2 or OWL-ViT model id from Hugging Face
                     (default google/owlv2-base-patch16-ensemble). Loaded on the first video request, downloaded
                     on first use to <project>/model_cache (IMGCLS_DETECTOR_CACHE overrides the folder).
  IMGCLS_DETECT_THRESHOLD  minimum box score for /detect-video (default: chosen per detector, see imgcls/detect.py)

In /detect-video the detector only finds WHERE our objects are. With a fine-tuned checkpoint or zero-shot CLIP, each
box is cropped and named by the /predict model (the detector's own label is returned alongside, with "agrees").
With a plain ImageNet model the detector's labels are used, and the ImageNet model names only the dog breeds.

Run from the project root:
    uvicorn api.main:app --env-file api/.env
Web page for trying it (upload or drop images): http://127.0.0.1:8000/  (api/static/index.html)
Interactive docs: http://127.0.0.1:8000/docs
"""

import io
import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "static" / "index.html"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imgcls  # noqa: E402  (needs ROOT on sys.path)
import timm  # noqa: E402
from imgcls.data import build_transforms, load_rgb  # noqa: E402
from imgcls.zeroshot import OPEN_CLIP_NAMES, ZeroShotCLIP  # noqa: E402
from timm.data import ImageNetInfo, infer_imagenet_subset  # noqa: E402

DEFAULT_MODEL = "mobilenetv3_large_100.ra_in1k"
MAX_UPLOAD_BYTES = 20 * 2**20
MAX_VIDEO_BYTES = 200 * 2**20
MAX_VIDEO_FRAMES = 300  # frames analysed per request (2.5 minutes of video at 2 frames per second)
DOG_NAMES = [imgcls.DISPLAY_NAMES[c] for c in ("golden_retriever", "labrador_retriever", "siberian_husky", "pug")]

# ImageNet classes that correspond to our categories; used only by the pretrained (not fine-tuned) mode.
# ImageNet-1k has no sunflower or pine cone class; sunflower exists only in ImageNet-21k/22k label sets.
IMAGENET_TO_CATEGORY = {
    "n02099601": "golden_retriever",
    "n02099712": "labrador_retriever",
    "n02110185": "siberian_husky",
    "n02109961": "siberian_husky",  # "Eskimo dog, husky": ImageNet's near-identical husky class
    "n02110958": "pug",
    "n02276258": "butterfly",  # admiral
    "n02277742": "butterfly",  # ringlet
    "n02279972": "butterfly",  # monarch
    "n02280649": "butterfly",  # cabbage butterfly
    "n02281406": "butterfly",  # sulphur butterfly
    "n02281787": "butterfly",  # lycaenid
    "n07753592": "banana",
    "n07747607": "orange",
    "n07734744": "mushroom",
    "n12985857": "mushroom",  # coral fungus
    "n12998815": "mushroom",  # agaric
    "n13037406": "mushroom",  # gyromitra
    "n13040303": "mushroom",  # stinkhorn
    "n13044778": "mushroom",  # earthstar
    "n13052670": "mushroom",  # hen-of-the-woods
    "n13054560": "mushroom",  # bolete
    "n11978233": "sunflower",  # sunflower, helianthus (21k/22k only)
    "n11978713": "sunflower",  # common sunflower (21k/22k only)
}


def _resolve(path_text):
    path = Path(path_text).expanduser()
    for candidate in (path, ROOT / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"IMGCLS_CHECKPOINT not found: {path_text} (looked in the current folder and {ROOT})")


class Predictor:
    """Loads the model once at startup and classifies PIL images."""

    def __init__(self):
        checkpoint = os.environ.get("IMGCLS_CHECKPOINT", "").strip()
        arch = os.environ.get("IMGCLS_MODEL", "").strip()
        self.top_k = max(1, int(os.environ.get("IMGCLS_TOP_K", "3") or 3))
        self.device = imgcls.get_device(os.environ.get("IMGCLS_DEVICE", "auto") or "auto")

        if checkpoint:
            path = _resolve(checkpoint)
            self.model, ckpt = imgcls.load_checkpoint(path, self.device)
            if arch and arch != ckpt["arch"]:
                raise RuntimeError(f"IMGCLS_MODEL={arch} but the checkpoint was trained with {ckpt['arch']}; "
                                   "leave IMGCLS_MODEL empty or make them match")
            self.mode, self.arch, self.checkpoint = "fine-tuned", ckpt["arch"], str(path)
            self.labels = [imgcls.DISPLAY_NAMES.get(c, c) for c in ckpt["classes"]]
            self.categories = list(self.labels)
            self.transform = build_transforms(ckpt["img_size"], ckpt["mean"], ckpt["std"], train=False)
        elif arch in OPEN_CLIP_NAMES:  # CLIP without a checkpoint: our categories from text prompts
            self.labels = list(imgcls.DISPLAY_NAMES.values())
            self.categories = list(self.labels)
            self.model = ZeroShotCLIP(arch, self.labels).to(self.device).eval()
            self.transform = self.model.transform
            self.mode, self.arch, self.checkpoint = "zero-shot CLIP (text prompts, not trained)", arch, None
        else:
            arch = arch or DEFAULT_MODEL
            no_head = (f"IMGCLS_MODEL={arch} has no ImageNet classifier, so it can't predict without training. "
                       "Set IMGCLS_CHECKPOINT to the best.pth of a model you trained from it, or choose an "
                       "ImageNet-trained timm model, or a CLIP backbone that runs zero-shot: "
                       + ", ".join(OPEN_CLIP_NAMES))
            cfg = timm.models.get_pretrained_cfg(arch)  # no download; None for names timm doesn't know
            if cfg is not None and not infer_imagenet_subset(cfg.to_dict()):
                raise RuntimeError(no_head)  # before create_model downloads the weights (CLIP: 350 MB)
            self.model = timm.create_model(arch, pretrained=True).to(self.device).eval()
            subset = infer_imagenet_subset(self.model)
            if not subset:
                raise RuntimeError(no_head)
            info = ImageNetInfo(subset)
            synsets = [info.index_to_label_name(i) for i in range(info.num_classes())]
            self.labels = [info.index_to_description(i).split(",")[0] for i in range(info.num_classes())]
            self.categories = [imgcls.DISPLAY_NAMES[IMAGENET_TO_CATEGORY[s]] if s in IMAGENET_TO_CATEGORY else None
                               for s in synsets]
            data_cfg = timm.data.resolve_model_data_config(self.model)
            self.mode, self.arch, self.checkpoint = f"pretrained ({subset}, not fine-tuned)", arch, None
            self.transform = build_transforms(data_cfg["input_size"][-1], data_cfg["mean"], data_cfg["std"],
                                              train=False)
        # fine-tuned and zero-shot models know all 10 categories; the ImageNet model has no sunflower or pine cone
        self.names_all_categories = set(imgcls.DISPLAY_NAMES.values()) <= set(self.categories)
        print(f"API ready: {self.arch}, {self.mode}, {len(self.labels)} classes, device {self.device}", flush=True)

    @torch.inference_mode()
    def predict(self, image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        probs = self.model(x).float().softmax(1)[0].cpu()
        conf, idx = probs.topk(min(self.top_k, len(probs)))
        best = int(idx[0])
        return {
            "category": self.categories[best],
            "label": self.labels[best],
            "confidence": round(float(conf[0]), 4),
            "top_k": [{"label": self.labels[int(i)], "category": self.categories[int(i)],
                       "confidence": round(float(c), 4)} for c, i in zip(conf, idx)],
            "model": self.arch,
            "mode": self.mode,
            "checkpoint": self.checkpoint,
        }

    @torch.inference_mode()
    def classify_crops(self, crops):
        """(label, probability) of the top class for each image, in one batch. /detect-video uses it to name
        every box with the fine-tuned model."""
        x = torch.stack([self.transform(c) for c in crops]).to(self.device)
        conf, idx = self.model(x).float().softmax(1).max(1)
        return [(self.labels[i], c) for c, i in zip(conf.cpu().tolist(), idx.cpu().tolist())]

    @torch.inference_mode()
    def classify_dog(self, image):
        """Which of our 4 breeds a dog crop shows: (breed, its share of the probability given to the 4 breeds)."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        probs = self.model(x).float().softmax(1)[0].cpu().tolist()
        totals = dict.fromkeys(DOG_NAMES, 0.0)
        for p, category in zip(probs, self.categories):  # pretrained mode: several ImageNet classes per breed
            if category in totals:
                totals[category] += p
        total = sum(totals.values())
        best = max(totals, key=totals.get)
        return (best, totals[best] / total) if total > 0 else (None, 0.0)


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor()
    app.state.detector = None  # loaded on the first /detect-video request
    yield


_detector_lock = threading.Lock()


def get_detector():
    if app.state.detector is None:
        with _detector_lock:
            if app.state.detector is None:
                try:
                    import cv2  # noqa: F401
                    import transformers  # noqa: F401
                except ImportError:
                    raise HTTPException(503, "Video detection needs two more packages: "
                                             "pip install transformers opencv-python-headless")
                from imgcls import detect

                model_id = os.environ.get("IMGCLS_DETECTOR", "").strip() or detect.DEFAULT_DETECTOR
                print(f"Loading detector {model_id} (downloads about 0.6 GB the first time)...", flush=True)
                app.state.detector = detect.Detector(model_id, device=app.state.predictor.device)
                print(f"Detector ready: {model_id}, threshold {app.state.detector.default_threshold}", flush=True)
    return app.state.detector


app = FastAPI(title="Natural objects image classifier", lifespan=lifespan,
              description="Upload an image to POST /predict to get its category, or a video to "
                          "POST /detect-video to find where and when our categories appear.")


@app.get("/", include_in_schema=False)
def page():
    """Small web page that sends images to POST /predict and shows the guesses (not part of the API docs)."""
    return FileResponse(PAGE)


@app.post("/predict")
def predict(file: UploadFile = File(..., description="image file (jpg, png, webp, ...)")):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "The upload is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
    try:
        image = load_rgb(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Could not read the upload as an image.")
    return {"filename": file.filename, **app.state.predictor.predict(image)}


@app.post("/detect-video")
def detect_video(
    file: UploadFile = File(..., description="video file (mp4, mov, avi, webm, ...)"),
    fps: float = Form(1.0, ge=0.1, le=10, description="frames analysed per second of video"),
    threshold: float | None = Form(None, gt=0, le=1, description="minimum box score; empty = detector default"),
):
    """Find our categories in a video: per-category time ranges (summary) and boxes for every analysed frame."""
    from imgcls import detect

    detector = get_detector()
    if threshold is None and os.environ.get("IMGCLS_DETECT_THRESHOLD", "").strip():
        threshold = float(os.environ["IMGCLS_DETECT_THRESHOLD"])
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)  # OpenCV reads videos from a file path
    try:
        size = 0
        with tmp:
            while chunk := file.file.read(2**20):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(413, f"Video larger than {MAX_VIDEO_BYTES // 2**20} MB.")
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(400, "The upload is empty.")
        info = detect.video_info(tmp.name)
        if info is None:
            raise HTTPException(400, "Could not read the upload as a video.")
        started = time.time()

        def progress(n, t):
            if n % 10 == 0:
                print(f"detect-video {file.filename}: {n} frames analysed ({t:.1f} s of video)", flush=True)

        # a fine-tuned or zero-shot model names every box; the plain ImageNet model can't name sunflowers or
        # pine cones, so with it the detector's labels stay and the ImageNet model only names dog breeds
        predictor = app.state.predictor
        every_box = predictor.names_all_categories
        result = detect.detect_video(detector, tmp.name, fps=fps, threshold=threshold,
                                     classify=predictor.classify_crops if every_box else None,
                                     classify_dog=None if every_box else predictor.classify_dog,
                                     max_frames=MAX_VIDEO_FRAMES, progress=progress)
    finally:
        os.unlink(tmp.name)
    kind = "fine-tuned" if predictor.checkpoint else predictor.mode
    labels_from = (f"{predictor.arch} ({kind}): every box" if every_box
                   else f"detector; dog breeds by {predictor.arch} ({predictor.mode})")
    return {"filename": file.filename, **info, "detector": detector.model_id, "labels_from": labels_from,
            "seconds": round(time.time() - started, 1), **result}
