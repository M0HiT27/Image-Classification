"""Prediction API with a single endpoint: POST /predict with an image, get back what it is.

Configured with environment variables (see api/.env.example):

  IMGCLS_CHECKPOINT  path to a best.pth from output/<model>/<timestamp>/ (absolute, or relative to the project
                     root). When set, the fine-tuned model predicts our 10 categories.
                     When empty, the pretrained ImageNet model from IMGCLS_MODEL is used as-is (not fine-tuned):
                     it returns ImageNet labels, and "category" is filled only when the ImageNet label matches one
                     of our categories.
  IMGCLS_MODEL       timm architecture. Used when there is no checkpoint (default: mobilenetv3_large_100.ra_in1k).
                     Optional with a checkpoint, but if set it must match the checkpoint's architecture.
  IMGCLS_DEVICE      auto (GPU if available, otherwise CPU; default) | cuda | cpu
  IMGCLS_TOP_K       how many guesses to return (default 3)

Run from the project root:
    uvicorn api.main:app --env-file api/.env
Interactive docs (upload an image in the browser): http://127.0.0.1:8000/docs
"""

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imgcls  # noqa: E402  (needs ROOT on sys.path)
import timm  # noqa: E402
from imgcls.data import build_transforms, load_rgb  # noqa: E402
from timm.data import ImageNetInfo, infer_imagenet_subset  # noqa: E402

DEFAULT_MODEL = "mobilenetv3_large_100.ra_in1k"
MAX_UPLOAD_BYTES = 20 * 2**20

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
        else:
            arch = arch or DEFAULT_MODEL
            self.model = timm.create_model(arch, pretrained=True).to(self.device).eval()
            subset = infer_imagenet_subset(self.model)
            if not subset:
                raise RuntimeError(f"{arch} has no ImageNet classifier to predict with; "
                                   "choose an ImageNet-trained timm model or set IMGCLS_CHECKPOINT")
            info = ImageNetInfo(subset)
            synsets = [info.index_to_label_name(i) for i in range(info.num_classes())]
            self.labels = [info.index_to_description(i).split(",")[0] for i in range(info.num_classes())]
            self.categories = [imgcls.DISPLAY_NAMES[IMAGENET_TO_CATEGORY[s]] if s in IMAGENET_TO_CATEGORY else None
                               for s in synsets]
            data_cfg = timm.data.resolve_model_data_config(self.model)
            self.mode, self.arch, self.checkpoint = f"pretrained ({subset}, not fine-tuned)", arch, None
            self.transform = build_transforms(data_cfg["input_size"][-1], data_cfg["mean"], data_cfg["std"],
                                              train=False)
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


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="Natural objects image classifier", lifespan=lifespan,
              description="Upload an image to POST /predict to get its category.")


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
