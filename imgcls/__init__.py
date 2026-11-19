"""Shared training code used by every model notebook in notebooks/."""

import os

# Windows without Developer Mode can't create symlinks; the model cache still works, so skip the warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from .config import DATA_DIR, DISPLAY_NAMES, EXPECTED_CLASSES, OUTPUT_DIR, ROOT
from .data import build_data, discover_classes
from .device import get_device
from .engine import config_from_globals, set_seed, train
from .evaluate import evaluate_run
from .model import create_model, load_checkpoint
from .predict import predict_folder

__all__ = [
    "ROOT", "DATA_DIR", "OUTPUT_DIR", "DISPLAY_NAMES", "EXPECTED_CLASSES",
    "build_data", "discover_classes", "get_device", "config_from_globals", "set_seed", "train",
    "evaluate_run", "create_model", "load_checkpoint", "predict_folder",
]
