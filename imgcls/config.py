"""Project-wide paths and the class list from the task briefing."""

import os
from pathlib import Path


def _path_from_env(name, default):
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default


ROOT = Path(__file__).resolve().parent.parent
# Both can be moved with environment variables, or with imgcls.setup_environment() / imgcls.use_paths()
# (Colab and Kaggle keep the data and the runs outside the code folder).
DATA_DIR = _path_from_env("IMGCLS_DATA_DIR", ROOT / "data")        # <class>/train and <class>/test
OUTPUT_DIR = _path_from_env("IMGCLS_OUTPUT_DIR", ROOT / "output")  # <model_name>/<timestamp>/

# Category 2 "Natural objects" from the briefing (folder name -> name used in the predictions file)
DISPLAY_NAMES = {
    "golden_retriever": "Golden retriever",
    "labrador_retriever": "Labrador retriever",
    "siberian_husky": "Siberian husky",
    "pug": "Pug",
    "sunflower": "Sunflower",
    "banana": "Banana",
    "orange": "Orange",
    "mushroom": "Mushroom",
    "butterfly": "Butterfly",
    "pine_cone": "Pine cone",
}
EXPECTED_CLASSES = sorted(DISPLAY_NAMES)
