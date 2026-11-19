"""Project-wide paths and the class list from the task briefing."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # data/<class>/train and data/<class>/test
OUTPUT_DIR = ROOT / "output"      # output/<model_name>/<timestamp>/

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
