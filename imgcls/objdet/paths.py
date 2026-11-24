"""Where data_object/ and the detection runs live: on this machine or on Kaggle.

Local:  data_object/ in the project (IMGCLS_OBJECT_DIR overrides); runs in output/ like the classifiers.
Kaggle: notebook 01 writes /kaggle/working/data_object; save it as a Kaggle dataset and attach it to notebooks
        02 and 03 with *Add Input*: they find it under /kaggle/input. Runs go to /kaggle/working/output.
"""

import os
from pathlib import Path

from .. import config
from ..environment import KAGGLE_INPUT, KAGGLE_OUTPUT, detect, use_paths

CLASSES = list(config.EXPECTED_CLASSES)  # detector class ids 0..9: the same order as the classifiers
OBJECT_DIR_NAME = "data_object"
KAGGLE_WORKING = "/kaggle/working"


def default_object_dir(env=None):
    """Where notebook 01 writes data_object/."""
    if os.environ.get("IMGCLS_OBJECT_DIR"):
        return Path(os.environ["IMGCLS_OBJECT_DIR"]).expanduser().resolve()
    env = env or detect()
    return Path(KAGGLE_WORKING) / OBJECT_DIR_NAME if env == "kaggle" else config.ROOT / OBJECT_DIR_NAME


def is_data_object(folder):
    folder = Path(folder)
    return (folder / "data.yaml").is_file() and (folder / "images" / "train").is_dir()


def _search(root, max_depth=6):
    queue = [(Path(root), 0)]
    while queue:
        folder, depth = queue.pop(0)
        if is_data_object(folder):
            return folder
        if depth < max_depth:
            try:
                queue += [(p, depth + 1) for p in sorted(folder.iterdir()) if p.is_dir()]
            except OSError:
                pass
    return None


def find_data_object(env=None):
    """The data_object/ folder made by notebook 01 (on Kaggle: an attached dataset, else /kaggle/working)."""
    env = env or detect()
    candidates = []
    if os.environ.get("IMGCLS_OBJECT_DIR"):
        candidates.append(default_object_dir(env))
    elif env == "kaggle":
        found = _search(KAGGLE_INPUT)
        if found:
            candidates.append(found)
    candidates.append(default_object_dir(env))
    for folder in candidates:
        if is_data_object(folder):
            return folder
    raise FileNotFoundError(
        f"No data_object/ found (looked in {', '.join(str(c) for c in candidates)}"
        + (f" and under {KAGGLE_INPUT}" if env == "kaggle" else "") + "). Run "
        "object_detection_notebooks/01_build_data_object.ipynb first; on Kaggle save its data_object folder as a "
        "dataset and attach it with 'Add Input'.")


def setup(object_dir=None):
    """Pick the runs folder (like imgcls.setup_environment, but without needing the classifier's data/) and find
    data_object/ (or check the folder given). Returns (environment, data_object folder)."""
    env = detect()
    output_dir = os.environ.get("IMGCLS_OUTPUT_DIR") or (KAGGLE_OUTPUT if env == "kaggle" else None)
    use_paths(output_dir=output_dir)
    if object_dir:
        object_dir = Path(object_dir).expanduser().resolve()
        if not is_data_object(object_dir):
            raise FileNotFoundError(f"{object_dir} has no data.yaml and images/train; run notebook 01 first")
    else:
        object_dir = find_data_object(env)
    print(f"Environment: {env}\n  data_object: {object_dir}\n  runs: {config.OUTPUT_DIR}")
    return env, object_dir


def detection_runs():
    """Every detection run under the runs folder with its scores (newest first), to compare detectors."""
    import json

    import pandas as pd

    rows = []
    for summary in Path(config.OUTPUT_DIR).glob("*/*/summary.json"):
        try:
            data = json.loads(summary.read_text())
        except ValueError:
            continue
        if data.get("kind"):
            rows.append({"run": f"{summary.parent.parent.name}/{summary.parent.name}", "detector": data["kind"],
                         **{k: data.get(k) for k in ("arch", "best_epoch", "best_val_map50_95", "test_map50",
                                                     "test_map50_95", "minutes")}})
    return pd.DataFrame(rows).sort_values("run", ascending=False) if rows else pd.DataFrame()
