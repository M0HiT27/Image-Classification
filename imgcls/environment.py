"""Where the data and the training runs live: on this machine, on Google Colab or on Kaggle.

Local:  data/ and output/ in the project folder (or IMGCLS_DATA_DIR / IMGCLS_OUTPUT_DIR).
Colab:  the data comes from imgcls_data.zip on Google Drive (made by scripts/pack_data.py) and is
        unpacked to the VM's own disk, because reading thousands of small files from Drive is slow.
        Runs are saved on Drive, so they survive a disconnect.
Kaggle: the data comes from a Kaggle dataset attached to the notebook (found under /kaggle/input).
        Runs are saved in /kaggle/working/output, which Kaggle keeps as the notebook's output.
"""

import importlib.util
import os
import shutil
import zipfile
from pathlib import Path

from . import config

COLAB_DRIVE_DIR = "/content/drive/MyDrive/imgcls"
COLAB_DATA_DEST = "/content/imgcls_data"  # the VM's own disk: fast, but emptied when the session ends
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_OUTPUT = "/kaggle/working/output"
DATA_ZIP_NAME = "imgcls_data.zip"


def detect():
    """'colab', 'kaggle' or 'local'."""
    try:
        in_colab = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        in_colab = False
    if in_colab or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").is_dir():
        return "kaggle"
    return "local"


def find_data_dir(search_root, max_depth=4):
    """First folder at or below search_root laid out like data/, i.e. holding <class>/train/."""
    queue = [(Path(search_root), 0)]
    while queue:
        folder, depth = queue.pop(0)
        if any((folder / c / "train").is_dir() for c in config.EXPECTED_CLASSES):
            return folder
        if depth < max_depth:
            try:
                queue += [(p, depth + 1) for p in sorted(folder.iterdir()) if p.is_dir()]
            except OSError:
                pass
    return None


def use_paths(data_dir=None, output_dir=None):
    """Point every imgcls function at another data folder and/or output folder."""
    if data_dir is not None:
        config.DATA_DIR = Path(data_dir).expanduser().resolve()
    if output_dir is not None:
        config.OUTPUT_DIR = Path(output_dir).expanduser().resolve()
    return config.DATA_DIR, config.OUTPUT_DIR


def unpack_data_zip(zip_path, dest):
    """Unpack the zip made by scripts/pack_data.py into dest (once) and return the data folder in it."""
    zip_path, dest = Path(zip_path), Path(dest)
    done = dest / ".unpacked"
    if not done.exists():
        if not zip_path.is_file():
            raise FileNotFoundError(
                f"{zip_path} not found. Make it locally with `python scripts/pack_data.py` and upload it "
                "to that Google Drive folder (README > Colab and Kaggle).")
        dest.mkdir(parents=True, exist_ok=True)
        local_copy = dest.parent / f"{dest.name}.zip"
        print(f"Copying {zip_path} to the VM and unpacking it to {dest} (about a minute)...", flush=True)
        shutil.copyfile(zip_path, local_copy)
        with zipfile.ZipFile(local_copy) as z:
            z.extractall(dest)
        local_copy.unlink()
        done.write_text(str(zip_path), encoding="utf-8")
    data_dir = find_data_dir(dest)
    if data_dir is None:
        raise FileNotFoundError(f"{zip_path} has no <class>/train folders; re-create it with scripts/pack_data.py")
    return data_dir


def setup_environment(colab_drive_dir=COLAB_DRIVE_DIR):
    """Find the data and pick the folder for runs on this machine, Colab or Kaggle.

    IMGCLS_DATA_DIR / IMGCLS_OUTPUT_DIR, when set, always win. Returns 'local', 'colab' or 'kaggle'.
    """
    env = detect()
    data_dir = os.environ.get("IMGCLS_DATA_DIR") or None
    output_dir = os.environ.get("IMGCLS_OUTPUT_DIR") or None

    if env == "colab" and colab_drive_dir:
        drive_dir = Path(colab_drive_dir)
        if str(drive_dir).startswith("/content/drive") and not Path("/content/drive/MyDrive").is_dir():
            from google.colab import drive  # asks for permission to use your Google Drive
            drive.mount("/content/drive")
        if data_dir is None:
            data_dir = unpack_data_zip(drive_dir / DATA_ZIP_NAME, COLAB_DATA_DEST)
        if output_dir is None:
            output_dir = drive_dir / "output"
    elif env == "kaggle":
        if data_dir is None:
            data_dir = find_data_dir(KAGGLE_INPUT)
            if data_dir is None:
                raise FileNotFoundError(
                    f"No dataset with <class>/train folders under {KAGGLE_INPUT}. In the notebook's side panel "
                    "use 'Add Input' and pick the dataset made from imgcls_data.zip (README > Colab and Kaggle).")
        if output_dir is None:
            output_dir = KAGGLE_OUTPUT

    use_paths(data_dir, output_dir)
    print(f"Environment: {env}\n  data: {config.DATA_DIR}\n  runs: {config.OUTPUT_DIR}")
    return env
