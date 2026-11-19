"""Pick the compute device: GPU when available, otherwise CPU."""

import os

import torch


def get_device(prefer="auto"):
    """prefer: "auto" (CUDA GPU, then Apple GPU, then CPU), "cuda", "mps" or "cpu".

    Asking for a GPU that isn't there falls back to the CPU with a warning instead of failing.
    """
    prefer = (prefer or "auto").lower()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if prefer in ("auto", "cuda", "gpu") and torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {props.name} ({props.total_memory / 2**30:.1f} GB)")
        return device
    if prefer in ("auto", "mps", "gpu") and mps_ok:
        print("Using Apple GPU (MPS)")
        return torch.device("mps")
    if prefer not in ("auto", "cpu"):
        print(f"WARNING: device '{prefer}' requested but no GPU is available; falling back to CPU.")
    print(f"Using CPU ({torch.get_num_threads()} threads, {os.cpu_count()} logical cores). "
          "Training will be slow; TRAIN_MODE='linear_probe' is much faster on CPU.")
    return torch.device("cpu")
