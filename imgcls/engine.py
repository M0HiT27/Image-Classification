"""Training loop. Every run gets its own folder: output/<MODEL_NAME>/<timestamp>/.

Files written to the run folder:
  config.json          settings, device, library versions, class counts
  history.csv          loss / accuracy per epoch (rewritten after every epoch)
  best.pth             weights from the epoch with the highest validation accuracy
  training_curves.png  loss and accuracy plots
  summary.json         best epoch, best val accuracy, run time (test metrics are added by evaluate_run)
"""

import csv
import json
import math
import platform
import random
import time
from datetime import datetime

import numpy as np
import timm
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_from_globals(namespace):
    """The notebook's UPPER_CASE settings that can be saved as JSON."""
    simple = (str, int, float, bool, type(None))
    return {k: v for k, v in namespace.items()
            if k.isupper() and not k.startswith("_") and isinstance(v, simple)}


def new_run_dir(model_name):
    run_dir = config.OUTPUT_DIR / model_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


_READY_LOADERS = set()  # loaders whose worker processes have already started


def _single_process(loader):
    """Same loader without worker processes (used when the workers hang or crash)."""
    return DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=loader.sampler,
                      drop_last=loader.drop_last, num_workers=0, pin_memory=loader.pin_memory)


def _run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None,
               linear_probe=False, desc=""):
    training = optimizer is not None
    model.train(training)
    if training and linear_probe:  # frozen backbone: keep its batch-norm statistics fixed
        model.eval()
        model.get_classifier().train()
    use_amp = device.type == "cuda"
    total_loss, correct, seen = 0.0, 0, 0
    starting = loader.num_workers > 0 and id(loader) not in _READY_LOADERS
    if starting:
        print(f"{desc}: starting {loader.num_workers} image-loading worker processes "
              "(can take a minute on Windows)...", flush=True)
    t_start = time.time()
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=training):
            if starting:
                print(f"{desc}: workers ready after {time.time() - t_start:.0f}s", flush=True)
                _READY_LOADERS.add(id(loader))
                starting = False
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def _plot_history(history, path):
    fig = Figure(figsize=(10, 4))
    ax1, ax2 = fig.subplots(1, 2)
    ep = [h["epoch"] for h in history]
    ax1.plot(ep, [h["train_loss"] for h in history], label="train")
    ax1.plot(ep, [h["val_loss"] for h in history], label="val")
    ax1.set_title("loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(ep, [h["train_acc"] for h in history], label="train")
    ax2.plot(ep, [h["val_acc"] for h in history], label="val")
    ax2.set_title("accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def train(cfg, model, data, device):
    """Train with AdamW + cosine schedule; keep the checkpoint of the best validation epoch.

    Stopping the cell (KeyboardInterrupt) ends training early but keeps best.pth and the history.
    Returns the run folder.
    """
    run_dir = new_run_dir(cfg["MODEL_NAME"])
    print(f"Run folder: {run_dir}")
    model.to(device)
    linear_probe = cfg.get("TRAIN_MODE") == "linear_probe"
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["LR"], weight_decay=cfg.get("WEIGHT_DECAY", 0.05))
    epochs = cfg["EPOCHS"]
    steps = len(data.train_loader)
    warmup = steps if epochs > 2 else 0
    total = max(1, epochs * steps)

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    def run_phase(attr, desc, **train_args):
        """Run one pass over data.<attr>; if its worker processes hang or crash, fall back to
        loading images in this process and redo the pass instead of waiting forever."""
        loader = getattr(data, attr)
        try:
            return _run_epoch(model, loader, device, criterion, linear_probe=linear_probe,
                              desc=desc, **train_args)
        except RuntimeError as e:
            if loader.num_workers == 0 or "DataLoader" not in str(e):
                raise
            print(f"WARNING: image-loading workers failed ({e}).\n"
                  f"Switching this loader to NUM_WORKERS = 0 and redoing '{desc}'. "
                  "Set NUM_WORKERS = 0 (or 2) in the settings cell to avoid the wait next time.",
                  flush=True)
            setattr(data, attr, _single_process(loader))
            return _run_epoch(model, getattr(data, attr), device, criterion, linear_probe=linear_probe,
                              desc=desc, **train_args)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get("LABEL_SMOOTHING", 0.0))

    info = {
        "config": cfg,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__},
        "classes": data.classes,
        "counts": data.counts(),
        "normalisation": {"mean": data.mean, "std": data.std, "img_size": data.img_size},
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    history, best_acc, best_epoch, stale = [], -1.0, 0, 0
    patience = cfg.get("PATIENCE")
    t_start = time.time()
    try:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_phase("train_loader", f"epoch {epoch}/{epochs} train",
                                        optimizer=optimizer, scheduler=scheduler, scaler=scaler)
            va_loss, va_acc = run_phase("val_loader", f"epoch {epoch}/{epochs} val")
            row = {"epoch": epoch, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                   "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
                   "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
            history.append(row)
            with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerows(history)

            improved = va_acc > best_acc
            if improved:
                best_acc, best_epoch, stale = va_acc, epoch, 0
                torch.save({
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "arch": cfg["ARCH"],
                    "classes": data.classes,
                    "img_size": data.img_size,
                    "mean": list(data.mean),
                    "std": list(data.std),
                    "epoch": epoch,
                    "val_acc": va_acc,
                    "config": cfg,
                }, run_dir / "best.pth")
            else:
                stale += 1
            print(f"epoch {epoch:3d}/{epochs}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | {row['seconds']:.0f}s"
                  + ("  * best, saved best.pth" if improved else ""))
            if patience and stale >= patience:
                print(f"No val improvement for {patience} epochs; stopping early.")
                break
    except KeyboardInterrupt:
        print("Training interrupted; keeping best.pth and the history so far.")

    if history:
        _plot_history(history, run_dir / "training_curves.png")
    summary = {"model_name": cfg["MODEL_NAME"], "arch": cfg["ARCH"], "best_epoch": best_epoch,
               "best_val_acc": round(best_acc, 4), "epochs_run": len(history),
               "minutes": round((time.time() - t_start) / 60, 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best val accuracy {best_acc:.4f} at epoch {best_epoch}; checkpoint: {run_dir / 'best.pth'}")
    return run_dir
