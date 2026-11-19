"""Create any timm model with a new classifier head, and load saved checkpoints."""

import timm
import torch


def create_model(arch, num_classes, pretrained=True, train_mode="finetune"):
    """arch: any timm model name (see timm.list_models(pretrained=True)).

    train_mode "finetune" trains every layer; "linear_probe" freezes the pretrained backbone and
    trains only the new classifier (far faster on CPU, usually a few points less accurate).
    """
    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    if train_mode == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
    elif train_mode != "finetune":
        raise ValueError(f"TRAIN_MODE must be 'finetune' or 'linear_probe', not {train_mode!r}")

    data_cfg = timm.data.resolve_model_data_config(model)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{arch}: {n_all / 1e6:.1f}M parameters, {n_train / 1e6:.2f}M trainable ({train_mode}); "
          f"pretrained input size {data_cfg['input_size']}")
    return model, data_cfg


def load_checkpoint(path, device="cpu"):
    """Rebuild the model saved by imgcls.train() and load its best weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
