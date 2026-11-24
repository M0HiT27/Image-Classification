"""Zero-shot CLIP: name our categories from text prompts, with no training and no checkpoint.

CLIP was trained to match images with their captions, so it can compare an image with sentences such as
"a photo of a pug." (one set per category) and pick the closest. The image tower is the same network the
CLIP linear-probe notebooks train a head on; here CLIP's text tower supplies the head instead.

The API uses this when IMGCLS_CHECKPOINT is empty and IMGCLS_MODEL is one of the CLIP backbones in
OPEN_CLIP_NAMES. The weights come from open_clip (image and text towers, about 0.6 GB for ViT-B-32;
scripts/screen_images.py uses the same DataComp.XL model), downloaded to the Hugging Face cache on first use.
"""
import torch

# timm CLIP backbone (the names used for linear probes) -> the same weights in open_clip, which has the text tower
OPEN_CLIP_NAMES = {
    "vit_base_patch32_clip_224.datacompxl": ("ViT-B-32", "datacomp_xl_s13b_b90k"),
    "vit_base_patch32_clip_224.laion2b": ("ViT-B-32", "laion2b_s34b_b79k"),
    "vit_base_patch32_clip_224.openai": ("ViT-B-32", "openai"),
    "vit_base_patch16_clip_224.datacompxl": ("ViT-B-16", "datacomp_xl_s13b_b90k"),
    "vit_base_patch16_clip_224.laion2b": ("ViT-B-16", "laion2b_s34b_b88k"),
    "vit_base_patch16_clip_224.openai": ("ViT-B-16", "openai"),
}
# averaged per category; "close-up" and "blurry" help with detector crops and video frames
TEMPLATES = ("a photo of a {}.", "a close-up photo of a {}.", "a blurry photo of a {}.")


class ZeroShotCLIP(torch.nn.Module):
    """Images -> one logit per name, like a classifier with a trained head. `transform` is CLIP's preprocessing
    (PIL image -> tensor); use it instead of imgcls.data.build_transforms."""

    def __init__(self, arch, names, templates=TEMPLATES):
        super().__init__()
        if arch not in OPEN_CLIP_NAMES:
            raise ValueError(f"no zero-shot weights known for {arch}; choose one of {', '.join(OPEN_CLIP_NAMES)}")
        try:
            import open_clip
        except ImportError:
            raise ImportError("zero-shot CLIP needs open_clip: pip install open_clip_torch") from None

        clip_arch, tag = OPEN_CLIP_NAMES[arch]
        model, _, self.transform = open_clip.create_model_and_transforms(clip_arch, pretrained=tag)
        model.eval()
        tokenizer = open_clip.get_tokenizer(clip_arch)
        with torch.no_grad():
            per_name = []
            for name in names:
                emb = model.encode_text(tokenizer([t.format(name.lower()) for t in templates]))
                emb = emb / emb.norm(dim=-1, keepdim=True)
                per_name.append(emb.mean(0))
            text = torch.stack(per_name)
        self.register_buffer("text", text / text.norm(dim=-1, keepdim=True))
        self.scale = float(model.logit_scale.detach().exp())  # 100 for these models: turns cosine similarity into logits
        self.visual = model.visual  # keep only the image tower; the text tower isn't needed after this

    def forward(self, x):
        features = self.visual(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return self.scale * features @ self.text.T
