"""
_vfm.py — shared loaders for the candidate extractors (not an extractor: no build()).

DINO v1 ViT (Caron et al., ICCV 2021): the official model code, vision_transformer.py of
github.com/facebookresearch/dino @ 7c446df (Apache-2.0), cloned to
/home/thanhnc/gauss_ws/third_party/dino/dino and imported under private module names (its
`import utils` would otherwise pick up any module called utils); weights are the official
dl.fbaipublicfiles.com/dino/dino_vitbase{8,16}_pretrain/*.pth files that torch.hub's
dino_vitb8 / dino_vitb16 load (strict). The sha256 of every weight file is pinned in
configs/extractor_prereg.json (weights) and checked by the drivers before any GPU step.

OpenCLIP ViT-H/14 laion2b_s32b_b79k (Ilharco et al. / LAION, MIT): open_clip_model.safetensors
of huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K @ 1c2b849 (the file open_clip's
pretrained tag 'laion2b_s32b_b79k' resolves to), loaded by path.
"""
from __future__ import annotations
import hashlib
import importlib.util
import os
import sys

import torch

DINO_REPO = "/home/thanhnc/gauss_ws/third_party/dino/dino"
WEIGHTS = "/home/thanhnc/gauss_ws/src/uags/data/weights"
DINO_FILES = {8: os.path.join(WEIGHTS, "dino", "dino_vitbase8_pretrain.pth"),
              16: os.path.join(WEIGHTS, "dino", "dino_vitbase16_pretrain.pth")}
CLIP_H = os.path.join(WEIGHTS, "clip", "ViT-H-14-laion2b_s32b_b79k", "open_clip_model.safetensors")
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dino_module():
    """DINO's vision_transformer module (its `utils` bound to DINO's utils.py)."""
    name = "uags_dino_vision_transformer"
    if name not in sys.modules:
        utils = _load_source("uags_dino_utils", os.path.join(DINO_REPO, "utils.py"))
        saved = sys.modules.get("utils")
        sys.modules["utils"] = utils
        try:
            sys.modules[name] = _load_source(name, os.path.join(DINO_REPO, "vision_transformer.py"))
        finally:
            if saved is None:
                sys.modules.pop("utils", None)
            else:
                sys.modules["utils"] = saved
    return sys.modules[name]


def load_dino_vitb(patch, device="cpu", dtype=torch.float32):
    """torch.hub 'facebookresearch/dino:main' dino_vitb<patch>: vit_base(patch_size, num_classes
    0) with the official backbone weights, strict."""
    path = DINO_FILES[int(patch)]
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    vt = dino_module()
    model = vt.vit_base(patch_size=int(patch), num_classes=0)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(device, dtype)


@torch.no_grad()
def dino_patch_features(model, x):
    """x: (B, 3, h, w) ImageNet-normalised, h, w multiples of the patch -> (B, C, h/p, w/p):
    the last block's normed patch tokens (get_intermediate_layers(x)[0][:, 1:]), as
    ProxyCLIP's and Trident's 'dino' branch."""
    p = model.patch_embed.patch_size
    feat = model.get_intermediate_layers(x)[0]
    B, h, w = x.shape[0], x.shape[-2] // p, x.shape[-1] // p
    return feat[:, 1:, :].reshape(B, h, w, -1).permute(0, 3, 1, 2)


def load_clip_h(device="cpu", dtype=torch.float32):
    """(model, tokenizer) of OpenCLIP ViT-H-14 laion2b_s32b_b79k from the local file."""
    import open_clip
    if not os.path.isfile(CLIP_H):
        raise FileNotFoundError(CLIP_H)
    model = open_clip.create_model("ViT-H-14", pretrained=CLIP_H)
    return model.eval().to(device, dtype), open_clip.get_tokenizer("ViT-H-14")


@torch.no_grad()
def clip_text_embedding(model, tokenizer, device, prompts, templates):
    """ProxyCLIP / Trident text queries: per prompt, the unit embeddings of every template,
    averaged, normalised -> (K, d) float32."""
    import torch.nn.functional as Fn
    if isinstance(prompts, str):
        prompts = [prompts]
    out = []
    for p in prompts:
        toks = tokenizer([t.format(p) for t in templates]).to(device)
        e = Fn.normalize(model.encode_text(toks).float(), dim=-1).mean(0)
        out.append(Fn.normalize(e, dim=0))
    return torch.stack(out)
