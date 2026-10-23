"""
radseg.py — RADSeg dense features (Alama et al., "RADSeg: Unleashing Parameter and Compute
Efficient Zero-Shot Open-Vocabulary Segmentation Using Agglomerative Models", CVPR 2026
Findings, arXiv 2511.19704; code github.com/RADSeg-OVSS/RADSeg @ 3fe8789, MIT licence),
training-free, WITHOUT the SAM refinement of RADSeg+ (that step prompts SAM with the coarse
class predictions, so it needs the class list and gives no class-agnostic features).

Model: C-RADIOv3-B (NVIDIA, 106M parameters; huggingface.co/nvidia/C-RADIOv3-B @ 44653a0,
file c-radio_v3-b_half.pth.tar, sha256 530b0e82..., NVIDIA Open Model License), built by the
official RADIO code (github.com/NVlabs/RADIO @ 4ffaf15, NVIDIA Source Code License for RADIO:
research / evaluation use only) imported from /home/thanhnc/gauss_ws/third_party/radio/RADIO.
Its "siglip2" adaptor head maps backbone tokens into the space of the SigLIP2 so400m text
tower (google/siglip2-so400m-patch16-naflex @ cc24074, Apache-2.0; the RADIO teacher named in
the checkpoint: siglip2-so400m), so every pixel feature is cosine-comparable to
text_embedding for any vocabulary.

Per frame (RADSeg encode_image_to_feat_map with predict=False, then
align_spatial_features_with_language; defaults = the paper's mid-resolution setting for
ADE20K / Pascal Context, Table A.1: short side 576, crop 336, stride 224, SCRA and SCGA
scaling 10, fp32 (amp False)):
  1. resize the frame so its short side is 576 (mmseg Resize keep_ratio, cv2 INTER_LINEAR on
     uint8, sizes int(x * s + 0.5)); RGB / 255 (RADIO's input conditioner normalises with the
     OpenAI CLIP mean / std, as RADSeg's mmseg preprocessor does);
  2. _preprocess_image: long side rounded UP to a multiple of the stride 224, the short side
     scaled with it and rounded up to a multiple of 224, both >= 336 (torch bilinear,
     align_corners False): 640x480 and 648x484 frames both become 896x672;
  3. 336x336 windows at stride 224 (mmseg slide order, last window clamped to the border), all
     windows in one batch through RADIO whose LAST block's attention is replaced by SCRA:
     standard attention -> proj -> per-head cosine self-similarity of that output * 10,
     negatives -> -inf, softmax -> applied to the block's values -> proj (no new parameters);
     the RADIO backbone features (summary tokens stripped, feature normaliser) of each window
     (21x21 tokens) are averaged where windows overlap on the 16-px token grid (42x56);
  4. SCGA: tokens re-aggregated over the whole frame with softmax(10 * (S - mean S)), S the
     cosine self-similarity of the tokens, negatives -> -inf;
  5. the siglip2 adaptor's summary MLP (head_mlp) on every token, L2-normalised ->
     (42, 56, 1152) = extract_tokens (extract() renormalises after bilinear upsampling).
RADSeg upsamples per-class softmax(100 cos) probabilities instead of unit features; the argmax
is the same up to bilinear border effects (cos is linear in the unit token).

Text (RADSeg encode_labels): each prompt filled into each template, canonicalised (lower case,
punctuation removed; RADIO's SigLIP2WrappedTokenizer), tokenised to 64 tokens (padding
max_length), the SigLIP2 text tower's pooler_output L2-normalised, averaged over templates,
then normalised (cosine-equivalent to RADSeg's unnormalised mean). The text tower runs in fp32
on `text_device` (default: the model device, as RADSeg; 80 templates take ~11 s per class on the
CPU); its output is moved to `device`.
"""
from __future__ import annotations
import argparse
import importlib.util
import math
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from usplat.clip_dense import upsample_tokens

REPO = "/home/thanhnc/gauss_ws/third_party/radio/RADIO"
WEIGHTS = "/home/thanhnc/gauss_ws/data/weights/radseg"
RADIO_CKPT = os.path.join(WEIGHTS, "c-radio_v3-b", "c-radio_v3-b_half.pth.tar")
RADIO_SHA256 = "530b0e827b1d409ca3b175466e11062885baedd823ec5dae3a88348e76d4c398"
TEXT_DIR = os.path.join(WEIGHTS, "siglip2-so400m-patch16-naflex")
VERSION = "c-radio_v3-b"
LANG = "siglip2"
TEXT_MODELS = {"siglip2-so400m": "siglip2-so400m-patch16-naflex"}   # RADIO's _VERSION_MAP


def _sha256(path, chunk=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _radio_hub():
    """NVlabs/RADIO hubconf.py, imported under its own name (other repos ship a hubconf)."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    name = "uags_radio_hubconf"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "hubconf.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def load_radio(path=RADIO_CKPT, lang=LANG, check_sha=True):
    """The RADIO model of torch.hub.load('NVlabs/RADIO', 'radio_model', version=VERSION,
    adaptor_names=[lang]) from the local checkpoint: the official hub function builds it; the
    file is read with weights_only=True (argparse.Namespace allowed) and handed to it in place
    of the download, and the language adaptor is built as RADIO's GenericAdaptor (its head and
    feature MLPs; the text tower is loaded separately, see load_text). Returns (model,
    adaptor, teacher config)."""
    if check_sha:
        got = _sha256(path)
        if got != RADIO_SHA256:
            raise ValueError(f"{path}: sha256 {got[:12]} != {RADIO_SHA256[:12]} (C-RADIOv3-B)")
    with torch.serialization.safe_globals([argparse.Namespace]):
        chk = torch.load(path, map_location="cpu", weights_only=True)
    hub = _radio_hub()
    from radio.adaptor_registry import adaptor_registry
    from radio.adaptor_generic import GenericAdaptor
    teacher = next((t for t in chk["args"].teachers if t["name"] == lang), None)
    if teacher is None:
        raise ValueError(f"{path}: no teacher named {lang!r}")
    saved_url = hub.load_state_dict_from_url
    saved_factory = adaptor_registry._registry.get(teacher["type"])
    hub.load_state_dict_from_url = lambda url, **kw: chk
    adaptor_registry._registry[teacher["type"]] = \
        lambda main, conf, state: GenericAdaptor(main, conf, state)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = hub.radio_model(version=VERSION, progress=False, adaptor_names=[lang])
    finally:
        hub.load_state_dict_from_url = saved_url
        if saved_factory is None:
            adaptor_registry._registry.pop(teacher["type"], None)
        else:
            adaptor_registry._registry[teacher["type"]] = saved_factory
    adaptor = model.adaptors[lang]
    model.adaptors = None                   # RADSeg: the adaptor runs only when asked
    return model.eval(), adaptor.eval(), teacher


def load_text(text_dir=TEXT_DIR):
    """(tokenizer, text tower) of RADIO's SigLIP2Adaptor: AutoModel(...).text_model and RADIO's
    SigLIP2WrappedTokenizer around AutoProcessor, from the local snapshot."""
    from transformers import AutoModel, AutoProcessor
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from radio.siglip2_adaptor import SigLIP2WrappedTokenizer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModel.from_pretrained(text_dir, local_files_only=True)
        proc = AutoProcessor.from_pretrained(text_dir, local_files_only=True)
    text = model.text_model.float().eval()
    del model
    return SigLIP2WrappedTokenizer(proc), text


class SelfCorrelatingRecursiveAttn(nn.Module):
    """RADSeg's SCRA (radseg/radseg.py SelfCorrelatingRecursiveAttn.custom_attn), the same
    operations in batch-first layout; q_norm / k_norm are identities as in RADSeg (qk_norm
    False), the original block's qkv and proj are reused."""

    def __init__(self, orig_attn, scaling=10.0):
        super().__init__()
        self.num_heads = orig_attn.num_heads
        self.qkv, self.proj = orig_attn.qkv, orig_attn.proj
        self.proj_drop = orig_attn.proj_drop
        self.scaling = float(scaling)

    def forward(self, x, *args, **kwargs):
        B, N, C = x.shape
        h, hd = self.num_heads, C // self.num_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        heads = lambda t: t.reshape(B, N, h, hd).transpose(1, 2)        # (B, h, N, hd)
        q, k, v = heads(q), heads(k), heads(v)
        a = torch.softmax((q @ k.transpose(-1, -2)) * hd ** -0.5, dim=-1)
        out = self.proj_drop(self.proj((a @ v).transpose(1, 2).reshape(B, N, C)))
        s = Fn.normalize(heads(out), dim=-1)
        sim = (s @ s.transpose(-1, -2)) * self.scaling
        sim = sim.masked_fill(sim < 0, float("-inf"))
        out = torch.softmax(sim, dim=-1) @ v
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, C)))


def scga(feat, scaling=10.0):
    """RADSeg _self_correlating_global_aggregation: feat (B, C, h, w) -> (B, C, h, w)."""
    B, C, h, w = feat.shape
    t = feat.flatten(2).transpose(1, 2)                                  # (B, hw, C)
    s = Fn.normalize(t, dim=-1)
    sim = s @ s.transpose(1, 2)
    sim = (sim - sim.mean()) * scaling
    sim = sim.masked_fill(sim < 0, float("-inf"))
    return (torch.softmax(sim, dim=-1) @ t).transpose(1, 2).reshape(B, C, h, w)


def rescale_size(h, w, short=576, long=4096):
    """mmcv rescale_size for Resize(scale=(long, short), keep_ratio=True)."""
    s = min(long / max(h, w), short / min(h, w))
    return int(h * s + 0.5), int(w * s + 0.5)


def preprocess_size(h, w, stride=224, crop=336):
    """RADSeg _preprocess_image target size (stride = the slide stride)."""
    if h % stride == 0 and w % stride == 0:
        return h, w
    longer = max(h, w)
    dst = (longer // stride + 1) * stride if longer % stride else longer
    nh, nw = int(h * dst / longer), int(w * dst / longer)
    if nh % stride:
        nh = (nh // stride + 1) * stride
    if nw % stride:
        nw = (nw // stride + 1) * stride
    return max(nh, crop), max(nw, crop)


def windows(h, w, crop=336, stride=224):
    """RADSeg _get_windowed_imgs / mmseg slide: [(y1, x1, y2, x2)] in pixels, row-major."""
    hg = max(h - crop + stride - 1, 0) // stride + 1
    wg = max(w - crop + stride - 1, 0) // stride + 1
    out = []
    for i in range(hg):
        for j in range(wg):
            y2, x2 = min(i * stride + crop, h), min(j * stride + crop, w)
            out.append((max(y2 - crop, 0), max(x2 - crop, 0), y2, x2))
    return out


class RADSegDense:
    """Dense RADSeg features behind the usplat extractor contract."""

    def __init__(self, device="cuda", short_side=576, crop=336, stride=224, scra_scaling=10.0,
                 scga_scaling=10.0, text_device=None, radio_ckpt=RADIO_CKPT, text_dir=TEXT_DIR):
        self.device = torch.device(device)
        self.short_side, self.crop, self.stride = int(short_side), int(crop), int(stride)
        self.crop_size, self.half = self.crop, False       # recorded by run_mapping (ext_opts)
        self.scra_scaling, self.scga_scaling = float(scra_scaling), float(scga_scaling)
        model, adaptor, teacher = load_radio(radio_ckpt)
        if TEXT_MODELS.get(teacher.get("model")) != os.path.basename(os.path.normpath(text_dir)):
            raise ValueError(f"RADIO's {LANG} teacher is {teacher.get('model')!r}; text weights "
                             f"{text_dir} do not match")
        blk = model.model.blocks[-1]
        blk.attn = SelfCorrelatingRecursiveAttn(blk.attn, self.scra_scaling)
        self.model = model.to(self.device)
        self.adaptor = adaptor.to(self.device)
        self.patch = int(model.patch_size)                                  # 16
        if self.crop % self.patch or self.stride % self.patch:
            raise ValueError("crop and stride must be multiples of the patch size")
        self.text_device = torch.device(text_device) if text_device else self.device
        self.tokenizer, self.text = load_text(text_dir)
        self.text = self.text.to(self.text_device)
        with torch.no_grad():
            probe = self.adaptor.head_mlp(torch.zeros(1, model.embed_dim, device=self.device))
        self.d = int(probe.shape[-1])                                       # 1152

    # ---------------------------------------------------------------- image side
    def _input(self, image):
        """(H, W, 3) uint8 -> (1, 3, h, w) float in [0, 1] at the RADSeg input size."""
        import cv2
        import numpy as np
        img = image.cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        img = np.ascontiguousarray(img.astype(np.uint8))
        H, W = img.shape[:2]
        h, w = rescale_size(H, W, self.short_side)
        if (h, w) != (H, W):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        ph, pw = preprocess_size(h, w, self.stride, self.crop)
        if (ph, pw) != (h, w):
            x = Fn.interpolate(x, size=(ph, pw), mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def backbone_map(self, image):
        """RADIO + SCRA features averaged over the windows: (1, C, h/16, w/16)."""
        x = self._input(image)
        _, _, h, w = x.shape
        win = windows(h, w, self.crop, self.stride)
        p = self.patch
        for y1, x1, y2, x2 in win:
            assert y1 % p == 0 and x1 % p == 0 and y2 % p == 0 and x2 % p == 0
        batch = torch.cat([x[..., y1:y2, x1:x2] for y1, x1, y2, x2 in win], 0)
        feats = self.model(batch).features                                 # (n, L, C)
        n, L, C = feats.shape
        gh, gw = self.crop // p, self.crop // p
        feats = feats.transpose(1, 2).reshape(n, C, gh, gw)
        acc = feats.new_zeros(1, C, math.ceil(h / p), math.ceil(w / p))
        cnt = feats.new_zeros(1, 1, acc.shape[2], acc.shape[3])
        for k, (y1, x1, y2, x2) in enumerate(win):
            acc[:, :, y1 // p:y2 // p, x1 // p:x2 // p] += feats[k]
            cnt[:, :, y1 // p:y2 // p, x1 // p:x2 // p] += 1
        return acc / cnt

    @torch.no_grad()
    def extract_tokens(self, image):
        """(H, W, 3) uint8 RGB -> (h/16, w/16, d) unit language-aligned tokens on the 16-px
        grid of the RADSeg input (640x480 -> 896x672 -> (42, 56, 1152))."""
        f = scga(self.backbone_map(image), self.scga_scaling)
        B, C, h, w = f.shape
        t = self.adaptor.head_mlp(f.flatten(2).transpose(1, 2))           # (1, hw, d)
        return Fn.normalize(t[0].float(), dim=-1).reshape(h, w, -1).contiguous()

    @torch.no_grad()
    def extract(self, image, out_hw=None):
        """(H, W, 3) uint8 RGB -> (h, w, d) unit per-pixel features (default: image size)."""
        tokens = self.extract_tokens(image)
        return upsample_tokens(tokens, out_hw or (int(image.shape[0]), int(image.shape[1])))

    # ---------------------------------------------------------------- text side
    @torch.no_grad()
    def text_embedding(self, prompts, templates=("a photo of a {}.",)):
        """prompts: str or list[str] -> (K, d) float32 unit text embeddings on `device`."""
        if isinstance(prompts, str):
            prompts = [prompts]
        out = []
        for p in prompts:
            tok = self.tokenizer([t.format(p) for t in templates]).to(self.text_device)
            e = self.text(**tok, return_dict=True).pooler_output.float()
            out.append(Fn.normalize(Fn.normalize(e, dim=-1).mean(0), dim=0))
        return torch.stack(out).to(self.device)


def build(device="cuda", **kw):
    """No environment overrides: every caller (run_mapping, distill_features, eval_seg) gets
    the same defaults."""
    return RADSegDense(device=device, **kw)
