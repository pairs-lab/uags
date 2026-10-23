"""
trident.py — Trident dense features (Shi, Dong, Xu, "Harnessing Vision Foundation Models for
High-Performance, Training-Free Open Vocabulary Segmentation", ICCV 2025, arXiv 2411.09219;
code github.com/YuHengsss/Trident @ 7feea10, Apache-2.0), training-free, at the paper's
OpenCLIP ViT-H/14 + SAM ViT-H setting WITHOUT the SAM refinement (that step prompts SAM with
the coarse class masks, so it needs the class list; Trident Table 4 reports this unrefined
"affinity" variant: 47.1 mean mIoU over eight benchmarks, ProxyCLIP ViT-H/14 44.4).

Per frame (Trident.get_trident_seg, non-Cityscapes path):
  SAM branch  the frame at its own resolution, both sides rounded UP to multiples of the CLIP
              patch 14 (cv2 INTER_LINEAR), through SamPredictor.set_image (resize long side to
              1024, SAM normalisation, pad); the LAST image-encoder block (global attention,
              64x64 tokens) gives its softmax attention A_sam (16 heads) and values v_sam,
              cropped to the valid token grid (input_size // 16: 640x480 -> 48x64).
  CLIP branch the frame resized to short side 576 (Trident's ADE20K / Context config, mmseg
              Resize keep_ratio, cv2 INTER_LINEAR), CLIP-normalised, preprocess_image (long
              side up to a multiple of 14, both sides multiples of 14, >= 336; torch bilinear:
              640x480 -> 768x576 -> 770x588), 336x336 windows at stride 224 (mmseg order,
              clamped at the border). Per window, DINO ViT-B/16 (ImageNet-normalised) gives the
              proxy features (21x21); CLIP ViT-H/14 runs to its last block (24x24 tokens, CLS
              dropped), whose attention is replaced by proxy attention (ProxyCLIP's custom_attn:
              softmax over (S - beta mean S) * gamma, S the DINO cosine similarity, negatives
              -inf; the mean is over ALL windows of the frame, which Trident batches; CLIP
              values bilinearly resized to 21x21) -> out_proj. Splice: each window's output
              resized back to 24x24 and averaged into the frame's CLIP token grid (42x55),
              then bilinearly resized to the SAM grid.
  Aggregate   guidance_attn / get_sam_sim: per SAM head, cos of v_sam; a key is kept when its
              cos exceeds t = min(cos_fac * mean over heads and keys of the cos, 0.9) (per
              query); kept A_sam entries averaged over the heads, rows renormalised -> W; the
              spliced CLIP tokens are aggregated with W (one global attention over the frame),
              then ln_post and the visual projection: (48, 64, 1024) tokens in CLIP space.
extract_tokens returns these tokens L2-normalised (Trident normalises them before its logits;
logits are linear in them, so bilinear upsampling of unit tokens = of Trident's logits up to
normalisation); extract() renormalises after upsampling.

Settings (Trident defaults / config): beta 1.2, gamma 3.0, cos_fac 1.2, refine_neg_cos True,
slide 336 / 224, CLIP ViT-H-14 laion2b_s32b_b79k, DINO ViT-B/16 (torch.hub dino_vitb16, hard-coded
in trident.py), SAM vit_h (sam_vit_h_4b8939.pth); fp16 weights on CUDA (Trident: CLIP
precision='fp16', vfm.half(), sam.half()), fp32 on the CPU. Deviations, numerically small: the
proxy similarity, its softmax and the SAM-weighted aggregation are computed in fp32 (Trident:
fp16), and the diagonal of the masked proxy similarity is kept finite (a NaN guard, as in
proxyclip.py). Text: the CLIP text tower, per-template unit embeddings averaged and normalised
(Trident's query features).

Code: SAM from github.com/facebookresearch/segment-anything @ dca509f (Apache-2.0) at
/home/thanhnc/gauss_ws/third_party/sam/segment-anything (sam_model_registry, SamPredictor,
add_decomposed_rel_pos); the last block's attention is wrapped to keep its attention and
values, as Trident's modified segment_anything does (its only change to SAM). DINO / CLIP
loaders: usplat/extractors/_vfm.py; CLIP last-block values: proxyclip.ProxyCLIPDense._clip_values.
"""
from __future__ import annotations
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from usplat.clip_dense import upsample_tokens
from usplat.extractors._vfm import (load_clip_h, load_dino_vitb, dino_patch_features,
                                    IMNET_MEAN, IMNET_STD, CLIP_MEAN, CLIP_STD)
from usplat.extractors.proxyclip import ProxyCLIPDense
from usplat.extractors.radseg import rescale_size, preprocess_size, windows

SAM_REPO = "/home/thanhnc/gauss_ws/third_party/sam/segment-anything"
SAM_CKPT = "/home/thanhnc/gauss_ws/data/weights/sam/sam_vit_h_4b8939.pth"
SAM_HEADS = {"vit_b": 12, "vit_l": 16, "vit_h": 16}


def _sam_modules():
    if SAM_REPO not in sys.path:
        sys.path.insert(0, SAM_REPO)
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
    return sam_model_registry, SamPredictor, add_decomposed_rel_pos


class CaptureAttention(nn.Module):
    """segment_anything Attention.forward, unchanged, that also keeps the softmax attention
    (B * heads, HW, HW) and the values (B, heads, H, W, hd) of its last call (Trident's
    return_attn branch: last_attn, last_v)."""

    def __init__(self, attn, rel_pos_fn):
        super().__init__()
        self.inner, self._rel = attn, rel_pos_fn
        self.last_attn = self.last_v = None

    def forward(self, x):
        a = self.inner
        B, H, W, _ = x.shape
        qkv = a.qkv(x).reshape(B, H * W, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * a.num_heads, H * W, -1).unbind(0)
        attn = (q * a.scale) @ k.transpose(-2, -1)
        if a.use_rel_pos:
            attn = self._rel(attn, q, a.rel_pos_h, a.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, a.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        self.last_attn, self.last_v = attn, v.view(B, a.num_heads, H, W, -1)
        return a.proj(x)


def sam_similarity(sam_v, sam_attn, cos_fac=1.2, refine_neg_cos=True):
    """Trident get_sam_sim (one copy; the caller broadcasts it over the CLIP heads):
    sam_v (1, h, vh, vw, c), sam_attn (1, h, n, n) -> (1, n, n) row-stochastic weights."""
    v = Fn.normalize(sam_v.flatten(2, 3).float(), dim=-1)                 # (1, h, n, c)
    h = v.shape[1]
    cos = lambda i: v[:, i] @ v[:, i].transpose(-1, -2)                  # (1, n, n), head i
    # per head, so that only one (n, n) cosine map is alive at a time (same values)
    thr = sum(cos(i).mean(dim=-1, keepdim=True) for i in range(h)) / h * cos_fac  # (1, n, 1)
    if not refine_neg_cos:
        thr = torch.where(thr < 0.0, torch.full_like(thr, 0.6), thr)
    thr = torch.where(thr > 0.9, torch.full_like(thr, 0.9), thr)
    w = sum(torch.where(cos(i) > thr, sam_attn[:, i].float(), 0.0) for i in range(h)) / h
    return w / w.sum(dim=-1, keepdim=True)


class TridentDense(ProxyCLIPDense):
    def __init__(self, device="cuda", short_side=576, crop=336, stride=224, beta=1.2, gamma=3.0,
                 cos_fac=1.2, refine_neg_cos=True, sam_type="vit_h", sam_ckpt=SAM_CKPT,
                 dtype=None):
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.short_side, self.crop, self.stride = int(short_side), int(crop), int(stride)
        self.crop_size, self.half = self.crop, dtype == torch.float16   # run_mapping ext_opts
        self.beta, self.gamma = float(beta), float(gamma)
        self.cos_fac, self.refine_neg_cos = float(cos_fac), bool(refine_neg_cos)
        self.model, self.tokenizer = load_clip_h(self.device, dtype)
        self.visual = self.model.visual
        self.cpatch = self.visual.conv1.kernel_size[0]                      # 14
        self.d = int(self.visual.proj.shape[1])                             # 1024
        self.vfm = load_dino_vitb(16, self.device, dtype)
        registry, Predictor, rel = _sam_modules()
        if not os.path.isfile(sam_ckpt):
            raise FileNotFoundError(sam_ckpt)
        sam = registry[sam_type](checkpoint=sam_ckpt).eval().to(self.device, dtype)
        last = sam.image_encoder.blocks[-1]
        if last.window_size != 0:
            raise ValueError("the last SAM block must use global attention")
        last.attn = CaptureAttention(last.attn, rel)
        self.sam_heads = SAM_HEADS[sam_type]
        self.sam_capture = last.attn
        self.predictor = Predictor(sam)
        f = dict(device=self.device, dtype=torch.float32)
        self._cm = torch.tensor(CLIP_MEAN, **f).view(1, 3, 1, 1)
        self._cs = torch.tensor(CLIP_STD, **f).view(1, 3, 1, 1)
        self._vm = torch.tensor(IMNET_MEAN, **f).view(1, 3, 1, 1)
        self._vs = torch.tensor(IMNET_STD, **f).view(1, 3, 1, 1)

    # ---------------------------------------------------------------- SAM branch
    @torch.no_grad()
    def sam_affinity(self, img):
        """img (H, W, 3) uint8 numpy RGB -> (W (1, n, n), (vh, vw)), n = vh * vw."""
        import cv2
        H, W = img.shape[:2]
        p = self.cpatch
        th, tw = -(-H // p) * p, -(-W // p) * p
        tmp = img if (th, tw) == (H, W) else cv2.resize(img, (tw, th))
        self.predictor.set_image(tmp)
        vh, vw = self.predictor.input_size[0] // 16, self.predictor.input_size[1] // 16
        A, V = self.sam_capture.last_attn, self.sam_capture.last_v
        g = int(round(math.sqrt(A.shape[-1])))                              # 64
        A = A.view(1, self.sam_heads, g, g, g, g)[:, :, :vh, :vw, :vh, :vw]
        A = A.flatten(2, 3).flatten(3, 4)                                   # (1, h, n, n)
        V = V[:, :, :vh, :vw, :]
        Wt = sam_similarity(V, A, self.cos_fac, self.refine_neg_cos)
        self.predictor.reset_image()
        self.sam_capture.last_attn = self.sam_capture.last_v = None
        return Wt, (vh, vw)

    # ---------------------------------------------------------------- CLIP branch
    def _clip_input(self, img):
        """uint8 RGB (H, W, 3) -> (1, 3, h, w) CLIP-normalised Trident input (multiples of 14)."""
        import cv2
        H, W = img.shape[:2]
        h, w = rescale_size(H, W, self.short_side, 2048)
        if (h, w) != (H, W):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        x = (x - self._cm) / self._cs
        ph, pw = preprocess_size(h, w, self.cpatch, self.crop)
        if (ph, pw) != (h, w):
            x = Fn.interpolate(x, size=(ph, pw), mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def clip_windows(self, x):
        """Trident's windowed CLIP pass with proxy attention and the splice: -> (1, D, gh, gw)
        last-block attention outputs (after out_proj, before ln_post) on the frame's CLIP
        token grid, and the attention layer."""
        _, _, h, w = x.shape
        win = windows(h, w, self.crop, self.stride)
        p = self.cpatch
        for y1, x1, y2, x2 in win:
            assert y1 % p == 0 and x1 % p == 0 and y2 % p == 0 and x2 % p == 0
        batch = torch.cat([x[..., y1:y2, x1:x2] for y1, x1, y2, x2 in win], 0)  # CLIP-normed
        raw = batch * self._cs + self._cm
        ex = dino_patch_features(self.vfm, ((raw - self._vm) / self._vs).to(self.dtype))
        B, _, I, J = ex.shape                                              # 21 x 21
        qk = Fn.normalize(ex.flatten(2).float(), dim=1)
        sim = qk.transpose(1, 2) @ qk                                      # (B, IJ, IJ)
        sim = (sim - sim.mean() * self.beta) * self.gamma                  # mean over ALL windows
        sim[sim < 0.0] = float("-inf")
        sim.diagonal(dim1=-2, dim2=-1).clamp_(min=0.0)                     # NaN guard only
        A = torch.softmax(sim, dim=-1)
        val, attn, nh, D = self._clip_values(batch)                        # (B, h, 24, 24, dh)
        gh, gw = val.shape[2:4]
        v = val.permute(0, 1, 4, 2, 3).reshape(B * nh, D // nh, gh, gw).float()
        v = Fn.interpolate(v, size=(I, J), mode="bilinear", align_corners=False)
        v = v.reshape(B, nh, D // nh, I * J).transpose(-1, -2)            # (B, h, IJ, dh)
        out = (A[:, None] @ v).transpose(1, 2).reshape(B, I * J, D)
        out = attn.out_proj(out.to(self.dtype)).float()                    # (B, IJ, D)
        vimg = out.transpose(1, 2).reshape(B, D, I, J)
        if J % gw != 0:                                                    # guidance_attn
            r = int(round(I / gh))
            vimg = Fn.interpolate(vimg, size=(gh * r, gw * r), mode="bilinear",
                                  align_corners=False)
        else:
            r = J // gw
        Hc, Wc = h // p, w // p
        dst = vimg.new_zeros(1, D, Hc * r, Wc * r)
        cnt = vimg.new_zeros(1, 1, Hc * r, Wc * r)
        for k, (y1, x1, y2, x2) in enumerate(win):
            sl = (..., slice(y1 // p * r, y2 // p * r), slice(x1 // p * r, x2 // p * r))
            dst[sl] += vimg[k]
            cnt[sl] += 1
        return dst / cnt, attn

    @torch.no_grad()
    def extract_tokens(self, image):
        """(H, W, 3) uint8 RGB -> (vh, vw, 1024) unit tokens on SAM's valid 16-px grid of the
        1024-px SAM input (640x480 -> (48, 64, 1024))."""
        import numpy as np
        img = image.cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        img = np.ascontiguousarray(img.astype(np.uint8))
        Wt, (vh, vw) = self.sam_affinity(img)
        dst, attn = self.clip_windows(self._clip_input(img))
        D = dst.shape[1]
        nh = attn.num_heads
        dst = Fn.interpolate(dst, size=(vh, vw), mode="bilinear", align_corners=False)
        dv = dst.reshape(1, nh, D // nh, vh * vw).transpose(-1, -2)       # (1, h, n, dh)
        out = (Wt[:, None] @ dv).permute(2, 0, 1, 3).reshape(vh * vw, 1, D)[:, 0]
        out = self.visual.ln_post(out.to(self.dtype)) @ self.visual.proj   # (n, d)
        return Fn.normalize(out.float(), dim=-1).reshape(vh, vw, -1)

    @torch.no_grad()
    def extract(self, image, out_hw=None):
        tokens = self.extract_tokens(image)
        return upsample_tokens(tokens, out_hw or (int(image.shape[0]), int(image.shape[1])))


def build(device="cuda", **kw):
    """No environment overrides (run_mapping, distill_features and eval_seg pass no kwargs)."""
    return TridentDense(device=device, **kw)
