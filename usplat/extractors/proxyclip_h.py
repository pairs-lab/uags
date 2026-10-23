"""
proxyclip_h.py — ProxyCLIP at its strongest published setting (Lan et al., ECCV 2024,
arXiv 2408.04883, Table 1: OpenCLIP ViT-H/14 + DINO ViT-B/8 proxy, 44.4 mean mIoU over eight
benchmarks vs 42.3 for ViT-B/16), training-free. usplat/extractors/proxyclip.py (unchanged)
implements the method with CLIP ViT-B/16 and a DINOv2 ViT-B/14 proxy; this file reuses its
code (proxy attention, CLIP last-block values, window averaging on the VFM grid, text) and
changes only the two backbones and the frame size:
  * CLIP: OpenCLIP ViT-H-14 laion2b_s32b_b79k (the repo's ('laion2b_s32b_b79k', 'ViT-H-14')),
    fp16 on CUDA as ProxyCLIP (precision='fp16'), fp32 on the CPU; d = 1024;
  * proxy: DINO ViT-B/8 (the repo's default vfm 'dino' = torch.hub dino_vitb8): the last
    block's normed patch tokens (get_intermediate_layers(x)[0][:, 1:]), ImageNet mean / std;
  * frame: short side 448 = ProxyCLIP's ADE20K setting (cfg_ade20k.py: Resize (2048, 448)),
    windows 336 / stride 112, beta 1.2, gamma 3.0 (the repo defaults). As in proxyclip.py, both
    sides are rounded to multiples of the VFM patch (8 px) so that the windows tile the VFM grid
    exactly: 640x480 -> 600x448 (mmseg: 597x448) -> (56, 75, 1024) tokens.
Code: github.com/mc-lan/ProxyCLIP @ a0e38d1 (NTU S-Lab License 1.0: non-commercial use only),
read at /home/thanhnc/gauss_ws/third_party/proxyclip/ProxyCLIP; the DINO and CLIP loaders are
in usplat/extractors/_vfm.py.
"""
from __future__ import annotations
import torch
import torch.nn.functional as Fn

from usplat.extractors._vfm import (load_clip_h, load_dino_vitb, dino_patch_features,
                                    IMNET_MEAN, IMNET_STD, CLIP_MEAN, CLIP_STD)
from usplat.extractors.proxyclip import ProxyCLIPDense


class ProxyCLIPHDense(ProxyCLIPDense):
    def __init__(self, device="cuda", vfm_patch=8, short_side=448, crop=336, stride=112,
                 beta=1.2, gamma=3.0, dtype=None):
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.pretrained, self.vfm_name = "laion2b_s32b_b79k", f"dino_vitb{int(vfm_patch)}"
        self.short_side, self.crop, self.stride = int(short_side), int(crop), int(stride)
        self.crop_size, self.half = self.crop, dtype == torch.float16   # run_mapping ext_opts
        self.beta, self.gamma = float(beta), float(gamma)
        self.model, self.tokenizer = load_clip_h(self.device, dtype)
        self.visual = self.model.visual
        self.cpatch = self.visual.conv1.kernel_size[0]                      # 14
        self.d = int(self.visual.proj.shape[1])                             # 1024
        self.vfm = load_dino_vitb(vfm_patch, self.device, dtype)
        self.vpatch = int(self.vfm.patch_embed.patch_size)                  # 8
        for s in (self.crop, self.stride):
            if s % self.vpatch:
                raise ValueError(f"crop/stride must be multiples of {self.vpatch}")
        f = dict(device=self.device, dtype=torch.float32)
        self._cm = torch.tensor(CLIP_MEAN, **f).view(1, 3, 1, 1)
        self._cs = torch.tensor(CLIP_STD, **f).view(1, 3, 1, 1)
        self._vm = torch.tensor(IMNET_MEAN, **f).view(1, 3, 1, 1)
        self._vs = torch.tensor(IMNET_STD, **f).view(1, 3, 1, 1)

    @torch.no_grad()
    def _window_tokens(self, crops):
        """proxyclip.ProxyCLIPDense._window_tokens with the DINO v1 proxy: crops (B, 3, h, w)
        in [0, 1], h, w multiples of 8 -> (B, I, J, d) unit tokens on the 8-px grid."""
        B, _, hh, ww = crops.shape
        ex = dino_patch_features(self.vfm, ((crops - self._vm) / self._vs).to(self.dtype))
        I, J = ex.shape[-2:]
        qk = Fn.normalize(ex.flatten(2).float(), dim=1)                    # (B, C, IJ)
        sim = qk.transpose(1, 2) @ qk                                      # (B, IJ, IJ)
        sim = (sim - sim.mean(dim=(1, 2), keepdim=True) * self.beta) * self.gamma
        sim[sim < 0.0] = float("-inf")
        sim.diagonal(dim1=-2, dim2=-1).clamp_(min=0.0)                     # NaN guard only
        A = torch.softmax(sim, dim=-1)
        hc = max(self.cpatch, round(hh / self.cpatch) * self.cpatch)
        wc = max(self.cpatch, round(ww / self.cpatch) * self.cpatch)
        xc = crops if (hc, wc) == (hh, ww) else Fn.interpolate(
            crops, size=(hc, wc), mode="bilinear", align_corners=False)
        val, attn, h, D = self._clip_values((xc - self._cm) / self._cs)    # (B,h,gh,gw,dh)
        gh, gw = val.shape[2:4]
        val = val.permute(0, 1, 4, 2, 3).reshape(B * h, D // h, gh, gw).float()
        val = Fn.interpolate(val, size=(I, J), mode="bilinear", align_corners=False)
        val = val.reshape(B, h, D // h, I * J).transpose(-1, -2)          # (B, h, IJ, dh)
        out = (A[:, None] @ val).transpose(1, 2).reshape(B, I * J, D).to(self.dtype)
        out = self.visual.ln_post(attn.out_proj(out)) @ self.visual.proj   # (B, IJ, d)
        return Fn.normalize(out.float(), dim=-1).reshape(B, I, J, -1)


def build(device="cuda", **kw):
    """No environment overrides (run_mapping, distill_features and eval_seg pass no kwargs)."""
    return ProxyCLIPHDense(device=device, **kw)
