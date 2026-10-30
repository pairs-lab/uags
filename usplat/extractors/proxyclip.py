"""
proxyclip.py — ProxyCLIP dense features (Lan et al., ECCV 2024, arXiv 2408.04883), training-free.

CLIP's last ViT block is replaced by *proxy attention*: the attention weights come from the
cosine similarity of a vision foundation model's (VFM) patch features, the values are CLIP's
last-block values (bilinearly resampled to the VFM grid); no residual, no FFN; then CLIP's
ln_post and visual projection, so every token lives in CLIP's joint image-text space and is
cosine-comparable to `text_embedding` for any vocabulary.

    A = softmax(mask((S - beta * mean(S)) * gamma)),  S = cos(f_vfm_i, f_vfm_j),
    mask: entries < 0 -> -inf;   out = ln_post(out_proj(A @ V_clip)) @ proj

Re-implemented on the installed open_clip (the official repo bundles a modified open_clip
that would shadow it). Settings follow the official code (github.com/mc-lan/ProxyCLIP @ a0e38d1,
proxyclip_segmentor.py and open_clip/transformer.py:custom_attn), copied to
/home/thanhnc/gauss_ws/third_party/proxyclip/ProxyCLIP (NTU S-Lab License 1.0: non-commercial
use only). Defaults = their CLIP ViT-B/16 recipe:
  * image resized to a short side of 336 (their VOC/Context/COCO configs; 448 for ADE20k),
  * sliding window 336 x 336, stride 112, both images fed to CLIP (patch 16 -> 21x21 grid)
    and to the VFM (DINOv2 ViT-B/14 with 4 registers, patch 14 -> 24x24 grid),
  * beta 1.2, gamma 3.0, fp16 weights on CUDA.
The per-window unit tokens are averaged on the global 14-px VFM grid (windows and the resized
image are multiples of 14 px, so windows align exactly); ProxyCLIP averages per-window logits
at pixel level instead, which is the same up to bilinear border effects (logits are linear in
the unit features). One guard not in the original: the diagonal of the masked similarity is
kept finite (a window whose mean similarity exceeds 1/beta would otherwise give NaN rows).

DINOv2 code: /home/thanhnc/gauss_ws/third_party/proxyclip/dinov2 (facebookresearch/dinov2,
Apache-2.0); weights /home/thanhnc/gauss_ws/src/uags/data/weights/proxyclip/dinov2_vitb14{_reg4,}_pretrain.pth
(Apache-2.0). CLIP weights through open_clip's HF cache (openai: MIT licence; laion2b: MIT).

Development overrides (for the frozen validation check, which passes no kwargs):
  UAGS_PROXYCLIP_PRETRAINED (openai | laion2b_s34b_b88k), UAGS_PROXYCLIP_VFM
  (dinov2_vitb14_reg | dinov2_vitb14), UAGS_PROXYCLIP_SHORT (short side in px).
"""
from __future__ import annotations
import os
import sys
import torch
import torch.nn.functional as Fn

from usplat.clip_dense import upsample_tokens

THIRD_PARTY = "/home/thanhnc/gauss_ws/third_party/proxyclip"
WEIGHTS = "/home/thanhnc/gauss_ws/src/uags/data/weights/proxyclip"
VFM_WEIGHTS = {"dinov2_vitb14_reg": "dinov2_vitb14_reg4_pretrain.pth",
               "dinov2_vitb14": "dinov2_vitb14_pretrain.pth"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def _load_dinov2(name, device, dtype):
    path = os.path.join(WEIGHTS, VFM_WEIGHTS[name])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    repo = os.path.join(THIRD_PARTY, "dinov2")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from dinov2.hub import backbones
    model = getattr(backbones, name)(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(device, dtype)


def _windows(n, crop, stride):
    """Start offsets of ProxyCLIP's sliding windows along one axis (forward_slide)."""
    c = min(crop, n)
    k = max(n - c + stride - 1, 0) // stride + 1
    return sorted({max(min(i * stride + c, n) - c, 0) for i in range(k)}), c


class ProxyCLIPDense:
    def __init__(self, device="cuda", pretrained="openai", vfm="dinov2_vitb14_reg",
                 short_side=336, crop=336, stride=112, beta=1.2, gamma=3.0, dtype=None):
        import open_clip
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.pretrained, self.vfm_name = pretrained, vfm
        self.short_side, self.crop, self.stride = int(short_side), int(crop), int(stride)
        self.beta, self.gamma = float(beta), float(gamma)
        # the openai weights need QuickGELU; open_clip 3.x does not switch it on by itself
        self.model = open_clip.create_model("ViT-B-16", pretrained=pretrained,
                                            force_quick_gelu=(pretrained == "openai"))
        self.model = self.model.eval().to(self.device, dtype)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.visual = self.model.visual
        self.cpatch = self.visual.conv1.kernel_size[0]                      # 16
        self.d = int(self.visual.proj.shape[1])                             # 512
        self.vfm = _load_dinov2(vfm, self.device, dtype)
        self.vpatch = int(self.vfm.patch_size)                              # 14
        for s in (self.crop, self.stride):
            if s % self.vpatch:
                raise ValueError(f"crop/stride must be multiples of {self.vpatch}")
        f = dict(device=self.device, dtype=torch.float32)
        self._cm = torch.tensor(CLIP_MEAN, **f).view(1, 3, 1, 1)
        self._cs = torch.tensor(CLIP_STD, **f).view(1, 3, 1, 1)
        self._vm = torch.tensor(IMNET_MEAN, **f).view(1, 3, 1, 1)
        self._vs = torch.tensor(IMNET_STD, **f).view(1, 3, 1, 1)

    # ---------------------------------------------------------------- image side
    def _resized(self, image):
        """(H, W, 3) uint8 -> (1, 3, Hr, Wr) float in [0, 1]; short side ~= short_side and
        both sides rounded to multiples of the VFM patch, so the windows tile the 14-px grid."""
        H, W = int(image.shape[0]), int(image.shape[1])
        s = self.short_side / min(H, W)
        p = self.vpatch
        Hr, Wr = max(p, round(H * s / p) * p), max(p, round(W * s / p) * p)
        x = torch.as_tensor(image).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        if (Hr, Wr) != (H, W):   # plain bilinear like mmseg's cv2 INTER_LINEAR (no antialias):
            x = Fn.interpolate(x, size=(Hr, Wr), mode="bilinear",   # antialias cut the argmax
                               align_corners=False).clamp_(0, 1)    # agreement 99.7 -> 98.1%
        return x

    def _clip_values(self, x):
        """x: (B, 3, h, w) CLIP-normalised, h, w multiples of 16. -> last-block values
        (B, heads, gh, gw, dh) and the (gh, gw) grid, via blocks[:-1] and ln_1 of the last."""
        v = self.visual
        x = v.conv1(x.to(self.dtype))
        B, _, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls = v.class_embedding.to(x.dtype).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = v.positional_embedding
        side = int(round((pos.shape[0] - 1) ** 0.5))
        grid = pos[1:]
        if (gh, gw) != (side, side):   # ProxyCLIP's interpolate_pos_encoding (+0.1 offset)
            g = grid.float().reshape(1, side, side, -1).permute(0, 3, 1, 2)
            g = Fn.interpolate(g, scale_factor=((gh + 0.1) / side, (gw + 0.1) / side),
                               mode="bicubic", align_corners=False)
            assert g.shape[-2:] == (gh, gw), (g.shape, gh, gw)
            grid = g.permute(0, 2, 3, 1).reshape(gh * gw, -1)
        x = x + torch.cat([pos[:1].float(), grid.float()], 0).to(x.dtype)
        x = v.ln_pre(x)
        batch_first = getattr(v.transformer, "batch_first", True)
        if not batch_first:
            x = x.permute(1, 0, 2)
        for blk in v.transformer.resblocks[:-1]:
            x = blk(x)
        if not batch_first:
            x = x.permute(1, 0, 2)
        blk = v.transformer.resblocks[-1]
        y = blk.ln_1(x)
        attn = blk.attn
        D = y.shape[-1]
        _, _, val = Fn.linear(y, attn.in_proj_weight, attn.in_proj_bias).split(D, dim=-1)
        h = attn.num_heads
        val = val[:, 1:].reshape(B, gh, gw, h, D // h).permute(0, 3, 1, 2, 4)
        return val, attn, h, D

    @torch.no_grad()
    def _window_tokens(self, crops):
        """crops: (B, 3, h, w) in [0, 1], h, w multiples of 14. -> (B, I, J, d) unit tokens
        on the VFM grid (I = h/14, J = w/14)."""
        B, _, hh, ww = crops.shape
        # VFM proxy: last-layer normed patch tokens (ProxyCLIP: get_intermediate_layers(...)[0])
        xv = ((crops - self._vm) / self._vs).to(self.dtype)
        ex = self.vfm.get_intermediate_layers(xv, reshape=True)[0]         # (B, C, I, J)
        I, J = ex.shape[-2:]
        qk = Fn.normalize(ex.flatten(2).float(), dim=1)                    # (B, C, IJ)
        sim = qk.transpose(1, 2) @ qk                                      # (B, IJ, IJ)
        sim = (sim - sim.mean(dim=(1, 2), keepdim=True) * self.beta) * self.gamma
        sim[sim < 0.0] = float("-inf")
        sim.diagonal(dim1=-2, dim2=-1).clamp_(min=0.0)                     # NaN guard only
        A = torch.softmax(sim, dim=-1)                                     # (B, IJ, IJ)
        # CLIP values: crop at a multiple of 16 px (336 -> 21x21 grid), resampled to (I, J)
        hc = max(self.cpatch, round(hh / self.cpatch) * self.cpatch)
        wc = max(self.cpatch, round(ww / self.cpatch) * self.cpatch)
        xc = crops if (hc, wc) == (hh, ww) else Fn.interpolate(
            crops, size=(hc, wc), mode="bilinear", align_corners=False)
        val, attn, h, D = self._clip_values((xc - self._cm) / self._cs)    # (B,h,gh,gw,dh)
        gh, gw = val.shape[2:4]
        val = val.permute(0, 1, 4, 2, 3).reshape(B * h, D // h, gh, gw).float()
        val = Fn.interpolate(val, size=(I, J), mode="bilinear", align_corners=False)
        val = val.reshape(B, h, D // h, I * J).transpose(-1, -2)          # (B, h, IJ, dh)
        out = A[:, None] @ val                                             # (B, h, IJ, dh)
        out = out.transpose(1, 2).reshape(B, I * J, D).to(self.dtype)
        out = attn.out_proj(out)
        out = self.visual.ln_post(out) @ self.visual.proj                  # (B, IJ, d)
        return Fn.normalize(out.float(), dim=-1).reshape(B, I, J, -1)

    @torch.no_grad()
    def extract_tokens(self, image):
        """(H, W, 3) uint8 RGB -> (gh, gw, d) float32 token grid on the 14-px grid of the
        resized image (640x480 -> 448x336 -> (24, 32, 512)); the mean of the overlapping
        windows' unit tokens, NOT normalised (normalise after upsampling)."""
        x = self._resized(image)
        Hr, Wr = x.shape[-2:]
        ys, ch = _windows(Hr, self.crop, self.stride)
        xs, cw = _windows(Wr, self.crop, self.stride)
        p = self.vpatch
        crops = torch.cat([x[:, :, y:y + ch, x0:x0 + cw] for y in ys for x0 in xs], 0)
        tok = self._window_tokens(crops)                                   # (n, ch/p, cw/p, d)
        acc = torch.zeros(Hr // p, Wr // p, self.d, device=self.device)
        cnt = torch.zeros(Hr // p, Wr // p, 1, device=self.device)
        k = 0
        for y in ys:
            for x0 in xs:
                acc[y // p:(y + ch) // p, x0 // p:(x0 + cw) // p] += tok[k]
                cnt[y // p:(y + ch) // p, x0 // p:(x0 + cw) // p] += 1
                k += 1
        return acc / cnt

    @torch.no_grad()
    def extract(self, image, out_hw=None):
        """(H, W, 3) uint8 RGB -> (h, w, d) unit per-pixel features (default: image size)."""
        tokens = self.extract_tokens(image)
        return upsample_tokens(tokens, out_hw or (int(image.shape[0]), int(image.shape[1])))

    # ---------------------------------------------------------------- text side
    @torch.no_grad()
    def text_embedding(self, prompts, templates=("a photo of a {}.",)):
        """prompts: str or list[str] -> (K, d) float32 unit text embeddings (mean over the
        per-template unit embeddings, then normalised; ProxyCLIP's own recipe)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        out = []
        for p in prompts:
            toks = self.tokenizer([t.format(p) for t in templates]).to(self.device)
            e = Fn.normalize(self.model.encode_text(toks).float(), dim=-1).mean(0)
            out.append(Fn.normalize(e, dim=0))
        return torch.stack(out)


def build(device="cuda", **kw):
    env = os.environ.get
    kw.setdefault("pretrained", env("UAGS_PROXYCLIP_PRETRAINED", "openai"))
    kw.setdefault("vfm", env("UAGS_PROXYCLIP_VFM", "dinov2_vitb14_reg"))
    kw.setdefault("short_side", int(env("UAGS_PROXYCLIP_SHORT", "336")))
    return ProxyCLIPDense(device=device, **kw)
