"""
lseg.py — LSeg dense features (Li et al., "Language-driven Semantic Segmentation", ICLR 2022).

Model: the official demo checkpoint demo_e200.ckpt (github.com/isl-org/lang-seg, MIT licence):
ViT-L/16 (384) image encoder with a DPT decoder whose 1x1 head `scratch.head1` outputs 512-d
per-pixel embeddings at 1/2 input resolution, trained so that they are cosine-comparable to the
frozen CLIP ViT-B/32 text encoder (the one stored in the same checkpoint). This is the
extractor used by CSS and Feature 3DGS.

The official code needs pytorch-lightning 1.4, torch-encoding and openai `clip`; none of it is
installed here. This file re-implements ONLY the inference path of LSegNet (backbone
`clip_vitl16_384`, readout "project", features 256, use_bn=True, arch_option 0) with the SAME
parameter names, so the checkpoint's state dict loads strictly:
    net.pretrained.model.*          timm vit_large_patch16_384 (blocks 5/11/17/23 hooked)
    net.pretrained.act_postprocessN readout projection + 1x1 conv + (de)conv resampling
    net.scratch.*                   DPT fusion blocks + head1 (256 -> 512)
    net.clip_pretrained.*           openai CLIP ViT-B/32 (only the text tower is used)
Code adapted from lang-seg modules/models/{lseg_net,lseg_vit,lseg_blocks}.py (MIT licence,
Copyright (c) 2021 Intelligent Systems Lab Org); the clone is at
/home/thanhnc/gauss_ws/third_party/lseg/lang-seg (commit bb00b219).

Weights: only the `state_dict` of demo_e200.ckpt (730 tensors, 1.73 GB), fetched with HTTP range
requests from huggingface.co/isl-org/lang-seg (file sha256 f21aeb55...), every record
CRC32-checked against the official zip directory; the optimizer state is skipped.

Inference (defaults chosen on the Replica VALIDATION scenes, see the report):
  * RGB in [0,1], normalised with mean = std = 0.5 (LSeg's transform);
  * the whole frame at its native resolution, padded (black, like LSeg's pad_image) to a
    multiple of 32 so that the DPT pyramid (strides 4/8/16/32) is consistent; the ViT
    positional embedding (24x24 grid) is bilinearly resized to the patch grid (as LSeg does);
  * optional horizontal-flip averaging (flip=True) and the official multi-scale sliding
    window (scales=..., base_size 520, crop 480, stride 320, like LSeg_MultiEvalModule);
  * head1 features (stride 2) are average-pooled to `token_stride` (default 8: a 640x480 frame
    gives a 60x80x512 token grid, 4.9 MB in fp16) -> extract_tokens.
"""
from __future__ import annotations
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from usplat.clip_dense import DenseCLIP, upsample_tokens

WEIGHTS = os.environ.get("UAGS_LSEG_WEIGHTS",
                         "/home/thanhnc/gauss_ws/src/uags/data/weights/lseg/demo_e200_state_dict.pt")
HOOKS = (5, 11, 17, 23)
VIT_W = 1024
FEATS = (256, 512, 1024, 1024)


# ----------------------------------------------------------------------------- model parts
class _ProjectReadout(nn.Module):
    """cat(patch token, CLS token) -> Linear(2C, C) -> GELU (lang-seg ProjectReadout)."""
    def __init__(self, c):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(2 * c, c), nn.GELU())

    def forward(self, x):                                # (B, 1+L, C)
        cls = x[:, :1].expand_as(x[:, 1:])
        return self.project(torch.cat([x[:, 1:], cls], -1))


def _postprocess(level):
    """act_postprocessN with the official indices: 0 readout, 1 Transpose, 2 Unflatten (no
    params; done in forward), 3 1x1 conv, 4 resampling (deconv x4 / x2, or stride-2 conv)."""
    out = FEATS[level]
    layers = [_ProjectReadout(VIT_W), nn.Identity(), nn.Identity(),
              nn.Conv2d(VIT_W, out, 1)]
    if level == 0:
        layers.append(nn.ConvTranspose2d(out, out, 4, stride=4))
    elif level == 1:
        layers.append(nn.ConvTranspose2d(out, out, 2, stride=2))
    elif level == 3:
        layers.append(nn.Conv2d(out, out, 3, stride=2, padding=1))
    return nn.Sequential(*layers)


class _ResidualConvUnit(nn.Module):
    """ResidualConvUnit_custom with bn=True, activation ReLU."""
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = self.bn1(self.conv1(Fn.relu(x)))
        out = self.bn2(self.conv2(Fn.relu(out)))
        return out + x


class _FusionBlock(nn.Module):
    """FeatureFusionBlock_custom (expand=False, align_corners=True)."""
    def __init__(self, c):
        super().__init__()
        self.out_conv = nn.Conv2d(c, c, 1)
        self.resConfUnit1 = _ResidualConvUnit(c)
        self.resConfUnit2 = _ResidualConvUnit(c)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        x = Fn.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class _LSegNet(nn.Module):
    def __init__(self, features=256, out_c=512):
        super().__init__()
        import timm
        pre = nn.Module()
        pre.model = timm.create_model("vit_large_patch16_384", pretrained=False)
        for i in range(4):
            setattr(pre, f"act_postprocess{i + 1}", _postprocess(i))
        self.pretrained = pre
        s = nn.Module()
        for i, cin in enumerate(FEATS):
            setattr(s, f"layer{i + 1}_rn", nn.Conv2d(cin, features, 3, padding=1, bias=False))
        for i in range(4):
            setattr(s, f"refinenet{i + 1}", _FusionBlock(features))
        s.head1 = nn.Conv2d(features, out_c, 1)
        self.scratch = s

    def _pos_embed(self, gh, gw):
        pe = self.pretrained.model.pos_embed               # (1, 1+24*24, C)
        tok, grid = pe[:, :1], pe[0, 1:]
        g0 = int(math.sqrt(grid.shape[0]))
        grid = grid.reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
        grid = Fn.interpolate(grid, size=(gh, gw), mode="bilinear")   # as lang-seg
        return torch.cat([tok, grid.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)], 1)

    def forward(self, x):
        """x: (B, 3, H, W) normalised, H and W multiples of 32 -> (B, 512, H/2, W/2)."""
        m = self.pretrained.model
        B, _, H, W = x.shape
        gh, gw = H // 16, W // 16
        t = m.patch_embed.proj(x).flatten(2).transpose(1, 2)
        t = torch.cat([m.cls_token.expand(B, -1, -1), t], 1) + self._pos_embed(gh, gw)
        acts = []
        for i, blk in enumerate(m.blocks):
            t = blk(t)
            if i in HOOKS:
                acts.append(t)
            if i == HOOKS[-1]:
                break
        layers = []
        for i, a in enumerate(acts):
            pp = getattr(self.pretrained, f"act_postprocess{i + 1}")
            y = pp[0](a).transpose(1, 2).reshape(B, VIT_W, gh, gw)
            layers.append(pp[3:](y))
        s = self.scratch
        l1, l2, l3, l4 = (getattr(s, f"layer{i + 1}_rn")(l) for i, l in enumerate(layers))
        p = s.refinenet4(l4)
        p = s.refinenet3(p, l3)
        p = s.refinenet2(p, l2)
        p = s.refinenet1(p, l1)
        return s.head1(p)


# ----------------------------------------------------------------------------- extractor
class LSegDense(DenseCLIP):
    """Dense LSeg features behind the usplat extractor contract."""
    def __init__(self, weights=WEIGHTS, device="cuda", token_stride=8, flip=False,
                 scales=None, base_size=520, crop_size=480, half=False):
        import open_clip
        if token_stride not in (2, 4, 8, 16):
            raise ValueError(token_stride)
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        super().__init__(feat_dim=512, device=device)
        self.net = _LSegNet()
        own = {k[len("net."):]: v for k, v in sd.items()
               if k.startswith("net.") and not k.startswith("net.clip_pretrained.")}
        own = {k: v.float() if v.is_floating_point() else v for k, v in own.items()}
        missing, unexpected = self.net.load_state_dict(own, strict=False)
        # timm's final norm / classifier head are never used by LSeg's DPT path
        missing = [k for k in missing if not k.startswith(("pretrained.model.norm.",
                                                           "pretrained.model.head.",
                                                           "pretrained.model.fc_norm."))]
        if missing or unexpected:
            raise RuntimeError(f"LSeg state dict mismatch: missing {missing[:8]}, "
                               f"unexpected {unexpected[:8]}")
        self.net = self.net.eval().to(device)
        clip_sd = {k[len("net.clip_pretrained."):]: v for k, v in sd.items()
                   if k.startswith("net.clip_pretrained.")}
        clip = open_clip.model.build_model_from_openai_state_dict(clip_sd, quick_gelu=True,
                                                                  cast_dtype=None)
        clip.visual = nn.Identity()                         # the text tower is all LSeg uses
        self.clip = clip.float().eval().to(device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        del sd, own, clip_sd
        self.token_stride, self.flip = token_stride, flip
        self.scales = tuple(scales) if scales else None
        self.base_size, self.crop_size = base_size, crop_size
        self.half = half

    # -- image side
    def _norm(self, image):
        x = torch.as_tensor(image).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        return (x - 0.5) / 0.5                              # LSeg: mean = std = 0.5

    def _forward_padded(self, x, min_hw=0):
        """(1,3,h,w) normalised -> (1,512,ceil(h/2),ceil(w/2)) raw head1 features. Pads
        bottom/right with the normalised value of a black pixel (LSeg's pad_image) to a
        multiple of 32 and at least min_hw, runs the net, crops the padding away."""
        h, w = x.shape[-2:]
        H, W = -(-max(h, min_hw) // 32) * 32, -(-max(w, min_hw) // 32) * 32
        if (H, W) != (h, w):
            x = Fn.pad(x, (0, W - w, 0, H - h), value=-1.0)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half and x.is_cuda):
            f = self.net(x).float()
        return f[..., : -(-h // 2), : -(-w // 2)]

    def _forward_tta(self, x, min_hw=0):
        """unit-normalised features (1,512,h/2,w/2), flip-averaged if self.flip."""
        f = Fn.normalize(self._forward_padded(x, min_hw), dim=1)
        if self.flip:
            f = f + Fn.normalize(self._forward_padded(x.flip(-1), min_hw), dim=1).flip(-1)
        return f

    def _forward_multiscale(self, x):
        """LSeg_MultiEvalModule on features: per scale, resize the long side to
        base_size*scale (bilinear, align_corners=True), then one padded 480 crop or 480
        sliding windows with stride 320 (every crop padded to 480), average unit features per
        pixel, resize to (h/2, w/2), sum unit features over scales."""
        _, _, h, w = x.shape
        oh, ow = -(-h // 2), -(-w // 2)
        cs, stride = self.crop_size, int(self.crop_size * 2 / 3)
        acc = 0
        for sc in self.scales:
            long = int(math.ceil(self.base_size * sc))
            if h > w:
                nh, nw = long, int(w * long / h + 0.5)
            else:
                nw, nh = long, int(h * long / w + 0.5)
            img = Fn.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=True)
            if long <= cs:
                f = self._forward_tta(img, min_hw=cs)
            else:
                ph, pw = max(nh, cs), max(nw, cs)
                img = Fn.pad(img, (0, pw - nw, 0, ph - nh), value=-1.0)
                fh, fw = -(-ph // 2), -(-pw // 2)
                f = x.new_zeros(1, 512, fh, fw)
                cnt = x.new_zeros(1, 1, fh, fw)
                for iy in range(int(math.ceil((ph - cs) / stride)) + 1):
                    for ix in range(int(math.ceil((pw - cs) / stride)) + 1):
                        y0, x0 = iy * stride, ix * stride
                        y1, x1 = min(y0 + cs, ph), min(x0 + cs, pw)
                        fc = self._forward_tta(img[..., y0:y1, x0:x1], min_hw=cs)
                        sl = (..., slice(y0 // 2, y0 // 2 + fc.shape[-2]),
                              slice(x0 // 2, x0 // 2 + fc.shape[-1]))
                        f[sl] += fc
                        cnt[sl] += 1
                f = f / cnt
            f = f[..., : -(-nh // 2), : -(-nw // 2)]
            f = Fn.interpolate(f, size=(oh, ow), mode="bilinear", align_corners=True)
            acc = acc + Fn.normalize(f, dim=1)
        return acc

    @torch.no_grad()
    def extract_tokens(self, image):
        """(H, W, 3) uint8 RGB -> (ceil(H/s), ceil(W/s), 512) LSeg features, s = token_stride,
        NOT normalised (average-pooled from the stride-2 head output)."""
        x = self._norm(image)
        f = self._forward_multiscale(x) if self.scales else \
            (self._forward_tta(x) if self.flip else self._forward_padded(x))
        k = self.token_stride // 2
        if k > 1:
            f = Fn.avg_pool2d(f, k, stride=k, ceil_mode=True)
        return f[0].permute(1, 2, 0).contiguous()

    @torch.no_grad()
    def extract(self, image, out_hw=None):
        tokens = self.extract_tokens(image)
        return upsample_tokens(tokens, out_hw or (int(image.shape[0]), int(image.shape[1])))

    # -- text side (frozen CLIP ViT-B/32 text encoder from the same checkpoint)
    @torch.no_grad()
    def text_embedding(self, prompts, templates=("a photo of a {}.",)):
        if isinstance(prompts, str):
            prompts = [prompts]
        embeds = []
        for p in prompts:
            toks = self.tokenizer([t.format(p) for t in templates]).to(self.device)
            e = Fn.normalize(self.clip.encode_text(toks).float(), dim=-1).mean(0)
            embeds.append(Fn.normalize(e, dim=0))
        return torch.stack(embeds)


def build(device="cuda", **kw):
    """Keyword arguments override the defaults; so do these environment variables (used to
    compare variants through the frozen scripts/check_clip2d.py, which passes no kwargs):
    UAGS_LSEG_STRIDE (2/4/8/16), UAGS_LSEG_FLIP (0/1), UAGS_LSEG_SCALES ("0.5,0.75,..."),
    UAGS_LSEG_HALF (0/1)."""
    env = os.environ
    if "UAGS_LSEG_STRIDE" in env:
        kw.setdefault("token_stride", int(env["UAGS_LSEG_STRIDE"]))
    if "UAGS_LSEG_FLIP" in env:
        kw.setdefault("flip", env["UAGS_LSEG_FLIP"] == "1")
    if env.get("UAGS_LSEG_SCALES"):
        kw.setdefault("scales", [float(s) for s in env["UAGS_LSEG_SCALES"].split(",")])
    if "UAGS_LSEG_HALF" in env:
        kw.setdefault("half", env["UAGS_LSEG_HALF"] == "1")
    return LSegDense(device=device, **kw)
