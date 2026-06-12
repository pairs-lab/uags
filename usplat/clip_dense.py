"""
clip_dense.py — produce a per-pixel CLIP feature map for a frame.

Open-vocabulary localization needs DENSE features, not one global CLIP vector
per image. In a real run, use a dense/region CLIP variant:
  - LSeg / OpenSeg / MaskCLIP-style dense features, or
  - SAM masks + per-mask CLIP (LangSplat-style), upsampled to pixels.
This module defines the interface the pipeline expects and a mock implementation
so everything runs on CPU. Swap MockDenseCLIP for your real extractor.

Contract:  extract(image_hw3_uint8) -> features (H, W, d), L2-normalized.
"""
from __future__ import annotations
import torch
import torch.nn.functional as Fn


class DenseCLIP:
    """Interface: extract(image_hw3_uint8) -> (H, W, d) unit features."""
    def __init__(self, feat_dim=512, device="cpu"):
        self.d, self.device = feat_dim, torch.device(device)

    def extract(self, image):  # image: (H, W, 3)
        raise NotImplementedError


class OpenCLIPDense(DenseCLIP):
    """Dense per-pixel CLIP features via the MaskCLIP value-projection trick.

    In the LAST transformer block of CLIP's ViT, the softmax attention pooling
    is replaced by the value path alone: each patch token is mapped through the
    value projection, the attention output projection, the residual MLP, then
    ln_post and the visual projection. The resulting patch features live in the
    joint image-text space (cosine-comparable to text embeddings) without any
    extra training [MaskCLIP, Zhou et al., ECCV 2022].

    Works with any open_clip ViT; non-square / non-224 inputs are handled by
    bicubic interpolation of the positional embedding.
    """
    def __init__(self, model_name="ViT-B-16", pretrained="openai",
                 device="cuda", dtype=torch.float32):
        import open_clip
        if pretrained in ("", "none", "None"):
            pretrained = None        # random init: offline pipeline smoke tests
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = model.eval().to(device, dtype)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.visual = self.model.visual
        self.patch = self.visual.conv1.kernel_size[0]
        d = self.visual.proj.shape[1] if self.visual.proj is not None \
            else self.visual.transformer.width
        super().__init__(feat_dim=d, device=device)
        self.dtype = dtype
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self._mean = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)

    def _interp_pos_embed(self, gh, gw):
        """Resample the (1+L, w) positional embedding to a (gh, gw) patch grid."""
        pos = self.visual.positional_embedding          # (1+L, w)
        cls_pos, grid_pos = pos[:1], pos[1:]
        L = grid_pos.shape[0]
        side = int(round(L ** 0.5))
        if side * side == L and (gh, gw) != (side, side):
            g = grid_pos.reshape(1, side, side, -1).permute(0, 3, 1, 2)
            g = Fn.interpolate(g, size=(gh, gw), mode="bicubic", align_corners=False)
            grid_pos = g.permute(0, 2, 3, 1).reshape(gh * gw, -1)
        return torch.cat([cls_pos, grid_pos], dim=0)

    @torch.no_grad()
    def extract(self, image):
        """image: (H, W, 3) uint8 (or float in [0,255]). -> (H, W, d) unit feats."""
        H, W = int(image.shape[0]), int(image.shape[1])
        # pad to a multiple of the patch size
        Hp = (H + self.patch - 1) // self.patch * self.patch
        Wp = (W + self.patch - 1) // self.patch * self.patch
        x = torch.as_tensor(image).to(self.device, self.dtype).permute(2, 0, 1)[None]
        x = x / 255.0
        if (Hp, Wp) != (H, W):
            x = Fn.interpolate(x, size=(Hp, Wp), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std

        v = self.visual
        x = v.conv1(x)                                   # (1, w, gh, gw)
        gh, gw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)                 # (1, gh*gw, w)
        cls = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._interp_pos_embed(gh, gw).to(x.dtype)
        x = v.ln_pre(x)

        x = x.permute(1, 0, 2)                           # LND (open_clip convention)
        blocks = v.transformer.resblocks
        for blk in blocks[:-1]:
            x = blk(x)
        # MaskCLIP trick on the last block: value path only, no softmax mixing
        blk = blocks[-1]
        y = blk.ln_1(x)
        attn = blk.attn
        w_v = attn.in_proj_weight[2 * y.shape[-1]:]
        b_v = attn.in_proj_bias[2 * y.shape[-1]:]
        val = Fn.linear(y, w_v, b_v)
        val = attn.out_proj(val)
        x = x + val
        x = x + blk.mlp(blk.ln_2(x))
        x = x.permute(1, 0, 2)                           # NLD

        x = v.ln_post(x)
        if v.proj is not None:
            x = x @ v.proj
        feats = x[0, 1:].reshape(gh, gw, -1)             # drop CLS
        feats = feats.permute(2, 0, 1)[None]
        feats = Fn.interpolate(feats, size=(H, W), mode="bilinear",
                               align_corners=False)[0].permute(1, 2, 0)
        return Fn.normalize(feats, dim=-1)               # (H, W, d)

    @torch.no_grad()
    def text_embedding(self, prompts, templates=("a photo of a {}.",)):
        """prompts: str or list[str]. -> (K, d) unit text embeddings."""
        if isinstance(prompts, str):
            prompts = [prompts]
        embeds = []
        for p in prompts:
            toks = self.tokenizer([t.format(p) for t in templates]).to(self.device)
            e = self.model.encode_text(toks).to(self.dtype)
            e = Fn.normalize(e, dim=-1).mean(0)
            embeds.append(Fn.normalize(e, dim=0))
        return torch.stack(embeds)                       # (K, d)


class MockDenseCLIP(DenseCLIP):
    """Deterministic mock: assigns each pixel one of K fixed semantic prototypes
    based on a provided integer label map, plus controllable noise. Lets us test
    the whole pipeline (and inject *conflicting* observations for dynamic-region
    behavior) without any network weights."""
    def __init__(self, feat_dim=512, num_protos=8, noise=0.1, device="cpu", seed=0):
        super().__init__(feat_dim, device)
        g = torch.Generator().manual_seed(seed)
        self.protos = Fn.normalize(
            torch.randn(num_protos, feat_dim, generator=g), dim=-1).to(self.device)
        self.noise = noise

    def extract_from_labels(self, label_map):
        """label_map: (H, W) int in [0, num_protos). returns (H, W, d) unit feats."""
        label_map = label_map.to(self.device).long()
        feats = self.protos[label_map]                        # (H, W, d)
        feats = feats + self.noise * torch.randn_like(feats)
        return Fn.normalize(feats, dim=-1)

    def extract(self, image):
        # mock: derive a pseudo-label map from image intensity buckets
        H, W = image.shape[:2]
        inten = image.float().mean(-1) / 255.0
        labels = (inten * (self.protos.shape[0] - 1)).round().long()
        return self.extract_from_labels(labels)

    def text_embedding(self, proto_idx):
        """Stand-in for CLIP text encoder: returns the prototype as the 'text' dir."""
        return self.protos[proto_idx]
