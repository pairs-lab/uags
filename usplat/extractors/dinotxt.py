"""
dinotxt.py — dense dino.txt features (Jose et al., "DINOv2 Meets Text: A Unified Framework for
Image- and Pixel-Level Vision-Language Alignment", CVPR 2025, arXiv 2412.16334).

Model: frozen DINOv2 ViT-L/14 with 4 registers (LVD-142M) + a trained vision head (2 attention
blocks) + a trained 24-layer text transformer (width 1280), from facebookresearch/dinov2
(hub entry dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, Apache-2.0 code and weights; the CLIP BPE
vocabulary is MIT). The text embedding is 2048-d: the first half is aligned to the CLS token, the
SECOND half to the patch tokens. Dense features = the head's patch tokens (1024-d); the matching
text embedding = text[:, 1024:] (the official notebooks/dinotxt.ipynb recipe).

Code: /home/thanhnc/gauss_ws/third_party/dinotxt/dinov2 (git 7764ea0f, imported via sys.path).
Weights: /home/thanhnc/gauss_ws/data/weights/dinotxt (see MANIFEST.json there: the backbone and
text-tower 2-D matrices are stored in fp16, everything else as released).

Inference (defaults chosen on the VALIDATION scenes office_0/room_2, see the run report):
  * RGB / 255, ImageNet mean/std;
  * resize so the short side is `short_side` px (None: keep the image scale), each side rounded to
    a multiple of the 14-px patch;
  * window=None: one forward pass over the whole resized image; window=W: W x W crops (multiple
    of 14) at `stride`, patch tokens averaged where crops overlap;
  * fp16 autocast on CUDA, fp32 on CPU.
"""
from __future__ import annotations
import os
import sys
import warnings

import torch
import torch.nn.functional as Fn

from usplat.clip_dense import upsample_tokens

REPO = "/home/thanhnc/gauss_ws/third_party/dinotxt/dinov2"
WEIGHTS = "/home/thanhnc/gauss_ws/data/weights/dinotxt"
FILES = dict(
    backbone="dinov2_vitl14_reg4_pretrain_fp16.pth",
    head="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
    text="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder_fp16.pth",
    vocab="bpe_simple_vocab_16e6.txt.gz",
)
PATCH = 14
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _repo():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    warnings.filterwarnings("ignore", message="xFormers is not available")


def weights_present(weights_dir=WEIGHTS):
    return all(os.path.isfile(os.path.join(weights_dir, f)) for f in FILES.values())


def build_model(backbone_sd, head_sd, text_sd):
    """The dino.txt model of dinov2.hub.dinotxt.dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, same
    config, but built on the meta device and filled from local state dicts (strict)."""
    _repo()
    import math
    from dinov2.hub.backbones import dinov2_vitl14_reg
    from dinov2.hub.text.dinotxt_model import DinoTxtConfig, DinoTxt
    from dinov2.hub.text.dinov2_wrapper import DINOv2Wrapper
    from dinov2.hub.text.text_transformer import TextTransformer
    cfg = DinoTxtConfig(
        embed_dim=2048, vision_model_freeze_backbone=True, vision_model_train_img_size=224,
        vision_model_use_class_token=True, vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2, vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False, vision_model_patch_tokens_pooler_type="mean",
        vision_model_patch_token_layer=1, text_model_freeze_backbone=False,
        text_model_num_head_blocks=0, text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0, text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True, init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None, freeze_logit_scale=False)
    with torch.device("meta"):
        vision = DINOv2Wrapper(dinov2_vitl14_reg(pretrained=False))
        text = TextTransformer(context_length=77, vocab_size=49408, dim=1280, num_heads=20,
                               num_layers=24, ffn_ratio=4, is_causal=True, ls_init_value=None,
                               dropout_prob=0.0)
        model = DinoTxt(cfg, vision, text)
    model.visual_model.backbone.model.load_state_dict(backbone_sd, strict=True, assign=True)
    model.visual_model.head.load_state_dict(head_sd, strict=True, assign=True)
    model.text_model.load_state_dict(text_sd, strict=True, assign=True)
    model.logit_scale = torch.nn.Parameter(torch.tensor([math.log(1 / 0.07)]))  # unused
    return model.float().eval()


class DinoTxtDense:
    """Dense dino.txt extractor behind the DenseCLIP contract (usplat/extractors/__init__.py)."""

    def __init__(self, device="cuda", short_side=None, window=None, stride=None, half=None,
                 weights_dir=WEIGHTS, state_dicts=None):
        self.device = torch.device(device)
        self.half = (self.device.type == "cuda") if half is None else bool(half)
        if window is not None and (window % PATCH or (stride or window) % PATCH):
            raise ValueError("window and stride must be multiples of 14")
        self.short_side, self.window = short_side, window
        self.stride = stride or (window // 2 if window else None)
        if state_dicts is None:
            ld = lambda f: torch.load(os.path.join(weights_dir, FILES[f]), map_location="cpu",
                                      weights_only=True)
            state_dicts = (ld("backbone"), ld("head"), ld("text"))
        self.model = build_model(*state_dicts).to(self.device)
        if self.half:
            self.model.half()
        _repo()
        from dinov2.hub.text.tokenizer import Tokenizer
        self.tokenizer = Tokenizer(vocab_path=os.path.join(weights_dir, FILES["vocab"]))
        self.d = self.model.visual_model.backbone.embed_dim          # 1024
        self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def _autocast(self):
        if self.half and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return torch.autocast(self.device.type, enabled=False)

    def _patch_tokens(self, x):
        """(B, 3, h, w) normalised, h and w multiples of 14 -> (B, h/14, w/14, d) fp32."""
        B, _, h, w = x.shape
        with self._autocast():
            _, patch = self.model.visual_model.get_class_and_patch_tokens(
                x.half() if self.half else x)
        return patch.float().reshape(B, h // PATCH, w // PATCH, -1)

    @torch.no_grad()
    def extract_tokens(self, image):
        """(H, W, 3) uint8 RGB -> (gh, gw, d) fp32 patch tokens, NOT normalised (cacheable)."""
        H, W = int(image.shape[0]), int(image.shape[1])
        s = 1.0 if self.short_side is None else self.short_side / min(H, W)
        Hr = max(PATCH, int(round(H * s / PATCH)) * PATCH)
        Wr = max(PATCH, int(round(W * s / PATCH)) * PATCH)
        x = torch.as_tensor(image).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        if (Hr, Wr) != (H, W):
            x = Fn.interpolate(x, size=(Hr, Wr), mode="bilinear", align_corners=False,
                               antialias=(Hr < H or Wr < W))
        x = (x - self._mean) / self._std
        if self.window is None or (Hr <= self.window and Wr <= self.window):
            return self._patch_tokens(x)[0]
        gh, gw = Hr // PATCH, Wr // PATCH
        wh, ww = min(self.window // PATCH, gh), min(self.window // PATCH, gw)
        sp = self.stride // PATCH

        def starts(n, k):
            s_ = list(range(0, n - k + 1, sp))
            return s_ + [n - k] if s_[-1] != n - k else s_
        boxes = [(y, x0) for y in starts(gh, wh) for x0 in starts(gw, ww)]
        crops = torch.cat([x[..., y * PATCH:(y + wh) * PATCH, x0 * PATCH:(x0 + ww) * PATCH]
                           for y, x0 in boxes])
        toks = torch.cat([self._patch_tokens(c) for c in crops.split(8)])
        acc = torch.zeros(gh, gw, toks.shape[-1], device=self.device)
        cnt = torch.zeros(gh, gw, 1, device=self.device)
        for (y, x0), t in zip(boxes, toks):
            acc[y:y + wh, x0:x0 + ww] += t
            cnt[y:y + wh, x0:x0 + ww] += 1
        return acc / cnt

    @torch.no_grad()
    def extract(self, image, out_hw=None):
        """(H, W, 3) uint8 -> (h, w, d) unit features at out_hw (default: the image size)."""
        tokens = self.extract_tokens(image)
        return upsample_tokens(tokens, out_hw or (int(image.shape[0]), int(image.shape[1])))

    @torch.no_grad()
    def text_embedding(self, prompts, templates=("a photo of a {}.",)):
        """prompts: str or list[str] -> (K, d) unit embeddings: the patch-aligned half of the
        2048-d dino.txt text embedding, normalised per template, averaged, re-normalised."""
        if isinstance(prompts, str):
            prompts = [prompts]
        texts = [t.format(p) for p in prompts for t in templates]
        toks = self.tokenizer.tokenize(texts).to(self.device)
        with self._autocast():
            e = self.model.encode_text(toks)
        e = Fn.normalize(e.float()[:, self.d:], dim=-1).reshape(len(prompts), len(templates), -1)
        return Fn.normalize(e.mean(1), dim=-1)                     # (K, d)


def build(device="cuda", **kw):
    """Registry entry (usplat.extractors.get_extractor("dinotxt"))."""
    return DinoTxtDense(device=device, **kw)
