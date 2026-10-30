"""
talk2dino.py -- dense DINOv2 patch features + CLIP text projected into the DINOv2 space.

Talk2DINO [Barsellotti et al., "Talking to DINO: Bridging Self-Supervised Vision Backbones
with Language for Open-Vocabulary Segmentation", ICCV 2025, arXiv 2411.19331;
code github.com/lorebianchi98/Talk2DINO, Apache-2.0] learns a small MLP that maps the CLIP
ViT-B/16 (OpenAI) text embedding into the patch-token space of DINOv2 ViT-B/14 with
registers. The image side is the FROZEN DINOv2 backbone, so:
    pixel feature  = DINOv2 x_norm_patchtokens               (d = 768 for ViT-B, 1024 for ViT-L)
    text embedding = normalise(MLP(mean_t CLIP_text(template_t(class))))
and the two are cosine-comparable for any vocabulary (the paper's DINOText.generate_masks
with ProjectionLayer, no PAMR, no background cleaning).

Inference (the defaults were picked on the validation scenes office_0 / room_2 only):
  * resize the frame so the short side is `short_side` px (448 as in the paper), long side
    rounded to a multiple of the patch size 14 (640x480 -> 602x448), bilinear + antialias,
    ImageNet mean/std;
  * mode "whole": one forward pass on the whole resized frame -> (32, 43, d) token grid;
    mode "slide": the paper's 448x448 windows with stride 224 (patch-aligned here), each
    window's tokens L2-normalised, then averaged where windows overlap (= averaging the
    per-window cosine logits, as mmseg's slide inference does).

Code used: the official DINOv2 model code (third_party/talk2dino/dinov2, Apache-2.0) via
sys.path; the Talk2DINO text projection is 12 lines re-implemented here (ProjectionLayer
with hidden_layer=True, act=tanh), loaded strict from the official weights file, to avoid
importing the repo's `src` package (it needs the unpinned `openai/CLIP` pip package).
The CLIP text tower is open_clip "ViT-B-16-quickgelu" / "openai" (the OpenAI weights that
`clip.load("ViT-B/16")` returns, with QuickGELU; the visual tower is dropped after loading).
"""
from __future__ import annotations
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from usplat.clip_dense import upsample_tokens

THIRD = "/home/thanhnc/gauss_ws/third_party/talk2dino"
WEIGHTS = "/home/thanhnc/gauss_ws/src/uags/data/weights/talk2dino"
BACKBONES = {   # name -> (dinov2 hub factory, DINOv2 weights file, Talk2DINO projection file)
    "vitb": ("dinov2_vitb14_reg", "dinov2_vitb14_reg4_pretrain.pth", "vitb_mlp_infonce.pth"),
    "vitl": ("dinov2_vitl14_reg", "dinov2_vitl14_reg4_pretrain.pth", "vitl_mlp_infonce.pth"),
}
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


class TextProjection(nn.Module):
    """Talk2DINO ProjectionLayer (src/model.py) with hidden_layer=True, act=tanh:
    CLIP text (512) -> Linear -> tanh -> Linear -> DINOv2 dim."""
    def __init__(self, clip_dim=512, dino_dim=768):
        super().__init__()
        self.linear_layer = nn.Linear(clip_dim, dino_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(dino_dim, dino_dim)])

    def forward(self, t):
        x = self.linear_layer(t.float())
        for h in self.hidden_layers:
            x = h(torch.tanh(x))
        return x


class Talk2DINODense:
    """DenseCLIP-contract extractor (see usplat/extractors/__init__.py)."""

    def __init__(self, device="cuda", backbone="vitb", short_side=448, mode="whole",
                 crop=448, stride=224, weights_dir=WEIGHTS, clip_pretrained="openai"):
        if backbone not in BACKBONES:
            raise ValueError(f"backbone {backbone!r} not in {sorted(BACKBONES)}")
        if mode not in ("whole", "slide"):
            raise ValueError(mode)
        self.device = torch.device(device)
        self.backbone, self.mode = backbone, mode
        self.short_side, self.crop, self.stride = int(short_side), int(crop), int(stride)
        hub_fn, dino_file, proj_file = BACKBONES[backbone]
        dino_path = os.path.join(weights_dir, dino_file)
        proj_path = os.path.join(weights_dir, proj_file)
        for p in (dino_path, proj_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)

        # DINOv2 backbone, official code, weights from the local file
        if os.path.join(THIRD, "dinov2") not in sys.path:
            sys.path.insert(0, os.path.join(THIRD, "dinov2"))
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        from dinov2.hub import backbones as dino_hub
        dino = getattr(dino_hub, hub_fn)(pretrained=False)
        dino.load_state_dict(torch.load(dino_path, map_location="cpu"), strict=True)
        self.dino = dino.eval().to(self.device)
        self.patch = int(self.dino.patch_size)
        self.d = int(self.dino.embed_dim)

        # CLIP ViT-B/16 (OpenAI) text tower
        import open_clip
        # "-quickgelu": the OpenAI weights were trained with QuickGELU; plain "ViT-B-16" +
        # "openai" builds nn.GELU in open_clip 3.3 (only a warning) and shifts the text
        # embeddings the projection was trained on.
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-quickgelu",
                                                                pretrained=clip_pretrained)
        clip_model.visual = None                         # text tower only
        self.clip = clip_model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")

        # Talk2DINO text -> DINOv2 projection
        self.proj = TextProjection(512, self.d)
        self.proj.load_state_dict(torch.load(proj_path, map_location="cpu"), strict=True)
        self.proj = self.proj.eval().to(self.device)

        self._mean = torch.tensor(IMNET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMNET_STD, device=self.device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------ image side
    def _resized(self, image):
        """(H, W, 3) uint8 -> (1, 3, Hr, Wr) normalised, short side = short_side, both
        sides multiples of the patch size."""
        H, W = int(image.shape[0]), int(image.shape[1])
        s = self.short_side / min(H, W)
        Hr = max(1, round(H * s / self.patch)) * self.patch
        Wr = max(1, round(W * s / self.patch)) * self.patch
        x = torch.as_tensor(image).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        x = Fn.interpolate(x, size=(Hr, Wr), mode="bilinear", align_corners=False,
                           antialias=True)
        return (x - self._mean) / self._std

    def _dino_tokens(self, x):
        """(B, 3, h, w) -> (B, h/14, w/14, d) DINOv2 x_norm_patchtokens."""
        B, _, h, w = x.shape
        t = self.dino.forward_features(x)["x_norm_patchtokens"]
        return t.reshape(B, h // self.patch, w // self.patch, self.d)

    @staticmethod
    def _offsets(g, c, s):
        """window start offsets along one axis (patch units), last window flush with the end"""
        if g <= c:
            return [0]
        offs = list(range(0, g - c + 1, s))
        if offs[-1] != g - c:
            offs.append(g - c)
        return offs

    @torch.no_grad()
    def extract_tokens(self, image):
        """(H, W, 3) uint8 RGB -> (gh, gw, d) token grid (not normalised); a 640x480 frame
        gives 32 x 43 x 768 at the default settings."""
        x = self._resized(image)
        if self.mode == "whole":
            return self._dino_tokens(x)[0].float()
        gh, gw = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        c, s = self.crop // self.patch, max(1, self.stride // self.patch)
        acc = torch.zeros(gh, gw, self.d, device=self.device)
        cnt = torch.zeros(gh, gw, 1, device=self.device)
        wins = [(y, x0) for y in self._offsets(gh, c, s) for x0 in self._offsets(gw, c, s)]
        ch, cw = min(c, gh), min(c, gw)
        crops = torch.cat([x[..., y * self.patch:(y + ch) * self.patch,
                             x0 * self.patch:(x0 + cw) * self.patch] for y, x0 in wins])
        toks = Fn.normalize(self._dino_tokens(crops).float(), dim=-1)
        for (y, x0), t in zip(wins, toks):
            acc[y:y + ch, x0:x0 + cw] += t
            cnt[y:y + ch, x0:x0 + cw] += 1
        return acc / cnt

    @torch.no_grad()
    def extract(self, image, out_hw=None):
        """(H, W, 3) uint8 -> (h, w, d) unit per-pixel features at out_hw (default image size)."""
        tokens = self.extract_tokens(image)
        return upsample_tokens(tokens, out_hw or (int(image.shape[0]), int(image.shape[1])))

    # ------------------------------------------------------------------ text side
    @torch.no_grad()
    def text_embedding(self, prompts, templates=("a photo of a {}.",)):
        """prompts: str or list[str] -> (K, d) unit embeddings in the DINOv2 token space.
        As in Talk2DINO's build_text_embedding: the RAW CLIP text embeddings are averaged
        over the templates, projected by the MLP, then normalised."""
        if isinstance(prompts, str):
            prompts = [prompts]
        raw = []
        for p in prompts:
            toks = self.tokenizer([t.format(p) for t in templates]).to(self.device)
            raw.append(self.clip.encode_text(toks).float().mean(0))
        return Fn.normalize(self.proj(torch.stack(raw)), dim=-1)          # (K, d)


def _env(kw, key, env, default, cast=str):
    """kwarg > environment variable UAGS_T2D_<KEY> > default (the env hook lets the frozen
    scripts/check_clip2d.py, which passes no kwargs, evaluate the variants)."""
    if key in kw:
        return cast(kw.pop(key))
    return cast(os.environ.get(env, default))


def build(device="cuda", **kw):
    kw = dict(kw)
    return Talk2DINODense(
        device=device,
        backbone=_env(kw, "backbone", "UAGS_T2D_BACKBONE", "vitb"),
        short_side=_env(kw, "short_side", "UAGS_T2D_SHORT", 448, int),
        mode=_env(kw, "mode", "UAGS_T2D_MODE", "whole"),
        crop=_env(kw, "crop", "UAGS_T2D_CROP", 448, int),
        stride=_env(kw, "stride", "UAGS_T2D_STRIDE", 224, int),
        **kw)
