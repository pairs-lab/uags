"""
Dense vision-language feature extractors behind one interface (the DenseCLIP contract):

    ext = get_extractor(name, device)
    ext.d                                   feature dimension
    ext.extract_tokens(image_hw3_uint8)     (gh, gw, d) cacheable token grid (not normalised)
    ext.extract(image, out_hw=None)         (h, w, d) unit per-pixel features
    ext.text_embedding(prompts, templates)  (K, d) unit text embeddings in the SAME space

"clearclip" is the training-free default (usplat.clip_dense.OpenCLIPDense). Every other name
maps to a module usplat/extractors/<name>.py that defines build(device) -> extractor.
Candidates are compared with the frozen validation protocol of scripts/check_clip2d.py.
"""
from __future__ import annotations
import importlib


def get_extractor(name="clearclip", device="cuda", **kw):
    if name in ("clearclip", "maskclip_value", "residual"):
        from usplat.clip_dense import OpenCLIPDense
        last = {"clearclip": "clearclip", "maskclip_value": "value", "residual": "residual"}[name]
        return OpenCLIPDense(kw.pop("model_name", "ViT-B-16"),
                             kw.pop("pretrained", "laion2b_s34b_b88k"), device=device,
                             last_block=last, **kw)
    mod = importlib.import_module(f"usplat.extractors.{name}")
    return mod.build(device=device, **kw)
