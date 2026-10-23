"""The candidate extractors of configs/extractor_prereg.json (usplat/extractors/radseg.py,
trident.py, proxyclip_h.py, _vfm.py), CPU only.

Without weights (always run): RADSeg's size / window rules, SCRA = RADSeg's own attention
code, SCGA properties, Trident's SAM similarity = a transcription of Trident's get_sam_sim,
the SAM attention capture = SAM's Attention.forward, registry wiring.
With weights (skipped when a file is absent): build each extractor on the CPU (fp32), token
grid shapes for 640x480 and 648x484 frames, unit norms, determinism (two calls identical),
unit text embeddings, and text/pixel alignment on a toy: a NON-dataset demo image (the RADSeg
repo's example1.jpg street scene) whose bottom-centre block is asphalt must be labelled 'road'
among {road, sky, tree, car} on >= 80% of its pixels. ProxyCLIP-H and Trident take ~1-2 min
per frame on the CPU: their weight tests run only with UAGS_SLOW_TESTS=1."""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as Fn

from usplat.extractors import radseg as R

SLOW = os.environ.get("UAGS_SLOW_TESTS") == "1"
DEMO = "/home/thanhnc/gauss_ws/third_party/radseg/RADSeg/assets/example1.jpg"


# ------------------------------------------------------------------------- no weights
def test_radseg_sizes_and_windows():
    assert R.rescale_size(480, 640, 576) == (576, 768)                  # mmcv rescale_size
    assert R.rescale_size(484, 648, 576) == (576, 771)
    assert R.preprocess_size(576, 768) == (672, 896)                     # multiples of 224
    assert R.preprocess_size(576, 771) == (672, 896)
    assert R.preprocess_size(672, 896) == (672, 896)
    assert R.preprocess_size(100, 120, 224, 336) == (336, 336)           # at least the crop
    for h, w in ((672, 896), (588, 770), (336, 336), (448, 600)):
        win = R.windows(h, w, 336, 224)
        cover = torch.zeros(h, w, dtype=torch.int32)
        for y1, x1, y2, x2 in win:
            assert y2 - y1 == min(336, h) and x2 - x1 == min(336, w)
            cover[y1:y2, x1:x2] += 1
        assert bool((cover > 0).all()), (h, w)
    assert len(R.windows(672, 896, 336, 224)) == 12


def _radseg_custom_attn(attn, x, scaling):
    """Transcription of RADSeg radseg/radseg.py SelfCorrelatingRecursiveAttn.forward /
    custom_attn @ 3fe8789 (LND layout, q_norm / k_norm identities)."""
    x = x.permute(1, 0, 2)
    num_heads = attn.num_heads
    num_tokens, bsz, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5
    q, k, v = attn.qkv(x).chunk(3, dim=-1)
    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale
    attn_weights = Fn.softmax(attn_weights, dim=-1)
    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
    attn_output = attn.proj(attn_output)
    attn_output = attn_output.view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    sim_tokens = Fn.normalize(attn_output, dim=-1)
    sim_matrix = torch.bmm(sim_tokens, sim_tokens.transpose(1, 2)) * scaling
    sim_matrix[sim_matrix < 0] = -torch.inf
    sim_matrix = Fn.softmax(sim_matrix, dim=-1)
    attn_output = torch.bmm(sim_matrix, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
    attn_output = attn.proj(attn_output)
    return attn_output.permute(1, 0, 2)


def test_scra_equals_official():
    from timm.models.vision_transformer import Attention
    torch.manual_seed(0)
    attn = Attention(48, num_heads=4, qkv_bias=True).eval()
    x = torch.randn(2, 17, 48)
    ours = R.SelfCorrelatingRecursiveAttn(attn, 10.0).eval()
    with torch.no_grad():
        a, b = _radseg_custom_attn(attn, x, 10.0), ours(x)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_scga_properties():
    torch.manual_seed(0)
    f = torch.randn(1, 8, 5, 6)
    out = R.scga(f, 10.0)
    assert out.shape == f.shape and torch.isfinite(out).all()
    # each output token is a convex combination of the input tokens: inside their hull per dim
    t = f.flatten(2)
    o = out.flatten(2)
    assert bool((o <= t.max(-1, keepdim=True).values + 1e-5).all())
    assert bool((o >= t.min(-1, keepdim=True).values - 1e-5).all())
    # identical tokens stay unchanged
    same = torch.ones(1, 8, 3, 3)
    assert torch.allclose(R.scga(same), same, atol=1e-6)


def _trident_get_sam_sim(sam_v, sam_attn, cos_fac, heads, refine_neg_cos=True):
    """Transcription of Trident open_clip/transformer.py get_sam_sim @ 7feea10."""
    sam_v = sam_v.flatten(2, 3)
    sam_v = Fn.normalize(sam_v, dim=-1)
    sam_sim = torch.einsum("b h m c, b h n c -> b h m n", sam_v, sam_v)
    sim_thresh = cos_fac * sam_sim.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
    if not refine_neg_cos:
        sim_thresh = torch.where(sim_thresh < 0.0, 0.6, sim_thresh)
    sim_thresh = torch.where(sim_thresh > 0.9, 0.9, sim_thresh)
    refined = torch.where(sam_sim > sim_thresh, sam_attn, torch.zeros_like(sam_attn))
    refined = refined.mean(dim=1)
    similarity = refined / refined.sum(dim=-1, keepdim=True)
    return similarity.unsqueeze(0).repeat(1, heads, 1, 1)


def test_trident_sam_similarity_equals_official():
    from usplat.extractors.trident import sam_similarity
    torch.manual_seed(0)
    v = torch.randn(1, 4, 5, 6, 8)
    A = torch.softmax(torch.randn(1, 4, 30, 30), -1)
    for neg in (True, False):
        ref = _trident_get_sam_sim(v, A, 1.2, 3, neg)
        got = sam_similarity(v, A, 1.2, neg)
        assert torch.allclose(ref[0, 0], got[0], atol=1e-6)
        assert torch.allclose(got.sum(-1), torch.ones(1, 30), atol=1e-5)


def test_trident_capture_attention_is_sams():
    sam_repo = "/home/thanhnc/gauss_ws/third_party/sam/segment-anything"
    if not os.path.isdir(sam_repo):
        pytest.skip("segment-anything absent")
    from usplat.extractors.trident import CaptureAttention, _sam_modules
    _, _, rel = _sam_modules()
    from segment_anything.modeling.image_encoder import Attention
    torch.manual_seed(0)
    a = Attention(32, num_heads=4, use_rel_pos=True, input_size=(6, 7))
    torch.nn.init.normal_(a.rel_pos_h, std=0.1)
    torch.nn.init.normal_(a.rel_pos_w, std=0.1)
    x = torch.randn(1, 6, 7, 32)
    cap = CaptureAttention(a, rel)
    with torch.no_grad():
        ref, got = a(x), cap(x)
    assert torch.allclose(ref, got, atol=1e-6)
    assert cap.last_attn.shape == (4, 42, 42) and cap.last_v.shape == (1, 4, 6, 7, 8)
    assert torch.allclose(cap.last_attn.sum(-1), torch.ones(4, 42), atol=1e-5)


def test_registry_names():
    """The registry imports usplat.extractors.<name>: each candidate module exposes build();
    the helper module _vfm is not an extractor."""
    import importlib
    for name in ("radseg", "trident", "proxyclip_h"):
        assert callable(importlib.import_module(f"usplat.extractors.{name}").build)
    assert not hasattr(importlib.import_module("usplat.extractors._vfm"), "build")


# ------------------------------------------------------------------------- with weights
WEIGHTS = {
    "radseg": [R.RADIO_CKPT, os.path.join(R.TEXT_DIR, "model.safetensors")],
    "proxyclip_h": ["/home/thanhnc/gauss_ws/data/weights/clip/ViT-H-14-laion2b_s32b_b79k/open_clip_model.safetensors",
                    "/home/thanhnc/gauss_ws/data/weights/dino/dino_vitbase8_pretrain.pth"],
    "trident": ["/home/thanhnc/gauss_ws/data/weights/clip/ViT-H-14-laion2b_s32b_b79k/open_clip_model.safetensors",
                "/home/thanhnc/gauss_ws/data/weights/dino/dino_vitbase16_pretrain.pth",
                "/home/thanhnc/gauss_ws/data/weights/sam/sam_vit_h_4b8939.pth"],
}
GRIDS = {"radseg": ((42, 56), (42, 56), 1152), "proxyclip_h": ((56, 75), (56, 75), 1024),
         "trident": ((48, 64), (47, 64), 1024)}


@pytest.fixture(scope="module", params=["radseg", "proxyclip_h", "trident"])
def ext(request):
    name = request.param
    if name != "radseg" and not SLOW:
        pytest.skip("set UAGS_SLOW_TESTS=1 (ViT-H on the CPU: ~1-2 min per frame)")
    missing = [p for p in WEIGHTS[name] if not os.path.isfile(p)]
    if missing:
        pytest.skip(f"weights absent: {missing}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from usplat.extractors import get_extractor
    return name, get_extractor(name, device="cpu")


def test_shapes_norms_determinism(ext):
    name, e = ext
    g480, g484, d = GRIDS[name]
    assert e.d == d
    torch.manual_seed(0)
    img = (torch.rand(480, 640, 3) * 255).to(torch.uint8)
    t1 = e.extract_tokens(img)
    t2 = e.extract_tokens(img)
    assert t1.shape == (*g480, d) and torch.isfinite(t1).all()
    assert torch.equal(t1, t2), "not deterministic"
    n = t1.norm(dim=-1)
    assert float(n.max()) <= 1.0 + 1e-4 and float(n.min()) > 0.5
    f = e.extract(img, out_hw=(240, 320))
    assert f.shape == (240, 320, d)
    assert torch.allclose(f.norm(dim=-1), torch.ones(240, 320), atol=1e-4)
    if name == "radseg":                       # the ScanNet frame size (cheap only for RADSeg)
        assert e.extract_tokens((torch.rand(484, 648, 3) * 255).to(torch.uint8)).shape == (*g484, d)
    t = e.text_embedding(["chair", "table"], templates=("a photo of a {}.", "a {} in a room."))
    assert t.shape == (2, d) and t.dtype == torch.float32
    assert torch.allclose(t.norm(dim=-1), torch.ones(2), atol=1e-4)
    assert not torch.allclose(t[0], t[1])
    print(f"PASS {name}: tokens {tuple(t1.shape)}, deterministic, unit text")


def test_text_pixel_alignment_toy(ext):
    name, e = ext
    if not os.path.isfile(DEMO):
        pytest.skip("demo image absent")
    import cv2
    frame = cv2.resize(cv2.cvtColor(cv2.imread(DEMO), cv2.COLOR_BGR2RGB), (640, 480),
                       interpolation=cv2.INTER_AREA)
    classes = ["road", "sky", "tree", "car"]
    t = e.text_embedding(classes, templates=("a photo of a {}.", "a photo of the {}."))
    pred = (e.extract(torch.from_numpy(frame)) @ t.t()).argmax(-1)
    block = pred[330:450, 220:420]                 # asphalt between the potholes, bottom centre
    frac = float((block == 0).float().mean())
    assert frac >= 0.8, f"{name}: only {frac:.2f} of the asphalt block is 'road'"
    print(f"PASS {name}: {frac * 100:.1f}% of the asphalt block labelled road")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
