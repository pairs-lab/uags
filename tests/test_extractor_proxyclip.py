"""ProxyCLIP extractor (usplat/extractors/proxyclip.py): output shapes, unit norm, and a real
semantic check (zero-shot pixel accuracy > 30% on room_2 frames 0 and 450, the classes present).
Skipped when Replica, the DINOv2 weights or the CLIP weights are absent. Runs on CPU (fp32,
~15 s) like tests/test_clip_real.py, so a full test run never touches the shared GPU outside
scripts/gpu_gate.sh; UAGS_TEST_DEVICE=cuda opts in to the GPU (fp16)."""
import json
import os
import pytest
import torch
from usplat.datasets import ReplicaSemNeRFDataset

SEQ = "/home/thanhnc/gauss_ws/src/uags/data/replica_semnerf/Replica_Dataset/room_2/Sequence_1"
VOCAB = os.path.join(os.path.dirname(__file__), "..", "configs", "vocab_replica.json")
DINO = "/home/thanhnc/gauss_ws/src/uags/data/weights/proxyclip/dinov2_vitb14_reg4_pretrain.pth"
DINO_CODE = "/home/thanhnc/gauss_ws/third_party/proxyclip/dinov2"


@pytest.fixture(scope="module")
def ext():
    if not (os.path.isfile(DINO) and os.path.isdir(DINO_CODE)):
        pytest.skip("DINOv2 weights or code absent")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from usplat.extractors import get_extractor
    try:
        return get_extractor("proxyclip", device=os.environ.get("UAGS_TEST_DEVICE", "cpu"),
                             pretrained="openai", vfm="dinov2_vitb14_reg", short_side=336)
    except Exception as e:
        pytest.skip(f"ProxyCLIP weights unavailable: {e}")


def test_proxyclip_shapes_and_norms(ext):
    img = (torch.rand(480, 640, 3) * 255).to(torch.uint8)
    tok = ext.extract_tokens(img)
    assert tok.shape == (24, 32, ext.d) and ext.d == 512          # 448x336 resize, 14-px grid
    assert torch.isfinite(tok).all()
    f = ext.extract(img)
    assert f.shape == (480, 640, 512)
    assert torch.allclose(f.norm(dim=-1), torch.ones(480, 640, device=f.device), atol=1e-4)
    assert ext.extract(img, out_hw=(60, 80)).shape == (60, 80, 512)
    t = ext.text_embedding(["chair", "table"], templates=("a photo of a {}.", "a {}."))
    assert t.shape == (2, 512) and t.dtype == torch.float32
    assert torch.allclose(t.norm(dim=-1), torch.ones(2, device=t.device), atol=1e-4)
    small = (torch.rand(200, 150, 3) * 255).to(torch.uint8)       # other size and aspect
    assert ext.extract(small).shape == (200, 150, 512)


@pytest.mark.skipif(not os.path.isdir(SEQ), reason="Replica not downloaded")
def test_proxyclip_is_semantic(ext):
    vocab = json.load(open(VOCAB))
    ds = ReplicaSemNeRFDataset(SEQ)
    correct = total = 0
    for i in (0, 450):
        fr = ds[i]
        gt = fr["labels"]
        present = sorted(set(torch.unique(gt).tolist()) - {0})
        text = ext.text_embedding([vocab["classes"][str(c)]["prompt"] for c in present])
        pred = torch.tensor(present)[(ext.extract(fr["image"]).cpu() @ text.cpu().t()).argmax(-1)]
        valid = gt > 0
        correct += int((pred[valid] == gt[valid]).sum()); total += int(valid.sum())
    acc = correct / total
    assert acc > 0.30, acc
    print(f"PASS ProxyCLIP is semantic: zero-shot pixel accuracy {acc * 100:.1f}% on 2 room_2 frames")
