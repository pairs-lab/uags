"""Real-feature test for the dino.txt dense extractor (usplat/extractors/dinotxt.py): output shapes,
unit norm, and zero-shot pixel accuracy > 30% on room_2 (validation) frames 0 and 450 over the
classes present, as tests/test_clip_real.py does for ClearCLIP. Skipped without weights or data.
Runs on the GPU when UAGS_TEST_DEVICE=cuda (use scripts/gpu_gate.sh), else on the CPU."""
import json
import os
import pytest
import torch
from usplat.datasets import ReplicaSemNeRFDataset
from usplat.extractors import dinotxt

SEQ = "/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset/room_2/Sequence_1"
VOCAB = os.path.join(os.path.dirname(__file__), "..", "configs", "vocab_replica.json")


@pytest.mark.skipif(not os.path.isdir(SEQ), reason="Replica not downloaded")
@pytest.mark.skipif(not dinotxt.weights_present(), reason="dino.txt weights absent")
@pytest.mark.skipif(not os.path.isdir(dinotxt.REPO), reason="dinov2 repo not cloned")
def test_dinotxt_is_semantic():
    ext = dinotxt.build(device=os.environ.get("UAGS_TEST_DEVICE", "cpu"))
    vocab = json.load(open(VOCAB))
    ds = ReplicaSemNeRFDataset(SEQ)
    correct = total = 0
    for i in (0, 450):
        fr = ds[i]
        gt = fr["labels"]
        H, W = gt.shape
        present = sorted(set(torch.unique(gt).tolist()) - {0})
        text = ext.text_embedding([vocab["classes"][str(c)]["prompt"] for c in present])
        assert text.shape == (len(present), ext.d)
        assert torch.allclose(text.norm(dim=-1), torch.ones(len(present), device=text.device), atol=1e-4)
        tok = ext.extract_tokens(fr["image"])
        assert tok.ndim == 3 and tok.shape[-1] == ext.d and tok.is_floating_point()
        feat = ext.extract(fr["image"])
        assert feat.shape == (H, W, ext.d)
        n = feat[::37, ::41].norm(dim=-1)
        assert torch.allclose(n, torch.ones_like(n), atol=1e-4)
        assert ext.extract(fr["image"], out_hw=(60, 80)).shape == (60, 80, ext.d)
        pred = torch.tensor(present)[(feat @ text.t()).argmax(-1).cpu()]
        valid = gt > 0
        correct += int((pred[valid] == gt[valid]).sum()); total += int(valid.sum())
    acc = correct / total
    assert acc > 0.30, acc
    print(f"PASS dino.txt is semantic: zero-shot pixel accuracy {acc * 100:.1f}% on 2 room_2 frames")


if __name__ == "__main__":
    test_dinotxt_is_semantic()
