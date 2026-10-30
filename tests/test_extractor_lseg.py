"""LSeg extractor (usplat/extractors/lseg.py): contract shapes, unit norm, and a real-feature
regression test. On 2 Replica VALIDATION frames (room_2, frames 0 and 450), zero-shot pixel
accuracy over the classes present must exceed 30% (same pattern as tests/test_clip_real.py).
Skipped when the weights or the data are absent."""
import json
import os
import pytest
import torch

SEQ = "/home/thanhnc/gauss_ws/src/uags/data/replica_semnerf/Replica_Dataset/room_2/Sequence_1"
VOCAB = os.path.join(os.path.dirname(__file__), "..", "configs", "vocab_replica.json")
WEIGHTS = os.environ.get("UAGS_LSEG_WEIGHTS",
                         "/home/thanhnc/gauss_ws/src/uags/data/weights/lseg/demo_e200_state_dict.pt")

pytestmark = pytest.mark.skipif(not (os.path.isfile(WEIGHTS) and os.path.isdir(SEQ)),
                                reason="LSeg weights or Replica not downloaded")

_EXT = {}


def _ext():
    if "e" not in _EXT:
        from usplat.extractors import get_extractor
        dev = os.environ.get("UAGS_TEST_DEVICE", "cpu")   # GPU only via scripts/gpu_gate.sh
        _EXT["e"] = get_extractor("lseg", device=dev)
    return _EXT["e"]


def test_lseg_contract_shapes_and_norm():
    from usplat.datasets import ReplicaSemNeRFDataset
    ext = _ext()
    assert ext.d == 512
    img = ReplicaSemNeRFDataset(SEQ)[0]["image"]                     # (480, 640, 3) uint8
    tok = ext.extract_tokens(img)
    s = ext.token_stride
    assert tok.shape == (-(-480 // s), -(-640 // s), 512), tok.shape
    assert torch.isfinite(tok).all()
    f = ext.extract(img)
    assert f.shape == (480, 640, 512)
    assert torch.allclose(f.norm(dim=-1), torch.ones(()), atol=1e-4)
    f2 = ext.extract(img, out_hw=(120, 160))
    assert f2.shape == (120, 160, 512)
    # odd, non-multiple-of-32 input sizes must work too (padding path)
    odd = img[:237, :301]
    assert ext.extract(odd).shape == (237, 301, 512)
    t = ext.text_embedding(["chair", "table", "wall"],
                           templates=("a photo of a {}.", "a photo of a {} in a room."))
    assert t.shape == (3, 512)
    assert torch.allclose(t.norm(dim=-1), torch.ones(()), atol=1e-5)
    assert t.device == f.device


def test_lseg_is_semantic():
    from usplat.datasets import ReplicaSemNeRFDataset
    ext = _ext()
    vocab = json.load(open(VOCAB))
    ds = ReplicaSemNeRFDataset(SEQ)
    correct = total = 0
    for i in (0, 450):
        fr = ds[i]
        gt = fr["labels"]
        present = sorted(set(torch.unique(gt).tolist()) - {0})
        text = ext.text_embedding([vocab["classes"][str(c)]["prompt"] for c in present])
        pred = torch.tensor(present)[(ext.extract(fr["image"]) @ text.t()).argmax(-1).cpu()]
        valid = gt > 0
        correct += int((pred[valid] == gt[valid]).sum())
        total += int(valid.sum())
    acc = correct / total
    assert acc > 0.30, acc
    print(f"PASS LSeg is semantic: zero-shot pixel accuracy {acc * 100:.1f}% on 2 room_2 frames")


if __name__ == "__main__":
    test_lseg_contract_shapes_and_norm()
    test_lseg_is_semantic()
