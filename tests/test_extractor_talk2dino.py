"""Talk2DINO extractor (usplat/extractors/talk2dino.py): contract shapes, unit norms, and a
real-feature semantic check (zero-shot pixel accuracy > 30% over the classes present in
room_2 frames 0 and 450, the pattern of tests/test_clip_real.py). Skipped when the Replica
renders, the DINOv2 / Talk2DINO weights, or the cached OpenAI CLIP text weights are absent."""
import json
import os
import pytest
import torch

SEQ = "/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset/room_2/Sequence_1"
VOCAB = os.path.join(os.path.dirname(__file__), "..", "configs", "vocab_replica.json")
WEIGHTS = "/home/thanhnc/gauss_ws/data/weights/talk2dino"
NEEDED = ("dinov2_vitb14_reg4_pretrain.pth", "vitb_mlp_infonce.pth")


def _extractor():
    if not all(os.path.isfile(os.path.join(WEIGHTS, f)) for f in NEEDED):
        pytest.skip("Talk2DINO / DINOv2 weights not downloaded")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from usplat.extractors import get_extractor
    dev = os.environ.get("UAGS_TEST_DEVICE", "cpu")   # GPU only via scripts/gpu_gate.sh
    try:
        return get_extractor("talk2dino", device=dev, backbone="vitb", mode="whole",
                             short_side=448)
    except Exception as e:                      # e.g. CLIP text weights not cached offline
        pytest.skip(f"Talk2DINO unavailable: {e}")


@pytest.mark.skipif(not os.path.isdir(SEQ), reason="Replica not downloaded")
def test_talk2dino_contract_and_semantics():
    ext = _extractor()
    from usplat.datasets import ReplicaSemNeRFDataset
    vocab = json.load(open(VOCAB))
    ds = ReplicaSemNeRFDataset(SEQ)
    assert ext.d == 768

    correct = total = 0
    for i in (0, 450):
        fr = ds[i]
        img, gt = fr["image"], fr["labels"]
        H, W = img.shape[:2]
        tok = ext.extract_tokens(img)
        assert tok.shape == (32, 43, ext.d), tok.shape            # 640x480 -> 602x448 / 14
        feat = ext.extract(img)
        assert feat.shape == (H, W, ext.d)
        assert torch.allclose(feat.norm(dim=-1), torch.ones(H, W, device=feat.device), atol=1e-4)
        assert ext.extract(img, out_hw=(60, 80)).shape == (60, 80, ext.d)

        present = sorted(set(torch.unique(gt).tolist()) - {0})
        text = ext.text_embedding([vocab["classes"][str(c)]["prompt"] for c in present])
        assert text.shape == (len(present), ext.d)
        assert torch.allclose(text.norm(dim=-1), torch.ones(len(present), device=text.device),
                              atol=1e-4)
        pred = torch.tensor(present)[(feat @ text.t()).argmax(-1).cpu()]
        valid = gt > 0
        correct += int((pred[valid] == gt[valid]).sum())
        total += int(valid.sum())

    # template averaging returns one unit vector per prompt
    t2 = ext.text_embedding("chair", templates=("a photo of a {}.", "a photo of a {} in a room."))
    assert t2.shape == (1, ext.d)
    assert abs(float(t2.norm()) - 1.0) < 1e-4

    acc = correct / total
    assert acc > 0.30, acc
    print(f"PASS talk2dino is semantic: zero-shot pixel accuracy {acc * 100:.1f}% "
          f"on 2 room_2 frames")


if __name__ == "__main__":
    test_talk2dino_contract_and_semantics()
