"""Real-feature regression test for the dense CLIP extractor (the shape test cannot catch
semantic bugs: a wrong token layout once gave 1% pixel accuracy). On 2 Replica validation
frames, zero-shot pixel accuracy over the classes present must exceed 30%."""
import json
import os
import pytest
import torch
from usplat.clip_dense import OpenCLIPDense
from usplat.datasets import ReplicaSemNeRFDataset

SEQ = "/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset/room_2/Sequence_1"
VOCAB = os.path.join(os.path.dirname(__file__), "..", "configs", "vocab_replica.json")


@pytest.mark.skipif(not os.path.isdir(SEQ), reason="Replica not downloaded")
def test_dense_clip_is_semantic():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        clip = OpenCLIPDense("ViT-B-16", "laion2b_s34b_b88k", device="cpu")
    except Exception as e:
        pytest.skip(f"CLIP weights unavailable: {e}")
    vocab = json.load(open(VOCAB))
    ds = ReplicaSemNeRFDataset(SEQ)
    correct = total = 0
    for i in (0, 450):
        fr = ds[i]
        gt = fr["labels"]
        present = sorted(set(torch.unique(gt).tolist()) - {0})
        text = clip.text_embedding([vocab["classes"][str(c)]["prompt"] for c in present])
        pred = torch.tensor(present)[(clip.extract(fr["image"]) @ text.t()).argmax(-1)]
        valid = gt > 0
        correct += int((pred[valid] == gt[valid]).sum()); total += int(valid.sum())
    acc = correct / total
    assert acc > 0.30, acc
    print(f"PASS dense CLIP is semantic: zero-shot pixel accuracy {acc * 100:.1f}% on 2 room_2 frames")


if __name__ == "__main__":
    test_dense_clip_is_semantic()
