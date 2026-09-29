"""F8: the fp16 token cache, upsampled at load time, matches direct extraction
(cosine >= 0.999 per pixel). Uses the cached laion2b weights; skipped when unavailable."""
import os
import pytest
import torch
from usplat.clip_dense import OpenCLIPDense, upsample_tokens

IMG = "/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset/office_0/Sequence_1/rgb/rgb_0.png"


@pytest.mark.skipif(not os.path.exists(IMG), reason="Replica not downloaded")
def test_token_cache_matches_direct():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        clip = OpenCLIPDense("ViT-B-16", "laion2b_s34b_b88k", device="cpu")
    except Exception as e:                                     # weights not cached
        pytest.skip(f"CLIP weights unavailable: {e}")
    import cv2
    img = torch.from_numpy(cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB))
    direct = clip.extract(img, out_hw=(120, 160))
    cached = upsample_tokens(clip.extract_tokens(img).half(), (120, 160))
    cos = (direct * cached).sum(-1)
    assert cos.min() > 0.999, cos.min()
    print(f"PASS token cache: min per-pixel cosine {cos.min():.6f} (fp16 tokens vs direct)")


if __name__ == "__main__":
    test_token_cache_matches_direct()
