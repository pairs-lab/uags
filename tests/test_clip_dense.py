"""OpenCLIPDense smoke test with RANDOM weights (no download): shapes + norms.

Uses a randomly initialized ViT-B-32 so the test runs offline on CPU. Semantic
quality obviously requires pretrained='openai' (or laion) weights.
"""
import torch


def test_dense_extract_shapes():
    from usplat.clip_dense import OpenCLIPDense
    clip = OpenCLIPDense("ViT-B-32", pretrained=None, device="cpu")
    H, W = 96, 128                       # non-square, multiple of patch
    img = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8)
    f = clip.extract(img)
    assert f.shape == (H, W, clip.d), f.shape
    assert torch.allclose(f.norm(dim=-1),
                          torch.ones(H, W), atol=1e-4)

    # non-multiple-of-patch input exercises pos-embed interpolation + padding
    img2 = torch.randint(0, 255, (70, 90, 3), dtype=torch.uint8)
    f2 = clip.extract(img2)
    assert f2.shape == (70, 90, clip.d)

    t = clip.text_embedding(["a chair", "a table"])
    assert t.shape == (2, clip.d)
    assert torch.allclose(t.norm(dim=-1), torch.ones(2), atol=1e-4)
    print(f"PASS: dense CLIP shapes ({clip.d}-d), unit-norm outputs")


if __name__ == "__main__":
    test_dense_extract_shapes()
