"""usplat.sam_lift (LangSplat and OpenGaussian re-implemented on SAM masks) on synthetic data
(CPU, linear toy rasterizers, no SAM weights), and scripts/train_sam_baselines.py on the toy
NeRF-Object-Removal scene of tests/test_eval_change.py (mock backend):
 * SAM levels: the three multimask outputs filtered apart (IoU, stability, NMS, small regions,
   min area; mock predictor); painting order (smaller masks win), the field-grid downscale;
   the masks file round trip and its refusals
 * targets: unit mask means on the painted pixels, no target outside every mask
 * LangSplat: autoencoder round trip (unit latent / output), training lowers the masked L1
   loss, the row renders the latent and decodes it to 64-d
 * OpenGaussian: L_s and L_c on hand examples, the min-pixel filter; training separates
   masks; codebook sizes; the association on a toy whose answer is known
 * row_factory: rows of the files next to the field (moved paths), the basis-hash refusal
 * resume equals an uninterrupted run (both methods); the scripts end to end
"""
import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

from usplat import distill as D
from usplat import sam_lift as SL

REPO = pathlib.Path(__file__).resolve().parents[1]
torch.set_num_threads(1)              # tiny ops: threads only contend on a shared machine
quiet = dict(log=lambda s: None)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- SAM masks
class MockPredictor:
    """SamPredictor stand-in: every point returns 3 logit masks (one per level) of the region
    that contains it, with set IoU predictions; transform = identity."""
    device = "cpu"

    class model:
        mask_threshold = 0.0

    class transform:
        @staticmethod
        def apply_coords(p, size):
            return p

    def __init__(self, regions, iou):
        self.regions, self.iou = regions, iou       # regions[l](x, y) -> (H, W) bool

    def set_image(self, image):
        self.hw = image.shape[:2]

    def reset_image(self):
        pass

    def predict_torch(self, pts, labels, multimask_output=True, return_logits=True):
        assert multimask_output and return_logits
        B = pts.shape[0]
        out = torch.zeros(B, 3, *self.hw)
        iou = torch.zeros(B, 3)
        for b in range(B):
            x, y = pts[b, 0].tolist()
            for lv in range(3):
                m = torch.from_numpy(self.regions[lv](x, y))
                out[b, lv] = torch.where(m, 10.0, -10.0)
                iou[b, lv] = self.iou[lv](x, y)
        return out, iou, None


def test_generate_levels_filters_each_level_apart():
    pytest.importorskip("segment_anything")
    H = W = 40
    yy, xx = np.mgrid[0:H, 0:W]

    def quad(x, y):          # subpart: the point's 20 x 20 quadrant, with a 2-px island in the next
        m = (xx // 20 == int(x) // 20) & (yy // 20 == int(y) // 20)
        if int(x) < 20 and int(y) < 20:
            m = m.copy()
            m[30:32, 30:32] = True                   # small island: SAM's clean-up removes it
        return m

    def half(x, y):          # part: the left / right half
        return xx // 20 == int(x) // 20

    def tiny(x, y):          # whole: a 9 x 9 square (81 px < min area 100) or the full image
        return (xx < 9) & (yy < 9) if int(x) < 20 else np.ones((H, W), bool)
    iou = (lambda x, y: 0.95, lambda x, y: 0.95 if x < 20 else 0.5, lambda x, y: 0.99)
    pred = MockPredictor((quad, half, tiny), iou)
    lv = SL.generate_levels(pred, np.zeros((H, W, 3), np.uint8), points_per_side=8,
                            points_per_batch=16)
    s, p, w = lv
    assert len(s) == 4                                    # NMS: one mask per quadrant
    assert all(int(m.sum()) == 400 for m in s)            # the island is gone
    assert len(p) == 1 and p[0][:, :20].all() and not p[0][:, 20:].any()   # right: IoU 0.5 < 0.88
    assert len(w) == 1 and w[0].all()                     # the 81-px square is under min area
    none = SL.generate_levels(pred, np.zeros((H, W, 3), np.uint8), points_per_side=8,
                              pred_iou_thresh=0.999)
    assert [len(x) for x in none] == [0, 0, 0]
    print("PASS SAM levels: per-level IoU / stability filter, NMS, small regions, min area")


def test_paint_labels_smaller_masks_win_and_downscale():
    H, W = 8, 8
    big = np.ones((H, W), bool)
    mid = np.zeros((H, W), bool)
    mid[:, :4] = True
    small = np.zeros((H, W), bool)
    small[:2, :2] = True
    lab = SL.paint_labels([small, big, mid], (H, W))       # list order does not matter
    assert lab.dtype == np.int16
    assert (lab[:2, :2] == 2).all()                        # smallest painted last (label 2)
    assert (lab[2:, :4] == 1).all() and (lab[:, 4:] == 0).all()
    assert (SL.paint_labels([], (3, 3)) == -1).all()
    half = SL.paint_labels([small, mid], (4, 4))           # factor 2: area average >= 1/2
    assert half.shape == (4, 4) and half[0, 0] == 1 and (half[1:, :2] == 0).all()
    assert (half[:, 2:] == -1).all()
    edge = np.zeros((8, 8), bool)
    edge[0, 0] = True                                      # a quarter of a coarse pixel
    assert not SL.downscale_mask(edge, (4, 4)).any()
    print("PASS painting: larger masks first, smaller win; field-grid downscale")


def test_masks_file_round_trip(tmp_path):
    lab = np.random.default_rng(0).integers(-1, 5, (3, 3, 4, 6)).astype(np.int16)
    p = tmp_path / "a_sam.npz"
    SL.save_masks(str(p), lab, [0, 5, 10], ["map"] * 3, dict(identity="X", settings=dict(k=1)))
    m = SL.load_masks(str(p))
    assert m["identity"] == "X" and set(m["labels"]) == {("map", 0), ("map", 5), ("map", 10)}
    assert np.array_equal(m["labels"][("map", 5)], lab[1])
    q = tmp_path / "b_sam_upd.npz"
    SL.save_masks(str(q), lab[:1], [20], ["update"], dict(identity="X"))
    assert len(SL.load_masks(f"{p},{q}")["labels"]) == 4
    r = tmp_path / "c.npz"
    SL.save_masks(str(r), lab[:1], [20], ["update"], dict(identity="Y"))
    with pytest.raises(ValueError, match="different SAM settings"):
        SL.load_masks([str(p), str(r)])
    with pytest.raises(ValueError, match="another masks file"):
        SL.load_masks([str(q), str(q)])
    with pytest.raises(ValueError, match="twice"):
        SL.save_masks(str(r), lab[:2], [1, 1], ["map", "map"], {})
    print("PASS masks file: round trip, two files, refusals")


# --------------------------------------------------------------------------- targets
def test_mask_means_and_targets():
    z = Fn.normalize(torch.randn(6, 4), dim=-1)
    lab = torch.tensor([3, 3, -1, 7, 7, 7])
    comp, means = SL.mask_means(lab, z)
    assert comp.tolist() == [0, 0, -1, 1, 1, 1] and means.shape == (2, 4)
    assert torch.allclose(means[0], Fn.normalize(z[:2].sum(0), dim=0), atol=1e-6)
    assert torch.allclose(means.norm(dim=-1), torch.ones(2))
    tgt, has = SL.mask_targets(lab, z)
    assert has.tolist() == [True, True, False, True, True, True]
    assert (tgt[2] == 0).all() and torch.allclose(tgt[4], means[1])
    print("PASS targets: unit mask means, no target outside the masks")


# --------------------------------------------------------------------------- toy views
def _toy(P=240, N=60, G=6, V=8, d=16, seed=0):
    """V views of G groups: Gaussian n is in group n % G; pixel p of view v sees only the
    Gaussians of one group g(p) (weights sum to 0.9), and the SAM mask of p is its group.
    The group features are unit vectors (d)."""
    g = torch.Generator().manual_seed(seed)
    grp = torch.arange(N) % G
    feat = Fn.normalize(torch.randn(G, d, generator=g), dim=-1)
    views = []
    for v in range(V):
        pg = torch.randint(G, (P,), generator=g)
        W = torch.rand(P, N, generator=g) * (pg[:, None] == grp[None, :])
        W = 0.9 * W / W.sum(1, keepdim=True)
        z = Fn.normalize(feat[pg] + 0.05 * torch.randn(P, d, generator=g), dim=-1)
        views.append(dict(W=W, ras=(lambda buf, W=W: W @ buf), lab=pg, z=z))
    return views, grp, feat


def test_autoencoder_round_trip():
    _, _, feat = _toy(d=64)
    g = torch.Generator().manual_seed(1)
    x = Fn.normalize(feat[torch.randint(6, (600,), generator=g)] + 0.03 * torch.randn(600, 64, generator=g), dim=-1)
    ae, hist = SL.train_autoencoder(x, epochs=20, lr=3e-3, batch=64, **quiet)
    assert len(hist) == SL.AE_KEEP and not ae.training
    with torch.no_grad():
        r = ae.encode(x)
        y = ae.decode(r)
    assert r.shape == (600, 3) and torch.allclose(r.norm(dim=-1), torch.ones(600), atol=1e-5)
    assert torch.allclose(y.norm(dim=-1), torch.ones(600), atol=1e-5)
    cos = Fn.cosine_similarity(y, x).mean()
    assert cos > 0.9, cos
    dec = SL.LatentDecoder(ae)
    assert torch.allclose(dec(r), y, atol=1e-6)
    names = [type(m).__name__ for m in ae.encoder]
    assert names == ["Linear", "BatchNorm1d", "ReLU", "Linear", "BatchNorm1d", "ReLU", "Linear",
                     "BatchNorm1d", "ReLU", "Linear"]
    assert [m.out_features for m in ae.decoder if hasattr(m, "out_features")] == [8, 16, 32, 64]
    print(f"PASS autoencoder: unit latent and output, reconstruction cos {cos:.3f}")


def test_langsplat_lowers_its_loss_and_the_row_decodes(tmp_path):
    views, grp, feat = _toy(d=64)
    means = []
    for v in views:
        comp, mu = SL.mask_means(v["lab"], v["z"])
        v["comp"], v["mu"] = comp, mu
        means.append(mu)
    ae, _ = SL.train_autoencoder(torch.cat(means).repeat(20, 1), epochs=30, lr=3e-3, **quiet)
    tv = [(v["ras"], dict(lab=v["comp"].to(torch.int16), tgt=SL.encode_means(ae, v["mu"])))
          for v in views]
    res = SL.train_langsplat(tv, 60, iters=800, lr=0.02, log_every=100, min_iters=10 ** 6, **quiet)
    h = res["history"]
    assert h[-1][1] < 0.3 * h[0][1], h
    assert res["features"].shape == (60, 3) and res["iters_run"] == 800
    # the row: rendered latent -> decoder -> 64-d, close to the group's feature
    main = str(tmp_path / "scene_lseg.pt")
    pca = dict(basis=torch.eye(64))
    rec = dict(family=SL.FAMILY, method="langsplat", name="langsplat_w", features=res["features"],
               ae=ae.state_dict(), ae_dims=list(SL.AE_DIMS), identity="id-ls",
               pca_basis_sha256=D.basis_sha256(pca))
    torch.save(rec, D.out_path(main, "langsplat_w"))
    rows = SL.row_factory(dict(config=dict(out=main), pca=pca), None, "cpu")
    assert [r.name for r in rows] == ["langsplat_w"] and rows[0].buffer().shape == (60, 3)
    from usplat.seg_eval import render_row
    rr = render_row(rows[0], views[0]["ras"])
    assert rr.feat.shape == (240, 64) and torch.allclose(rr.feat.norm(dim=-1), torch.ones(240), atol=1e-5)
    cos = Fn.cosine_similarity(rr.feat, feat[views[0]["lab"]]).mean()
    assert cos > 0.8, cos
    print(f"PASS LangSplat: L1 {h[0][1]:.3f} -> {h[-1][1]:.3f}, decoded cos to the truth {cos:.3f}")


# --------------------------------------------------------------------------- OpenGaussian
def test_opengs_losses_on_hand_examples():
    F = torch.tensor([[1.0, 0], [1, 0], [0, 1], [0, 2], [5, 5]])
    lab = torch.tensor([0, 0, 1, 1, 2])
    cov = torch.tensor([True, True, True, True, True])
    ls, lc, means = SL.opengs_terms(F, lab, cov, min_px=2)        # mask 2 has 1 px: left out
    assert means.shape == (2, 2) and torch.allclose(means[1], torch.tensor([0.0, 1.5]))
    assert math.isclose(float(ls), (0.0 + 0.5) / 2, rel_tol=1e-6)  # mask means of |F - Mbar|
    d2 = 1.0 + 1.5 ** 2
    assert math.isclose(float(lc), 1.0 / (d2 + 1.0), rel_tol=1e-6)  # 2 / (2 (2 - 1)) pairs
    assert math.isclose(float(SL.opengs_loss(F, lab, cov, 0.1, 2)), float(lc + 0.1 * ls), rel_tol=1e-6)
    cov2 = torch.tensor([True, False, True, True, True])
    ls2, lc2, m2 = SL.opengs_terms(F, lab, cov2, min_px=2)          # mask 0: 1 covered px
    assert m2.shape == (1, 2) and float(lc2) == 0.0
    assert SL.opengs_terms(F, torch.full((5,), -1), cov, 1) is None
    assert SL.opengs_loss(F, lab, torch.zeros(5, dtype=torch.bool)) is None
    # the gradient pushes two close mask means apart
    G = torch.tensor([[0.1, 0.0], [0.0, 0.1]], requires_grad=True)
    SL.opengs_terms(G, torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool), 1)[1].backward()
    step = -G.grad
    assert float((step[0] - step[1]) @ (G[0] - G[1]).detach()) > 0
    print("PASS OpenGaussian losses: L_s, L_c, min-pixel filter, separation gradient")


def test_opengs_training_separates_the_masks():
    views, grp, _ = _toy(G=4, N=40)
    tv = [(v["ras"], dict(lab=v["lab"].to(torch.int16))) for v in views]
    res = SL.train_opengs(tv, 40, iters=600, lr=0.02, min_px=5, log_every=100, min_iters=10 ** 6, **quiet)
    f = res["features"]
    assert f.shape == (40, 6) and torch.allclose(f.norm(dim=-1), torch.ones(40), atol=1e-5)
    assert res["history"][-1][1] < res["history"][0][1]
    c = Fn.normalize(torch.stack([f[grp == k].mean(0) for k in range(4)]), dim=-1)
    within = torch.stack([Fn.cosine_similarity(f[grp == k], c[k][None]).mean() for k in range(4)])
    between = (c @ c.t())[~torch.eye(4, dtype=torch.bool)].max()
    assert within.min() > 0.9 and between < within.min() - 0.2, (within, between)
    print(f"PASS OpenGaussian training: within-mask cos {within.min():.3f}, between {between:.3f}")


def test_codebook_sizes():
    g = torch.Generator().manual_seed(0)
    f = Fn.normalize(torch.randn(300, 6, generator=g), dim=-1)
    xyz = torch.randn(300, 3, generator=g) * 3
    cb = SL.two_level_codebook(f, xyz, k1=8, k2=3, iters=5)
    assert cb["n_instances"] == 24 and cb["codes"].shape == (24, 6)
    assert cb["coarse"].max() < 8 and cb["instance"].min() >= 0 and cb["instance"].max() < 24
    assert torch.equal(cb["instance"] // 3, cb["coarse"])           # fine ids inside the coarse one
    for c in range(8):
        assert torch.unique(cb["instance"][cb["coarse"] == c]).numel() <= 3
    for c in range(8):                    # each Gaussian: the nearest fine centre of its cluster
        sel = cb["coarse"] == c
        near = torch.cdist(f[sel], cb["codes"][c * 3:(c + 1) * 3]).argmin(1)
        assert torch.equal(cb["instance"][sel], c * 3 + near)
    small = SL.two_level_codebook(f[:5], xyz[:5], k1=64, k2=5)       # k capped at the points
    assert small["n_instances"] == 320 and torch.unique(small["coarse"]).numel() <= 5
    c, a = SL.kmeans(torch.tensor([[0.0], [0.1], [10.0], [10.1]]), 2, seed=3)
    assert torch.allclose(c.flatten().sort().values, torch.tensor([0.05, 10.05])) and a[0] == a[1] != a[2]
    print("PASS codebook: k1 x k2 instances, fine ids inside coarse clusters, caps")


def test_association_toy_with_a_known_answer():
    """4 groups = 4 instances; masks = groups except that mask 3 also covers half of group 2's
    pixels... (group 2 matches its own mask, the larger IoU). Instance 4 has no Gaussian in view
    and instance 5 is empty: both keep the zero vector."""
    views, grp, feat = _toy(G=4, N=40, d=16)
    inst = grp.clone()
    inst[:2] = 4                                    # two Gaussians of group 0 -> instance 4 ...
    for v in views:
        v["W"][:, :2] = 0.0                         # ... that no view renders
    codes = torch.zeros(6, 6)
    codes[:4] = torch.eye(6)[:4]
    f6 = codes[inst]
    zbar = Fn.normalize(torch.randn(5, 16), dim=-1)
    tv = []
    for k, v in enumerate(views):
        lab = v["lab"].clone()
        if k % 2 == 0:                              # mask 3 grabs some of group 2's pixels
            two = torch.nonzero(lab == 2).flatten()
            lab[two[: len(two) // 3]] = 3
        tv.append(((lambda buf, W=v["W"]: W @ buf), dict(lab=lab, zbar=zbar[:4])))
    # pixels whose group 0 weights were zeroed keep alpha below 0.9: renormalize the toy so
    # every pixel is covered
    for (ras, vd), v in zip(tv, views):
        v["W"] *= 0.9 / v["W"].sum(1, keepdim=True).clamp_min(1e-9)
    out = SL.associate(tv, inst, 6, codes, f6, alpha_min=0.5, min_score=0.2)
    F = out["features"]
    for k in range(4):
        assert torch.allclose(F[k], zbar[k], atol=1e-5), k
    assert (F[4] == 0).all() and (F[5] == 0).all()
    assert out["n_matched"][:4].tolist() == [len(views)] * 4 and out["n_matched"][4:].tolist() == [0, 0]
    # a feature far from the masks' rendered features loses the match (S <= 0.2)
    far = SL.associate(tv, inst, 6, codes + 2.0, f6, alpha_min=0.5, min_score=0.2)
    assert (far["features"] == 0).all() and int(far["n_matched"].sum()) == 0
    pix = SL.instance_pixels(tv[0][0], inst, 6, torch.full((240,), 0.9),
                             torch.ones(240, dtype=torch.bool))
    assert torch.equal(pix, views[0]["lab"])        # each pixel: its group's instance
    print("PASS association: every instance gets its mask's feature, unmatched ones zero")


# --------------------------------------------------------------------------- files
def _files(main, pca, N=5):
    ae = SL.LangSplatAE()
    out = {}
    for lv in SL.LEVELS:
        name = SL.variant_name("langsplat", lv)
        out[name] = dict(family=SL.FAMILY, method="langsplat", name=name,
                         features=torch.randn(N, 3), ae=ae.eval().state_dict(),
                         ae_dims=list(SL.AE_DIMS), identity=f"id-{name}",
                         pca_basis_sha256=D.basis_sha256(pca))
    out["opengs"] = dict(family=SL.FAMILY, method="opengs", name="opengs",
                         instance=torch.tensor([0, 1, 1, 2, 0], dtype=torch.int32)[:N],
                         instance_features=Fn.normalize(torch.randn(3, 8), dim=-1),
                         identity="id-opengs", pca_basis_sha256=D.basis_sha256(pca))
    out["opengs"]["instance_features"][2] = 0.0
    for name, rec in out.items():
        torch.save(rec, D.out_path(main, name))
    return out, ae


def test_row_factory_moved_paths_and_basis_refusal(tmp_path):
    main = str(tmp_path / "scene_lseg.pt")
    basis = torch.linalg.qr(torch.randn(16, 8))[0]
    pca = dict(basis=basis)
    recs, ae = _files(main, pca)
    torch.save(dict(name="f3dgs", features=torch.randn(5, 8), pca_basis_sha256=D.basis_sha256(pca)),
               D.out_path(main, "f3dgs"))                           # distill's rows stay apart
    open(D.out_path(main, "opengs") + ".ckpt", "w").close()       # a training state is no row
    torch.save(dict(family=SL.FAMILY + "_ae"), D.out_path(main, "langsplat_ae"))
    torch.save(recs["opengs"], D.out_path(main, "opengs_upd"))      # Table IV files are not rows
    ck = dict(config=dict(out=main), pca=pca)
    rows = SL.row_factory(ck, None, "cpu")
    assert [r.name for r in rows] == ["langsplat_s", "langsplat_p", "langsplat_w", "opengs"]
    by = {r.name: r for r in rows}
    assert all(r.uncertainties() == {} for r in rows)
    r = torch.randn(4, 3)
    assert torch.allclose(by["langsplat_p"].decoder(r), ae.decode(r), atol=1e-6)
    og = by["opengs"]
    assert og.decoder is None and og.buffer().shape == (5, 8)
    assert torch.equal(og.buffer()[1], recs["opengs"]["instance_features"][1])
    assert (og.buffer()[3] == 0).all()                              # unmatched instance
    assert sorted(ck["_aux_identities"]) == sorted(f"id-{n}" for n in recs)
    with pytest.raises(ValueError, match="PCA basis"):
        SL.row_factory(dict(config=dict(out=main), pca=dict(basis=basis.flip(1))), None, "cpu")

    class FakeField:
        N = 7
    with pytest.raises(ValueError, match="Gaussians"):
        SL.row_factory(dict(config=dict(out=main), pca=pca), FakeField(), "cpu")
    assert SL.row_factory(dict(config=dict(out=str(tmp_path / "other.pt")), pca=pca), None, "cpu") == []
    assert [r.name for r in D.row_factory(dict(config=dict(out=main), pca=pca), None, "cpu")] == ["f3dgs"]
    import usplat.paths as P                                         # fields built before the move
    old = P.OLD_ROOT + "/uags_fields/scene_lseg.pt"
    saved = P._RENAMED["uags_fields"]
    P._RENAMED["uags_fields"] = str(tmp_path)
    try:
        assert len(SL.row_factory(dict(config=dict(out=old), pca=pca), None, "cpu")) == 4
    finally:
        P._RENAMED["uags_fields"] = saved
    fn = SL.render_fn(recs["opengs"])
    W = torch.rand(9, 5)
    assert torch.allclose(fn(lambda b: W @ b), W @ og.buffer())
    print("PASS sam_lift row factory: four rows, decoder, instance features, refusals, moved paths")


def test_resume_matches_uninterrupted(tmp_path):
    views, grp, _ = _toy(G=4, N=40, V=6)
    lat = Fn.normalize(torch.randn(4, 3), dim=-1)
    tl = [(v["ras"], dict(lab=v["lab"], tgt=lat)) for v in views]
    to = [(v["ras"], dict(lab=v["lab"])) for v in views]
    for fn, tv, kw in ((SL.train_langsplat, tl, {}), (SL.train_opengs, to, dict(min_px=5))):
        common = dict(lr=1e-2, log_every=100, min_iters=10 ** 6, **quiet, **kw)
        full = fn(tv, 40, iters=600, **common)
        ck = str(tmp_path / f"{fn.__name__}.ckpt")
        fn(tv, 40, iters=400, ckpt_path=ck, ckpt_every=200, **common)
        res = fn(tv, 40, iters=600, ckpt_path=ck, ckpt_every=200, **common)
        assert torch.allclose(res["features"], full["features"], atol=1e-5), fn.__name__
        assert [h[0] for h in res["history"]] == [h[0] for h in full["history"]]
    with pytest.raises(ValueError, match="another run"):
        SL.train_langsplat(tl, 41, iters=10, ckpt_path=str(tmp_path / "train_langsplat.ckpt"), **quiet)
    print("PASS resume: identical to an uninterrupted run (LangSplat, OpenGaussian)")


# --------------------------------------------------------------------------- scripts
TOY = _load("toy_eval_change_sam", REPO / "tests" / "test_eval_change.py")
TS = _load("train_sam_baselines", REPO / "scripts" / "train_sam_baselines.py")


def _toy_masks(root, path, parts=("map",)):
    """Three levels per frame of the toy scene: w = object (red) / wall, p = object / left
    and right wall, s = object / wall quadrants, at the field's grid (downscale 1)."""
    from usplat.datasets import make_dataset
    labels, index, part = [], [], []
    for pt in parts:
        ds = make_dataset("nor", str(root), stride=1, part=pt)
        for i in range(len(ds)):
            fr = ds[i]
            img = fr["image"].numpy()
            red = img[..., 0] > 160
            h, w = red.shape
            yy, xx = np.mgrid[0:h, 0:w]
            lv = [np.where(red, 0, 1 + 2 * (yy >= h // 2) + (xx >= w // 2)),
                  np.where(red, 0, 1 + (xx >= w // 2)), np.where(red, 0, 1)]
            labels.append(np.stack(lv).astype(np.int16))
            index.append(int(fr["index"]))
            part.append(pt)
    SL.save_masks(str(path), np.stack(labels), index, part,
                  dict(identity="toy-masks", settings=dict(toy=True)))


def test_train_script_end_to_end(tmp_path, monkeypatch):
    _, _, root, align, sp, n_wall, n_obj, _ = TOY.build(tmp_path)
    monkeypatch.setattr(TS, "load_extractor", lambda cfg, device: TOY.Fake())
    masks = tmp_path / "masks.npz"
    _toy_masks(root, masks)
    field = tmp_path / "field.pt"
    common = ["--field", str(field), "--masks", str(masks), "--backend", "mock", "--device", "cpu",
              "--iters", "20", "--min-iters", "1000000"]
    recs = TS.main(common + ["--method", "langsplat", "--level", "all", "--ae-epochs", "3"])
    assert [r["name"] for r in recs] == ["langsplat_s", "langsplat_p", "langsplat_w"]
    ae_path = D.out_path(str(field), "langsplat_ae")
    ae0 = torch.load(ae_path, weights_only=False)
    assert all(r["ae_identity"] == ae0["identity"] for r in recs)
    assert ae0["per_level"][0] > ae0["per_level"][2]             # more masks at the fine level
    r = recs[0]
    assert r["family"] == SL.FAMILY and r["n_frames"] == r["n_map_frames"] == 10
    assert r["iters_run"] == 20 and r["config"]["lr"] == SL.LANGSPLAT_LR
    assert r["masks"]["identity"] == "toy-masks" and r["pca_basis_sha256"]
    assert len({x["identity"] for x in recs}) == 3                 # one identity per level
    assert TS.main(common + ["--method", "langsplat", "--level", "s"]) == []   # kept
    og = TS.main(common[:-4] + ["--iters", "200", "--min-iters", "1000000", "--lr", "0.05",
                                "--log-every", "50", "--method", "opengs", "--k1", "4", "--k2", "2"])[0]
    assert og["instance"].shape == (len(sp["means"]),) and og["instance_features"].shape == (8, 8)
    assert og["config"]["lr"] == 0.05 and og["association"]["n_matched"] >= 1
    assert og["history"][-1][1] < og["history"][0][1]
    ck = torch.load(field, weights_only=False)
    rows = SL.row_factory(ck, None, "cpu", field_path=str(field))
    assert [x.name for x in rows] == SL.row_names()
    # the object's Gaussians that got a feature carry the object's mask feature (Fake channel 1)
    buf = rows[3].buffer()
    obj = buf[n_wall:n_wall + n_obj]
    hit = obj.norm(dim=-1) > 0
    assert bool(hit.any()) and float(obj[hit][:, 1].mean()) > 0.5
    with pytest.raises(SystemExit, match="lack frame"):
        bad = tmp_path / "bad.npz"
        z = SL.load_masks(str(masks))
        keys = sorted(z["labels"])[1:]
        SL.save_masks(str(bad), np.stack([z["labels"][k] for k in keys]), [k[1] for k in keys],
                      [k[0] for k in keys], dict(identity="toy-masks"))
        TS.main(["--field", str(field), "--masks", str(bad), "--backend", "mock", "--device", "cpu",
                 "--method", "opengs", "--out-dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="out-dir"):
        TS.main(common + ["--method", "opengs", "--smoke", "1"])
    print("PASS train_sam_baselines.py: three LangSplat levels on one autoencoder, OpenGaussian, rows")


@pytest.fixture(scope="module")
def nor(tmp_path_factory):
    """The toy scene with its Table III record and its dump (tests/test_dump_nor_qual.py), the
    masks of both parts, and two sam_lift files of the mapping field: an OpenGaussian file with
    one instance per Gaussian that holds the saved field's mu* (it must give the record's 'feat'
    row and the dump's 'before' counts exactly) and a LangSplat file (the decode path)."""
    TDQ = _load("toy_dump_nor_qual_sam", REPO / "tests" / "test_dump_nor_qual.py")
    t = TDQ.toy_inputs(tmp_path_factory.mktemp("sam_nor"))   # patches eval_change_nor.load_extractor
    t["dump"] = t["tmp"] / "dump"
    TDQ.DQ.main(TDQ.argv(t, t["dump"]))
    t["masks_map"], t["masks_upd"] = t["tmp"] / "m.npz", t["tmp"] / "u.npz"
    _toy_masks(t["root"], t["masks_map"], ("map",))
    _toy_masks(t["root"], t["masks_upd"], ("update",))
    from usplat.vmf_field import DirectionalSemanticField
    ck = torch.load(t["tmp"] / "field.pt", map_location="cpu", weights_only=False)
    mu = DirectionalSemanticField.from_state(ck["field"], device="cpu").mu_post().float()
    sha = D.basis_sha256(ck["pca"])
    N, d = mu.shape
    t["og"], t["ls"] = t["tmp"] / "mu_opengs.pt", t["tmp"] / "rand_langsplat_w.pt"
    torch.save(dict(family=SL.FAMILY, method="opengs", name="opengs", instance=torch.arange(N),
                    instance_features=mu.clone(), identity="toy", pca_basis_sha256=sha), t["og"])
    ae = SL.LangSplatAE((d,) + SL.AE_DIMS[1:]).eval()
    torch.save(dict(family=SL.FAMILY, method="langsplat", name="langsplat_w",
                    features=torch.randn(N, 3, generator=torch.Generator().manual_seed(0)),
                    ae=ae.state_dict(), ae_dims=[d] + list(SL.AE_DIMS[1:]), identity="toy",
                    pca_basis_sha256=sha), t["ls"])
    t["align"] = t["tmp"] / "align.json"
    return t


def test_train_script_table_mode(nor):
    """--table: the mapping and update keyframes of a Table III record (Table IV files)."""
    t = nor
    rec = TS.main(["--table", str(t["rec"]), "--masks", f"{t['masks_map']},{t['masks_upd']}",
                   "--method", "langsplat", "--level", "w", "--backend", "mock", "--device", "cpu",
                   "--iters", "20", "--min-iters", "1000000", "--ae-epochs", "2"])[0]
    views = json.loads(t["rec"].read_text())
    views = (views.get("metrics") or views)["views"]
    assert rec["name"] == "langsplat_w_upd" and rec["config"]["frames"] == "mapping+update"
    assert rec["n_map_frames"] == len(views["mapping"]) and rec["n_update_frames"] == len(views["update"])
    assert rec["table_sha256"] and TS.code_sha256(True) != TS.code_sha256(False)
    field = t["tmp"] / "field.pt"
    assert pathlib.Path(D.out_path(str(field), "langsplat_w_upd")).exists()
    assert pathlib.Path(D.out_path(str(field), "langsplat_ae_upd")).exists()
    with pytest.raises(SystemExit, match="lack frame"):
        TS.main(["--table", str(t["rec"]), "--masks", str(t["masks_map"]), "--method", "opengs",
                 "--backend", "mock", "--device", "cpu", "--iters", "5"])
    print("PASS train_sam_baselines.py --table: mapping + update keyframes")


def _wrapper(monkeypatch):
    W = _load("sam_change_nor", REPO / "scripts" / "sam_change_nor.py")
    for mod in (W.ECD, W.SB):                       # undo install() after the test
        monkeypatch.setattr(mod, "CODE", mod.CODE)
    monkeypatch.setattr(W.ECD, "load_sources", W.ECD.load_sources)
    return W


def test_change_wrapper_table_iii(nor, monkeypatch):
    t, W = nor, _wrapper(monkeypatch)
    field = str(t["tmp"] / "field.pt")
    base = ["change", "--field", field, "--backend", "mock", "--device", "cpu", "--plain-json",
            "--align", str(t["align"])]
    res = W.main(base + ["--distilled", f"{t['og']},{t['ls']}", "--reference", str(t["rec"]),
                         "--out", str(t["tmp"] / "change.json")])
    s = res["summary"]
    assert set(s) == {"feat_control", "opengs_feat", "langsplat_w_feat"}
    for k in ("g_ap", "g_auroc", "fa", "rec", "n_observed"):    # mu* per instance = the feat row
        assert s["opengs_feat"][k] == s["feat_control"][k], k
    assert abs(s["opengs_feat"]["threshold"] - s["feat_control"]["threshold"]) <= 1e-7
    assert abs(s["opengs_feat"]["p_ap"] - s["feat_control"]["p_ap"]) <= 1e-6
    assert res["control_check"]["max_abs_diff"] <= 1e-6
    assert 0.0 <= s["langsplat_w_feat"]["g_ap"] <= 1.0
    code = res["provenance"]["eval_code"]
    assert "scripts/sam_change_nor.py" in code and "usplat/sam_lift.py" in code
    assert res["provenance"]["distilled"]["opengs_feat"]["name"] == "opengs"
    bad = t["tmp"] / "bad_basis.pt"
    torch.save(dict(torch.load(t["og"], weights_only=False), pca_basis_sha256="x"), bad)
    with pytest.raises(SystemExit, match="PCA basis"):
        W.main(base + ["--distilled", str(bad), "--out", str(t["tmp"] / "x.json")])
    print("PASS sam_change_nor.py change: OpenGaussian with mu* = the feat row; LangSplat decodes")


def test_stale_wrapper_table_iv(nor, monkeypatch):
    t, W = nor, _wrapper(monkeypatch)
    res = W.main(["stale", "--table", str(t["rec"]), "--dump", str(t["dump"]), "--vocab",
                  str(t["vocab"]), "--text-cache", str(t["cache"]), "--stage", "toy", "--out",
                  str(t["tmp"] / "stale.json"), "--backend", "mock", "--device", "cpu", "--no-2d",
                  "--distilled", f"{t['og']},{t['ls']}"])
    rows = res["rows"]
    assert res["exit"] == 0 and set(rows) == {"before", "opengs", "langsplat_w"}
    assert rows["opengs"] == rows["before"]                      # the same features, rendered
    assert rows["langsplat_w"]["n_mask_cov"] == rows["before"]["n_mask_cov"]
    assert set(res["distilled"]) == {"opengs", "langsplat_w"}
    print("PASS sam_change_nor.py stale: OpenGaussian with mu* = the 'before' row")
