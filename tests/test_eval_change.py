"""scripts/eval_change_nor.py (Table II, stale content on NeRF-Object-Removal) on the CPU, mock
backend, on a toy scene in the dataset's layout:
  a textured wall of Gaussians 3 m away and a red square "object" of Gaussians at 2 m in front
  of it. The mapping frames see the object; the update frames (the object removed) see the
  wall where it was, so the object's Gaussians change both content (feature) and depth. The
  update poses in images.txt are offset by 5 cm along z, and the scene's alignment record
  undoes it (D3). The box (bbox.txt) reaches back to the wall, so it also holds wall
  Gaussians that are still there after the removal (D2 must keep them static). A field file
  in run_mapping.py's format is built by run_mapping.py's mapping loop; a fake extractor
  stands in for LSeg.
Checks:
  * ground truth: in-box = object + in-box wall Gaussians, stale = the object only (the wall
    is within 2 cm of the after-removal depth); without the alignment the wall moves 5 cm
    and becomes "stale" too; the unobserved Gaussian is counted, not scored; the static-shell
    gate is 1 with the alignment and 0 without (the 5 cm offset); with every point at high
    confidence the any-confidence variant equals D2, and with the after-removal points
    behind the object at medium confidence D2 calls part of the in-box wall stale while the
    variant does not (update_depth_points on a copy of the scene)
  * coverage: the observed set is the Gaussians some update keyframe gives a full view weight
    (m >= a0 H W, the mock weights at the true poses); the merely seen ones are counted apart;
    FA / Rec are shares of exactly these sets, also on the common tested set; the realised
    per-view rates are fires / tests with NormLift's tests = the covered static
    Gaussian-views, and every other row tests a subset of them
  * populations: the stale / static counts and the pixel population (covered pixels,
    masks_real positives) equal an independent computation with the mock weights
  * scores: the stale Gaussians score higher (AUROC > 0.9) for ours, the depth and feature
    residuals, the dispersion and the fused row; NormLift's drop only > 0.65 here: it weighs
    a view by its pixel mass, so the wall Gaussians hidden behind the object (static by D2)
    and those at its silhouette, whose few mapping pixels showed the object, flip like stale
    ones (measured: 35 of its 60 highest static scores are hidden in-box wall Gaussians)
  * the mapping replay reproduces the saved field exactly; thresholds come from mapping-pass
    null scores (NormLift and the dispersion per history bucket, NormLift over the last
    min(V, M // 2) mapping keyframes); refusals of a field that is not mapping-only or was
    built on another map; the record names the alignment by sha256 with its fit settings
  * stale_ground_truth on hand-made points
"""
import importlib.util
import json
import math
import os
import pathlib

import numpy as np
import pytest
import torch

from usplat import baselines as B
from usplat.box3d import points_in_box
from usplat.clip_dense import upsample_tokens
from usplat.datasets import make_dataset, downscale
from usplat.feature_pca import StreamingPCA
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = pathlib.Path(__file__).resolve().parents[1]
H, W = 72, 96                                 # image; depth / masks on the 192x256 grid
K = np.array([[80.0, 0, 48.0], [0, 80.0, 36.0], [0, 0, 1.0]])
KD = K * np.array([[256 / W], [192 / H], [1.0]])
D = 8                                         # token / field size (identity PCA)
N_MAP, N_BETWEEN, N_UPD = 10, 2, 8
OFFSET_Z = 0.05                               # images.txt error of the update part [m]
OBJ, WALL, OBJ_HALF = 2.0, 3.0, 0.28          # object plane, wall plane, object half size


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Fake:
    """Stands in for LSeg: per-pixel tokens, class one-hot (red = object) + the texture."""
    d = D

    def extract_tokens(self, image):
        img = torch.as_tensor(image).float()
        tok = torch.zeros(img.shape[0], img.shape[1], D)
        red = img[..., 0] > 160
        tok[..., 0], tok[..., 1] = (~red).float(), red.float()
        tok[..., 2:5] = (img - 128.0) / 255.0 * 0.8
        return tok


def centres():
    """True camera centres (identity rotation, looking along +z)."""
    c = {i: np.array([-0.3 + 0.065 * i, 0.0, 0.0]) for i in range(N_MAP)}
    c.update({N_MAP + i: np.array([-0.3, 0.0, 0.0]) for i in range(N_BETWEEN)})
    c.update({N_MAP + N_BETWEEN + j: np.array([-0.25 + 0.07 * j, 0.04, 0.0])
              for j in range(N_UPD)})
    return c


def texture(x, y):
    return 30.0 * np.sin(2 * np.pi * x / 0.25) * np.sin(2 * np.pi * y / 0.21)


def cast(C, Km, h, w, with_object):
    """Ray cast of the true scene: depth (h, w), object hit (h, w), world x, y of the hit."""
    v, u = np.mgrid[0:h, 0:w]
    dx, dy = (u + 0.5 - Km[0, 2]) / Km[0, 0], (v + 0.5 - Km[1, 2]) / Km[1, 1]
    xo, yo = C[0] + dx * (OBJ - C[2]), C[1] + dy * (OBJ - C[2])
    hit = with_object & (np.abs(xo) <= OBJ_HALF) & (np.abs(yo) <= OBJ_HALF)
    xw, yw = C[0] + dx * (WALL - C[2]), C[1] + dy * (WALL - C[2])
    depth = np.where(hit, OBJ - C[2], WALL - C[2])
    return depth, hit, np.where(hit, xo, xw), np.where(hit, yo, yw)


def write_scene(root, seed=0):
    """The toy scene in the NeRF-Object-Removal layout; returns the true camera centres."""
    import cv2
    rng = np.random.default_rng(seed)
    for d in ("images", "depth", "confidence", "masks_real", "sparse", "boxes"):
        os.makedirs(root / d)
    n = N_MAP + N_BETWEEN + N_UPD
    json.dump({"split_start": N_MAP, "split_end": N_MAP + N_BETWEEN, "synthetic_end": -1},
              open(root / "meta.json", "w"))
    box = np.array([[x, y, z] for x in (-0.3, 0.3) for y in (-0.3, 0.3) for z in (1.93, 3.03)])
    np.savetxt(root / "boxes" / "bbox.txt", box)
    C = centres()
    cams, ims = [], []
    for i in range(n):
        obj = i < N_MAP                                      # removed from frame N_MAP on
        _, hit, x, y = cast(C[i], K, H, W, obj)
        base = np.where(hit[..., None], [220.0, 40.0, 40.0], [90.0, 120.0, 150.0])
        img = base + texture(x, y)[..., None] + rng.normal(0, 3, (H, W, 1))
        cv2.imwrite(str(root / "images" / f"{i:03d}.jpg"),
                    np.clip(img, 0, 255).astype(np.uint8)[..., ::-1])
        depth, hit_d, _, _ = cast(C[i], KD, 192, 256, obj)
        depth.astype(np.float32).tofile(root / "depth" / f"depth_{i}.bin")
        np.full((192, 256), 2, np.uint8).tofile(root / "confidence" / f"depthConfidence_{i}.bin")
        _, where, _, _ = cast(C[i], KD, 192, 256, True)      # where the object is / was
        np.save(root / "masks_real" / f"{i:03d}.npy", where)
        file_c = C[i] + (np.array([0, 0, OFFSET_Z]) if i >= N_MAP + N_BETWEEN else 0.0)
        cams.append(f"{i + 1} PINHOLE {W} {H} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}")
        ims.append(f"{i + 1} 1 0 0 0 {' '.join(map(str, -file_c))} 1 {i:03d}.jpg")
        ims.append("")
    (root / "sparse" / "cameras.txt").write_text("\n".join(cams) + "\n")
    (root / "sparse" / "images.txt").write_text("\n".join(ims) + "\n")
    T = np.eye(4)
    T[2, 3] = -OFFSET_Z
    align = root.parent / "align.json"
    align.write_text(json.dumps(dict(scene=root.name, T=T.tolist(), settings=dict(tau=0.1))))
    return C, align


def splats():
    """The map: wall Gaussians (textured colours), the object's Gaussians, one off-screen."""
    gx, gy = np.meshgrid(np.arange(-1.5, 1.501, 0.1), np.arange(-1.1, 1.101, 0.1))
    wall = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, WALL)], 1)
    ox, oy = np.meshgrid(np.arange(-0.25, 0.251, 0.0625), np.arange(-0.25, 0.251, 0.0625))
    obj = np.stack([ox.ravel(), oy.ravel(), np.full(ox.size, OBJ)], 1)
    means = np.concatenate([wall, obj, [[6.0, 6.0, WALL]]])
    col = np.concatenate([np.array([90.0, 120.0, 150.0]) + texture(wall[:, 0], wall[:, 1])[:, None],
                          np.array([220.0, 40.0, 40.0]) + texture(obj[:, 0], obj[:, 1])[:, None],
                          [[90.0, 120.0, 150.0]]]) / 255.0
    n = len(means)
    scale = np.concatenate([np.full(len(wall), 0.06), np.full(len(obj), 0.04), [0.06]])
    return dict(means=torch.tensor(means, dtype=torch.float32),
                quats=torch.tensor([1.0, 0, 0, 0]).expand(n, 4).clone(),
                scales=torch.tensor(scale, dtype=torch.float32)[:, None].repeat(1, 3),
                opacities=torch.full((n,), 0.95), colors=torch.tensor(col, dtype=torch.float32)
                .clamp(0, 1)), len(wall), len(obj)


def build(tmp):
    """Scene, map and a mapping-only field file as run_mapping.py writes it."""
    root = tmp / "005"
    C, align = write_scene(root)
    sp, n_wall, n_obj = splats()
    torch.save(sp, tmp / "map.pt")
    ec, eval_seg = _load("eval_change_nor"), _load("eval_seg")
    ds = make_dataset("nor", str(root), stride=1, part="map")
    idx = ec.select_keyframes(ds, 0.05, 5.0, 0)
    pca = StreamingPCA(D, D)
    pca.basis = torch.eye(D)
    cfg = FieldConfig()
    cfg.change_thr = float("inf")                          # mapping pass: record only
    field = DirectionalSemanticField(len(sp["means"]), D, cfg)
    views = []
    for i in idx:                                          # run_mapping.py run_sequence
        full = ds[i]
        fr = downscale(full, 1)
        h, w = fr["image"].shape[:2]
        z = pca.project(upsample_tokens(Fake().extract_tokens(full["image"]), (h, w)))
        views.append((full, fr, z.reshape(-1, D)))
    field.set_centre(torch.cat([z for _, _, z in views]).mean(0))
    bs = B.default_baselines(field)
    for full, fr, z in views:
        h, w = fr["image"].shape[:2]
        viewmat = torch.linalg.inv(full["c2w"].double()).float()
        ras = eval_seg.PinholeMockRasterizer(sp, viewmat, fr["K"], w, h)
        gz = sp["means"] @ viewmat[2, :3] + viewmat[2, 3]
        field.update_view(ras, z, num_pixels=h * w, depth_obs=fr["depth"].reshape(-1),
                          gauss_depth=gz)
    run_cfg = dict(backend="gsplat", ply=str(tmp / "map.pt"), data=str(root), dataset="nor",
                   part="map", stride=1, downscale=1, extractor="fake", feat_dim=D,
                   kf_trans=0.05, kf_rot=5.0, max_keyframes=0, use_depth=True,
                   baselines=True, update_data="", update_part="", align="",
                   channel_chunk=32, field_config="", change_quantile=0.999)
    ck = dict(field=field.state_dict(), pca=pca.state_dict(), config=run_cfg,
              timing=dict(keyframes=len(idx)), field_config=field.cfg.to_dict(),
              extractor="fake", extractor_options={}, code=ec.RM.mapping_code_state(),
              inputs=dict(ply=ec.RM.input_entry(str(tmp / "map.pt"))),
              baselines=B.baselines_state(bs), css_class_ids=None)
    torch.save(ck, tmp / "field.pt")
    return ec, eval_seg, root, align, sp, n_wall, n_obj, C


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval_change")
    ec, eval_seg, root, align, sp, n_wall, n_obj, C = build(tmp)
    ec.load_extractor = lambda cfg, device: Fake()
    argv = ["--field", str(tmp / "field.pt"), "--backend", "mock", "--device", "cpu",
            "--plain-json"]
    out = tmp / "out.json"
    ec.main(argv + ["--align", str(align), "--out", str(out)])
    rec = json.loads(out.read_text())
    out_none = tmp / "out_none.json"
    ec.main(argv + ["--align", "none", "--out", str(out_none)])
    return dict(ec=ec, eval_seg=eval_seg, rec=rec, rec_none=json.loads(out_none.read_text()),
                root=root, sp=sp, n_wall=n_wall, n_obj=n_obj, C=C, tmp=tmp, argv=argv)


def _update_weights(r):
    """Mock weights (P, N) of the update keyframes at the TRUE (= aligned) poses."""
    ds = make_dataset("nor", str(r["root"]), stride=1, part="update")
    Ws, masks = [], []
    for i in r["rec"]["views"]["update"]:
        f = ds[i - ds.start]
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor(r["C"][i], dtype=torch.float32)
        ras = r["eval_seg"].PinholeMockRasterizer(r["sp"], torch.linalg.inv(c2w), f["K"], W, H)
        Ws.append(ras._r.weights())
        masks.append(r["eval_seg"]._nearest(f["mask_real"], H, W).reshape(-1))
    return Ws, masks


def _coverage(Ws):
    """Per update keyframe the (N,) seen (m > 0) and covered (full view weight m >= a0 H W,
    the gate of our score's recent bucket with fast_min = 1) masks of the mock weights."""
    cfg = FieldConfig()
    ref = cfg.view_mass_frac * H * W
    m = [Wv.sum(0) for Wv in Ws]
    seen = [x > 1e-12 for x in m]
    cov = [s & ((x / ref).clamp(max=1.0) >= cfg.fast_min) for s, x in zip(seen, m)]
    return seen, cov


# ------------------------------------------------------------------- ground truth
def test_ground_truth_and_alignment(run):
    rec, n_wall, n_obj, sp = run["rec"], run["n_wall"], run["n_obj"], run["sp"]
    c = rec["counts"]
    in_box = points_in_box(sp["means"].numpy(), np.loadtxt(run["root"] / "boxes" / "bbox.txt"),
                           0.02)
    assert in_box[n_wall:n_wall + n_obj].all() and in_box[:n_wall].sum() == 49
    assert c["n_in_box"] == int(in_box.sum()) == n_obj + 49
    assert c["n_stale"] == n_obj and c["n_in_box_near_after"] == 49     # the wall stays static
    assert c["n_stale_observed"] == n_obj and c["n_stale_unobserved"] == 0
    assert c["after_removal"]["n_frames"] == N_UPD
    assert rec["alignment"]["T"][2][3] == -OFFSET_Z
    cn = run["rec_none"]["counts"]                 # raw poses: the wall points 5 cm off
    assert cn["n_stale"] == n_obj + 49 and cn["n_in_box_near_after"] == 0
    assert run["rec_none"]["alignment"] is None
    # D2 gate: the wall around the box is static; it matches the after-removal depth only
    # with the alignment (the raw update poses are 5 cm off along the viewing direction)
    sh, shn = rec["ground_truth"]["static_shell"], run["rec_none"]["ground_truth"]["static_shell"]
    assert sh["n_points"] > 1000 and sh["n_points"] == shn["n_points"]
    assert sh["share_near"] == 1.0 and shn["share_near"] == 0.0 and shn["share_10cm"] == 1.0
    assert sh["inner_m"] == 0.05 and sh["outer_m"] == 0.3 and sh["min_confidence"] == 2
    # every toy depth pixel has high confidence: the any-confidence variant equals D2
    a = rec["ground_truth"]["any_confidence"]
    assert a["n_stale"] == c["n_stale"] and a["n_in_box_near_after"] == 49
    g = rec["gaussian"]["rows"]
    assert all(a["rows"][r]["ap"] == g[r]["ap"] for r in rec["rows"])
    assert c["after_removal"]["n_points"] == c["after_removal"]["n_points_any_confidence"]
    # provenance: the alignment file by sha256, with the settings of its fit
    pa = rec["provenance"]["align"]
    assert pa["sha256"] == rec["alignment"]["sha256"] and pa["settings"] == dict(tau=0.1)
    assert run["rec_none"]["provenance"]["align"] is None


def test_any_confidence_variant(run, tmp_path):
    """After-removal points behind the object at medium confidence: D2 (high confidence only)
    calls in-box wall Gaussians stale that the any-confidence variant keeps static."""
    import shutil
    ec, sp = run["ec"], run["sp"]
    root = tmp_path / run["root"].name
    shutil.copytree(run["root"], root)
    for i in range(N_MAP + N_BETWEEN, N_MAP + N_BETWEEN + N_UPD):
        where = np.load(root / "masks_real" / f"{i:03d}.npy")       # where the object was
        conf = np.where(where, 1, 2).astype(np.uint8)
        conf.tofile(root / "confidence" / f"depthConfidence_{i}.bin")
    corners = np.loadtxt(root / "boxes" / "bbox.txt")
    P, C, info = ec.update_depth_points(str(root), str(run["tmp"] / "align.json"), corners, 0.04)
    assert set(np.unique(C).tolist()) == {1, 2}
    assert info["n_points_any_confidence"] == N_UPD * 192 * 256 > info["n_points"]
    means = sp["means"].numpy()
    hi = ec.stale_ground_truth(means, corners, P[C >= 2])
    anyc = ec.stale_ground_truth(means, corners, P)
    assert int(anyc["stale"].sum()) == run["n_obj"] and int(anyc["near_after"].sum()) == 49
    assert int(hi["stale"].sum()) > run["n_obj"]                   # the hidden wall: "stale"
    assert not (anyc["stale"] & ~hi["stale"]).any()                 # high only adds stale


def test_populations_match_an_independent_count(run):
    rec = run["rec"]
    Ws, masks = _update_weights(run)
    seen_v, cov_v = _coverage(Ws)
    seen, obs = torch.stack(seen_v).any(0), torch.stack(cov_v).any(0)
    N = len(seen)
    assert not seen[-1]                                          # the off-screen Gaussian
    c = rec["counts"]
    assert 0 < int((seen & ~obs).sum()) and c["n_seen"] == int(seen.sum())
    assert c["n_observed"] == int(obs.sum()) and c["n_unobserved"] == N - int(obs.sum())
    assert c["n_seen_uncovered"] == int((seen & ~obs).sum())
    assert c["n_static_observed"] == int(obs.sum()) - run["n_obj"]
    g = rec["gaussian"]["rows"]
    for r in rec["rows"]:
        assert g[r]["n"] == c["n_observed"] and g[r]["n_pos"] == run["n_obj"], r
    cov = [Wv.sum(1) >= 0.5 for Wv in Ws]
    n_pix = sum(int(m.sum()) for m in cov)
    n_pos = sum(int(mk.reshape(-1)[m].sum()) for mk, m in zip(masks, cov))
    p = rec["pixel"]
    assert p["n_pix"] == n_pix and p["n_pos"] == n_pos and p["n_frames"] == N_UPD
    assert 0 < n_pos < n_pix
    for r in rec["rows"]:
        assert p["rows"][r]["n"] == n_pix and p["rows"][r]["n_pos"] == n_pos, r
    o = rec["operating_point"]
    for r in ("ncc", "depth", "feat", "normlift", "dispersion", "ours"):
        assert math.isclose(o[r]["rec"], o[r]["n_flagged_stale"] / run["n_obj"]), r
        assert math.isclose(o[r]["fa"], o[r]["n_flagged_static"] / c["n_static_observed"]), r
    assert o["fused"]["fa"] is None and o["fused"]["rec"] is None
    assert rec["gaussian"]["common"]["n"] <= c["n_observed"]
    # every score is NaN outside the observed (covered) Gaussians: rows rank one population
    for r in rec["rows"]:
        assert g[r]["n_nan"] <= c["n_observed"], r


def test_coverage_common_set_and_realised_rates(run):
    rec = run["rec"]
    Ws, _ = _update_weights(run)
    _, cov_v = _coverage(Ws)
    n_obj, N = run["n_obj"], len(cov_v[0])
    stale = torch.zeros(N, dtype=torch.bool)
    stale[run["n_wall"]:run["n_wall"] + n_obj] = True
    obs = torch.stack(cov_v).any(0)
    covered_static = sum(int((m & obs & ~stale).sum()) for m in cov_v)
    covered_stale = sum(int((m & stale).sum()) for m in cov_v)
    o, c = rec["operating_point"], rec["counts"]
    # NormLift is tested in every covered view, every other row in a subset of them
    assert o["normlift"]["tests_static"] == covered_static
    assert o["normlift"]["tests_stale"] == covered_stale
    for r in ("ncc", "depth", "feat", "dispersion", "ours"):
        assert 0 < o[r]["tests_static"] <= covered_static, r
        assert 0 < o[r]["tests_stale"] <= covered_stale, r
    for r in ("ncc", "depth", "feat", "normlift", "dispersion", "ours"):
        for name in ("static", "stale"):
            t, f = o[r][f"tests_{name}"], o[r][f"fires_{name}"]
            assert 0 <= f <= t and math.isclose(o[r][f"rate_{name}"], f / t), (r, name)
        assert o[r]["n_flagged_static"] <= o[r]["fires_static"], r
    # the common tested set: observed Gaussians every row with a threshold tested
    assert 0 < c["n_common_tested"] <= c["n_observed"]
    assert c["n_common_tested_stale"] <= c["n_stale_observed"]
    for r in ("ncc", "depth", "feat", "normlift", "dispersion", "ours"):
        assert 0 <= o[r]["fa_common"] <= 1 and 0 <= o[r]["rec_common"] <= 1, r
    assert o["fused"]["fa_common"] is None
    # ours: the realised per-view rate on static content is of the order of 1 - q
    assert o["ours"]["rate_static"] < 0.01 and o["ours"]["rate_stale"] > 0.05


# ------------------------------------------------------------------- scores
def test_stale_gaussians_score_higher(run):
    rec = run["rec"]
    g, p = rec["gaussian"]["rows"], rec["pixel"]["rows"]
    for r in ("ours", "depth", "feat", "dispersion", "fused"):
        assert g[r]["auroc"] > 0.9, (r, g[r])
    assert g["normlift"]["auroc"] > 0.65, g["normlift"]
    for r in ("ours", "depth", "feat", "fused"):
        assert g[r]["ap"] > 0.5 and p[r]["auroc"] > 0.8, (r, g[r], p[r])
    assert g["ours"]["n_nan_pos"] == 0 and g["depth"]["n_nan_pos"] == 0
    o = rec["operating_point"]
    assert o["ours"]["rec"] > 0.5 and o["depth"]["rec"] > 0.5, o
    assert 0 <= rec["ours"]["kept"] <= rec["ours"]["kept_after_mapping"] <= 1
    assert rec["ours"]["kept"] < 0.5
    th = rec["thresholds"]
    for r in ("ncc", "depth", "feat", "normlift", "dispersion"):
        assert isinstance(th["rows"][r]["pooled"], float) and th["rows"][r]["n"] > 0, r
        assert o[r]["threshold"] == th["rows"][r]["pooled"] and o[r]["n_null"] > 0, r
    for r in ("ncc", "depth", "feat"):                         # vs the frozen map: pooled
        assert th["rows"][r]["by_history"] is None, r
    for r in ("normlift", "dispersion"):                       # per history bucket
        assert len(th["rows"][r]["by_history"]) == 6 and sum(th["rows"][r]["n_by_history"]) \
            == th["rows"][r]["n"], r
    seg = th["normlift_segment"]                               # M = 10, V = 8: L = 5
    assert seg == dict(start=5, length=5, n_mapping=N_MAP, n_update=N_UPD)
    assert th["ours"]["by_history"] is not None and len(th["ours"]["by_history"]) == 6
    assert set(rec["gaussian"]["extra"]) == {"depth_gauss/kmin", "ncc/max", "depth_render/max",
                                              "feat/max", "depth_gauss/max"}


def test_replay_and_settings(run):
    rec = run["rec"]
    rc = rec["replay_check"]
    assert rc["ok"] and rc["max_abs_dW1"] == 0 and rc["c_hist_l1_share"] == 0, rc
    assert rc["c_hist_total_saved"] > 0
    v = rec["views"]
    assert v["n_mapping"] == N_MAP and v["n_update"] == N_UPD and v["grid_hw"] == [H, W]
    assert v["update"][0] == N_MAP + N_BETWEEN
    s = rec["settings"]
    assert s["k"] == 2 and s["update_stride"] == 1 and s["change_quantile"] == 0.999
    assert rec["provenance"]["field"]["sha256"] and rec["scene"] == "005"


def test_refusals(run, tmp_path):
    ec, tmp = run["ec"], run["tmp"]
    ck = torch.load(tmp / "field.pt", weights_only=False)
    bad = dict(ck, config=dict(ck["config"], update_data=str(run["root"])))
    torch.save(bad, tmp_path / "upd.pt")
    with pytest.raises(SystemExit, match="update-data"):
        ec.main(["--field", str(tmp_path / "upd.pt"), "--backend", "mock", "--device", "cpu",
                 "--plain-json", "--align", "none", "--out", str(tmp_path / "x.json")])
    f = DirectionalSemanticField.from_state(ck["field"])
    f.calibrate_change_threshold()
    torch.save(dict(ck, field=f.state_dict()), tmp_path / "cal.pt")
    with pytest.raises(SystemExit, match="calibrated"):
        ec.main(["--field", str(tmp_path / "cal.pt"), "--backend", "mock", "--device", "cpu",
                 "--plain-json", "--align", "none", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="gsplat"):
        ec.main(["--field", str(tmp / "field.pt"), "--device", "cpu", "--align", "none",
                 "--out", str(tmp_path / "x.json")])
    other = dict(ck, inputs=dict(ply=dict(ck["inputs"]["ply"], sha256="0" * 64)))
    torch.save(other, tmp_path / "other_map.pt")               # a field of an earlier map
    with pytest.raises(SystemExit, match="built on a map with sha256 000000000000"):
        ec.main(["--field", str(tmp_path / "other_map.pt"), "--backend", "mock", "--device",
                 "cpu", "--plain-json", "--align", "none", "--out", str(tmp_path / "x.json")])


def test_stale_ground_truth_by_hand():
    ec = _load("eval_change_nor")
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    means = np.array([[0.5, 0.5, 0.5],        # inside, far from the after points: stale
                      [0.5, 0.5, 0.99],       # inside, 1 cm from an after point: static
                      [1.01, 0.5, 0.5],       # inside the 2 cm margin only: stale
                      [1.05, 0.5, 0.5]])      # outside: static
    after = np.array([[0.5, 0.5, 1.0]])
    gt = ec.stale_ground_truth(means, corners, after, inflate=0.02, near=0.02)
    assert gt["in_box"].tolist() == [True, True, True, False]
    assert gt["near_after"].tolist() == [False, True, False, False]
    assert gt["stale"].tolist() == [True, False, True, False]
    gt = ec.stale_ground_truth(means, corners, np.zeros((0, 3)))
    assert gt["stale"].tolist() == [True, True, True, False]
