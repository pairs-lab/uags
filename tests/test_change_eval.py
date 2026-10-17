"""Labelled change evaluation (usplat.change_eval, scripts/eval_change.py), CPU only.

  geometry   nearest_within against brute force (inclusive radius, ties -> lowest index);
             ICP recovers a 30 deg turn; object status from point clouds (static, moved,
             turned, removed, small move, few points, ignored id) and from 3RScan lists;
             3RScan ids absent from labels.ply refused; run_mapping's keyframe rule
  GT         per-Gaussian labels with the exclusive categories unobserved > hidden > far >
             object unjudged, nearest ANNOTATED vertex only; visibility from a rescan depth
             (visible, hidden, outside, no depth, seen in one of two views); alignment check
             refuses d_move <= p95
  metrics    AP / AUROC exactly equal to brute force (distinct thresholds; all pairs) with
             ties and NaN ranked last; object level against a hand computation; operating
             point, reset masses, the FA threshold and region accuracy by hand
  criterion  conditions per scene, NaN fails, wrong scene set refused
  CLI        eval on a synthetic field after an update pass (rows, missing rows, fused, oracle,
             label-free flags, thresholds, segmentation block, dumps -> thresholds ->
             criterion -> test with the validation reference) and gt-rio10 on a synthetic
             RIO10 layout (labels property, keyframes, mass floor)
  protocol   a test scene never enters as validation; thresholds only from this config's
             full validation set; the test split refuses a dirty tree, a missing or foreign
             validation reference, missing thresholds and foreign code; a compare field must
             have seen the field's features; residual state needs the update-only marker;
             the RIO10 update frames must match the field run; G3 preconditions
"""
import importlib.util
import json
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

from usplat import change_eval as CE
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_cli(name="eval_change"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ brute force
def bf_ap(s, y):
    s = np.where(np.isfinite(s), s, -np.inf)
    P = int((y == 1).sum())
    if P == 0:
        return float("nan")
    ap, prev = 0.0, 0.0
    for t in sorted(set(s.tolist()), reverse=True):
        pred = s >= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        r, p = tp / P, tp / (tp + fp)
        ap += (r - prev) * p
        prev = r
    return ap


def bf_auroc(s, y):
    s = np.where(np.isfinite(s), s, -np.inf)
    pos, neg = s[y == 1], s[y == 0]
    tot = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return tot / (len(pos) * len(neg))


def bf_nearest(q, r, radius):
    idx, dist = [], []
    for x in q:
        d = np.linalg.norm(r - x, axis=1)
        j = int(np.argmin(d))                           # lowest index among exact ties
        if d[j] <= radius:
            idx.append(j); dist.append(d[j])
        else:
            idx.append(-1); dist.append(np.inf)
    return np.array(idx), np.array(dist)


# ------------------------------------------------------------------ geometry
def test_nearest_within_matches_brute_force():
    rng = np.random.default_rng(0)
    r = rng.random((300, 3))
    q = rng.random((200, 3))
    for radius in (0.02, 0.05, 0.2):
        i1, d1 = CE.nearest_within(q, r, radius)
        i2, d2 = bf_nearest(q, r, radius)
        assert np.array_equal(i1, i2) and np.allclose(d1, d2)
    # exact ties on a grid: a query at a cell centre is equidistant to 8 corners
    g = np.stack(np.meshgrid(*[np.arange(3.0)] * 3, indexing="ij"), -1).reshape(-1, 3)
    perm = np.random.default_rng(1).permutation(len(g))
    g = g[perm]
    q = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 1.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5 + 1e-9]])
    i1, d1 = CE.nearest_within(q, g, np.sqrt(3) / 2)     # radius exactly the tie distance
    i2, d2 = bf_nearest(q, g, np.sqrt(3) / 2)
    assert np.array_equal(i1, i2) and np.allclose(d1, d2), (i1, i2)
    assert CE.nearest_within(q, np.zeros((0, 3)), 1.0)[0].tolist() == [-1] * 4


def _cloud(rng, n, center, scale=(0.3, 0.1, 0.05)):
    """An asymmetric (L-shaped) object cloud."""
    a = rng.random((n, 3)) * scale
    b = rng.random((n // 2, 3)) * (0.05, 0.25, 0.05)
    return np.concatenate([a, b]) + center


def _rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])


def test_object_status_from_points_and_transforms():
    rng = np.random.default_rng(0)
    objs = {1: np.array([0.0, 0, 0]), 2: np.array([1.0, 0, 0]), 3: np.array([0.0, 1, 0]),
            4: np.array([1.0, 1, 0]), 5: np.array([2.0, 0, 0])}
    ref = {o: _cloud(rng, 400, c) for o, c in objs.items()}
    ref[6] = rng.random((10, 3)) + 5.0                 # few points
    ref[0] = rng.random((100, 3)) + 7.0                 # unlabelled: ignored
    res = {}
    res[1] = ref[1] + rng.normal(0, 0.002, ref[1].shape)            # static (resampled noise)
    res[2] = ref[2] + np.array([0.2, 0.0, 0.0])                     # moved 20 cm
    c3 = ref[3].mean(0)
    res[3] = (ref[3] - c3) @ _rot_z(30).T + c3                      # turned 30 deg in place
    res[5] = ref[5] + np.array([0.03, 0.0, 0.0])                    # 3 cm: below d_move
    res[6], res[0] = ref[6], ref[0]                                 # (4 removed)
    cat = lambda d: (np.concatenate(list(d.values())),
                     np.concatenate([np.full(len(v), k) for k, v in d.items()]))
    rp, ri = cat(ref)
    qp, qi = cat(res)
    mo = CE.object_motion_from_points(rp, ri, qp, qi, min_points=50, ignore_ids=(0,))
    assert 0 not in mo
    assert mo[2]["shift_m"] == pytest.approx(0.2, abs=1e-9)
    assert mo[3]["rot_deg"] == pytest.approx(30.0, abs=1.0) and mo[3]["shift_m"] < 1e-9
    assert mo[1]["rot_deg"] < 1.0 and mo[4]["removed"] and mo[5]["shift_m"] == pytest.approx(0.03)
    st = CE.object_status(mo, d_move=0.10, r_deg=10.0, min_points=50)
    assert {o: s[1] for o, s in st.items()} == {1: "static", 2: "moved", 3: "rotated",
                                                 4: "removed", 5: "static", 6: "few_points"}
    assert {o: s[0] for o, s in st.items()} == {1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: -1}
    # 3RScan route: transforms about the world origin; shift of the centroid
    T = np.eye(4)
    T[:3, :3] = _rot_z(15)
    ch = dict(removed={4}, nonrigid={5}, rigid={2: (2, T)})
    mt = CE.object_motion_from_transforms(ch, rp, ri, min_points=50)
    c2 = ref[2].mean(0)
    assert mt[2]["shift_m"] == pytest.approx(np.linalg.norm(_rot_z(15) @ c2 - c2))
    assert mt[2]["rot_deg"] == pytest.approx(15.0)
    st = CE.object_status(mt, 0.10, 10.0, 50)
    assert {o: s[1] for o, s in st.items()} == {1: "static", 2: "moved", 3: "static",
                                                 4: "removed", 5: "nonrigid", 6: "few_points"}
    # V10: the 3RScan lists must index the labels.ply ids
    assert CE.rscan_id_check(ch, ri)["absent"] == []
    bad = dict(removed={4, 99}, nonrigid=set(), rigid={2: (2, T), 98: (98, T)})
    with pytest.raises(ValueError, match="2 of 4 instance ids"):
        CE.rscan_id_check(bad, ri)
    assert CE.rscan_id_check(bad, ri, max_absent_share=0.5)["absent"] == [98, 99]
    with pytest.raises(ValueError, match="V10"):                  # 0 is unlabelled, not an id
        CE.rscan_id_check(dict(removed={0}, nonrigid=set(), rigid={}), ri)


def test_select_keyframes_rule():
    def pose(t=0.0, deg=0.0):
        m = torch.eye(4)
        m[:3, :3] = torch.tensor(_rot_z(deg), dtype=torch.float32)
        m[0, 3] = t
        return m
    nan = torch.full((4, 4), float("nan"))
    c2ws = [nan, pose(), pose(0.01), pose(0.06), pose(0.07, 2.0), pose(0.07, 7.5), nan,
            pose(0.07, 7.5), pose(0.2, 7.5)]
    # first finite pose kept; then a frame is dropped iff BOTH gaps to the last kept frame are
    # below kf_trans [m] and kf_rot [deg]; non-finite poses never count
    assert CE.select_keyframes(c2ws, 0.05, 5.0) == [1, 3, 5, 8]
    assert CE.select_keyframes(c2ws, 0.0, 0.0) == [1, 2, 3, 4, 5, 7, 8]
    assert CE.select_keyframes([nan, nan], 0.05, 5.0) == []


def test_gaussian_labels_and_categories():
    ref = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    ids = np.array([1, 2, 3, 0])
    status = {1: (CE.STALE, "moved"), 2: (CE.STATIC, "static"), 3: (CE.UNJUDGED, "few_points")}
    means = torch.tensor([[0.01, 0, 0], [1.02, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [5.0, 0, 0],
                          [0.0, 0.01, 0], [1.0, 0.01, 0], [5.0, 0, 0]])
    obs = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    hid = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.bool)
    out = CE.gaussian_stale_labels(means, ref, ids, status, d_near=0.05, observed=obs, hidden=hid)
    assert out["label"].tolist() == [1, 0, -1, -1, -1, -1, -1, -1]
    names = [CE.CAT_NAMES[int(c)] for c in out["category"]]
    assert names == ["judged", "judged", "object_unjudged", "object_unjudged", "far", "hidden",
                     "unobserved", "unobserved"]
    assert out["object_id"].tolist() == [1, 2, -1, -1, -1, -1, -1, -1]
    s = out["shares"]
    assert s["n_judged"] == 2 and s["n_hidden"] == 1 and s["n_unobserved"] == 2
    assert s["share_far"] == pytest.approx(1 / 8) and s["n_stale"] == 1 and s["n_static"] == 1
    # an observed Gaussian is never hidden (observed wins over a hidden flag)
    out = CE.gaussian_stale_labels(means[:1], ref, ids, status, 0.05,
                                   observed=torch.tensor([True]), hidden=torch.tensor([True]))
    assert out["label"].tolist() == [1]
    # only ANNOTATED vertices are searched: an unlabelled vertex (id 0) next to a Gaussian
    # neither labels it nor hides a labelled one within d_near
    ref2 = np.concatenate([ref, [[1.0, 0.03, 0.0]]])
    ids2 = np.concatenate([ids, [0]])
    q = torch.tensor([[3.0, 0, 0], [1.0, 0.025, 0]])
    a = CE.gaussian_stale_labels(q, ref2, ids2, status, 0.05)
    b = CE.gaussian_stale_labels(q, ref2, ids2, status, 0.05, ignore_ids=(0,))
    assert [CE.CAT_NAMES[int(c)] for c in a["category"]] == ["object_unjudged", "object_unjudged"]
    assert [CE.CAT_NAMES[int(c)] for c in b["category"]] == ["far", "judged"]
    assert b["label"].tolist() == [-1, 0] and b["object_id"].tolist() == [-1, 2]


def test_visibility_from_depth():
    K = torch.tensor([[10.0, 0, 5.0], [0, 10.0, 4.0], [0, 0, 1]])
    c2w = torch.eye(4)
    D = torch.full((8, 10), 3.0)
    D[:, 8:] = 0.0                                       # no surface on the right
    D[0:2, 0:2] = 1.0                                    # a new object near the camera
    means = torch.tensor([[0.0, 0.0, 2.0],               # in front of the wall: visible
                          [0.0, 0.0, 3.04],              # on the wall (within the margin)
                          [0.0, 0.0, 3.5],               # behind the wall: hidden
                          [-0.9, -0.7, 2.0],             # behind the new object: hidden
                          [0.75, 0.0, 2.0],              # projects into the no-depth band
                          [9.0, 0.0, 2.0],               # outside the image
                          [0.0, 0.0, -1.0]])             # behind the camera
    v = CE.visibility_from_depth(means, [dict(c2w=c2w, K=K, depth=D, hw=(8, 10))],
                                 margin=0.05, near=0.05)
    assert v["observed"].tolist() == [True, True, False, False, False, False, False]
    assert v["hidden"].tolist() == [False, False, True, True, False, False, False]
    assert v["no_depth"].tolist() == [False, False, False, False, True, False, False]
    assert v["in_frustum"].tolist() == [True] * 5 + [False, False]
    # a second view from the side sees the hidden one: observed in >= 1 view wins
    c2 = torch.eye(4)
    c2[2, 3] = 1.0                                       # camera 1 m closer
    D2 = torch.full((8, 10), 4.0)
    v = CE.visibility_from_depth(means, [dict(c2w=c2w, K=K, depth=D, hw=(8, 10)),
                                         dict(c2w=c2, K=K, depth=D2, hw=(8, 10))])
    assert bool(v["observed"][2]) and not bool(v["hidden"][2]) and v["n_frames"] == 2
    assert v["n_views"][0].item() == 2
    # depth at half resolution: pixel coordinates are scaled to the depth grid
    v = CE.visibility_from_depth(means[:3], [dict(c2w=c2w, K=K, depth=D[::2, ::2], hw=(8, 10))])
    assert v["observed"].tolist() == [True, True, False]


def test_alignment_check():
    rng = np.random.default_rng(0)
    wall = rng.random((500, 3)) * (2, 0, 2)
    a = CE.alignment_check(wall, np.ones(500), wall + rng.normal(0, 0.003, wall.shape),
                           np.ones(500), static_ids={1})
    assert a["p95_m"] < 0.02 and a["median_m"] < a["p95_m"]
    CE.check_alignment(a["p95_m"], 0.10)
    with pytest.raises(ValueError, match="not above the alignment p95"):
        CE.check_alignment(0.12, 0.10)
    with pytest.raises(ValueError, match="no static-structure"):
        CE.alignment_check(wall, np.ones(500), wall, np.ones(500), static_ids={7})


# ------------------------------------------------------------------ metrics
def test_ranking_metrics_equal_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = int(rng.integers(5, 40))
        s = rng.integers(0, 6, n).astype(float)          # many ties
        s[rng.random(n) < 0.2] = np.nan                  # no score
        y = (rng.random(n) < 0.4).astype(int)
        if y.sum() == 0 or y.sum() == n:
            continue
        m = CE.ranking_metrics(torch.tensor(s), torch.tensor(y))
        assert m["ap"] == pytest.approx(bf_ap(s, y), abs=1e-12)
        assert m["auroc"] == pytest.approx(bf_auroc(s, y), abs=1e-12)
        assert m["n_nan"] == int(np.isnan(s).sum())
        assert m["coverage"] == pytest.approx(1 - np.isnan(s).mean())
    # order of tied elements does not matter (deterministic)
    s = np.array([1, 1, 1, 0.5, 0.5, np.nan, np.nan])
    y = np.array([1, 0, 1, 0, 1, 1, 0])
    a = CE.ranking_metrics(s, y)
    p = np.random.default_rng(0).permutation(7)
    b = CE.ranking_metrics(s[p], y[p])
    assert a["ap"] == b["ap"] and a["auroc"] == b["auroc"]
    # all-NaN: AP = positive rate, AUROC = 0.5
    m = CE.ranking_metrics([np.nan] * 4, [1, 0, 0, 1])
    assert m["ap"] == pytest.approx(0.5) and m["auroc"] == pytest.approx(0.5)


def test_object_ranking_by_hand():
    score = torch.tensor([0.9, 0.7, float("nan"), 0.1, 0.2, float("nan"), 0.95, 0.5])
    label = torch.tensor([1, 1, 1, 0, 0, 0, -1, 0])
    obj = torch.tensor([10, 10, 11, 12, 12, 13, 14, -1])
    out = CE.object_ranking(score, label, obj, [10, 11, 12, 13, 14, 15], [1, 1, 0, 0, 1, 0])
    # objects: 10 -> 0.8 (stale), 11 -> NaN (stale), 12 -> 0.15 (static), 13 -> NaN (static);
    # 14 has no judged Gaussian (its only one is unjudged), 15 none at all: both dropped
    so = np.array([0.8, np.nan, 0.15, np.nan])
    yo = np.array([1, 1, 0, 0])
    assert out["n_objects"] == 4 and out["n_dropped"] == 2
    assert out["ap"] == pytest.approx(bf_ap(so, yo)) and out["auroc"] == pytest.approx(bf_auroc(so, yo))
    with pytest.raises(ValueError, match="no object label"):
        CE.object_ranking(score, label, obj, [10], [1])


def test_operating_point_threshold_and_accuracy():
    label = torch.tensor([1, 1, 1, 0, 0, 0, 0, -1])
    flags = torch.tensor([1, 0, 1, 1, 0, 0, 0, 1], dtype=torch.bool)
    mass = torch.tensor([1.0, 1, 2, 3, 1, 1, 5, 9])
    op = CE.operating_point(flags, label, mass)
    assert op["recall"] == pytest.approx(2 / 3) and op["fa"] == pytest.approx(0.25)
    assert op["kept"] == pytest.approx(0.75)
    assert op["static_reset_mass"] == pytest.approx(3 / 10)
    assert op["stale_reset_mass"] == pytest.approx(3 / 4)
    s = torch.tensor([0.1, float("nan"), 0.3, 0.3, 0.9, 0.5, 0.2, float("nan"), 0.7, 0.4])
    for target in (0.0, 0.1, 0.2, 0.25, 0.5, 0.95):
        t = CE.fa_threshold(s, target)
        fired = CE.flags_at(s, t)
        assert fired.double().mean() <= target + 1e-12
        lower = [v for v in s.tolist() if math.isfinite(v) and v < t]
        if lower:                                         # the next lower value fires too many
            assert CE.flags_at(s, max(lower)).double().mean() > target
    assert CE.fa_threshold(s, 1.0) == float("-inf")
    f = torch.tensor([[1.0, 0], [0, 1], [0, 0], [1, 1], [0.2, 0.9]])
    t = torch.tensor([[1.0, 0], [0, 1.0]])
    pred = CE.classify(f, t)
    assert pred.tolist() == [0, 1, -1, -1, 1]             # zero feature and a tie: no decision
    acc = CE.region_accuracy(pred, torch.tensor([0, 0, 1, 1, -1]), torch.tensor([1, 1, 0, 0, 0]),
                             mass=torch.tensor([1.0, 3, 1, 1, 1]))
    assert acc["acc_changed"] == 0.5 and acc["acc_static"] == 0.0 and acc["n_static"] == 2
    assert acc["acc_changed_mass"] == pytest.approx(0.25)


def test_check_criterion():
    crit = dict(scenes=["a", "b", "c"], min_pass=2, conditions=[
        dict(left="x.v", op=">", right="y.v"), dict(left="z", op="<=", right=0.01),
        dict(left="x.v", op=">=", right="y.v", offset=0.03)])
    rec = {"a": dict(x=dict(v=0.5), y=dict(v=0.4), z=0.0),
           "b": dict(x=dict(v=0.5), y=dict(v=0.48), z=0.0),          # fails the offset
           "c": dict(x=dict(v=0.9), y=dict(v=0.1), z=float("nan"))}  # NaN fails
    r = CE.check_criterion(rec, crit)
    assert [r["scenes"][s]["ok"] for s in "abc"] == [True, False, False]
    assert r["n_pass"] == 1 and not r["passed"]
    rec["c"]["z"] = 0.005
    assert CE.check_criterion(rec, crit)["passed"]
    with pytest.raises(ValueError, match="scenes"):
        CE.check_criterion({"a": rec["a"]}, crit)


# ------------------------------------------------------------------ CLI: eval
def _field_after_update(N=60, d=8, seed=0):
    """Mapping: every Gaussian sees its prototype; calibrate; update: 0..19 switch prototype,
    50..59 are never seen in the update. Returns field, baselines, change baselines, text."""
    from usplat.baselines import NormLift, CSS
    from usplat.change_baselines import ChangeBaselines
    g = torch.Generator().manual_seed(seed)
    protos = Fn.normalize(torch.randn(4, d, generator=g), dim=-1)
    own = torch.arange(N) % 4
    new = own.clone()
    new[:20] = (own[:20] + 2) % 4
    field = DirectionalSemanticField(N, d, FieldConfig(readmit_depth=None))
    nl = NormLift.for_field(field, with_averaging=True)
    css = CSS.for_field(field, protos, class_ids=[11, 12, 13, 14])
    rep = 9                                               # pixels per Gaussian and view

    def view(proto_of, ids):
        gid = ids.repeat_interleave(rep)
        z = protos[proto_of[gid]] + 0.15 * torch.randn(len(gid), d, generator=g)
        w = torch.ones(len(gid))
        field.update_sparse(gid, w, z, num_pixels=100)
        css.update_sparse(gid, w, z)
    for _ in range(12):
        view(own, torch.arange(N))
    field.calibrate_change_threshold(0.99)
    seen = torch.arange(50)
    cb = ChangeBaselines(N)
    for _ in range(8):
        view(new, seen)
        obs = torch.zeros(N, dtype=torch.bool)
        obs[seen] = True
        stale = (torch.arange(N) < 20).double()
        r = 0.05 + 0.6 * stale + 0.05 * torch.rand(N, generator=g, dtype=torch.float64)
        cb._series("feat").push(r.float(), obs)
        cb._series("depth_gauss").push((0.3 * stale + 0.2 * torch.rand(N, generator=g,
                                                                       dtype=torch.float64)).float(), obs)
    return field, [nl, css], cb, protos, own, new


def _save_case(tmp_path, with_cb=True):
    from usplat.baselines import baselines_state
    from usplat.feature_pca import StreamingPCA
    field, bs, cb, protos, _, new = _field_after_update()
    N, d = field.N, field.d
    pca = StreamingPCA(d, d)
    pca.basis = torch.eye(d)
    ckpt = dict(field=field.state_dict(), pca=pca.state_dict(), baselines=baselines_state(bs),
                config=dict(feature_transform="", feature_transform_arg=""))
    if with_cb:                                           # update-pass state, marked as such
        ckpt.update(change_baselines=cb.state_dict(), change_baselines_scope="update")
    torch.save(ckpt, tmp_path / "field.pt")
    label = torch.zeros(N, dtype=torch.int8)
    label[:20] = 1
    label[56:] = -1                                       # a few unjudged
    olab = {k: int(k < 20) for k in range(N)}
    gt = CE.make_gt("toy", "toy", label, torch.arange(N), olab, mass=torch.ones(N) * 2,
                    target=new, classes=dict(ids=[11, 12, 13, 14], names=list("abcd")),
                    text_raw=protos)
    torch.save(gt, tmp_path / "gt.pt")
    torch.save(dict(gt, scene="toy_t"), tmp_path / "gt_t.pt")     # the same case as a test scene
    cfg = dict(schema="toy", rows=dict(residual_filter="kmin", fused=["ours_c_max", "depth_residual"],
                                       common_rows=["ours_c_max", "feature_residual"]),
               scenes=dict(val=["toy"], test=["toy_t"]),
               criterion=dict(id="c", split="test", min_pass=1, conditions=[
                   dict(left="rows.ours_c_max.gaussian.ap", op=">", right=0.5)]))
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    return field, gt


def test_eval_cli_rows_and_record(tmp_path):
    cli = _load_cli()
    field, _ = _save_case(tmp_path)
    assert int((field.n_resets[:20] >= 1).sum()) >= 15   # the update fired on stale ones
    out = tmp_path / "rec" / "eval.json"
    cli.main(["eval", "--field", str(tmp_path / "field.pt"), "--gt", str(tmp_path / "gt.pt"),
              "--config", str(tmp_path / "cfg.json"), "--scene", "toy", "--split", "val",
              "--compare-field", f"ours_noreset={tmp_path / 'field.pt'}",
              "--dump-scores", str(tmp_path / "dump_toy.pt"), "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rec["status"] == "ok" and rec["schema"].startswith("uags.runrecord")
    m = rec["metrics"]
    rows = m["rows"]
    for k in ("ours_c_max", "ours_c_last", "ours_fired", "u_ale", "1-rbar", "feature_residual",
              "depth_residual", "css_var", "yager_conflict", "fused", "oracle_gt"):
        assert rows[k]["status"] == "ok", (k, rows[k]["status"])
    for k in ("ncc", "depth_render_residual", "depth_oracle", "normlift_drop"):
        assert rows[k]["status"].startswith("missing"), k
    assert rows["depth_oracle"]["oracle"] and rows["oracle_gt"]["oracle"]
    assert rows["oracle_gt"]["gaussian"]["ap"] == pytest.approx(1.0)
    assert rows["oracle_gt"]["object"]["auroc"] == pytest.approx(1.0)
    c = rows["ours_c_max"]["gaussian"]
    assert c["ap"] > 0.9 and c["n"] == 56 and c["n_pos"] == 20
    # Gaussians 50..55 are never seen in the update: no c, ranked last, counted
    assert c["n_nan"] == 6 and c["coverage"] == pytest.approx(50 / 56)
    assert rows["ours_c_max"]["op"]["source"] == "label_free"
    assert rows["ours_c_max"]["op"]["recall"] >= 0.75 and "static_reset_mass" in rows["ours_c_max"]["op"]
    assert "op" not in rows["feature_residual"]          # no threshold given
    assert rows["feature_residual"]["common"]["rows"] == ["ours_c_max", "feature_residual"]
    assert rows["feature_residual"]["common"]["n"] == 50
    assert m["reset"]["c_coverage_judged"] == pytest.approx(50 / 56)
    seg = m["seg"]
    assert seg["status"] == "ok" and set(seg["rows"]) == {"ours", "averaging", "ours_noreset"}
    assert seg["rows"]["ours"]["acc_changed"] > seg["rows"]["averaging"]["acc_changed"]
    assert m["gt"]["n_judged"] == 56 and len(rec["inputs"]) >= 3
    # thresholds from the dump (validation), then the validation criterion
    thr_path = tmp_path / "thr.json"
    cli.main(["thresholds", "--config", str(tmp_path / "cfg.json"), "--dumps",
              str(tmp_path / "dump_toy.pt"), "--target-fa", "0.05", "--out", str(thr_path)])
    thr = json.loads(thr_path.read_text())["metrics"]["thresholds"]
    assert "feature_residual" in thr and "ours_c_max" not in thr and "oracle_gt" not in thr
    crv = tmp_path / "crit_val.json"
    cli.main(["criterion", "--config", str(tmp_path / "cfg.json"), "--split", "val",
              "--records", str(out), "--out", str(crv)])
    cv = json.loads(crv.read_text())["metrics"]
    assert not cv["criteria"]["c"]["binding"] and cv["mapping_codes"] == ["None|None"]
    # the test scene, after validation and on a clean tree (git state stubbed here)
    cli._dirty_tree = lambda: []
    out2 = tmp_path / "eval_t.json"
    test = ["eval", "--field", str(tmp_path / "field.pt"), "--gt", str(tmp_path / "gt_t.pt"),
            "--config", str(tmp_path / "cfg.json"), "--scene", "toy_t", "--split", "test",
            "--thresholds", str(thr_path)]
    cli.main(test + ["--reference", str(crv), "--out", str(out2)])
    m2 = json.loads(out2.read_text())["metrics"]
    r2 = m2["rows"]["feature_residual"]["op"]
    assert r2["source"] == "validation_threshold" and r2["fa"] <= 0.05 and r2["recall"] > 0.9
    assert m2["reference"] == str(crv)
    # thresholds only from validation dumps
    torch.save(dict(torch.load(tmp_path / "dump_toy.pt", weights_only=False), split="test"),
               tmp_path / "dump_test.pt")
    with pytest.raises(ValueError, match="validation only"):
        cli.main(["thresholds", "--config", str(tmp_path / "cfg.json"), "--dumps",
                  str(tmp_path / "dump_test.pt"), "--out", str(tmp_path / "t2.json")])
    # a finished test record is never overwritten (the test split runs once)
    with pytest.raises(SystemExit, match="finished test record"):
        cli.main(test + ["--reference", str(crv), "--out", str(out2)])
    # criterion over the per-scene records of one split, same config: validation first
    cr = tmp_path / "crit.json"
    with pytest.raises(ValueError, match="needs --reference"):
        cli.main(["criterion", "--config", str(tmp_path / "cfg.json"), "--split", "test",
                  "--records", str(out2), "--out", str(cr)])
    with pytest.raises(ValueError, match="not validation"):        # a test record as reference
        cli.main(["criterion", "--config", str(tmp_path / "cfg.json"), "--split", "test",
                  "--records", str(out2), "--reference", str(out2), "--out", str(cr)])
    cli.main(["criterion", "--config", str(tmp_path / "cfg.json"), "--split", "test",
              "--records", str(out2), "--reference", str(crv), "--out", str(cr)])
    cm = json.loads(cr.read_text())["metrics"]
    assert cm["passed"] and cm["criteria"]["c"]["binding"] and cm["reference"] == str(crv)
    with pytest.raises(ValueError, match="split"):
        cli.main(["criterion", "--config", str(tmp_path / "cfg.json"), "--split", "test",
                  "--records", str(out), "--reference", str(crv), "--force-test",
                  "--out", str(tmp_path / "c2.json")])


def test_protocol_refusals(tmp_path):
    """The test split cannot leak in through validation, and test needs a finished,
    matching validation selection on a clean tree."""
    cli = _load_cli()
    _save_case(tmp_path)
    cfg = str(tmp_path / "cfg.json")
    ev = lambda gt, split, *extra: ["eval", "--field", str(tmp_path / "field.pt"), "--gt",
                                    str(tmp_path / gt), "--config", cfg, "--split", split,
                                    "--plain-json", *extra]
    # a test scene is never scored as validation (no dump, no thresholds from it)
    with pytest.raises(ValueError, match="not a val scene"):
        cli.main(ev("gt_t.pt", "val", "--dump-scores", str(tmp_path / "leak.pt"),
                    "--out", str(tmp_path / "leak.json")))
    assert not (tmp_path / "leak.pt").exists()
    gt = torch.load(tmp_path / "gt.pt", weights_only=False)
    torch.save(dict(gt, scene=None), tmp_path / "gt_noscene.pt")
    with pytest.raises(ValueError, match="names no scene"):
        cli.main(ev("gt_noscene.pt", "val", "--scene", "toy", "--out", str(tmp_path / "x.json")))
    c = json.loads(open(cfg).read())
    (tmp_path / "overlap.json").write_text(json.dumps(dict(c, scenes=dict(val=["toy"],
                                                                          test=["toy", "b"]))))
    with pytest.raises(ValueError, match="in splits"):
        cli.main(["eval", "--field", str(tmp_path / "field.pt"), "--gt", str(tmp_path / "gt.pt"),
                  "--config", str(tmp_path / "overlap.json"), "--split", "val", "--plain-json",
                  "--out", str(tmp_path / "x.json")])
    # thresholds: dumps of this config, of validation scenes only, all of them
    cli.main(ev("gt.pt", "val", "--dump-scores", str(tmp_path / "d.pt"),
                "--out", str(tmp_path / "v.json")))
    d = torch.load(tmp_path / "d.pt", weights_only=False)
    th = lambda *dumps: cli.main(["thresholds", "--config", cfg, "--plain-json", "--dumps",
                                  *map(str, dumps), "--out", str(tmp_path / "thr.json")])
    torch.save(dict(d, config_sha256="0" * 64), tmp_path / "d_cfg.pt")
    with pytest.raises(ValueError, match="another config"):
        th(tmp_path / "d_cfg.pt")
    torch.save(dict(d, scene="toy_t"), tmp_path / "d_test.pt")
    with pytest.raises(ValueError, match="not a validation scene"):
        th(tmp_path / "d_test.pt")
    (tmp_path / "two.json").write_text(json.dumps(dict(c, scenes=dict(val=["toy", "v2"],
                                                                      test=["toy_t"]))))
    torch.save(dict(d, config_sha256=CE.load_config(tmp_path / "two.json")[1]),
               tmp_path / "d_two.pt")
    with pytest.raises(ValueError, match="pool every validation scene"):
        cli.main(["thresholds", "--config", str(tmp_path / "two.json"), "--plain-json",
                  "--dumps", str(tmp_path / "d_two.pt"), "--out", str(tmp_path / "thr2.json")])
    th(tmp_path / "d.pt")
    # a thresholds record of another config / other scenes is refused at eval
    t = json.loads((tmp_path / "thr.json").read_text())
    (tmp_path / "thr_cfg.json").write_text(json.dumps(dict(t, config_sha256="0" * 64)))
    with pytest.raises(ValueError, match="thresholds of another config"):
        cli.main(ev("gt.pt", "val", "--thresholds", str(tmp_path / "thr_cfg.json"),
                    "--out", str(tmp_path / "x.json")))
    (tmp_path / "thr_sc.json").write_text(json.dumps(dict(t, scenes=["toy_t"])))
    with pytest.raises(ValueError, match="not the validation scenes"):
        cli.main(ev("gt.pt", "val", "--thresholds", str(tmp_path / "thr_sc.json"),
                    "--out", str(tmp_path / "x.json")))
    # the test split: clean tree, finished validation reference, its code versions
    cli.main(["criterion", "--config", cfg, "--split", "val", "--plain-json", "--records",
              str(tmp_path / "v.json"), "--out", str(tmp_path / "crv.json")])
    t_out = str(tmp_path / "t.json")
    cli._dirty_tree = lambda: [" M usplat/vmf_field.py"]
    with pytest.raises(SystemExit, match="uncommitted changes"):
        cli.main(ev("gt_t.pt", "test", "--reference", str(tmp_path / "crv.json"), "--out", t_out))
    with pytest.raises(SystemExit, match="uncommitted changes"):
        cli.main(["criterion", "--config", cfg, "--split", "test", "--plain-json", "--records",
                  t_out, "--reference", str(tmp_path / "crv.json"), "--out", str(tmp_path / "c.json")])
    cli._dirty_tree = lambda: []
    with pytest.raises(SystemExit, match="--reference"):
        cli.main(ev("gt_t.pt", "test", "--out", t_out))
    with pytest.raises(SystemExit, match="not a finished validation criterion"):
        cli.main(ev("gt_t.pt", "test", "--reference", str(tmp_path / "v.json"), "--out", t_out))
    crv = json.loads((tmp_path / "crv.json").read_text())
    (tmp_path / "crv_code.json").write_text(json.dumps(dict(crv, mapping_codes=["abc|None"])))
    with pytest.raises(ValueError, match="not among the validation codes"):
        cli.main(ev("gt_t.pt", "test", "--reference", str(tmp_path / "crv_code.json"),
                    "--out", t_out))
    (tmp_path / "op.json").write_text(json.dumps(dict(c, operating_point=dict(baseline_target_fa=0.01))))
    with pytest.raises(SystemExit, match="--thresholds"):
        cli.main(["eval", "--field", str(tmp_path / "field.pt"), "--gt", str(tmp_path / "gt_t.pt"),
                  "--config", str(tmp_path / "op.json"), "--split", "test", "--plain-json",
                  "--reference", str(tmp_path / "crv.json"), "--out", t_out])
    assert not os.path.exists(t_out)                      # every refusal before the record
    out = cli.main(ev("gt_t.pt", "test", "--reference", str(tmp_path / "crv.json"),
                      "--thresholds", str(tmp_path / "thr.json"), "--out", t_out))
    assert out["split"] == "test" and out["scene"] == "toy_t"


def test_compare_field_provenance(tmp_path):
    """--compare-field must have seen the field's features; residual rows need state that is
    marked update-pass only; a RIO10 field must have ingested the ground truth's frames."""
    cli = _load_cli()
    _save_case(tmp_path)
    ck = torch.load(tmp_path / "field.pt", weights_only=False)
    log = lambda name, frames: (tmp_path / name).write_text(json.dumps(dict(frames=frames)))
    frames = [dict(i=0, index=0, n_px={"3": 10}), dict(i=1, index=5, n_px={"3": 0})]
    log("a.frames.json", frames)
    log("b.frames.json", frames)
    arg = lambda name: f"bank=b.pt,spec=s.json,scene=toy,log={tmp_path / name}"
    run = dict(feature_transform="usplat.swap:swap_factory", ply="map.pt", update_stride=5)
    main = dict(ck, config=dict(run, feature_transform_arg=arg("a.frames.json")),
                code=dict(mapping_code_sha256="m" * 64))
    torch.save(main, tmp_path / "main.pt")
    base = ["eval", "--field", str(tmp_path / "main.pt"), "--gt", str(tmp_path / "gt.pt"),
            "--config", str(tmp_path / "cfg.json"), "--split", "val", "--plain-json",
            "--out", str(tmp_path / "e.json")]

    def compare(**kw):
        c = dict(main, config=dict(main["config"], feature_transform_arg=arg("b.frames.json")))
        for k, v in kw.items():
            if k == "config":
                c["config"] = dict(c["config"], **v)
            else:
                c[k] = v
        torch.save(c, tmp_path / "cmp.pt")
        return cli.main(base + ["--compare-field", f"ours_noreset={tmp_path / 'cmp.pt'}"])
    out = compare()                                       # only the log differs: accepted
    cf = out["compare_fields"]["ours_noreset"]
    assert cf["mapping_code"] == "m" * 64 and cf["n_resets_total"] > 0
    pca = dict(ck["pca"], basis=torch.eye(ck["field"]["d"]).flip(0))
    for kw, msg in ((dict(code=dict(mapping_code_sha256="x" * 64)), "mapping code"),
                    (dict(config=dict(update_stride=10)), "update_stride"),
                    (dict(config=dict(feature_transform_arg="bank=other.pt,scene=toy,"
                                                            f"log={tmp_path / 'b.frames.json'}")),
                     "feature_transform_arg"),
                    (dict(pca=pca), "PCA basis")):
        with pytest.raises(ValueError, match=msg):
            compare(**kw)
    log("b.frames.json", frames[:1])
    with pytest.raises(ValueError, match="frames logs differ"):
        compare()
    (tmp_path / "b.frames.json").unlink()
    with pytest.raises(ValueError, match="frames log .* is missing"):
        compare()
    # residual state without the update-pass marker is never read
    torch.save(dict(ck, change_baselines_scope=None), tmp_path / "nomark.pt")
    out = cli.main(base[:2] + [str(tmp_path / "nomark.pt")] + base[3:])
    assert out["rows"]["feature_residual"]["status"].startswith(
        "missing: ChangeBaselines state not marked update-pass only")
    assert out["rows"]["fused"]["status"].startswith("missing: inputs")
    # RIO10: the update frames of the ground truth against the field's run arguments
    gt = torch.load(tmp_path / "gt.pt", weights_only=False)
    uf = dict(sequence="seq01_02", part="first_half", stride=5, kf_trans=0.05, kf_rot=5.0)
    torch.save(dict(gt, update_frames=uf), tmp_path / "gt_uf.pt")
    ok = dict(update_data="/d/seq01/seq01_02::first_half", update_stride=5, kf_trans=0.05,
              kf_rot=5.0)
    for bad, msg in ((dict(kf_trans=0.1), "kf_trans"), (dict(update_stride=1), "update_stride"),
                     (dict(update_data="/d/seq01/seq01_02"), "part"),
                     (dict(update_data="/d/seq01/seq01_01::first_half"), "sequence")):
        torch.save(dict(ck, config=dict(ok, **bad)), tmp_path / "f_uf.pt")
        with pytest.raises(ValueError, match=msg):
            cli.main(["eval", "--field", str(tmp_path / "f_uf.pt"), "--gt", str(tmp_path / "gt_uf.pt"),
                      "--config", str(tmp_path / "cfg.json"), "--split", "val", "--plain-json",
                      "--out", str(tmp_path / "u.json")])
    torch.save(dict(ck, config=ok), tmp_path / "f_uf.pt")
    cli.main(["eval", "--field", str(tmp_path / "f_uf.pt"), "--gt", str(tmp_path / "gt_uf.pt"),
              "--config", str(tmp_path / "cfg.json"), "--split", "val", "--plain-json",
              "--out", str(tmp_path / "u.json")])


def test_criterion_preconditions(tmp_path):
    """G3's precondition (plan section 6) read from another finished record: every listed
    scene must pass; without its record the criterion fails."""
    cli = _load_cli()
    _save_case(tmp_path)
    c = json.loads((tmp_path / "cfg.json").read_text())
    c["preconditions"] = dict(fa=dict(split="val", statement="field rate <= NCC rate",
                                      scenes_path="val_scenes", scene_list=["a", "b"],
                                      rows_path="rates.static_fa", min_pass="all",
                                      conditions=[dict(left="rate", op="<=",
                                                       right="baselines.ncc")]))
    (tmp_path / "pc.json").write_text(json.dumps(c))
    cli.main(["eval", "--field", str(tmp_path / "field.pt"), "--gt", str(tmp_path / "gt.pt"),
              "--config", str(tmp_path / "pc.json"), "--split", "val", "--plain-json",
              "--out", str(tmp_path / "v.json")])
    crit = lambda *extra: cli.main(["criterion", "--config", str(tmp_path / "pc.json"),
                                    "--split", "val", "--plain-json", "--records",
                                    str(tmp_path / "v.json"), *extra,
                                    "--out", str(tmp_path / "c.json")])
    r = crit()
    assert r["criteria"]["c"]["passed"] and not r["passed"]
    assert r["preconditions"]["fa"]["status"].startswith("missing")
    sel = lambda ncc_b, scenes=("a", "b"): (tmp_path / "sel.json").write_text(json.dumps(dict(
        status="ok", metrics=dict(val_scenes=list(scenes), rates=dict(static_fa=dict(
            a=dict(rate=0.001, baselines=dict(ncc=0.004)),
            b=dict(rate=0.002, baselines=dict(ncc=ncc_b))))))))
    sel(0.003)
    r = crit("--precondition", f"fa={tmp_path / 'sel.json'}")
    assert r["passed"] and r["preconditions"]["fa"]["n_pass"] == 2
    sel(0.0015)                                           # one scene above NCC: fails
    r = crit("--precondition", f"fa={tmp_path / 'sel.json'}")
    assert not r["passed"] and r["preconditions"]["fa"]["n_pass"] == 1
    sel(0.003, scenes=("a",))
    with pytest.raises(ValueError, match="needs"):
        crit("--precondition", f"fa={tmp_path / 'sel.json'}")
    with pytest.raises(ValueError, match="not in the config"):
        crit("--precondition", f"other={tmp_path / 'sel.json'}")


def test_eval_cli_refusals(tmp_path):
    cli = _load_cli()
    field, _ = _save_case(tmp_path, with_cb=False)
    # a field that never saw an update pass (no calibration) is refused
    f0 = DirectionalSemanticField(field.N, field.d, FieldConfig(readmit_depth=None))
    ck = torch.load(tmp_path / "field.pt", weights_only=False)
    torch.save(dict(ck, field=f0.state_dict()), tmp_path / "f0.pt")
    base = ["eval", "--gt", str(tmp_path / "gt.pt"), "--config", str(tmp_path / "cfg.json"),
            "--split", "val", "--plain-json"]
    with pytest.raises(ValueError, match="never calibrated"):
        cli.main(base + ["--field", str(tmp_path / "f0.pt"), "--out", str(tmp_path / "a.json")])
    with pytest.raises(ValueError, match="scene"):
        cli.main(base + ["--field", str(tmp_path / "field.pt"), "--scene", "other",
                         "--out", str(tmp_path / "b.json")])
    out = cli.main(base + ["--field", str(tmp_path / "field.pt"), "--out", str(tmp_path / "c.json")])
    assert out["rows"]["feature_residual"]["status"].startswith("missing: no ChangeBaselines")
    assert out["rows"]["fused"]["status"].startswith("missing: inputs")
    small = CE.make_gt("swap", "toy", torch.zeros(5), torch.zeros(5), {0: 0})
    torch.save(small, tmp_path / "small.pt")
    with pytest.raises(ValueError, match="Gaussians"):
        cli.main(["eval", "--gt", str(tmp_path / "small.pt"), "--config", str(tmp_path / "cfg.json"),
                  "--split", "val", "--plain-json", "--field", str(tmp_path / "field.pt"),
                  "--out", str(tmp_path / "d.json")])


# ------------------------------------------------------------------ CLI: gt-rio10
def test_gt_rio10_cli(tmp_path):
    from test_rio_loader import write_rio
    cli = _load_cli()
    rng = np.random.default_rng(0)
    plane = lambda n, x0, x1, y0, y1, z: np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n),
                                                   np.full(n, z)], -1)
    ref = {1: plane(3000, -1.2, 1.2, -1.8, 1.8, 1.5),           # wall behind everything
           2: _cloud(rng, 600, np.array([-0.6, -0.8, 0.5])),    # chair: static
           3: _cloud(rng, 600, np.array([0.2, -0.8, 0.5])),     # box: moved 20 cm
           4: _cloud(rng, 600, np.array([-0.6, 0.6, 0.5])),     # lamp: removed
           5: rng.uniform(0, 0.02, (8, 3)) + np.array([0.4, 0.6, 0.5])}   # few points
    res = {1: ref[1] + rng.normal(0, 0.002, ref[1].shape), 2: ref[2].copy(),
           3: ref[3] + np.array([0.2, 0.0, 0.0]), 5: ref[5]}
    cat = lambda d: (np.concatenate(list(d.values())),
                     np.concatenate([np.full(len(v), k) for k, v in d.items()]))
    cam = np.eye(4)
    cam[2, 3] = -3.0                                             # looks along +z at the scene
    poses = {(k, i): cam for k in (1, 2) for i in range(4)}

    def depth(k, i):                                             # rescan surface: wall at 4.5 m
        d = np.full((48, 27), 4500, np.uint16)
        d[:6, :] = 2500                                          # a new shelf in front (top rows)
        return d
    write_rio(str(tmp_path / "rio"), scene="01", n=4, poses=poses, depth_fn=depth,
              ply={1: cat(ref), 2: cat(res)}, names={1: "wall", 2: "chair", 3: "box", 4: "lamp",
                                                      5: "cup"})
    # Gaussians: on each object, one far from every vertex, one outside every image, one
    # behind the new shelf (top rows of the image)
    pick = lambda P, n: P[rng.choice(len(P), n, replace=False)]
    means = np.concatenate([pick(ref[2], 20), pick(ref[3], 20), pick(ref[4], 20), pick(ref[1], 20),
                            ref[5][:2], [[0.9, 0.0, 0.9]], [[9.0, 0.0, 0.5]],
                            [[0.0, -2.3, 1.5]]])
    N = len(means)
    torch.save(dict(means=torch.tensor(means, dtype=torch.float32),
                    quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                    scales=torch.full((N, 3), 0.01), opacities=torch.full((N,), 0.9)),
               tmp_path / "gs.pt")
    cfg = json.load(open(os.path.join(REPO, "configs", "rio10_change.json")))
    (tmp_path / "rio.json").write_text(json.dumps(cfg))
    out = tmp_path / "gt" / "gt01.pt"
    cli.main(["gt-rio10", "--root", str(tmp_path / "rio"), "--scene", "01", "--config",
              str(tmp_path / "rio.json"), "--ply", str(tmp_path / "gs.pt"), "--out", str(out)])
    gt = CE.load_gt(str(out))
    lab = gt["label"].tolist()
    assert lab[0:20] == [0] * 20                                 # chair static
    assert lab[20:40] == [1] * 20 and lab[40:60] == [1] * 20      # box moved, lamp removed
    assert all(v in (0, -1) for v in lab[60:80]) and lab[60:80].count(0) >= 15   # wall
    assert lab[80:82] == [-1, -1]                                # few points: unjudged
    cats = [CE.CAT_NAMES[int(c)] for c in gt["category"]]
    assert cats[80] == "object_unjudged" and cats[82] == "far"
    assert cats[83] == "unobserved" and cats[84] == "hidden"
    assert gt["object_status"][3][1] == "moved" and gt["object_status"][4][1] == "removed"
    assert gt["alignment"]["p95_m"] < 0.02
    assert set(int(o) for o in gt["object_ids"]) == {1, 2, 3, 4}
    rec = json.loads(open(f"{out}.json").read())
    assert rec["status"] == "ok" and rec["metrics"]["shares"]["n_hidden"] >= 1
    assert gt["target"] is None                                  # no class map: no targets
    assert gt["labels_property"] == "objectId" and rec["metrics"]["labels_property"] == "objectId"
    # the judged update frames are the ones run_mapping ingests: first half of 4, stride 5
    uf = gt["update_frames"]
    assert uf["indices"] == [0] and uf["numbers"] == [0] and uf["part"] == "first_half"
    assert (uf["sequence"], uf["stride"], uf["kf_trans"], uf["kf_rot"]) == ("seq01_02", 5, 0.05, 5.0)
    # the update-pass mass floor (mock renderer, full resolution, 8 cm splats): unobserved
    # below observed_min_mass_px
    torch.save(dict(torch.load(tmp_path / "gs.pt"), scales=torch.full((N, 3), 0.08)),
               tmp_path / "gs_big.pt")
    (tmp_path / "rio_m.json").write_text(json.dumps(dict(cfg, visibility=dict(cfg["visibility"],
                                                                              downscale=1))))
    gm = cli.main(["gt-rio10", "--root", str(tmp_path / "rio"), "--scene", "01", "--config",
                   str(tmp_path / "rio_m.json"), "--ply", str(tmp_path / "gs_big.pt"),
                   "--mass-backend", "mock", "--plain-json", "--out", str(tmp_path / "gm.pt")])
    floor = gm["mass"] >= cfg["visibility"]["observed_min_mass_px"]
    assert gm["mass"].shape == (N,) and 0 < int(floor[:80].sum()) < 80, gm["mass"][:80]
    assert gm["label"].tolist() == [v if f else -1 for v, f in zip(lab, floor.tolist())]
    # with a class map (RIO10 label -> vocabulary id) the GT carries segmentation targets:
    # static keeps its object's class, stale takes what the location shows in the rescan
    vocab = dict(templates=["a photo of a {}."], ignore=[0],
                 classes={20: "chair", 15: "box", 47: "lamp", 93: "wall"})
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    emb = torch.eye(4, 8)
    torch.save({f"lseg|a photo of a {{}}.|{n}": emb[k] for k, n in
                enumerate(["chair", "box", "lamp", "wall"])}, tmp_path / "text.pt")
    (tmp_path / "cmap.json").write_text(json.dumps({"Wall": 93, "chair": 20, "box": 15,
                                                    "lamp": 47}))
    out3 = tmp_path / "gt" / "gt01_cls.pt"
    cli.main(["gt-rio10", "--root", str(tmp_path / "rio"), "--scene", "01", "--config",
              str(tmp_path / "rio.json"), "--ply", str(tmp_path / "gs.pt"), "--class-map",
              str(tmp_path / "cmap.json"), "--vocab", str(tmp_path / "vocab.json"),
              "--text-cache", str(tmp_path / "text.pt"), "--plain-json", "--out", str(out3)])
    g3 = CE.load_gt(str(out3))
    names = g3["classes"]["names"]
    assert names == ["chair", "box", "lamp", "wall"] and torch.equal(g3["text_raw"], emb)
    t = g3["target"].tolist()
    assert t[0:20] == [names.index("chair")] * 20
    assert set(t[20:40]) <= {-1, names.index("box")} and names.index("box") in t[20:40]
    assert t[40:60] == [-1] * 20                                 # removed: nothing there now
    assert all(v == names.index("wall") for v, l in zip(t[60:80], lab[60:80]) if l == 0)
    # the two scans must name their instance ids with one vertex property (V7)
    from test_rio_loader import write_ply
    lp = tmp_path / "rio" / "models01" / "seq01_02" / "labels.ply"
    rp, ri = cat(res)
    write_ply(str(lp), rp, ri, prop="instance")
    with pytest.raises(ValueError, match="labels_id_property"):
        cli.main(["gt-rio10", "--root", str(tmp_path / "rio"), "--scene", "01", "--config",
                  str(tmp_path / "rio.json"), "--ply", str(tmp_path / "gs.pt"),
                  "--plain-json", "--out", str(tmp_path / "x.pt")])
    write_ply(str(lp), rp, ri)
    # a stricter alignment than the scan noise: d_move below p95 is refused
    cfg["object"]["d_move_cm"] = 0.1
    (tmp_path / "rio2.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="alignment p95"):
        cli.main(["gt-rio10", "--root", str(tmp_path / "rio"), "--scene", "01", "--config",
                  str(tmp_path / "rio2.json"), "--ply", str(tmp_path / "gs.pt"),
                  "--plain-json", "--out", str(tmp_path / "x.pt")])
