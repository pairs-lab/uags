"""scripts/eval_views.py added analyses against brute force on toy cases.

 * (a) reducibility AUROC: among pixels covered in both maps and wrong at k, e_red (right at
   full) against e_irr, for every ranking signal and -u_ale, equals a pairwise brute force on
   the three-tier key (uncovered > undefined > finite u, written independently here as
   tuples), also with covered_only=False (the uncovered tier on top, not last); AUROC(-s) =
   1 - AUROC(s); "-u_ale" inside the defined tier (auroc_defined) against the pairwise
   brute force on the finite u_ale only;
 * (b) AUSE / random gap equals a loop brute force of the tie-aware sparsification curve; the
   oracle order gives 0, a constant score 1;
 * (c) the u_ale split: AUSE on the defined pixels equals the brute force there; on the
   undefined tier it equals the tier's random gap; counts, shares and error rates;
 * the pre-registered criterion: its constants and header text, kept / dropped / undecided
   (NaN fails, extra scenes ignored, a k-view field that records other than M = 10 mapping
   keyframes leaves its scene missing);
 * the CLI on the mock backend: the new keys are present, consistent with the old ones (the
   wrong-at-k pixels are e_red + e_irr), and `eval_views.py verdict` merges per-scene files,
   refuses duplicates and never overwrites a verdict.
"""
import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest
import torch

from usplat.seg_eval import views_cross_ause
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV = _load("eval_views")


def _outs(M=240, seed=0):
    """Two evaluate_view-like outputs of one row on the same pixels (k-view and full map),
    with ties in every score, undefined u_ale, and uncovered pixels in both maps."""
    g = torch.Generator().manual_seed(seed)
    r = lambda: torch.rand(M, generator=g)
    pix, gt = torch.arange(M), torch.randint(0, 4, (M,), generator=g)
    cov_k, cov_f = r() > 0.12, r() > 0.12
    corr_k, corr_f = (r() > 0.45) & cov_k, (r() > 0.35) & cov_f      # uncovered = wrong
    conf = 0.3 + 0.7 * torch.randint(0, 6, (M,), generator=g).float() / 6
    u_epi = torch.randint(0, 5, (M,), generator=g).float() / 5
    u_ale = torch.randint(0, 7, (M,), generator=g).float() / 10
    u_ale[r() < 0.3] = float("nan")
    ent = torch.randint(0, 4, (M,), generator=g).float()

    def out(corr, cov):
        return dict(pix=pix, gt=gt, correct=corr, covered=cov, conf=conf, entropy=ent,
                    u=dict(u_epi=u_epi, u_ale=u_ale))
    return out(corr_k, cov_k), out(corr_f, cov_f)


def _tier_key(d, name, i):
    """Independent three-tier key as a comparable tuple: (tier, value)."""
    cov = bool(d["covered"][i])
    if name == "msp":
        return (0, 1.0 - float(d["conf"][i])) if cov else (0, 1.0)
    if name == "entropy":
        return (0, float(d["entropy"][i]))
    u = float(d["u"][name][i])
    if not cov:
        return (2, 0.0)
    return (1, 0.0) if math.isnan(u) else (0, u)


def _bf_auroc(keys, labels, negate=False):
    pos = [k for k, y in zip(keys, labels) if y]
    neg = [k for k, y in zip(keys, labels) if not y]
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for a in pos:
        for b in neg:
            hi = (a < b) if negate else (a > b)
            s += 1.0 if hi else (0.5 if a == b else 0.0)
    return s / (len(pos) * len(neg))


def _bf_sparsification(err, keys, bins=20):
    """Loop brute force: drop the most uncertain first, equal keys averaged (tie block)."""
    M = len(err)
    order = sorted(range(M), key=lambda i: keys[i], reverse=True)
    e_model, i = [], 0
    while i < M:
        j = i
        while j < M and keys[order[j]] == keys[order[i]]:
            j += 1
        blk = [err[order[t]] for t in range(i, j)]
        e_model += [sum(blk) / len(blk)] * (j - i)
        i = j
    e_oracle = sorted(err, reverse=True)
    ks = [int(b * M / bins) for b in range(bins)]
    curve = lambda es: [sum(es[k:]) / (M - k) for k in ks]
    model, oracle = curve(e_model), curve(e_oracle)
    ebar = sum(err) / M
    return (sum(m - o for m, o in zip(model, oracle)) / bins,
            sum(ebar - o for o in oracle) / bins)


# ------------------------------------------------------------------------------- (a)
@pytest.mark.parametrize("covered_only", [True, False])
def test_reducibility_auroc_brute_force(covered_only):
    ok, of = _outs(seed=1)
    res = EV.reducibility_auroc(ok, of, covered_only=covered_only)
    M = ok["gt"].numel()
    both = ok["covered"] & of["covered"]
    keep = both if covered_only else torch.ones(M, dtype=torch.bool)
    idx = [i for i in range(M) if keep[i] and not ok["correct"][i]]
    lab = [bool(of["correct"][i]) for i in idx]
    assert res["n_pixels"] == len(idx) and res["n_e_red"] == sum(lab)
    assert res["n_e_irr"] == len(idx) - sum(lab) and 0 < sum(lab) < len(idx)
    assert set(res["auroc"]) == {"msp", "entropy", "u_epi", "u_ale", "-u_ale"}
    for name in ("msp", "entropy", "u_epi", "u_ale"):
        keys = [_tier_key(ok, name, i) for i in idx]
        assert abs(res["auroc"][name] - _bf_auroc(keys, lab)) < 1e-12, name
    keys = [_tier_key(ok, "u_ale", i) for i in idx]
    assert abs(res["auroc"]["-u_ale"] - _bf_auroc(keys, lab, negate=True)) < 1e-12
    assert abs(res["auroc"]["-u_ale"] + res["auroc"]["u_ale"] - 1.0) < 1e-12
    dfn = [i for i in idx if ok["covered"][i] and not math.isnan(float(ok["u"]["u_ale"][i]))]
    want = _bf_auroc([float(ok["u"]["u_ale"][i]) for i in dfn],
                     [bool(of["correct"][i]) for i in dfn], negate=True)
    assert set(res["auroc_defined"]) == {"-u_ale"}
    assert abs(res["auroc_defined"]["-u_ale"] - want) < 1e-12 and len(dfn) > 0
    n_und = sum(1 for i in idx if ok["covered"][i] and math.isnan(float(ok["u"]["u_ale"][i])))
    assert res["n_undefined"] == {"u_epi": 0, "u_ale": n_und} and n_und > 0
    if not covered_only:                    # the uncovered tier exists and is ranked on top
        assert any(not ok["covered"][i] for i in idx)
    print(f"PASS reducibility AUROC (covered_only={covered_only}): {len(idx)} wrong pixels, "
          + ", ".join(f"{k} {v:.3f}" for k, v in res["auroc"].items()))


def test_reducibility_auroc_degenerate():
    """No e_irr (or no wrong pixel at all): AUROC is NaN, counted, never an error."""
    ok, of = _outs(seed=2)
    of = dict(of, correct=torch.ones_like(of["correct"]), covered=torch.ones_like(of["covered"]))
    res = EV.reducibility_auroc(ok, of)
    assert res["n_e_irr"] == 0 and all(math.isnan(v) for v in res["auroc"].values())
    assert math.isnan(res["auroc_defined"]["-u_ale"])
    ok2 = dict(ok, correct=torch.ones_like(ok["correct"]), covered=torch.ones_like(ok["covered"]))
    res = EV.reducibility_auroc(ok2, of)
    assert res["n_pixels"] == 0 and all(math.isnan(v) for v in res["auroc"].values())
    assert math.isnan(res["auroc_defined"]["-u_ale"])         # no defined wrong pixel


# ------------------------------------------------------------------------------- (b)
def test_ause_over_random_gap_brute_force():
    ok, of = _outs(seed=3)
    v = views_cross_ause(ok, of, ause_bins=10)
    ex = EV.extra_analyses(ok, of, v["ause"], ause_bins=10)
    keep = [i for i in range(ok["gt"].numel()) if ok["covered"][i] and of["covered"][i]]
    tg = {"e_red": [float((not ok["correct"][i]) and bool(of["correct"][i])) for i in keep],
          "e_irr": [float((not ok["correct"][i]) and not of["correct"][i]) for i in keep]}
    for t, err in tg.items():
        for name in ("msp", "entropy", "u_epi", "u_ale"):
            a, gap = _bf_sparsification(err, [_tier_key(ok, name, i) for i in keep], bins=10)
            assert abs(v["ause"][name][t] - a) < 1e-9, (name, t)
            assert abs(ex["random_gap"][t] - gap) < 1e-12, t
            assert abs(ex["ause_norm"][name][t] - a / gap) < 1e-9, (name, t)
    # oracle order -> 0, constant score -> 1 (ties are a random order)
    e = torch.tensor(tg["e_irr"], dtype=torch.float64)
    from usplat.evaluation import sparsification_curve
    gaps = EV.random_gaps({"e_irr": e}, 10)
    orc = sparsification_curve(e, e, num_bins=10)["ause"]
    cst = sparsification_curve(e, torch.zeros_like(e), num_bins=10)["ause"]
    n = EV.normalised({"oracle": {"e_irr": orc}, "const": {"e_irr": cst}}, gaps)
    assert abs(n["oracle"]["e_irr"]) < 1e-12 and abs(n["const"]["e_irr"] - 1.0) < 1e-12
    z = EV.normalised({"s": {"e_irr": 0.0}}, EV.random_gaps({"e_irr": torch.zeros(5)}, 10))
    assert math.isnan(z["s"]["e_irr"])                        # no error: no gap, NaN ratio
    assert math.isnan(EV.random_gaps({"e": torch.zeros(0)})["e"])
    print(f"PASS AUSE / random gap = brute force (gap e_red {ex['random_gap']['e_red']:.4f}, "
          f"e_irr {ex['random_gap']['e_irr']:.4f}); oracle 0, constant 1")


# ------------------------------------------------------------------------------- (c)
def test_ale_split_brute_force():
    ok, of = _outs(seed=4)
    sp = EV.ale_split(ok, of, ause_bins=10)
    M = ok["gt"].numel()
    keep = [i for i in range(M) if ok["covered"][i] and of["covered"][i]]
    u = ok["u"]["u_ale"]
    tiers = {"defined": [i for i in keep if not math.isnan(float(u[i]))],
             "undefined": [i for i in keep if math.isnan(float(u[i]))]}
    for tier, idx in tiers.items():
        d = sp[tier]
        assert d["n_pixels"] == len(idx) > 0 and abs(d["share"] - len(idx) / len(keep)) < 1e-12
        tg = {"e_red": [float((not ok["correct"][i]) and bool(of["correct"][i])) for i in idx],
              "e_irr": [float((not ok["correct"][i]) and not of["correct"][i]) for i in idx]}
        for t, err in tg.items():
            a, gap = _bf_sparsification(err, [_tier_key(ok, "u_ale", i) for i in idx], bins=10)
            assert abs(d["ause"][t] - a) < 1e-9 and abs(d["random_gap"][t] - gap) < 1e-12
            assert abs(d["rate"][t] - sum(err) / len(err)) < 1e-12
            assert d[f"n_{t}"] == int(sum(err))
            if tier == "undefined":                           # one key: a random order
                assert abs(d["ause"][t] - d["random_gap"][t]) < 1e-12
                assert abs(d["ause_norm"][t] - 1.0) < 1e-9
    assert EV.ale_split(dict(ok, u={"u_epi": ok["u"]["u_epi"]}), of) is None
    nan_all = dict(ok, u=dict(ok["u"], u_ale=torch.full((M,), float("nan"))))
    sp2 = EV.ale_split(nan_all, of)
    assert sp2["defined"] == dict(n_pixels=0, share=0.0, n_e_red=0, n_e_irr=0)
    assert sp2["undefined"]["share"] == 1.0
    print(f"PASS u_ale split: defined {sp['defined']['n_pixels']} px (AUSE/gap e_irr "
          f"{sp['defined']['ause_norm']['e_irr']:.3f}), undefined {sp['undefined']['n_pixels']} px "
          f"(AUSE = gap)")


# ---------------------------------------------------------------- the pre-registered rule
def _scene(u_epi, msp, k=10, row="ours_P0", n_map=None):
    return {"k": {str(k): {"n_mapping_views": n_map,
                           "rows": {row: {"reducibility": {"auroc": {"u_epi": u_epi,
                                                                     "msp": msp}}}}}}}


def test_reducibility_criterion_is_preregistered():
    c = EV.REDUCIBILITY_CRITERION
    assert c == dict(row="ours_P0", score="u_epi", against="msp", k=10, auroc_min=0.60,
                     scenes=("office_0", "room_2"), stage="val")
    doc = EV.__doc__
    for s in ("PRE-REGISTERED REDUCIBILITY CRITERION", "M = 10", "> 0.60", "office_0, room_2",
              "MSP", "u_epi", "ours_P0"):
        assert s in doc, s
    V = EV.reducibility_verdict
    kept = V({"office_0": _scene(0.70, 0.65), "room_2": _scene(0.61, 0.55), "toy": {}})
    assert kept["status"] == "kept" and kept["kept"] is True and kept["ignored_scenes"] == ["toy"]
    assert kept["per_scene"]["room_2"] == {"u_epi": 0.61, "msp": 0.55, "passed": True}
    assert V({"office_0": _scene(0.70, 0.65), "room_2": _scene(0.60, 0.50)})["status"] == "dropped"
    assert V({"office_0": _scene(0.70, 0.71), "room_2": _scene(0.90, 0.50)})["status"] == "dropped"
    assert V({"office_0": _scene(float("nan"), 0.5), "room_2": _scene(0.9, 0.5)})["kept"] is False
    und = V({"office_0": _scene(0.9, 0.5)})
    assert und["status"] == "undecided" and und["kept"] is None and und["missing"] == ["room_2"]
    assert V({"office_0": _scene(0.9, 0.5), "room_2": _scene(0.9, 0.5, k=30)})["status"] == "undecided"
    assert V({"office_0": _scene(0.9, 0.5), "room_2": _scene(0.9, 0.5, row="ours_P2")})["missing"] == ["room_2"]
    # M is the field label; a field with other recorded mapping keyframes is not M = 10
    assert V({"office_0": _scene(0.9, 0.5, n_map=10), "room_2": _scene(0.9, 0.5, n_map=10)})["kept"]
    mis = V({"office_0": _scene(0.9, 0.5, n_map=10), "room_2": _scene(0.9, 0.5, n_map=9)})
    assert mis["status"] == "undecided" and mis["missing"] == ["room_2"]
    print("PASS criterion: constants and header fixed; kept / dropped (threshold, MSP, NaN) / "
          "undecided (scene, k, row missing; field keyframes != M)")


# ---------------------------------------------------------------------------- CLI
def _fake_7scenes(root, labels_fn, n=2):
    import cv2
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        base = root / f"frame-{i:06d}"
        cv2.imwrite(str(base) + ".color.png", np.zeros((480, 640, 3), np.uint8))
        cv2.imwrite(str(base) + ".depth.png", np.full((480, 640), 2000, np.uint16))
        c2w = np.eye(4)
        c2w[0, 3] = 0.05 * i
        np.savetxt(str(base) + ".pose.txt", c2w)
        cv2.imwrite(str(base) + ".label.png", labels_fn().astype(np.uint16))


def _toy(tmp_path):
    """A toy 7-Scenes scene (chair left, table right) with the full field (4 views) and a
    noisy 3-view field; returns the common CLI argv and the scenes file of the views run."""
    from usplat.feature_pca import StreamingPCA

    def labels():
        lab = np.zeros((480, 640), np.int64)
        lab[40:, :320] = 3
        lab[40:, 320:] = 7
        return lab
    _fake_7scenes(tmp_path / "data", labels)
    D, d = 16, 8
    pca = StreamingPCA(D, d)
    pca.basis = torch.eye(D)[:, :d]
    xs = torch.arange(-1.15, 1.2, 0.04)
    ys = torch.arange(-0.9, 0.95, 0.04)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    N = means.shape[0]
    splats = dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                  scales=torch.full((N, 3), 0.03), opacities=torch.full((N,), 0.9))
    torch.save(splats, tmp_path / "gs.pt")
    right = means[:, 0] > 0
    feat = torch.where(right[:, None], pca.project(torch.eye(D)[1][None]),
                       pca.project(torch.eye(D)[0][None]))
    full = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    full.set_centre(torch.zeros(d))
    for _ in range(4):
        full.update_sparse(torch.arange(N), torch.ones(N), feat, num_pixels=100)
    torch.save(dict(field=full.state_dict(), pca=pca.state_dict(), timing=dict(keyframes=4)),
               tmp_path / "field.pt")
    g = torch.Generator().manual_seed(0)
    fk = DirectionalSemanticField(N, d, FieldConfig(change_test=False, tau_min=3.0))
    fk.set_centre(torch.zeros(d))
    for v in range(3):                    # random labels on random subsets: mixed tiers
        lab = torch.randint(0, 3, (N,), generator=g)
        seen = torch.nonzero(torch.rand(N, generator=g) < 0.7).squeeze(1)
        fk.update_sparse(seen, torch.rand(seen.numel(), generator=g),
                         pca.project(torch.eye(D)[lab[seen]]), num_pixels=100)
    torch.save(dict(field=fk.state_dict(), pca=pca.state_dict(), timing=dict(keyframes=3)),
               tmp_path / "field_k3.pt")
    vocab = dict(templates=["a photo of a {}."], ignore=[0],
                 classes={3: {"name": "chair"}, 7: {"name": "table"}, 9: "sofa"})
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    key = lambda p: f"m|p|a photo of a {{}}.|{p}"
    torch.save({key(n): torch.eye(D)[i] for i, n in enumerate(("chair", "table", "sofa"))},
               tmp_path / "text.pt")
    sc = dict(name="toy", field=str(tmp_path / "field.pt"), ply=str(tmp_path / "gs.pt"),
              data=str(tmp_path / "data"), dataset="7scenes", stride=1,
              label_pattern="{root}/frame-{idx:06d}.label.png")
    (tmp_path / "scenes.json").write_text(json.dumps([sc]))
    # "10" labels the 3-view field so the criterion's M = 10 entry exists in the toy
    (tmp_path / "views.json").write_text(json.dumps([dict(sc, fields={
        "10": str(tmp_path / "field_k3.pt"), "4": str(tmp_path / "field.pt")})]))
    return ["--vocab", str(tmp_path / "vocab.json"), "--backend", "mock", "--device", "cpu",
            "--downscale", "16", "--tau-cov", "0.5", "--extractor", "clearclip",
            "--clip-model", "m", "--clip-pretrained", "p", "--text-cache", str(tmp_path / "text.pt")]


def test_eval_views_cli_added_keys_and_verdict(tmp_path):
    pytest.importorskip("cv2")
    base = _toy(tmp_path)
    es = _load("eval_seg")
    fit = tmp_path / "fit.json"
    es.main(base + ["--val-scenes", str(tmp_path / "scenes.json"), "--plain-json",
                    "--out", str(fit)])
    out = EV.main(base + ["--scenes", str(tmp_path / "views.json"), "--betas", str(fit),
                          "--rows", "ours_P0,ours_scalar", "--out", str(tmp_path / "v.json")])
    for k in ("ks", "scenes", "summary", "summary_on_ale_defined", "summary_ause_norm",
              "summary_reducibility", "summary_ale_split", "reducibility_criterion"):
        assert k in out, k
    r = out["scenes"]["toy"]["k"]["10"]["rows"]["ours_P0"]
    old = {"ause", "n_pixels", "n_hole", "n_e_red", "n_e_irr", "n_undefined", "miou_k",
           "on_ale_defined"}
    assert old <= set(r) and {"random_gap", "ause_norm", "reducibility", "ale_split"} <= set(r)
    red = r["reducibility"]
    assert red["n_e_red"] == r["n_e_red"] and red["n_e_irr"] == r["n_e_irr"]   # wrong at k
    assert red["n_pixels"] == r["n_e_red"] + r["n_e_irr"] > 0
    assert set(red["auroc"]) == {"msp", "entropy", "u_epi", "u_ale", "-u_ale"}
    assert set(red["auroc_defined"]) == {"-u_ale"}
    for s, t in r["ause"].items():
        for tn, a in t.items():
            want = a / r["random_gap"][tn] if r["random_gap"][tn] > 1e-12 else float("nan")
            got = r["ause_norm"][s][tn]
            assert (math.isnan(want) and math.isnan(got)) or abs(got - want) < 1e-12
    sp = r["ale_split"]
    assert sp["defined"]["n_pixels"] + sp["undefined"]["n_pixels"] == r["n_pixels"]
    assert sp["defined"]["n_pixels"] > 0 and sp["undefined"]["n_pixels"] > 0
    assert "ale_split" not in out["scenes"]["toy"]["k"]["10"]["rows"]["ours_scalar"]  # no u_ale
    assert {"random_gap", "ause_norm"} <= set(r["on_ale_defined"])
    assert out["summary_reducibility"]["10"]["ours_P0"]["u_epi"]["n"] == 1
    assert out["summary_reducibility"]["10"]["ours_P0"]["-u_ale@defined"]["n"] == 1
    assert out["reducibility_criterion"]["status"] == "undecided"
    # the verdict from per-scene files (as scripts/replica_views.sh writes them)
    saved = json.loads((tmp_path / "v.json").read_text())
    toy = saved["scenes"]["toy"]
    assert toy["k"]["10"]["n_mapping_views"] == 3                 # the toy's "10" has 3 views
    assert EV.reducibility_verdict({"office_0": toy, "room_2": toy})["status"] == "undecided"
    toy["k"]["10"]["n_mapping_views"] = 10                        # as a real 10-view field
    paths = []
    for name in ("office_0", "room_2"):
        d = dict(saved, scenes={name: toy})
        p = tmp_path / f"views_val_{name}.json"
        p.write_text(json.dumps(d))
        paths.append(str(p))
    vout = tmp_path / "reducibility_val.json"
    v = EV.main(["verdict"] + paths + ["--out", str(vout)])
    a = red["auroc"]
    want = a["u_epi"] > 0.60 and a["u_epi"] > a["msp"]
    assert v["status"] == ("kept" if want else "dropped") and v["kept"] == want
    assert set(v["inputs"]) == {str(pathlib.Path(p).resolve()) for p in paths}
    su = v["summary_reducibility"]["10"]["ours_P0"]["u_epi"]          # over the 2 scenes
    assert su["n"] == 2 and abs(su["mean"] - a["u_epi"]) < 1e-12 and su["std"] == 0.0
    assert v["summary_ale_split"]["4"]["ours_P0"]["defined"]["share"]["n"] == 2
    assert json.loads(vout.read_text())["status"] == v["status"]
    with pytest.raises(SystemExit, match="written once"):
        EV.main(["verdict"] + paths + ["--out", str(vout)])
    with pytest.raises(SystemExit, match="in both"):
        EV.main(["verdict", paths[0], paths[0]])
    assert EV.main(["verdict", paths[0]])["status"] == "undecided"
    und = tmp_path / "undecided.json"                         # an undecided verdict is not
    with pytest.raises(SystemExit, match="undecided"):         # written: it would lock the
        EV.main(["verdict", paths[0], "--out", str(und)])     # path that is written once
    assert not und.exists()
    nored = json.loads(pathlib.Path(paths[0]).read_text())    # files of pre-T4b code
    for kv in nored["scenes"]["office_0"]["k"].values():
        for r in kv["rows"].values():
            r.pop("reducibility", None)
    (tmp_path / "nored.json").write_text(json.dumps(nored))
    with pytest.raises(SystemExit, match="undecided"):
        EV.main(["verdict", str(tmp_path / "nored.json"), paths[1], "--out", str(und)])
    assert not und.exists()
    print(f"PASS eval_views CLI: added keys, AUROC u_epi {a['u_epi']:.3f} / msp {a['msp']:.3f}, "
          f"verdict {v['status']}")


if __name__ == "__main__":
    import tempfile
    test_reducibility_auroc_brute_force(True)
    test_reducibility_auroc_brute_force(False)
    test_reducibility_auroc_degenerate()
    test_ause_over_random_gap_brute_force()
    test_ale_split_brute_force()
    test_reducibility_criterion_is_preregistered()
    with tempfile.TemporaryDirectory() as tmp:
        test_eval_views_cli_added_keys_and_verdict(pathlib.Path(tmp))


# ------------------------------------------------------------------ scripts/replica_views.sh
V1_VIEWS = REPO / "docs" / "reports" / "2026-09-29-replica-tab1" / "logs" / "views" / \
    "views_val_206c0b7.json"


def _views_repo(tmp_path):
    """A copy of replica_views.sh and eval_views.py (usplat, eval_seg linked) with a
    gpu_gate.sh stub that logs each call, fakes run_mapping (writes --out) and eval_views
    (a per-scene file from the v1 record, with or without the reducibility analysis)."""
    import os
    import shutil
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for rel in ("replica_views.sh", "eval_views.py", "eval_seg.py"):
        shutil.copy2(REPO / "scripts" / rel, root / "scripts" / rel)
    os.symlink(REPO / "usplat", root / "usplat")
    gate = root / "scripts" / "gpu_gate.sh"
    gate.write_text("""#!/usr/bin/env bash
need=$1; shift
echo "GPU $*" >> "$STUB_LOG"
args=("$@"); out=""; scenes=""; cache=""
for ((i=0;i<${#args[@]};i++)); do
  case "${args[$i]}" in --out) out=${args[$((i+1))]};; --scenes) scenes=${args[$((i+1))]};;
                        --text-cache) cache=${args[$((i+1))]};; esac
done
case "$*" in
  *run_mapping.py*) echo new > "$out"; echo "saved $out";;
  *eval_views.py*) echo prompts >> "$cache"; python3 - "$scenes" "$out" <<'PY'
import json, os, sys
sc = json.load(open(sys.argv[1]))[0]
for k, f in sc["fields"].items():
    assert open(f).read().strip() == "new", f        # the field this run mapped
v1 = json.load(open(os.environ["V1_VIEWS"]))
d = dict(ks=v1["ks"], scenes={sc["name"]: v1["scenes"][sc["name"]]})
for kv in d["scenes"][sc["name"]]["k"].values():
    for r in kv["rows"].values():
        r["reducibility"] = dict(auroc=dict(u_epi=0.7, msp=0.6))
json.dump(d, open(sys.argv[2], "w"))
PY
  ;;
esac
""")
    gate.chmod(0o755)
    return root


@pytest.mark.skipif(not V1_VIEWS.exists(), reason="v1 views record not in the checkout")
def test_replica_views_sh_never_touches_v1(tmp_path):
    """replica_views.sh: the k-view fields go to KFIELDS (default OUT/kfields), never into the
    v1 fields directory (a v1 k-view field there is neither reused nor deleted); the text
    cache is a copy in OUT; OUT = the v1 views directory and KFIELDS = FIELDS are refused; a
    per-scene file without the reducibility analysis is refused, not skipped."""
    import hashlib
    import os
    import subprocess
    root = _views_repo(tmp_path)
    fields, maps, tab1, out = (tmp_path / n for n in ("fields", "maps", "tab1", "views_v2"))
    for d in (fields, maps, tab1):
        d.mkdir()
    (tab1 / "eval_val.json").write_text("{}")
    v1_files = {"office_0_lseg.pt": "v1 full", "room_2_lseg.pt": "v1 full",
                "office_0_lseg_k10.pt": "v1 k10", "text_cache_lseg.pt": "v1 cache"}
    for n, c in v1_files.items():
        (fields / n).write_text(c)
    digest = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(fields.iterdir())}
    before = digest()
    log = tmp_path / "gpu.log"
    env = dict(os.environ, STAGE="val", ROOT=str(tmp_path / "Replica"), MAPS=str(maps),
               FIELDS=str(fields), TAB1=str(tab1), OUT=str(out), STUB_LOG=str(log),
               V1_VIEWS=str(V1_VIEWS))
    env.pop("PYTHONPATH", None)
    sh = lambda **kw: subprocess.run(["bash", str(root / "scripts" / "replica_views.sh")],
                                     env=dict(env, **kw), capture_output=True, text=True,
                                     timeout=300)
    r = sh(OUT="/home/thanhnc/gauss_ws/src/uags/results/views")
    assert r.returncode == 1 and "v1 views directory" in r.stdout, r.stdout + r.stderr
    r = sh(KFIELDS=str(fields))
    assert r.returncode == 1 and "KFIELDS is the fields directory" in r.stdout, r.stdout
    assert not out.exists()
    r = sh()
    assert r.returncode == 0, r.stdout + r.stderr
    calls = log.read_text().splitlines()
    maps_ = [c for c in calls if "run_mapping.py" in c]
    assert len(maps_) == 6                                      # every k of both scenes
    assert all(f"--out {out}/kfields/" in c for c in maps_)
    assert all(f"--pca-from {fields}/" in c for c in maps_)
    evals = [c for c in calls if "eval_views.py" in c]
    assert len(evals) == 2 and all(f"--text-cache {out}/text_cache_lseg.pt" in c for c in evals)
    assert digest() == before                                  # v1 fields and cache untouched
    assert list((out / "kfields").iterdir()) == []             # KEEP=0 deletes its own fields
    assert (out / "text_cache_lseg.pt").read_text().startswith("v1 cache")
    assert json.loads((out / "views_val.json").read_text())["ks"]
    log.unlink()
    r = sh()                                                    # done scenes are skipped
    assert r.returncode == 0 and r.stdout.count(": done") == 2 and not log.exists()
    d = json.loads((out / "views_val_room_2.json").read_text())
    for kv in d["scenes"]["room_2"]["k"].values():
        for row in kv["rows"].values():
            row.pop("reducibility")
    (out / "views_val_room_2.json").write_text(json.dumps(d))
    r = sh()
    assert r.returncode == 1 and "no reducibility analysis" in r.stdout, r.stdout
