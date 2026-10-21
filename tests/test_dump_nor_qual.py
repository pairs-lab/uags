"""scripts/dump_nor_qual.py on the CPU (mock backend) on the toy NeRF-Object-Removal scene of
tests/test_eval_change.py (a red object in front of a textured wall, removed in the update).
A Table III record of the toy is made by eval_change_nor.py itself; the dump must reproduce it
exactly (CPU sums are deterministic) and its captured fields must give the scores that
run_update returns without any patch.
Checks:
  * the run reproduces the record (every AP/AUROC delta 0, frames identical) and the patches
    are undone (EC.DirectionalSemanticField, EC.residuals_view, EC.ranking_table)
  * the dumped per-Gaussian scores of the 7 Table III rows equal an unpatched run_update's;
    the 4 fused baselines equal fuse_rank([X, depth], observed); fused.json keeps Table III's
    own cells and common sets (equal to the record)
  * the rank population has n_pix - n_nan values; midrank on a hand example
  * the frame and turn rules on hand-made masks and cameras; the toy's rule frame
  * the query is the object's class; after the update ours answers it less often than
    before on the removed object's pixels; the curves start at the mapping state and end at
    the end state
  * the text cache is never written by a dump; a missing prompt, a finished test dump and a
    record the run does not reproduce are refused (exit 2 for the last)
"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOY = _load("toy_eval_change", REPO / "tests" / "test_eval_change.py")
DQ = _load("dump_nor_qual", REPO / "scripts" / "dump_nor_qual.py")
EC = DQ.EC
CLASSES = ("wall", "object", "other")


def toy_inputs(tmp):
    """Toy scene, field, its Table III record (eval_change_nor.py), vocabulary, text cache,
    betas and tolerance files."""
    _, _, root, align, sp, n_wall, n_obj, _ = TOY.build(tmp)
    EC.load_extractor = lambda cfg, device: TOY.Fake()
    rec = tmp / "rec.json"
    EC.main(["--field", str(tmp / "field.pt"), "--backend", "mock", "--device", "cpu",
             "--plain-json", "--align", str(align), "--out", str(rec)])
    vocab = tmp / "vocab.json"
    vocab.write_text(json.dumps(dict(templates=["a photo of a {}."],
                                     classes={str(i + 1): c for i, c in enumerate(CLASSES)})))
    cache = {}
    for i, c in enumerate(CLASSES):
        e = torch.zeros(TOY.D)
        e[(0, 1, 5)[i]] = 1.0                         # Fake: channel 0 = wall, 1 = object
        cache[f"fake|a photo of a {{}}.|{c}"] = e
    torch.save(cache, tmp / "text.pt")
    (tmp / "betas.json").write_text(json.dumps(dict(
        betas=dict(ours_P0=20.0, averaging=21.0),
        beta_fit_settings=dict(undef_frac=0.5, tau_cov=0.5))))
    (tmp / "tol.json").write_text(json.dumps(dict(tolerance=0.002)))
    return dict(tmp=tmp, rec=rec, vocab=vocab, cache=tmp / "text.pt", betas=tmp / "betas.json",
                tol=tmp / "tol.json", n_wall=n_wall, n_obj=n_obj, root=root)


def argv(t, out, stage="toy", table=None, extra=()):
    return ["--table", str(table or t["rec"]), "--tolerance", str(t["tol"]), "--vocab",
            str(t["vocab"]), "--text-cache", str(t["cache"]), "--betas", str(t["betas"]),
            "--out", str(out), "--stage", stage, "--backend", "mock", "--device", "cpu",
            *extra]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    t = toy_inputs(tmp_path_factory.mktemp("dump_nor_qual"))
    sha = DQ.sha256_file(t["cache"])
    out = t["tmp"] / "dump"
    res = DQ.main(argv(t, out, extra=["--curves", "--save-features"]))
    t.update(out=out, res=res, cache_sha=sha,
             meta=json.loads((out / "meta.json").read_text()),
             G=torch.load(out / "gauss.pt", weights_only=False),
             record=json.loads(t["rec"].read_text()))
    return t


def test_reproduces_the_record_and_restores_patches(run):
    meta = run["meta"]
    assert meta["status"] == "done" and run["res"]["exit"] == 0
    rep = meta["reproduction"]
    assert rep["ok"] and rep["frames_equal"] and all(rep["views_equal"].values())
    assert rep["max_ap_auroc_delta"] == 0.0 and rep["max_operating_point_delta"] == 0.0
    assert all(v is True for k, v in meta["capture_checks"].items() if k != "curve_label_agreement")
    assert EC.DirectionalSemanticField is DQ.DirectionalSemanticField
    assert EC.residuals_view.__module__ == EC.__name__
    assert EC.ranking_table.__module__ == EC.__name__ and EC.ranking_table.__name__ == "ranking_table"
    assert DQ.sha256_file(run["cache"]) == run["cache_sha"]          # the cache was only read


def test_scores_equal_an_unpatched_run_update(run):
    rec = run["record"]
    ns = DQ.ec_namespace(rec, DQ.build_parser().parse_args(argv(run, run["out"])))
    ctx = EC.load_context(ns)
    k = int(rec["settings"]["k"])
    replay, null, _ = EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    thr = {r: nl.threshold(0.999) for r, nl in null.items()}
    with torch.no_grad():
        upd = EC.run_update(ctx, ctx["upd_views"], k, thr, 0.999)
    G = run["G"]
    for r in ("ncc", "depth", "feat", "normlift", "dispersion", "ours"):
        assert DQ.nan_equal(G["scores"][r], upd["scores"][r]), r
    for r, f in upd["flags"].items():
        assert torch.equal(G["flags"][r], f), r
    obs = G["observed"]
    assert torch.equal(obs, upd["observed"])
    for name in DQ.NEW_FUSED:
        x = name.split("+")[0]
        ref = EC.fuse_rank([upd["scores"][x], upd["scores"]["depth"]], mask=obs)
        assert DQ.nan_equal(G["scores"][name], ref), name
    assert DQ.nan_equal(G["scores"]["fused"],
                        EC.fuse_rank([upd["scores"]["ours"], upd["scores"]["depth"]], mask=obs))
    fused = json.loads((run["out"] / "fused.json").read_text())
    for lvl in ("gaussian", "pixel"):          # Table III's own cells and common sets untouched
        for r in EC.ROWS:
            a, b = fused["table3"][lvl][r], rec[lvl]["rows"][r]
            assert a["ap"] == b["ap"] and a["common"]["ap"] == b["common"]["ap"], (lvl, r)
    assert fused["gaussian"]["rows"]["fused"]["ap"] == rec["gaussian"]["rows"]["fused"]["ap"]
    assert fused["gaussian"]["common"]["rows"] == list(DQ.FUSED_ROWS)
    for r in DQ.FUSED_ROWS:
        assert 0 <= fused["gaussian"]["rows"][r]["ap"] <= 1
        assert 0 <= fused["pixel"]["rows"][r]["auroc"] <= 1


def test_rank_population_and_midrank(run):
    pc = run["meta"]["rank_population"]
    for r, c in pc.items():
        assert c["n_defined"] == c["n_pix_minus_nan"] == c["record_n_pix_minus_nan"], r
    pop = np.array([1.0, 2.0, 2.0, 3.0])
    got = DQ.midrank(pop, np.array([2.0, 0.0, 4.0, np.nan, 1.0]))
    assert np.allclose(got[:3], [0.5, 0.0, 1.0]) and np.isnan(got[3]) and got[4] == 0.125
    fr = int(run["meta"]["selection"]["index"])
    z = np.load(run["out"] / f"frame_{fr}.npz")
    rk = z["grid_rank/ours"]
    cov = z["grid_covered"].astype(bool)
    assert np.nanmin(rk) >= 0 and np.nanmax(rk) <= 1
    assert z["full_photo"].shape == (TOY.H, TOY.W, 3) and z["grid_tint"].shape == (TOY.H, TOY.W, 3)
    f = run["meta"]["frames"][str(fr)]
    assert (f["n_covered"], f["n_pos"]) == (f["record_n_covered"], f["record_n_pos"])


def test_frame_and_turn_rules(run):
    H, W = 192, 256
    def box(r0, r1, c0, c1):
        m = np.zeros((H, W), bool)
        m[r0:r1, c0:c1] = True
        return m
    masks = {10: box(0, 100, 50, 150),        # largest, but touches the top border
             12: box(5, 60, 5, 60),           # inside, 55 x 55
             14: box(100, 155, 100, 155),     # inside, same area: the lower index (12) wins
             16: None, 18: box(2, 10, 2, 10)}
    i, info = DQ.pick_frame([10, 12, 14, 16, 18], masks.get, {10: 9, 12: 1})
    assert i == 12 and info["rule"].startswith("largest") and info["n_inside"] == 3
    i, info = DQ.pick_frame([10, 16], masks.get, {10: 9, 16: 3})
    assert i == 10 and info["rule"].startswith("fallback")
    assert DQ.pick_frame([18], masks.get)[0] == 18           # exactly 2 px inside qualifies
    edge = box(1, 10, 5, 20)
    with pytest.raises(ValueError):
        DQ.pick_frame([1], {1: edge}.get)
    # turns: image-down = camera +y, +x, -y, -x after 0..3 clockwise turns
    for k, down in enumerate(([0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0])):
        R = np.eye(4)
        # a camera whose axis `down` (in camera coordinates) points to world +y
        if k == 1:
            R[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]      # camera x -> world +y
        elif k == 2:
            R[:3, :3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        elif k == 3:
            R[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        assert DQ.quarter_turns(R) == k, k
    # the toy's rule frame: an independent pass over the masks of the record's keyframes
    rec = run["record"]
    root = run["root"]
    area = {}
    for i in rec["views"]["update"]:
        m = np.load(root / "masks_real" / f"{i:03d}.npy")
        ys, xs = np.nonzero(m)
        if m.any() and ys.min() >= 2 and xs.min() >= 2 and ys.max() < H - 2 and xs.max() < W - 2:
            area[i] = int(m.sum())
    want = (min((i for i in area if area[i] == max(area.values()))) if area else
            max(rec["pixel"]["frames"], key=lambda f: (f["n_pos"], -f["index"]))["index"])
    assert run["meta"]["selection"]["index"] == want


def test_query_contrast_and_curves(run):
    q = run["meta"]["query"]
    assert q["name"] == "object" and q["top3"][0]["share"] > 0.5
    con = json.loads((run["out"] / "contrast.json").read_text())["sets"]
    assert con["before"]["stale_answer_rate"] > 0.5
    assert con["ours"]["stale_answer_rate"] < con["before"]["stale_answer_rate"]
    assert con["ours"]["delta_cos"] < con["before"]["delta_cos"]
    for s in DQ.SETS:
        assert con[s]["n_mask_cov"] > 0
    cur = json.loads((run["out"] / "curves.json").read_text())
    G = run["G"]
    st, sp = G["observed"] & G["stale"], G["observed"] & ~G["stale"]
    assert len(cur["ours"]["stale_is_query"]) == len(run["record"]["views"]["update"]) + 1
    first = float((G["top_class"]["before"][st] == q["index"]).double().mean())
    last = float((G["top_class"]["ours"][st] == q["index"]).double().mean())
    assert cur["ours"]["stale_is_query"][0] == pytest.approx(first)
    assert cur["ours"]["stale_is_query"][-1] == pytest.approx(last)
    assert cur["ours"]["static_changed"][0] == 0.0 and cur["averaging"]["static_changed"][0] == 0.0
    ag = run["meta"]["capture_checks"]["curve_label_agreement"]
    assert ag["ours"] == 1.0 and ag["averaging"] == 1.0
    assert set(G["features"]) == set(DQ.SETS) and G["features"]["ours"].dtype == torch.float16
    fa = json.loads((run["out"] / "fa_frames.json").read_text())
    assert len(fa["frames"]) == len(run["record"]["views"]["update"])


def test_refusals(run, tmp_path):
    # a prompt missing from the cache: refused (the dump never writes the cache)
    t = dict(run)
    torch.save({k: v for k, v in torch.load(run["cache"]).items() if not k.endswith("other")},
               tmp_path / "short.pt")
    t["cache"] = tmp_path / "short.pt"
    with pytest.raises(SystemExit, match="lacks 1 prompts"):
        DQ.main(argv(t, tmp_path / "a"))
    # a finished test dump is never redone
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "meta.json").write_text(json.dumps(dict(status="done", stage="test")))
    with pytest.raises(SystemExit, match="run once"):
        DQ.main(argv(run, tmp_path / "b", stage="test"))
    # a record the run does not reproduce: exit 2, status FAIL
    bad = json.loads(run["rec"].read_text())
    bad["gaussian"]["rows"]["ours"]["ap"] += 0.01
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit) as e:
        DQ.main(argv(run, tmp_path / "c", table=tmp_path / "bad.json"))
    assert e.value.code == 2
    meta = json.loads((tmp_path / "c" / "meta.json").read_text())
    assert meta["status"] == "FAIL" and any("gaussian ours ap" in f
                                            for f in meta["reproduction"]["failures"])
