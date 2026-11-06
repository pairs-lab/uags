"""Attempt 08, label-free decision rules (usplat/decoding.py, scripts/decode_tab1.py,
scripts/decode_select.py, configs/decoding_prereg.json), on the CPU:
 * the rules on hand cases: base = argmax with ties and uncovered pixels as no decision; centring
   removes a class's offset; prior balancing; the expected mean IoU and its coordinate ascent;
 * decision_metrics of the base rule equals usplat.seg_eval.summarize on the same pixels;
 * decode_tab1.py on the toy scene of tests/test_seg_eval.py (mock backend): the base rule
   reproduces eval_seg's own record exactly, and every rule writes an eval_seg-format record;
 * the selection rule (score, tie order, margins), the adoption rule, and the test gates.
"""
import importlib.util
import json
import math
import pathlib
import shutil
import subprocess

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from usplat import decoding as D                                           # noqa: E402


def test_rules_on_hand_cases():
    s = torch.tensor([[2.0, 1.0], [1.0, 1.0], [0.5, 0.4], [3.0, 0.0]], dtype=torch.float64)
    p = torch.softmax(s, 1)
    cov = torch.tensor([True, True, True, False])
    lab, conf, tied = D.decide("base", s, p, cov)
    assert lab.tolist() == [0, -1, 0, -1] and tied.tolist() == [False, True, False, False]
    assert conf[1] == pytest.approx(0.5) and conf[3] == pytest.approx(0.5)          # tie: max p; uncovered: 1/K
    # centring: class 0's scene-wide offset of +0.3 is removed (alpha = 1)
    lab, _, _ = D.decide("centre", s, p, cov, torch.ones(4), torch.tensor([0.3, 0.0]))
    assert lab.tolist() == [0, 1, 1, -1]
    # prior balancing: pi = (0.9, 0.1), tau = 1 favours class 1 where p is close
    pp = torch.tensor([[0.6, 0.4], [0.95, 0.05]], dtype=torch.float64)
    par = 1.0 * torch.tensor([0.9, 0.1]).log()
    lab, conf, _ = D.decide("prior", pp.log(), pp, torch.tensor([True, True]), None, par.double())
    assert lab.tolist() == [1, 0] and conf.tolist() == pytest.approx([0.4, 0.95])
    with pytest.raises(ValueError, match="unknown rule"):
        D.decide("argmax2", s, p, cov)


def test_expected_miou_and_offsets():
    # class 1 has 45% probability on half the pixels but never wins the argmax
    p = torch.tensor([[0.55, 0.45]] * 50 + [[0.99, 0.01]] * 50, dtype=torch.float64)
    lp = p.log()
    e0 = D.expected_miou(lp, torch.zeros(2, dtype=torch.float64))
    tp0, g0 = p[:, 0].sum(), p[:, 0].sum()
    assert e0 == pytest.approx(float(tp0 / (g0 + 100 - tp0)) / 2)          # class 1: E[TP] = 0
    b, best = D.eiou_offsets(p)
    assert best > e0 and b[1] > b[0]
    assert D.expected_miou(lp, b) == pytest.approx(best)
    lab = (lp + b).argmax(1)
    assert int((lab[:50] == 1).sum()) == 50 and int((lab[50:] == 0).sum()) == 50
    st = D.SceneStats(K=2, n_frames=2, subsample=10, seed=0)
    st.add(lp[:60], p[:60], torch.ones(60))
    st.add(lp[60:], p[60:], torch.ones(40))
    f = st.finalize()
    assert f["n_pixels"] == 100 and f["n_subsample"] == 10 and torch.allclose(f["pi"], p.mean(0))
    assert torch.allclose(f["m"], lp.sum(0) / 100)
    assert D.scene_parameters(f, "base") is None
    assert torch.allclose(D.scene_parameters(f, "prior", 0.5), 0.5 * p.mean(0).log())


def test_decision_metrics_equal_summarize():
    from usplat.seg_eval import summarize
    g = torch.Generator().manual_seed(0)
    M, K = 3000, 5
    s = torch.randn(M, K, generator=g, dtype=torch.float64)
    p = torch.softmax(2.0 * s, 1)
    gt = torch.randint(0, K, (M,), generator=g)
    cov = torch.rand(M, generator=g) > 0.05
    p = torch.where(cov[:, None], p, torch.full_like(p, 1.0 / K))
    u = dict(u_ale=torch.where(torch.rand(M, generator=g) > 0.1, torch.rand(M, generator=g),
                               torch.full((M,), float("nan"))))
    lab, conf, tied = D.decide("base", s, p, cov)
    py = p.gather(1, gt[:, None]).squeeze(1)
    d = dict(gt=gt, pred=lab, conf=conf, entropy=(-torch.special.xlogy(p, p).sum(1)).float(),
             nll=(-py.clamp_min(1e-12).log()).float(), brier=((p * p).sum(1) - 2 * py + 1).float(),
             correct=(lab == gt), covered=cov, tied=tied, alpha=torch.ones(M), u=u)
    ref = summarize(d, [str(i) for i in range(K)])
    got = D.decision_metrics(gt, lab, conf, cov, u=u, class_names=[str(i) for i in range(K)])
    for var in ("all", "covered"):
        for k in ("miou", "acc", "macc", "ece", "aurc", "n_tied", "n_uncovered", "coverage"):
            assert got[var][k] == pytest.approx(ref[var][k], abs=1e-12), (var, k)
        assert got[var]["iou"] == pytest.approx(ref[var]["iou"])
        for ch in ("msp", "u_ale"):
            assert got[var]["ranking"][ch]["ause"] == pytest.approx(ref[var]["ranking"][ch]["ause"], abs=1e-12)


def _toy_prereg(tmp_path, rec, scenes):
    p = json.load(open(REPO / "configs" / "decoding_prereg.json"))
    p["datasets"] = dict(toy=dict(record_val=str(rec), record_test=str(rec), scenes_val=str(scenes),
                                  scenes_test=str(scenes), rows=["ours_P0", "ours_P2"],
                                  selection_rows=["ours_P2"]))
    p["reproduction"]["tolerance"] = 1e-12
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(p))
    return path


def test_toy_scene_reproduces_and_writes_every_rule(tmp_path):
    TS = _load("test_seg_eval_for_decoding", REPO / "tests" / "test_seg_eval.py")
    cli = TS._load_cli()
    base = TS._cli_toy(tmp_path)
    scenes = tmp_path / "scenes.json"
    rec = tmp_path / "out" / "eval.json"
    cli.main(base + ["--scenes", str(scenes), "--val-scenes", str(scenes), "--out", str(rec)])
    pre = _toy_prereg(tmp_path, rec, scenes)
    DT = _load("decode_tab1", REPO / "scripts" / "decode_tab1.py")
    out = tmp_path / "dec" / "toy_val.json"
    code = DT.run(DT.build_parser().parse_args(["--dataset", "toy", "--stage", "toy", "--prereg", str(pre),
                                               "--out", str(out), "--device", "cpu", "--backend", "mock"]))
    assert code == 0
    r = json.loads(out.read_text())
    assert r["status"] == "done" and r["rules"] == ["base", "centre_0.5", "centre_1", "prior_0.5", "prior_1", "eiou"]
    rep = r["reproduction"]["toy"]
    assert rep["ok"] and rep["max_delta"] <= 1e-12 and rep["base_label_mismatches"] == 0
    pub = json.loads(rec.read_text())["metrics"]["scenes"]["toy"]["rows"]
    for k, path in r["outputs"].items():
        b = json.loads(pathlib.Path(path).read_text())
        assert b["status"] == "ok" and b["metrics"]["decoding"]["rule"] == k
        row = b["metrics"]["scenes"]["toy"]["rows"]["ours_P2"]["all"]
        assert 0 <= row["miou"] <= 1 and row["nll"] == pytest.approx(pub["ours_P2"]["all"]["nll"])
        assert set(b["metrics"]["summary"]) == {"ours_P0", "ours_P2"}
    st = r["stats"]["toy"]["ours_P2"]
    assert st["n_pixels"] > 0 and len(st["params"]["eiou"]) == 2


def _fake_runs(d, prereg_path, stage, values):
    """values: {dataset: {rule: {scene: mIoU of every selection row}}}."""
    pre = json.load(open(prereg_path))
    DS = _load("decode_select", REPO / "scripts" / "decode_select.py")
    rules = ["base"] + DS.rule_order(pre)
    sha = DS.sha256_file(prereg_path)
    for ds, by_rule in values.items():
        outs = {}
        for k in rules:
            scenes = {s: dict(rows={r: dict(all=dict(miou=v)) for r in pre["datasets"][ds]["selection_rows"]})
                      for s, v in by_rule[k].items()}
            q = d / f"{ds}_{stage}_{k}.json"
            q.write_text(json.dumps(dict(status="ok", metrics=dict(scenes=scenes))))
            outs[k] = str(q)
        (d / f"{ds}_{stage}.json").write_text(json.dumps(dict(
            status="done", stage=stage, prereg_sha256=sha, rules=rules, outputs=outs,
            reproduction={s: dict(ok=True) for s in next(iter(by_rule.values()))})))
    return DS


def _vals(base, gains, scenes):
    return {k: {s: base + g for s in scenes} for k, g in gains.items()}


def test_selection_and_adoption(tmp_path):
    pre = REPO / "configs" / "decoding_prereg.json"
    rules0 = dict(base=0.0, **{"centre_0.5": 0.004, "centre_1": 0.012, "prior_0.5": 0.0115, "prior_1": 0.0,
                               "eiou": 0.0})
    vals = dict(replica=_vals(0.30, rules0, ["office_0", "room_2"]),
                scannet=_vals(0.60, rules0, ["scene0000_00", "scene0062_00"]))
    DS = _fake_runs(tmp_path, pre, "val", vals)
    sel = tmp_path / "selection.json"
    out = DS.main(["select", "--prereg", str(pre), "--dir", str(tmp_path), "--out", str(sel)])
    assert out["chosen"] == "centre_1" and out["accepted"]       # prior_0.5 within 0.25 pt, later in order
    # a dataset below base refuses the acceptance
    v2 = json.loads(json.dumps(vals))
    v2["scannet"]["centre_1"] = {s: 0.59 for s in v2["scannet"]["centre_1"]}
    v2["replica"]["centre_1"] = {s: 0.34 for s in v2["replica"]["centre_1"]}
    _fake_runs(tmp_path, pre, "val", v2)
    out = DS.main(["select", "--prereg", str(pre), "--dir", str(tmp_path), "--out", str(sel)])
    assert not out["accepted"] or out["chosen"] != "centre_1"
    _fake_runs(tmp_path, pre, "val", vals)
    DS.main(["select", "--prereg", str(pre), "--dir", str(tmp_path), "--out", str(sel)])
    # test: S gain >= 0.5 point, ours_P2 higher on both datasets and in >= 4/6 and >= 5/8 scenes
    rs = ["room_0", "room_1", "office_1", "office_2", "office_3", "office_4"]
    ss = [f"scene{i:04d}_00" for i in range(8)]
    tv = dict(replica={k: {s: 0.28 + (0.01 if k == "centre_1" and i < 4 else 0.0) for i, s in enumerate(rs)}
                       for k in rules0},
              scannet={k: {s: 0.66 + (0.006 if k == "centre_1" and i < 5 else 0.0) for i, s in enumerate(ss)}
                       for k in rules0})
    _fake_runs(tmp_path, pre, "test", tv)
    t = DS.main(["table", "--prereg", str(pre), "--selection", str(sel), "--dir", str(tmp_path),
                 "--out-dir", str(tmp_path / "table")])
    assert t["adopt"] and t["wins"] == dict(replica=4, scannet=5)
    tv["scannet"]["centre_1"] = {s: 0.66 + (0.006 if i < 4 else 0.0) for i, s in enumerate(ss)}
    _fake_runs(tmp_path, pre, "test", tv)
    t = DS.main(["table", "--prereg", str(pre), "--selection", str(sel), "--dir", str(tmp_path),
                 "--out-dir", str(tmp_path / "table")])
    assert not t["adopt"] and t["wins"]["scannet"] == 4


def test_check_test_gates(tmp_path):
    repo = tmp_path / "repo"
    DS = _load("decode_select", REPO / "scripts" / "decode_select.py")
    for rel in ("configs/decoding_prereg.json",) + DS.CODE:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, repo / rel)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "registration")
    pre = repo / "configs" / "decoding_prereg.json"
    d = tmp_path / "runs"
    d.mkdir()
    rules0 = dict(base=0.0, **{"centre_0.5": 0.02, "centre_1": 0.0, "prior_0.5": 0.0, "prior_1": 0.0, "eiou": 0.0})
    _fake_runs(d, pre, "val", dict(replica=_vals(0.3, rules0, ["a", "b"]), scannet=_vals(0.6, rules0, ["c", "d"])))
    sel = d / "selection.json"
    DS.main(["select", "--prereg", str(pre), "--dir", str(d), "--out", str(sel), "--repo", str(repo)])
    args = ["check-test", "--prereg", str(pre), "--selection", str(sel), "--dir", str(d), "--repo", str(repo)]
    assert DS.main(args) == 0
    (repo / "usplat" / "decoding.py").write_text("# changed\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        DS.main(args)
    git("checkout", "--", "usplat/decoding.py")
    (d / "replica_test.json").write_text(json.dumps(dict(status="done")))
    with pytest.raises(SystemExit, match="run once"):
        DS.main(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rules_on_gpu_tensors():
    """decode_tab1.py passes GPU scores with GPU masks and CPU parameters (the 2026-11-06 crash)."""
    s = torch.randn(500, 4, device="cuda", dtype=torch.float64)
    p = torch.softmax(s, 1)
    cov = torch.rand(500, device="cuda") > 0.1
    for rule, par in (("base", None), ("centre", torch.randn(4)), ("prior", torch.randn(4)),
                      ("eiou", torch.randn(4))):
        lab, conf, _ = D.decide(rule, s, p, cov, torch.rand(500, device="cuda"), par)
        assert lab.device.type == "cuda" and conf.device.type == "cuda" and int((lab >= 0).sum()) > 0
