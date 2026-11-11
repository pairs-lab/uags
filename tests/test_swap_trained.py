"""scripts/distill_update_swap.py, scripts/swap_trained_eval.py and scripts/swap_trained.sh
(configs/swap_trained_prereg.json) on the toy Replica swap scene of tests/test_swap.py (CPU:
the swap field mapped by the real run_gsplat with CPU stand-ins, mock backend, fake extractor):
 * training reads the field's mapping keyframes and its update keyframes with the field's swap
   transform replayed: the update targets are the transform's output (swapped rows exactly on
   the source pixels, as many as the frames log), the mapping targets carry no swap, and
   train_features gets both parts;
 * the scorer: a trained file that holds the field's mu* reproduces the row ours exactly; both
   trained methods are scored on the same Gaussians as the eval record's rows;
 * the control: averaging recomputed equals the eval record (pass, 1e-10 off passes), 1e-6 off
   fails with exit 2 ('control failed'; --check-only writes nothing);
 * refusals: another PCA basis (trained file, existing output, resume checkpoint), a field without
   the swap transform, another frames log, other settings than the registration, the test gate
   (finished record, dirty tree, no registration, no or failed validation records);
 * the driver (DRY_RUN): the rebuild, control, training and scoring commands with their own
   parsers, the reset field's arguments equal to scripts/replica_swap.sh's, reuse of existing
   files, and the refusal rules.
"""
import importlib.util
import json
import os
import pathlib
import shlex
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


TS = _load("toy_swap", REPO / "tests" / "test_swap.py")
DUS = _load("distill_update_swap", REPO / "scripts" / "distill_update_swap.py")
SE = _load("swap_trained_eval", REPO / "scripts" / "swap_trained_eval.py")
EC = TS._load_cli("eval_change")
PREREG = REPO / "configs" / "swap_trained_prereg.json"


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("swap_trained")
    root, spec = TS._toy_scene(tmp)
    with pytest.MonkeyPatch.context() as mp:
        bank, F, F0 = TS._swap_fields(tmp, mp, root, spec)
    log = tmp / "fields" / "toy_swap.frames.json"
    gt = tmp / "gt" / "toy.pt"
    EC.main(["gt-swap", "--ply", str(tmp / "gs.pt"), "--data", f"{root}/toy/Sequence_2",
             "--update-stride", "1", "--frames-log", str(log), "--bank", bank, "--spec", spec,
             "--scene", "toy", "--downscale", "16", "--backend", "mock", "--device", "cpu",
             "--out", str(gt)])
    rec = tmp / "eval_val_toy.json"
    EC.main(["eval", "--field", F, "--gt", str(gt), "--config", spec, "--scene", "toy",
             "--split", "val", "--compare-field", f"ours_noreset={F0}", "--out", str(rec)])
    t = dict(tmp=tmp, root=root, spec=spec, bank=bank, F=F, F0=F0, log=log, gt=gt, rec=rec)
    t["files"] = {}
    for method, extra in (("f3dgs", []), ("legs", ["--codes", "8"])):
        t["files"][method] = DUS.main(_train_argv(t, method, extra), extractor=TS.FakeExtractor())
    return t


def _train_argv(t, method, extra=(), out=None):
    a = ["--field", t["F"], "--method", method, "--iters", "40", "--min-iters", "1000000",
         "--lr", "0.02", "--backend", "mock", "--device", "cpu", "--frames-ref", str(t["log"]),
         *extra]
    return a + (["--out", str(out)] if out else [])


def _trained(t, method):
    name = "f3dgs_upd" if method == "f3dgs" else "legs_l4_upd"
    return f"{t['F'][:-3]}_{name}.pt"


def _score_argv(t, out=None, trained=None, record=None, extra=()):
    a = ["--field", t["F"], "--gt", str(t["gt"]), "--config", t["spec"], "--scene", "toy",
         "--split", "val", "--record", str(record or t["rec"]), *extra]
    if trained is None:
        trained = [_trained(t, "f3dgs"), _trained(t, "legs")]
    if trained:
        a += ["--trained", ",".join(map(str, trained))]
    return a + (["--out", str(out)] if out else [])


# ------------------------------------------------------------------ training
def test_views_are_mapping_plus_transformed_update(toy):
    from usplat import swap as SW
    from usplat.clip_dense import upsample_tokens
    from usplat.datasets import downscale
    from usplat.feature_pca import StreamingPCA
    from usplat.gs_io import load_gaussians
    ckpt = torch.load(toy["F"], map_location="cpu", weights_only=False)
    sw = DUS.swap_inputs(ckpt)
    map_ds, upd_ds = DUS.datasets(ckpt["config"])
    map_idx, upd_idx = DUS.select_frames(ckpt, map_ds, upd_ds, sw["log"])
    assert len(map_idx) == len(map_ds) == 8 and upd_idx == [e["i"] for e in sw["log"]["frames"]]
    pca = StreamingPCA.from_state(ckpt["pca"])
    ext = TS.FakeExtractor()
    views, n_rows = DUS.build_views(ckpt, sw, map_ds, upd_ds, map_idx, upd_idx, ext, pca,
                                    load_gaussians(ckpt["config"]["ply"]), "mock", "cpu")
    assert len(views) == len(map_idx) + len(upd_idx) and len(n_rows) == len(upd_idx)
    src = torch.tensor([p["source_id"] for p in sw["bank"]["pairs"]])
    fresh = SW.swap_factory(f"bank={toy['bank']},spec={toy['spec']},scene=toy", pca, "cpu")
    for k, (ds, i) in enumerate([(map_ds, i) for i in map_idx] + [(upd_ds, i) for i in upd_idx]):
        fr_full = ds[i]
        fr = downscale(fr_full, 16)
        h, w = fr["image"].shape[:2]
        raw = pca.project(upsample_tokens(ext.extract_tokens(fr_full["image"]), (h, w))).reshape(-1, 8)
        z = views[k][1]()
        if k < len(map_idx):
            assert torch.equal(z, raw)                          # no swap in the mapping pass
            continue
        j = k - len(map_idx)
        assert torch.equal(z, fresh(raw, fr, i, (h, w)))        # the field's transform
        swapped = (z != raw).any(1)
        lab = SW.nearest_labels(fr_full["labels"], h, w).reshape(-1)
        assert torch.equal(swapped, torch.isin(lab, src))       # exactly the source pixels
        logged = sum(sw["log"]["frames"][j]["n_px"].values())
        assert n_rows[j] == int(swapped.sum()) == logged > 0


def test_training_records_both_parts(toy, monkeypatch, tmp_path):
    log = json.loads(toy["log"].read_text())
    for method, rec in toy["files"].items():
        assert rec["n_map_frames"] == 8 and rec["n_update_frames"] == len(log["frames"]) == 6
        assert rec["n_frames"] == 14 and rec["frames"]["update"] == DUS.log_frames(log)
        assert rec["frames"]["mapping"] == list(range(8))
        assert rec["name"] == ("f3dgs_upd" if method == "f3dgs" else "legs_l4_upd")
        assert rec["config"]["frames"] == DUS.FRAMES and rec["iters_run"] == 40
        assert rec["bank"]["sha256"] == DUS.sha256_file(toy["bank"])
        assert rec["frames_ref"]["sha256"] == rec["frames_log"]["sha256"]
        assert rec["transform_code_sha256"] == log["code_sha256"]
        out = _trained(toy, method)
        assert os.path.exists(out) and not os.path.exists(f"{out}.ckpt")
        assert not os.path.exists(f"{out}.ckpt.json")
    assert toy["files"]["legs"]["config"]["lam_u"] == 4.0 and toy["files"]["legs"]["u"] is not None
    seen = {}
    orig = DUS.D.train_features

    def spy(views, *a, **k):
        seen["n"] = len(views)
        return orig(views, *a, **k)
    monkeypatch.setattr(DUS.D, "train_features", spy)
    DUS.main(_train_argv(toy, "f3dgs", ["--iters", "2"], out=tmp_path / "x.pt"),
             extractor=TS.FakeExtractor())
    assert seen["n"] == 14
    assert DUS.code_sha256() != DUS.DF.code_sha256()


def test_training_refusals(toy, tmp_path):
    ck = torch.load(toy["F"], map_location="cpu", weights_only=False)
    bad = tmp_path / "plain.pt"
    torch.save(dict(ck, config=dict(ck["config"], feature_transform="")), bad)
    with pytest.raises(SystemExit, match="not a swap field"):
        DUS.main(_train_argv(dict(toy, F=str(bad)), "f3dgs", out=tmp_path / "a.pt"),
                 extractor=TS.FakeExtractor())
    other = json.loads(toy["log"].read_text())
    other["frames"][2]["n_px"] = {k: v + 1 for k, v in other["frames"][2]["n_px"].items()}
    (tmp_path / "other.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="is not"):
        DUS.main(_train_argv(toy, "f3dgs", ["--frames-ref", str(tmp_path / "other.json")],
                             out=tmp_path / "b.pt"), extractor=TS.FakeExtractor())
    with pytest.raises(SystemExit, match="same basis"):          # never overwritten
        DUS.main(_train_argv(toy, "f3dgs"), extractor=TS.FakeExtractor())
    d = torch.load(_trained(toy, "f3dgs"), weights_only=False)
    torch.save(dict(d, pca_basis_sha256="0" * 64), tmp_path / "c.pt")
    with pytest.raises(SystemExit, match="another PCA basis"):
        DUS.main(_train_argv(toy, "f3dgs", out=tmp_path / "c.pt"), extractor=TS.FakeExtractor())
    (tmp_path / "d.pt.ckpt").write_bytes(b"")                    # a resume state of unknown basis
    with pytest.raises(SystemExit, match="resume checkpoint"):
        DUS.main(_train_argv(toy, "f3dgs", out=tmp_path / "d.pt"), extractor=TS.FakeExtractor())
    (tmp_path / "d.pt.ckpt.json").write_text(json.dumps(dict(pca_basis_sha256="0" * 64)))
    with pytest.raises(SystemExit, match="resume checkpoint"):
        DUS.main(_train_argv(toy, "f3dgs", out=tmp_path / "d.pt"), extractor=TS.FakeExtractor())


# ------------------------------------------------------------------ scoring + control
def test_scores_and_control_pass(toy, tmp_path):
    out = tmp_path / "trained_val_toy.json"
    res = SE.main(_score_argv(toy, out))
    assert res["exit"] == 0 and res["status"] == "ok"
    c = res["control"]
    assert c["ok"] and c["max_abs_diff"] == 0.0 and c["ours_info"]["max_abs_diff"] == 0.0
    assert c["field_config_equal"]
    rec = json.loads(toy["rec"].read_text())["metrics"]["seg"]["rows"]
    assert set(res["rows"]) == {"f3dgs_upd", "legs_l4_upd"}
    for r in res["rows"].values():
        assert r["n_changed"] == rec["averaging"]["n_changed"] > 0
        assert r["n_static"] == rec["averaging"]["n_static"] > 0
        assert 0.0 <= r["acc_changed"] <= 1.0 and 0.0 <= r["acc_static"] <= 1.0
    saved = json.loads(out.read_text())
    assert saved["status"] == "ok" and "exit" not in saved
    assert saved["record_rows"] == rec and set(saved["record_rows"]) == {"ours", "ours_noreset", "averaging"}
    assert saved["trained"]["f3dgs_upd"]["n_update_frames"] == 6
    assert "scripts/swap_trained_eval.py" in saved["provenance"]["code"]
    # a trained file holding the field's mu*: the row ours, exactly (the same scoring path)
    from usplat.vmf_field import DirectionalSemanticField
    ck = torch.load(toy["F"], map_location="cpu", weights_only=False)
    mu = DirectionalSemanticField.from_state(ck["field"]).mu_post()
    d = torch.load(_trained(toy, "f3dgs"), weights_only=False)
    torch.save(dict(d, name="mu_upd", features=mu), tmp_path / "mu.pt")
    res = SE.main(_score_argv(toy, tmp_path / "mu.json", trained=[tmp_path / "mu.pt"]))
    assert res["rows"]["mu_upd"] == res["recomputed"]["ours"] == rec["ours"]
    # LEGaussians: each Gaussian's code decoded by its MLP
    dl = torch.load(_trained(toy, "legs"), weights_only=False)
    f = SE.gaussian_features(dl)
    assert f.shape == (d["features"].shape[0], 8)
    dec = SE.D.LEGDecoder(dl["decoder"]["codebook"], dim=8)
    dec.load_state_dict(dl["decoder"])
    assert torch.allclose(f, dec(dl["features"].float()))


def test_control_fails_and_check_only(toy, tmp_path):
    rec = json.loads(toy["rec"].read_text())
    for eps, ok in ((1e-10, True), (1e-6, False)):
        r = json.loads(json.dumps(rec))
        r["metrics"]["seg"]["rows"]["averaging"]["acc_changed"] += eps
        p = tmp_path / f"rec_{eps:g}.json"
        p.write_text(json.dumps(r))
        out = tmp_path / f"out_{eps:g}.json"
        res = SE.main(_score_argv(toy, out, record=p))
        assert res["control"]["ok"] is ok and res["exit"] == (0 if ok else 2)
        assert json.loads(out.read_text())["status"] == ("ok" if ok else "control failed")
        chk = SE.main(_score_argv(toy, trained=[], record=p, extra=["--check-only"]))
        assert chk["exit"] == (0 if ok else 2) and "rows" not in chk
    r = json.loads(json.dumps(rec))
    r["metrics"]["seg"]["rows"]["averaging"]["n_static"] += 1
    (tmp_path / "n.json").write_text(json.dumps(r))
    assert SE.main(_score_argv(toy, trained=[], record=tmp_path / "n.json",
                               extra=["--check-only"]))["exit"] == 2
    assert not SE.control_check(dict(a=1.0), dict(a=1.0, b=2.0))["ok"]      # a missing key
    assert SE.control_check(dict(a=float("nan")), dict(a="nan"))["ok"]
    r = json.loads(json.dumps(rec))
    r["metrics"]["scene"] = "other"
    (tmp_path / "s.json").write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="not the eval record"):
        SE.main(_score_argv(toy, trained=[], record=tmp_path / "s.json", extra=["--check-only"]))


def test_scorer_refusals(toy, tmp_path, monkeypatch):
    d = torch.load(_trained(toy, "f3dgs"), weights_only=False)
    for name, change, msg in (("basis", dict(pca_basis_sha256="0" * 64), "another PCA basis"),
                              ("frames", dict(frames=dict(d["frames"], update=d["frames"]["update"][1:])),
                               "frames log"),
                              ("bank", dict(bank=dict(d["bank"], sha256="0" * 64)), "another bank"),
                              ("code", dict(transform_code_sha256="0" * 64), "transform code")):
        torch.save(dict(d, **change), tmp_path / f"{name}.pt")
        with pytest.raises(SystemExit, match=msg):
            SE.main(_score_argv(toy, tmp_path / "x.json", trained=[tmp_path / f"{name}.pt"]))
    with pytest.raises(SystemExit, match="registered settings"):  # toy iterations, not the prereg's
        SE.main(_score_argv(toy, tmp_path / "x.json", extra=["--prereg", str(PREREG)]))
    # the test gate (a spec copy with the toy scene in the test split)
    spec = json.loads(pathlib.Path(toy["spec"]).read_text())
    spec["scenes"] = dict(val=["toy_v"], test=["toy"])
    sp = tmp_path / "spec_test.json"
    sp.write_text(json.dumps(spec))
    argv = lambda out, *extra: ["--field", toy["F"], "--gt", str(toy["gt"]), "--config", str(sp),
                                "--scene", "toy", "--split", "test", "--record", str(toy["rec"]),
                                "--trained", _trained(toy, "f3dgs"), "--out", str(out), *extra]
    done = tmp_path / "done.json"
    done.write_text(json.dumps(dict(status="ok")))
    with pytest.raises(SystemExit, match="finished test record"):
        SE.main(argv(done))
    monkeypatch.setattr(SE.EC, "_dirty_tree", lambda: ["M scripts/x.py"])
    with pytest.raises(SystemExit, match="uncommitted"):
        SE.main(argv(tmp_path / "t.json"))
    monkeypatch.setattr(SE.EC, "_dirty_tree", lambda: [])
    with pytest.raises(SystemExit, match="--prereg"):
        SE.main(argv(tmp_path / "t.json"))
    with pytest.raises(SystemExit, match="--reference"):
        SE.main(argv(tmp_path / "t.json", "--prereg", str(PREREG)))
    _, sha = SE.CE.load_config(str(sp))
    val = dict(status="ok", split="val", scene="toy_v", config_sha256=sha, control=dict(ok=True),
               provenance=dict(code=SE.code_state()))
    for bad, msg in ((dict(control=dict(ok=False)), "control passed"),
                     (dict(provenance=dict(code={})), "other code"),
                     (dict(scene="toy_w"), "validation scenes")):
        (tmp_path / "v.json").write_text(json.dumps(dict(val, **bad)))
        with pytest.raises(SystemExit, match=msg):
            SE.main(argv(tmp_path / "t.json", "--prereg", str(PREREG), "--reference", str(tmp_path / "v.json")))
    (tmp_path / "v.json").write_text(json.dumps(val))           # the gate passes; the val record
    with pytest.raises(SystemExit, match="not the eval record"):  # is not a test eval record
        SE.main(argv(tmp_path / "t.json", "--prereg", str(PREREG), "--reference", str(tmp_path / "v.json")))


# ------------------------------------------------------------------ the driver
def _run_sh(tmp_path, stage, **env):
    e = dict(os.environ, DRY_RUN="1", STAGE=stage, ROOT=str(tmp_path / "Replica"),
             MAPS=str(tmp_path / "maps"), SWAP=str(tmp_path / "swap"), OUT=str(tmp_path / "swap"),
             PYTHONDONTWRITEBYTECODE="1")
    e.update(env)
    return subprocess.run(["bash", str(REPO / "scripts" / "swap_trained.sh")], capture_output=True,
                          text=True, env=e, timeout=300)


def _commands(stdout):
    return [shlex.split(line[2:]) for line in stdout.splitlines() if line.startswith("+ ")]


def _committed(path):
    r = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", str(path)],
                       capture_output=True, text=True)
    t = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", str(path)],
                       capture_output=True, text=True)
    return not r.stdout.strip() and t.returncode == 0


def _ok(path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(status="ok", **kw)))


def test_swap_trained_sh_dry_run(tmp_path):
    spec = json.loads((REPO / "configs" / "swap_replica.json").read_text())
    pre = json.loads(PREREG.read_text())
    assert pre["splits"] == spec["scenes"] and pre["training"]["settings"]["lam_u"] == 4
    r = _run_sh(tmp_path, "val")
    if not _committed(PREREG):
        assert r.returncode == 1 and "is not committed" in r.stdout
        return
    assert r.returncode == 1 and "refused: no map" in r.stdout
    sw = tmp_path / "swap"
    (tmp_path / "maps").mkdir()
    for s in spec["scenes"]["val"] + spec["scenes"]["test"]:
        (tmp_path / "maps" / f"{s}_seq1.pt").write_bytes(b"")
        (sw / "fields").mkdir(parents=True, exist_ok=True)
        (sw / "fields" / f"{s}_swap.frames.json").write_text("{}")
    r = _run_sh(tmp_path, "val")
    assert r.returncode == 1 and "no finished eval record" in r.stdout
    for s in spec["scenes"]["val"] + spec["scenes"]["test"]:
        for st in ("val", "test"):
            _ok(sw / f"eval_{st}_{s}.json")
    r = _run_sh(tmp_path, "val")
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = _commands(r.stdout)
    assert len(cmds) == 7 * 2                                   # 7 steps per val scene
    mb = TS._load_cli("make_swap_bank").build_parser()
    du, se = DUS.build_parser(), SE.build_parser()
    rs = _commands(TS._run_sh(tmp_path / "rs", "val", ROOT=str(tmp_path / "Replica"),
                              MAPS=str(tmp_path / "maps"), SWAP=str(sw), OUT=str(tmp_path / "rs_out"),
                              FIELD_CONFIG=pre_field_config()).stdout)
    root = str(tmp_path / "Replica")
    for k, s in enumerate(spec["scenes"]["val"]):
        bank, field, gt, ctl, t1, t2, score = cmds[7 * k: 7 * k + 7]
        assert bank[:4] == ["scripts/gpu_gate.sh", "5000", "python3", "scripts/make_swap_bank.py"]
        a = mb.parse_args(bank[4:])
        assert a.scene == s and a.out == f"{sw}/banks/{s}.pt"
        assert a.record == f"{sw}/trained_rebuild/bank_{s}.json"          # never the kept record
        assert field[:4] == ["scripts/gpu_gate.sh", "7000", "python3", "scripts/run_mapping.py"]
        ref = [c for c in rs if c[3:4] == ["scripts/run_mapping.py"] and "--baselines" in c
               and f"scene={s}," in " ".join(c)][0]
        arg = lambda c: dict(zip(c[4:], c[5:]))["--feature-transform-arg"]
        strip = lambda c: [x for x in c[4:] if x != arg(c)]
        assert strip(field) == strip(ref)                       # replica_swap.sh's reset field
        assert SW_parse(arg(field))["log"] == f"{sw}/fields/{s}_swap.rebuild.frames.json"
        assert {k2: v for k2, v in SW_parse(arg(field)).items() if k2 != "log"} == \
            {k2: v for k2, v in SW_parse(arg(ref)).items() if k2 != "log"}
        v = dict(zip(field[4:], field[5:]))
        assert json.loads(v["--field-config"]) == json.loads(pre_field_config())
        assert gt[:4] == ["scripts/gpu_gate.sh", "7000", "python3", "scripts/eval_change.py"]
        g = EC.build_parser().parse_args(gt[4:])
        assert g.cmd == "gt-swap" and g.frames_log == SW_parse(arg(field))["log"]
        assert g.record == f"{sw}/trained_rebuild/gt_{s}.json" and g.out == f"{sw}/gt/{s}.pt"
        c = se.parse_args(ctl[2:])
        assert ctl[:2] == ["python3", "scripts/swap_trained_eval.py"] and c.check_only
        assert c.record == f"{sw}/eval_val_{s}.json" and c.field == v["--out"] and c.gt == g.out
        F = v["--out"]
        for t, mth, name in ((t1, "f3dgs", "f3dgs"), (t2, "legs", "legs_l4")):
            assert t[:4] == ["scripts/gpu_gate.sh", "5000", "python3", "scripts/distill_update_swap.py"]
            ta = du.parse_args(t[4:])
            assert ta.method == mth and ta.lam_u == 4.0 and ta.field == F and ta.out == ""
            assert ta.frames_ref == f"{sw}/fields/{s}_swap.frames.json" and ta.channel_chunk == 32
        sc = se.parse_args(score[2:])
        assert sc.trained.split(",") == [f"{F[:-3]}_f3dgs_upd.pt", f"{F[:-3]}_legs_l4_upd.pt"]
        assert sc.out == f"{sw}/trained_val_{s}.json" and sc.prereg == "configs/swap_trained_prereg.json"
        assert sc.split == "val" and sc.reference == [] and not sc.check_only
    # existing bank / field / gt / trained files are reused; a field without its bank is refused
    s = spec["scenes"]["val"][0]
    for p in (f"fields/{s}_swap.pt", f"gt/{s}.pt", f"fields/{s}_swap_f3dgs_upd.pt"):
        (sw / p).parent.mkdir(parents=True, exist_ok=True)
        (sw / p).write_bytes(b"")
    r = _run_sh(tmp_path, "val")
    assert r.returncode == 1 and "exists without its bank" in r.stdout
    (sw / "banks").mkdir(exist_ok=True)
    (sw / "banks" / f"{s}.pt").write_bytes(b"")
    cmds = _commands(_run_sh(tmp_path, "val").stdout)
    assert len(cmds) == 3 + 7 and [c[1] if c[0] == "python3" else c[3] for c in cmds[:3]] == [
        "scripts/swap_trained_eval.py", "scripts/distill_update_swap.py", "scripts/swap_trained_eval.py"]
    _ok(sw / f"trained_val_{s}.json")
    assert len(_commands(_run_sh(tmp_path, "val").stdout)) == 7    # a finished record is kept
    # the test stage
    r = _run_sh(tmp_path, "test")
    assert r.returncode == 1
    if TS._tracked_dirty():
        assert "uncommitted changes" in r.stdout
        return
    assert "run STAGE=val first" in r.stdout
    for v in spec["scenes"]["val"]:
        _ok(sw / f"trained_val_{v}.json")
    r = _run_sh(tmp_path, "test")
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = _commands(r.stdout)
    assert len(cmds) == 7 * 6
    scores = [se.parse_args(c[2:]) for c in cmds if c[:2] == ["python3", "scripts/swap_trained_eval.py"]
              and "--check-only" not in c]
    assert len(scores) == 6 and all(x.split == "test" and x.reference == [
        f"{sw}/trained_val_{v}.json" for v in spec["scenes"]["val"]] for x in scores)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    t0 = spec["scenes"]["test"][0]
    _ok(sw / f"trained_test_{t0}.json", provenance=dict(git=dict(commit=head)))
    assert len(_commands(_run_sh(tmp_path, "test").stdout)) == 7 * 5     # kept, the rest run
    _ok(sw / f"trained_test_{t0}.json", provenance=dict(git=dict(commit="0" * 40)))
    r = _run_sh(tmp_path, "test")
    assert r.returncode == 1 and "another commit" in r.stdout
    for t in spec["scenes"]["test"]:
        _ok(sw / f"trained_test_{t}.json", provenance=dict(git=dict(commit=head)))
    r = _run_sh(tmp_path, "test")
    assert r.returncode == 1 and "every test record is finished" in r.stdout


def pre_field_config():
    return '{"change_weight": "relmax", "fast_min": 0.5, "slow_min": 0.5}'


def SW_parse(s):
    from usplat import swap as SW
    return SW.parse_arg(s)
