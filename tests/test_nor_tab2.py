"""Table II pipeline on NeRF-Object-Removal: configs/nor_{val,test}.json, scripts/nor_tab2.py
(scenes, reuse, frozen, merge) and scripts/nor_tab2.sh, on the CPU with the fake outputs of
tests/nor_tab2_fakes.py (no GPU, no data needed except for the config check on the data).
 * configs: validation 001-004 and test 005-016 (D6), disjoint; each stride is the one whose
   mapping frame count is nearest 125 (D4), update_stride = stride, and the recorded frame
   and keyframe counts are what the loader and eval_change_nor.py's keyframe rule give with the
   D3 alignment (checked on the data when it is present);
 * merge: means and population stds equal a direct computation over the fake scenes; the
   fused row has no operating point; it refuses a missing or unfinished scene, a keyframe
   count, stride or path that is not the config's, settings or code that differ between
   scenes, a field or map changed after the evaluation, and a test merge whose pipeline
   settings or code differ from the validation merge;
 * reuse: an output built with the same options and code is reused; a map, field or
   evaluation built with another option (a smoke run) or by other code, an unfinished
   evaluation, or one whose field changed since, is refused;
 * a real eval_change_nor.py record (the stage-2 GPU smoke of scene 001, when present) is read
   the same way, its keyframe subsets are the only problems against the protocol config,
   and the fake records have its layout wherever the merge reads;
 * the shell pipeline in a throw-away git copy of the repo with a mock GPU gate: DRY_RUN
   prints the protocol's commands and writes nothing; validation then test run end to end
   (a crashed evaluation is retried); a rerun reuses every output; a smoke map at a protocol
   path stops the run; the test is refused when its shared options differ from validation's,
   when it already ran, when the tree is dirty, and when the code changed after validation.
"""
import importlib.util
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path("/home/thanhnc/gauss_ws/data/nerf_object_removal/nerf-object-removal")
ALIGN = pathlib.Path("/home/thanhnc/gauss_ws/data/uags_results/nor/align")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(REPO / "scripts"))
NT = _load(REPO / "scripts" / "nor_tab2.py", "nor_tab2")
FK = _load(REPO / "tests" / "nor_tab2_fakes.py", "nor_tab2_fakes")


# ------------------------------------------------------------------------------ configs
def _cfg(split):
    return json.loads((REPO / "configs" / f"nor_{split}.json").read_text())


def test_configs_split_and_paths():
    val, test = _cfg("val"), _cfg("test")
    names = lambda c: [e["name"] for e in c["scenes"]]
    assert val["split"] == "val" and test["split"] == "test"
    assert names(val) == ["001", "002", "003", "004"]
    # 012 and 016 are left out: before->after offset above 1 cm (protocol of 2026-10-18)
    assert names(test) == [f"{i:03d}" for i in range(5, 17) if i not in (12, 16)]
    assert set(test["excluded"]) == {"012", "016"}
    for c in (val, test):
        for e in c["scenes"]:
            s = e["name"]
            assert e["dataset"] == "nor" and e["data"] == f"{c['root']}/{s}"
            assert e["update_stride"] == e["stride"] >= 1
            assert e["align"] == "none"                     # no D3 alignment
            assert e["ply"].endswith(f"/nor_{s}_map.pt") and e["field"].endswith(f"/nor_{s}_lseg.pt")
            assert 0 < e["n_map_keyframes"] <= e["n_map_frames"]
            assert 0 < e["n_update_keyframes"] <= e["n_update_frames"]
    print("PASS configs: 001-004 val, 005-016 test without 012/016, one path set per scene")


class Poses:
    """ds[i]["c2w"] of the loader without reading the images (its own c2w array)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds
        return {"c2w": torch.from_numpy(d.c2w[d.start + i * d.stride]).float()}


@pytest.mark.skipif(not DATA.is_dir(), reason="NeRF-Object-Removal data not found")
def test_configs_match_the_data():
    from check_nor import suggested_stride
    from time_field import select_keyframes
    from usplat.datasets import make_dataset
    for split in ("val", "test"):
        for e in _cfg(split)["scenes"]:
            meta = json.loads((pathlib.Path(e["data"]) / "meta.json").read_text())
            assert e["stride"] == suggested_stride(int(meta["split_start"])), e["name"]
            al = e["align"] if os.path.isfile(e["align"]) else None
            m = make_dataset("nor", e["data"], stride=e["stride"], part="map", align=al)
            u = make_dataset("nor", e["data"], stride=e["update_stride"], part="update", align=al)
            assert (len(m), len(u)) == (e["n_map_frames"], e["n_update_frames"]), e["name"]
            if al is None:
                continue
            assert len(select_keyframes(Poses(m), 0.05, 5.0, 0)) == e["n_map_keyframes"]
            assert len(select_keyframes(Poses(u), 0.05, 5.0, 0)) == e["n_update_keyframes"]
    print("PASS configs: strides nearest 125 frames, counts = loader + keyframe rule")


# ------------------------------------------------------------------------ fake scenes
def fit_args(e):
    return ["--dataset", "nor", "--data", e["data"], "--part", "map", "--stride", str(e["stride"]),
            "--iters", "30000", "--max-gaussians", "1000000", "--max-res", "720", "--out", e["ply"]]


def map_args(e):
    return ["--backend", "gsplat", "--dataset", "nor", "--data", e["data"], "--part", "map",
            "--stride", str(e["stride"]), "--ply", e["ply"], "--downscale", "2", "--extractor",
            "lseg", "--baselines", "--use-depth", "--feat-dim", "64", "--pca-frames", "20",
            "--seed", "0", "--out", e["field"]]


def eval_args(e, out):
    return ["--field", e["field"], "--align", e["align"], "--update-stride", str(e["update_stride"]),
            "--tau-cov", "0.5", "--out", str(out)]


def entries(tmp, names, split):
    """Config entries of fake scenes under tmp (data folder, alignment, map and field paths)."""
    out = []
    for k, s in enumerate(names):
        (tmp / "data" / s).mkdir(parents=True, exist_ok=True)
        al = tmp / "align" / f"{s}.json"
        al.parent.mkdir(parents=True, exist_ok=True)
        al.write_text(json.dumps(dict(scene=s, T=np.eye(4).tolist())))
        out.append(dict(name=s, dataset="nor", data=str(tmp / "data" / s), stride=1 + k % 2,
                        update_stride=1 + k % 2, n_map_frames=100 + k, n_map_keyframes=30 + k,
                        n_update_frames=90 + k, n_update_keyframes=25 + k, align=str(al),
                        ply=str(tmp / "maps" / f"nor_{s}_map.pt"),
                        field=str(tmp / "fields" / f"nor_{s}_lseg.pt")))
    cfg = tmp / "configs" / f"nor_{split}.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(dict(split=split, rule="test", root=str(tmp / "data"),
                                   scenes=out), indent=1))
    return cfg, out


def write_record(e, i, out, edit=None):
    """A finished eval_change_nor.py record of scene i (fake numbers), as the runlog writes it."""
    import eval_change_nor as eval_change
    import run_mapping
    from usplat.runlog import to_jsonable
    args = eval_change.build_parser().parse_args(eval_args(e, out))
    ck = torch.load(e["field"], map_location="cpu", weights_only=False)
    m = FK.eval_metrics(args, i, e, ck, eval_change.code_state(),
                        run_mapping.mapping_code_state()["mapping_code_sha256"])
    rec = dict(schema="fake", status="ok", config=vars(args), metrics=m)
    if edit:
        edit(rec)
    out.write_text(json.dumps(to_jsonable(rec)))
    return rec


def build(tmp, names=("101", "102", "103"), split="val"):
    """Config, maps, fields and finished records of fake scenes; returns (config, dir, entries)."""
    cfg, es = entries(tmp, names, split)
    d = tmp / "records" / split
    d.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(es):
        FK.fake_fit(fit_args(e))
        FK.fake_map(map_args(e))
        write_record(e, i, d / f"{e['name']}.json")
    return cfg, d, es


# -------------------------------------------------------------------------------- merge
def test_merge_means_and_population_std(tmp_path):
    cfg, d, es = build(tmp_path)
    out = NT.merge(str(cfg), str(d))
    n = len(es)
    vals = [FK.fake_values(i) for i in range(n)]
    for r in FK.ROWS:
        for k in ("g_ap", "g_auroc", "p_ap", "p_auroc", "g_ap_common", "p_auroc_common"):
            x = [v[r][k] for v in vals]
            st = out["summary"][r][k]
            assert st["n"] == n and st["n_nan"] == 0, (r, k)
            assert math.isclose(st["mean"], statistics.mean(x), rel_tol=1e-12), (r, k)
            assert math.isclose(st["std"], statistics.pstdev(x), rel_tol=1e-9, abs_tol=1e-15)
        if r == "fused":                                         # no operating point
            assert out["summary"][r]["fa"]["n"] == 0 and out["summary"][r]["fa"]["n_nan"] == n
        else:
            x = [v[r]["fa"] for v in vals]
            assert math.isclose(out["summary"][r]["fa"]["mean"], statistics.mean(x))
        g = out["summary"][r]["g_unscored"]["mean"]
        assert math.isclose(g, statistics.mean(v[r]["n_nan"] / FK.fake_counts(i)["n_observed"]
                                               for i, v in enumerate(vals)))
    c = [FK.fake_counts(i) for i in range(n)]
    assert math.isclose(out["counts"]["unobserved_share"]["mean"],
                        statistics.mean(x["n_unobserved"] / x["n_gaussians"] for x in c))
    assert math.isclose(out["counts"]["pooled_unobserved_share"],
                        sum(x["n_unobserved"] for x in c) / sum(x["n_gaussians"] for x in c))
    assert math.isclose(out["ours"]["kept"]["mean"], statistics.mean(0.05 + 0.01 * i
                                                                      for i in range(n)))
    assert out["scenes_order"] == [e["name"] for e in es] and out["split"] == "val"
    assert out["rows"] == list(FK.ROWS) and out["replay"]["all_ok"]
    assert set(out["code"]) == {"eval_code", "field_code", "fit_code", "fit_config_sha256"}
    assert out["fit_config"]["iters"] == 30000 and out["fit_config"]["max_res"] == 720
    assert out["settings"]["k"] == 2 and out["settings"]["backend"] == "gsplat"
    print("PASS merge: means / population stds per row, counts, Kept, code identity")


def _refused(cfg, d, match, **kw):
    with pytest.raises(SystemExit, match=match):
        NT.merge(str(cfg), str(d), **kw)


def test_merge_refusals(tmp_path):
    cfg, d, es = build(tmp_path)
    r1 = d / f"{es[1]['name']}.json"
    keep = r1.read_text()

    def edited(fn):
        rec = json.loads(keep)
        fn(rec)
        r1.write_text(json.dumps(rec))

    r1.unlink()
    _refused(cfg, d, "102: no record")
    edited(lambda r: r.update(status="failed"))
    _refused(cfg, d, "not finished")
    edited(lambda r: r["metrics"]["views"].update(n_update=99))
    _refused(cfg, d, "n_update_keyframes 99 in the record, 26 in the config")
    edited(lambda r: r["metrics"]["settings"].update(mapping_stride=7))
    _refused(cfg, d, "102: stride 7 in the record")
    edited(lambda r: r["metrics"]["settings"].update(max_update_keyframes=10))
    _refused(cfg, d, "max_update_keyframes 10")
    edited(lambda r: r["metrics"]["alignment"].update(path="/elsewhere.json"))
    _refused(cfg, d, "align '/elsewhere.json'")
    edited(lambda r: r["metrics"]["settings"].update(tau_cov=0.4))
    _refused(cfg, d, r"settings differs between scenes: \[101, 103\] vs \[102\]")
    edited(lambda r: r["metrics"]["provenance"]["eval_code"].update({"usplat/box3d.py": "x"}))
    _refused(cfg, d, "eval_code differs between scenes")
    edited(lambda r: r["metrics"]["provenance"].update(field_code="other"))
    _refused(cfg, d, "field_code differs between scenes")
    r1.write_text(keep)
    NT.merge(str(cfg), str(d))                                      # back to a clean merge
    f0 = pathlib.Path(es[0]["field"])
    field0 = f0.read_bytes()
    f0.write_bytes(field0 + b"x")                                   # a rebuilt field
    _refused(cfg, d, f"101: its field {f0} changed after the evaluation")
    f0.write_bytes(field0)
    torch.save(torch.load(es[2]["ply"], weights_only=True) | {"means": torch.ones(64, 3)},
               es[2]["ply"])
    _refused(cfg, d, "103: map .* changed after the evaluation")
    print("PASS merge refusals")


def test_frozen_test_merge(tmp_path):
    cfg_v, d_v, _ = build(tmp_path / "v", ("101", "102"), "val")
    val = tmp_path / "change_val.json"
    assert NT.main(["merge", "--config", str(cfg_v), "--dir", str(d_v), "--out", str(val),
                    "--pipeline", "iters=30000"]) == 0
    rec = json.loads(val.read_text())
    assert rec["status"] == "ok" and rec["metrics"]["pipeline"] == "iters=30000"
    cfg_t, d_t, es = build(tmp_path / "t", ("201", "202"), "test")
    out = NT.merge(str(cfg_t), str(d_t), frozen=str(val), pipeline="iters=30000")
    assert out["frozen"]["path"] == str(val) and out["split"] == "test"
    _refused(cfg_t, d_t, "pipeline settings 'iters=300' differ", frozen=str(val),
             pipeline="iters=300")
    for name in ("201", "202"):                       # the same in both test scenes, not in val
        q = d_t / f"{name}.json"
        r = json.loads(q.read_text())
        r["metrics"]["settings"]["change_quantile"] = 0.99
        q.write_text(json.dumps(r))
    _refused(cfg_t, d_t, r"settings differ from validation \(change_quantile\)", frozen=str(val),
             pipeline="iters=30000")
    vm = json.loads(val.read_text())["metrics"]
    assert vm["split"] == "val"
    assert NT.frozen_problems(vm, code=NT.code_now(), pipeline="iters=30000") == []
    vm["code"]["field_code"] = "older"                # the fields were built by other code
    assert NT.frozen_problems(vm, code=NT.code_now()) == ["field_code differs from validation"]
    vm["code"]["eval_code"]["usplat/box3d.py"] = "older"
    assert NT.frozen_problems(vm, code=NT.code_now())[0] == \
        "eval_code differs from validation (usplat/box3d.py)"
    with pytest.raises(SystemExit, match="no validation merge"):
        NT.finished_merge(str(tmp_path / "missing.json"))
    print("PASS frozen: the test merge must match the validation merge")


# -------------------------------------------------------------------------------- reuse
def test_reuse(tmp_path):
    cfg, d, es = build(tmp_path, ("101",))
    e = es[0]
    assert NT.reuse_problems("map", e["ply"], fit_args(e)) == []
    assert NT.reuse_problems("field", e["field"], map_args(e)) == []
    rec = d / "101.json"
    assert NT.reuse_problems("eval", str(rec), eval_args(e, rec)) == []
    smoke = fit_args(e)
    smoke[smoke.index("--iters") + 1] = "300"
    assert NT.reuse_problems("map", e["ply"], smoke) == ["iters: 30000 saved, 300 now"]
    smoke = map_args(e)
    smoke[smoke.index("--pca-frames") + 1] = "3"
    assert NT.reuse_problems("field", e["field"], smoke + ["--max-keyframes", "6"]) == [
        "pca_frames: 20 saved, 3 now", "max_keyframes: 0 saved, 6 now"]
    assert NT.reuse_problems("field", e["field"], [a for a in map_args(e) if a != "--use-depth"]) \
        == ["use_depth: True saved, False now"]
    assert NT.reuse_problems("eval", str(rec), eval_args(e, rec) + ["--max-update-keyframes",
                                                                   "10"]) == [
        "max_update_keyframes: 0 saved, 10 now"]
    keep = rec.read_text()
    r = json.loads(keep)
    r["status"] = "failed"
    rec.write_text(json.dumps(r))
    assert NT.reuse_problems("eval", str(rec), eval_args(e, rec)) == ["status 'failed', not finished"]
    rec.write_text(keep)
    assert NT.main(["reuse", "map", e["ply"], "--", *fit_args(e)]) == 0
    assert NT.main(["reuse", "map", e["ply"], "--", *fit_args(e), "--iters", "300"]) == 1
    ck = torch.load(e["field"], weights_only=False)            # a field from older code
    ck["code"]["mapping_code_sha256"] = "older"
    torch.save(ck, e["field"])
    assert NT.reuse_problems("field", e["field"], map_args(e)) == ["built by other code (field_code)"]
    assert NT.reuse_problems("eval", str(rec), eval_args(e, rec)) == [    # scored another field
        f"its field {e['field']} changed after the evaluation (sha256)"]
    m = torch.load(e["ply"], weights_only=True)
    m["meta"]["code_sha256"]["usplat/gs_fit.py"] = "older"
    torch.save(m, e["ply"])
    assert NT.reuse_problems("map", e["ply"], fit_args(e)) == [
        "built by other code (fit_code: usplat/gs_fit.py)"]
    print("PASS reuse: same options reused, smoke options and unfinished records refused")


def test_scenes_lines(tmp_path, capsys):
    cfg, es = entries(tmp_path, ("101", "102"), "val")
    assert NT.main(["scenes", str(cfg)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [" ".join(str(e[k]) for k in ("name", "data", "stride", "update_stride",
                                                     "align", "ply", "field")) for e in es]


# ------------------------------------------------------------------------ the pipeline
def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    """A throw-away git copy of scripts/ usplat/ configs/ with fake val / test configs, and a
    mock GPU gate."""
    tmp = tmp_path_factory.mktemp("nor_tab2_pipe")
    repo = tmp / "repo"
    for sub in ("scripts", "usplat", "configs"):
        shutil.copytree(REPO / sub, repo / sub, ignore=shutil.ignore_patterns("__pycache__"))
    _, val = entries(tmp, ("001", "002"), "val")
    _, test = entries(tmp, ("005", "006"), "test")
    for split in ("val", "test"):
        shutil.copy(tmp / "configs" / f"nor_{split}.json", repo / "configs" / f"nor_{split}.json")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "copy")
    gate = tmp / "gate.sh"
    gate.write_text(f"#!/usr/bin/env bash\nexec python3 {REPO / 'tests' / 'nor_tab2_fakes.py'} \"$@\"\n")
    gate.chmod(0o755)
    return dict(tmp=tmp, repo=repo, val=val, test=test, gate=gate, out=tmp / "out",
                calls=tmp / "calls.txt")


def _run(p, stage, gate=True, **env):
    e = dict(os.environ, STAGE=stage, OUT=str(p["out"]), FAKE_CALLS=str(p["calls"]),
             CUDA_VISIBLE_DEVICES="", UAGS_GPU_POLL_S="1", **env)
    e.pop("GPU_GATE", None)
    if gate:
        e["GPU_GATE"] = str(p["gate"])
    r = subprocess.run(["bash", str(p["repo"] / "scripts" / "nor_tab2.sh")], env=e,
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def _calls(p):
    return p["calls"].read_text().splitlines() if p["calls"].exists() else []


def test_pipeline_end_to_end(pipe):
    p = pipe
    v0 = p["val"][0]
    # dry run: the protocol's commands, nothing written (one alignment missing: printed)
    al = pathlib.Path(v0["align"])
    keep = al.read_text()
    al.unlink()
    rc, log = _run(p, "val", gate=False, DRY_RUN="1")
    al.write_text(keep)
    assert rc == 0, log
    dry = [ln for ln in log.splitlines() if ln.startswith("[dry run]")]
    assert any("scripts/align_nor.py" in ln and f"--scenes {v0['name']}" in ln for ln in dry)
    fit = [ln for ln in dry if "scripts/fit_gs.py" in ln]
    mp = [ln for ln in dry if "scripts/run_mapping.py" in ln]
    assert len(fit) == 2 and fit[0].startswith("[dry run] scripts/gpu_gate.sh 3000 python3")
    assert (f"--part map --stride {v0['stride']} --iters 30000 --max-gaussians 1000000 "
            "--max-res 720") in fit[0] and "--align" not in fit[0] + mp[0]   # map part: no D3
    assert len(mp) == 2 and mp[0].startswith("[dry run] scripts/gpu_gate.sh 7000 python3")
    assert (f"--part map --stride {v0['stride']} --ply {v0['ply']} --downscale 2 --extractor lseg "
            "--baselines --use-depth --feat-dim 64 --pca-frames 20 --seed 0") in mp[0]
    ev = [ln for ln in dry if "scripts/eval_change_nor.py" in ln]
    assert len(ev) == 2 and ev[0].startswith("[dry run] scripts/gpu_gate.sh 6500 python3")
    assert f"--update-stride {v0['update_stride']} --tau-cov 0.5" in ev[0]
    assert any("nor_tab2.py merge --config configs/nor_val.json" in ln for ln in dry)
    assert not pathlib.Path(v0["ply"]).exists() and not _calls(p)

    # validation, end to end; the first evaluation of 001 crashes once and is retried
    rc, log = _run(p, "val", FAKE_CRASH_ONCE="001")
    assert rc == 0, log
    assert "eval 001: attempt 1 failed (" in log and "the launch timed out" in log, log
    calls = _calls(p)
    assert sorted(c.split()[0] for c in calls) == ["eval_change_nor.py"] * 2 + ["fit_gs.py"] * 2 \
        + ["run_mapping.py"] * 2
    assert {c.split()[0]: c.split()[1] for c in calls} == {
        "fit_gs.py": "3000", "run_mapping.py": "7000", "eval_change_nor.py": "6500"}
    rec = json.loads((p["out"] / "change_val.json").read_text())
    assert rec["status"] == "ok" and rec["metrics"]["scenes_order"] == ["001", "002"]
    tex = sorted(p["out"].glob("tab2_val.*.tex"))
    assert len(tex) == 1
    rows = [ln for ln in tex[0].read_text().splitlines() if ln.endswith("\\\\")]
    assert len(rows) == 7 and rows[0].startswith("NCC photometric residual & I & ")
    slog = sorted(p["out"].glob("nor_tab2_val.*.log"))
    assert slog and "git " in slog[-1].read_text().splitlines()[0]

    # a rerun reuses every output (no GPU step)
    rc, log = _run(p, "val")
    assert rc == 0, log
    assert len(_calls(p)) == len(calls) and log.count("reusing ") == 6, log

    # a smoke map at a protocol path stops the run
    ply = pathlib.Path(v0["ply"])
    saved = ply.read_bytes()
    smoke = fit_args(v0)
    smoke[smoke.index("--iters") + 1] = "300"
    subprocess.run([sys.executable, str(REPO / "tests" / "nor_tab2_fakes.py"), "0", "python3",
                    "scripts/fit_gs.py", *smoke], cwd=p["repo"], check=True, capture_output=True)
    rc, log = _run(p, "val")
    assert rc == 1 and "refused: " in log and "iters: 300 saved, 30000 now" in log, log
    ply.write_bytes(saved)

    # test: dry run (frozen check passes), then the run, then it is refused
    rc, log = _run(p, "test", gate=False, DRY_RUN="1")
    assert rc == 0, log
    assert "frozen: the same code and pipeline settings" in log
    assert sum("scripts/eval_change_nor.py" in ln for ln in log.splitlines()) == 2
    rc, log = _run(p, "test", gate=False, DRY_RUN="1", TAU_COV="0.4")   # not the val setting
    assert rc == 1 and "the method frozen on validation" in log and "--tau-cov 0.4" in log, log
    n_calls = len(_calls(p))
    rc, log = _run(p, "test")
    assert rc == 0, log
    assert len(_calls(p)) == n_calls + 6
    rec = json.loads((p["out"] / "change_test.json").read_text())
    assert rec["status"] == "ok" and rec["metrics"]["frozen"]["path"] == str(p["out"] / "change_val.json")
    rc, log = _run(p, "test")
    assert rc == 1 and "holds a finished test run" in log, log

    # a dirty tree is refused, even with FORCE_TEST=1
    f = p["repo"] / "scripts" / "nor_tab2.py"
    f.write_text(f.read_text() + "# edit\n")
    rc, log = _run(p, "test", FORCE_TEST="1")
    assert rc == 1 and "refused: uncommitted changes or untracked source" in log, log
    _git(p["repo"], "checkout", "--", "scripts/nor_tab2.py")
    (p["repo"] / "usplat" / "new_module.py").write_text("x = 1\n")
    rc, log = _run(p, "test", FORCE_TEST="1")
    assert rc == 1 and "usplat/new_module.py" in log, log
    (p["repo"] / "usplat" / "new_module.py").unlink()

    # code changed (and committed) after validation: the frozen check refuses the test
    g = p["repo"] / "usplat" / "datasets.py"
    g.write_text(g.read_text() + "# a change after validation\n")
    _git(p["repo"], "commit", "-qam", "change")
    rc, log = _run(p, "test", FORCE_TEST="1")
    assert rc == 1 and "the method frozen on validation" in log and "field_code" in log, log
    print("PASS pipeline: dry run, val, reuse, smoke refusal, test once, dirty, frozen")


SMOKE = pathlib.Path("/home/thanhnc/gauss_ws/data/uags_results/nor/smoke/eval_change_001_smoke.json")


@pytest.mark.skipif(not SMOKE.is_file(), reason="no real eval_change_nor.py record (stage-2 smoke)")
def test_real_record_layout(tmp_path):
    """The merge reads a real eval_change_nor.py record (the stage-2 GPU smoke of scene 001), and
    the fake records have the same layout wherever the merge reads them."""
    real = json.loads(SMOKE.read_text())
    m = real["metrics"]
    for r in m["rows"]:
        v = NT.row_values(m, r)
        assert v["g_ap"] == m["gaussian"]["rows"][r]["ap"]
        assert v["p_auroc"] == m["pixel"]["rows"][r]["auroc"]
        assert v["g_unscored"] == m["gaussian"]["rows"][r]["n_nan"] / m["gaussian"]["rows"][r]["n"]
        assert v["fa"] == m["operating_point"][r]["fa"]
    c = NT.scene_counts(m)
    assert c["n_stale_observed"] == m["counts"]["n_stale_observed"]
    assert c["unobserved_share"] == m["counts"]["n_unobserved"] / m["counts"]["n_gaussians"]
    assert NT.ours_values(m)["kept"] == m["ours"]["kept"]
    cfg = _cfg("val")["scenes"][0]
    e = dict(cfg, ply=m["provenance"]["ply"]["path"], field=m["provenance"]["field"]["path"])
    assert NT.check_scene(e, real) == [                    # the smoke's subsets, nothing else
        "001: align '/home/thanhnc/gauss_ws/data/uags_results/nor/align/001.json' in the record, "
        "None in the config",
        "001: n_map_keyframes 6 in the record, 65 in the config",
        "001: n_update_keyframes 10 in the record, 55 in the config",
        "001: max_keyframes 6 in the record, 0 in the config",
        "001: max_update_keyframes 10 in the record, 0 in the config"]
    cfg_f, d, es = build(tmp_path, ("101",))
    fake = json.loads((d / "101.json").read_text())["metrics"]

    def keys(x, depth=2):
        if not isinstance(x, dict) or depth == 0:
            return None
        return {k: keys(v, depth - 1) for k, v in x.items()}
    for part in ("settings", "views", "replay_check", "ours"):
        assert set(fake[part]) <= set(m[part]) | {"device"}, part
    assert set(m) == set(fake), set(m) ^ set(fake)
    for part in ("gaussian", "pixel"):
        assert set(keys(fake[part]["rows"]["ours"])) == set(keys(m[part]["rows"]["ours"]))
    assert set(fake["counts"]) == set(m["counts"])
    assert set(fake["provenance"]) == set(m["provenance"])
    assert set(fake["alignment"]) == set(m["alignment"])
