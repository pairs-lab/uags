"""Table I v2: the pre-registration, the validation selection, the test gate and the pipeline
(configs/tab1_v2_prereg.json, scripts/make_tables.py, scripts/eval_seg.py --v2-rows,
scripts/replica_tab1_v2.sh).

 * the prereg names exactly the rows usplat.rows.v2_rows builds (plus ours_P2), with the
   grids of usplat.rows, dated 2026-10-10;
 * Table I v1 is reproduced byte for byte from the v1 records (make_tables defaults unchanged);
 * '@p1' / '@p4' / '@ours_v2' take the lowest validation NLL pooled over pixels (not the scene
   mean), ties within tol go to the candidate listed first, a missing or NaN row refuses;
 * check-test refuses: no / unfinished validation record, a v2 row without beta, no
   selection, a validation record or prereg changed after the selection, a selection the rule
   does not reproduce, a finished test record; it lets a crashed test record be rerun;
 * eval_seg --v2-rows fits and evaluates the v2 rows on the CPU toy scene; without the flag
   the row set is v1's;
 * replica_tab1_v2.sh end to end in a scratch git repository (mock backend, CPU): STAGE=val
   writes the selection; STAGE=test refuses a dirty tree, an uncommitted prereg and the v1
   output directory, logs the SECOND look, evaluates once, and refuses a second test run and
   a later validation run.
"""
import copy
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess

import pytest
import torch

from usplat.rows import v2_rows, P1_C0S, P4_ALPHA0S, P4_LAMBDAS, P4_INF_ALPHA0, P4_SAMPLES
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = pathlib.Path(__file__).resolve().parents[1]
PREREG = REPO / "configs" / "tab1_v2_prereg.json"
V1_LOGS = REPO / "docs" / "reports" / "2026-09-29-replica-tab1" / "logs"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mt = _load("make_tables_v2", REPO / "scripts" / "make_tables.py")


# --------------------------------------------------------------------------- prereg
def test_prereg_matches_the_code():
    p = json.loads(PREREG.read_text())
    assert p["registered"] == "2026-10-10"
    g = p["grids"]
    assert tuple(g["p1_c0"]) == P1_C0S and tuple(g["p4_alpha0"]) == P4_ALPHA0S
    assert tuple(g["p4_lambda"]) == P4_LAMBDAS and g["p4_inf_alpha0"] == P4_INF_ALPHA0
    assert g["p4_samples"] == P4_SAMPLES and g["p4_seed"] == 0
    f = DirectionalSemanticField(5, 4, FieldConfig(change_test=False))
    names = [r.name for r in v2_rows(f, torch.eye(4)[:3])]
    assert set(mt.prereg_rows(p)) == set(names) | {"ours_P2"}
    ch = p["choices"]
    assert set(ch["@p1"]["rows"]) == {n for n in names if n.startswith("ours_P1_c")}
    assert set(ch["@p4"]["rows"]) == {n for n in names if n.startswith("ours_P4_a")}
    assert ch["@ours_v2"]["rows"] == ["ours_P2", "ours_P2p1", "@p1", "@p4"]
    assert ch["@p1"]["rows"][0] == "ours_P1_c1" and ch["@p4"]["rows"][0] == "ours_P4_a1_s1"
    assert p["ablations"] == ["ours_P4_inf"]
    assert p["metric"]["name"] == "nll" and p["metric"]["pooling"] == "pixels"
    assert p["metric"]["variant"] == "all" and 0 < p["tie"]["tol"] < 1e-2
    assert p["v1"]["test_commit"] == "224fe1e"
    print(f"PASS prereg: {len(mt.prereg_rows(p))} rows, grids match usplat.rows")


def test_v1_table_is_unchanged(monkeypatch):
    """make_tables' default output still equals the committed v1 LaTeX rows."""
    tex = V1_LOGS / "table1_rows.tex"
    if not tex.exists():
        pytest.skip("v1 records not present")
    monkeypatch.chdir(REPO)
    rel = "docs/reports/2026-09-29-replica-tab1/logs"
    out = mt.tab1(f"{rel}/eval_test_224fe1e.json", "all", f"{rel}/eval_val_224fe1e.json")
    assert out + "\n" == tex.read_text()
    assert mt.choose(mt.body(f"{rel}/eval_val_224fe1e.json")["summary"], "@css") == "css_a0.01"
    print("PASS v1 Table I rows reproduced byte for byte")


# ---------------------------------------------------------------------- selection rule
def _val_body(nll, n=(100, 900), rows=None, betas=True):
    """A validation body with two scenes; nll: {row: (scene0, scene1)}; other prereg rows
    get a high NLL."""
    p = json.loads(PREREG.read_text())
    rows = rows or mt.prereg_rows(p)
    scenes = {}
    for j, name in enumerate(("office_0", "room_2")):
        scenes[name] = dict(rows={r: {"all": dict(nll=nll.get(r, (2.0, 2.0))[j], n_pixels=n[j],
                                                   miou=0.3, acc=0.8)} for r in rows})
    summ = {r: {"all": {"nll": {"mean": sum(nll.get(r, (2.0, 2.0))) / 2, "n": 2}}} for r in rows}
    b = dict(scenes=scenes, summary=summ)
    if betas:
        b["betas"] = {r: ({"beta0": 1.0, "a": 0, "b": 0, "g": 0} if r.startswith("ours_P2")
                          else 1.0) for r in rows}
    return b


def test_choose_v2_pooled_ties_and_refusals():
    p = json.loads(PREREG.read_text())
    # pooled over pixels: X wins the scene mean (0.75 vs 0.95) but loses pooled (0.95 vs 0.91)
    b = _val_body({"ours_P4_a1_s1": (0.5, 1.0), "ours_P4_a0.01_s2": (1.0, 0.9)})
    row, vals = mt.choose_v2(b, "@p4", p)
    assert row == "ours_P4_a0.01_s2", vals
    assert vals["ours_P4_a1_s1"][1] == pytest.approx(0.95)
    # a tie within tol goes to the candidate listed first (ours_P1_c1 before ours_P1_c0.01)
    b = _val_body({"ours_P1_c1": (0.80005, 0.80005), "ours_P1_c0.01": (0.8, 0.8)})
    assert mt.choose_v2(b, "@p1", p)[0] == "ours_P1_c1"
    b = _val_body({"ours_P1_c1": (0.8002, 0.8002), "ours_P1_c0.01": (0.8, 0.8)})
    assert mt.choose_v2(b, "@p1", p)[0] == "ours_P1_c0.01"
    # @ours_v2: ours_P2 keeps the row unless a v2 candidate beats it by more than tol
    b = _val_body({"ours_P2": (0.82, 0.82), "ours_P4_a0.1_s1": (0.80, 0.80),
                   "ours_P1_c0.3": (0.81, 0.81)})
    row, vals = mt.choose_v2(b, "@ours_v2", p)
    assert row == "ours_P4_a0.1_s1" and vals["@p4"][0] == "ours_P4_a0.1_s1"
    assert vals["@p1"][0] == "ours_P1_c0.3"
    b = _val_body({"ours_P2": (0.80, 0.80), "ours_P2p1": (0.80003, 0.80003)})
    assert mt.choose_v2(b, "@ours_v2", p)[0] == "ours_P2"
    # refusals: a missing candidate, a NaN value
    rows = [r for r in mt.prereg_rows(p) if r != "ours_P4_a1_s2"]
    with pytest.raises(SystemExit, match="ours_P4_a1_s2"):
        mt.choose_v2(_val_body({}, rows=rows), "@p4", p)
    with pytest.raises(SystemExit, match="non-finite"):
        mt.choose_v2(_val_body({"ours_P2p1": (float("nan"), 1.0)}), "@ours_v2", p)
    print("PASS v2 choices: pooled NLL, first-listed tie-break, refusals")


def _write_record(path, b, status="ok"):
    path.write_text(json.dumps(dict(status=status, run_id="r1", metrics=b)))


def test_select_and_check_test(tmp_path):
    prereg = tmp_path / "prereg.json"
    shutil.copy(PREREG, prereg)
    val, sel, test = tmp_path / "eval_val.json", tmp_path / "sel.json", tmp_path / "eval_test.json"
    gate = lambda: mt.check_test(str(val), str(sel), str(test), str(prereg))
    with pytest.raises(SystemExit, match="no validation record"):
        gate()
    b = _val_body({"ours_P2p1": (0.7, 0.7)})
    _write_record(val, b, status="running")
    with pytest.raises(SystemExit, match="not a finished validation"):
        gate()
    nob = copy.deepcopy(b)
    del nob["betas"]["ours_P4_inf"]
    _write_record(val, nob)
    with pytest.raises(SystemExit, match="ours_P4_inf"):
        gate()
    _write_record(val, b)
    with pytest.raises(SystemExit, match="no validation selection"):
        gate()
    mt.main(["select", str(val), "--prereg", str(prereg), "--out", str(sel)])
    rec = json.loads(sel.read_text())
    assert rec["status"] == "ok" and rec["name"] == "tab1_v2_select"
    s = rec["metrics"]["selection"]
    assert s["chosen"] == {"@p1": "ours_P1_c1", "@p4": "ours_P4_a1_s1", "@ours_v2": "ours_P2p1"}
    assert s["prereg"]["sha256"] == mt.sha256_file(prereg) and s["validation"]["run_id"] == "r1"
    assert gate() == s["chosen"]
    _write_record(test, {}, status="failed")                    # a crashed test run: rerun
    assert gate() == s["chosen"]
    _write_record(test, {}, status="ok")
    with pytest.raises(SystemExit, match="finished v2 test run"):
        gate()
    test.unlink()
    val.write_text(val.read_text() + " ")                       # validation changed afterwards
    with pytest.raises(SystemExit, match="changed after the selection"):
        gate()
    _write_record(val, b)
    prereg.write_text(prereg.read_text().replace('"tol": 0.0001', '"tol": 0.01'))
    with pytest.raises(SystemExit, match="changed after the selection"):
        gate()
    shutil.copy(PREREG, prereg)
    forged = json.loads(sel.read_text())
    forged["metrics"]["selection"]["chosen"]["@p4"] = "ours_P4_a0.01_s2"
    sel.write_text(json.dumps(forged))
    with pytest.raises(SystemExit, match="the rule gives"):
        gate()
    print("PASS select record and check-test refusals")


def test_tab1_v2_rows(tmp_path):
    b = _val_body({"ours_P2p1": (0.7, 0.7), "ours_P4_a0.1_s2": (0.6, 0.6)})
    stat = lambda v: dict(mean=v, std=0.01, n=2)
    metrics = {k: stat(0.5) for k in ("miou", "acc", "ece", "nll", "brier", "aurc")}
    rows = list(b["betas"]) + ["css", "css_a0.01", "ours_P0", "ours_P1"]
    tb = dict(summary={r: {"all": dict(metrics, ause={"u_ale": stat(0.05)})} for r in rows})
    val, test = tmp_path / "val.json", tmp_path / "test.json"
    _write_record(val, b)
    _write_record(test, tb)
    out = mt.tab1(str(test), "all", str(val), v2=True)
    lines = out.splitlines()
    assert "'@ours_v2': 'ours_P4_a0.1_s2'" in lines[0] and "--v2" in lines[0]
    p2p1 = next(x for x in lines if x.startswith("Ours P2 on P1"))
    assert "\\multicolumn{2}{c}{as P1}" in p2p1
    assert any(x.startswith("Ours P4 (vote predictive)") for x in lines)
    assert any(x.startswith("Ours P4, zero noise") for x in lines)
    assert not any(x.startswith("Ours P4") for x in mt.tab1(str(test), "all", str(val)).splitlines())
    with pytest.raises(SystemExit, match="--val"):
        mt.tab1(str(test), "all", None, v2=True)
    print("PASS tab1 --v2 rows")


# ------------------------------------------------------------------------ eval_seg CLI
def _seg_tests():
    return _load("test_seg_eval_v2", REPO / "tests" / "test_seg_eval.py")


def test_eval_seg_v2_rows(tmp_path):
    t = _seg_tests()
    cli = t._load_cli()
    base = t._cli_toy(tmp_path)
    scenes = str(tmp_path / "scenes.json")
    out = cli.main(base + ["--val-scenes", scenes, "--scenes", scenes, "--v2-rows", "--plain-json",
                           "--out", str(tmp_path / "v2.json")])
    p = json.loads(PREREG.read_text())
    assert set(mt.prereg_rows(p)) <= set(out["betas"])
    rows = out["scenes"]["toy"]["rows"]
    for r in mt.prereg_rows(p):
        assert rows[r]["all"]["miou"] > 0.9, (r, rows[r]["all"]["miou"])
        assert set(rows[r]["all"]["ranking"]) >= {"u_epi", "u_ale", "msp"}
    assert set(out["betas"]["ours_P2p1"]) == {"beta0", "a", "b", "g"}
    assert rows["ours_P1_c1"]["all"]["nll"] == pytest.approx(rows["ours_P1"]["all"]["nll"])
    assert rows["ours_P2p1"]["all"]["miou"] == rows["ours_P1"]["all"]["miou"]
    v1 = cli.main(base + ["--val-scenes", scenes, "--scenes", scenes, "--plain-json",
                          "--out", str(tmp_path / "v1.json")])
    assert set(v1["betas"]) == {"ours_P0", "ours_P1", "ours_P2", "ours_scalar", "ours_P3_b10",
                                "ours_P3_b20", "ours_P3_b40"}
    for r in v1["betas"]:                                       # v1 rows: same numbers
        assert v1["betas"][r] == out["betas"][r], r
    assert mt.choose_v2(dict(out, scenes=out["scenes"]), "@ours_v2", p)[0] in out["betas"]
    print(f"PASS eval_seg --v2-rows: {len(out['betas'])} rows fit and evaluated")


# --------------------------------------------------------------------- pipeline script
def _fake_repo(root):
    """A git repository with the v2 pipeline, the real eval_seg / make_tables / usplat and a
    gpu_gate.sh stub that runs the command (no GPU)."""
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    for f in ("replica_tab1_v2.sh", "make_tables.py", "eval_seg.py"):
        shutil.copy2(REPO / "scripts" / f, root / "scripts" / f)
    (root / "scripts" / "gpu_gate.sh").write_text("#!/usr/bin/env bash\nshift\nexec \"$@\"\n")
    (root / "scripts" / "gpu_gate.sh").chmod(0o755)
    shutil.copytree(REPO / "usplat", root / "usplat",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(PREREG, root / "configs" / "tab1_v2_prereg.json")
    git = lambda *a: subprocess.run(["git", "-C", str(root), "-c", "user.name=t",
                                     "-c", "user.email=t@example.com", *a],
                                    check=True, capture_output=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "init")
    return git


def test_replica_tab1_v2_script(tmp_path):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    t = _seg_tests()
    toy = tmp_path / "toy"
    toy.mkdir()
    t._cli_toy(toy)
    repo = tmp_path / "repo"
    git = _fake_repo(repo)
    out = tmp_path / "out"
    env = dict(os.environ, OUT=str(out), SCENES=str(toy / "scenes.json"),
               VOCAB=str(toy / "vocab.json"), DATASET="7scenes", STRIDE="1",
               EXTRACTOR="clearclip", ROW_FACTORY="", TEXT_CACHE=str(toy / "text.pt"),
               WITH_2D="0", BACKEND="mock", DEVICE="cpu", DOWNSCALE="16",
               EVAL_EXTRA="--clip-model m --clip-pretrained p")
    env.pop("PRIOR_LOOK", None)

    def run(stage, **kw):
        e = dict(env, STAGE=stage, **kw)
        return subprocess.run(["bash", str(repo / "scripts" / "replica_tab1_v2.sh")], env=e,
                              capture_output=True, text=True, timeout=600)

    def refused(r, msg):                 # refusals print to stdout (shell) or stderr (python)
        return r.returncode == 1 and msg in r.stdout + r.stderr

    r = run("test")                                             # nothing validated yet
    assert refused(r, "refused: no validation record"), r.stdout + r.stderr
    r = run("val", OUT="/home/thanhnc/gauss_ws/data/uags_results/tab1")
    assert refused(r, "v1 result directory"), r.stdout + r.stderr
    prereg = repo / "configs" / "tab1_v2_prereg.json"
    prereg.write_text(prereg.read_text().replace("2026-10-10", "2026-10-11"))
    r = run("val")
    assert refused(r, "is not committed"), r.stdout + r.stderr
    git("checkout", "--", "configs/tab1_v2_prereg.json")
    r = run("val")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    sel = json.loads((out / "selection_val.json").read_text())
    assert sel["status"] == "ok" and set(sel["metrics"]["selection"]["chosen"]) == set(mt.V2_KEYS)
    val = json.loads((out / "eval_val.json").read_text())
    assert val["status"] == "ok" and "ours_P4_inf" in val["metrics"]["betas"]
    assert "--v2-rows" in val["argv"]
    # the test stage refuses a dirty tree
    mtp = repo / "scripts" / "make_tables.py"
    mtp.write_text(mtp.read_text() + "\n# local edit\n")
    r = run("test")
    assert refused(r, "uncommitted changes"), r.stdout + r.stderr
    git("checkout", "--", "scripts/make_tables.py")
    r = run("test")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "SECOND LOOK" in r.stdout and "224fe1e" in r.stdout
    look = json.loads((out / "look_test.json").read_text())
    assert look["look"] == 2 and look["prior_looks"] == ["224fe1e"]
    assert look["chosen"] == sel["metrics"]["selection"]["chosen"]
    test = json.loads((out / "eval_test.json").read_text())
    assert test["status"] == "ok" and "--betas" in test["argv"]
    assert test["metrics"]["betas"] == val["metrics"]["betas"]
    assert "Ours P4 (vote predictive)" in (out / "table1_v2_rows.tex").read_text()
    # once: a second test run and a later validation run are refused
    r = run("test")
    assert refused(r, "finished v2 test run"), r.stdout + r.stderr
    r = run("val")
    assert refused(r, "selection is final"), r.stdout + r.stderr
    print("PASS replica_tab1_v2.sh: val -> selection; test gated, second look logged, once")
