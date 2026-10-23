"""Stage A/B/C code of configs/extractor_prereg.json on CPU toys (no GPU, no real frame):
scripts/extractor_2d.py (Table I's 2D-row path without a map), scripts/extractor_select.py
(the stage B rule, ties, the margin gate, the R0 reproduction gate, stage C choices, the
test gates, claim C1), and the refusals of scripts/extractor_select.sh / extractor_tab1.sh
end to end in a scratch git repo with toy extractors and a toy prereg."""
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
PREREG = REPO / "configs" / "extractor_prereg.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------- toy data
TOY_EXTRACTOR = '''
"""Toy extractor for tests: the colour of each 8-px cell, blurred over BLUR cells (mixes the
colours near a boundary) and with RED added everywhere (a bias toward 'chair')."""
import torch, torch.nn.functional as Fn
BLUR, RED = {blur}, {red}
D = 16
class Toy:
    def __init__(self, device="cpu"):
        self.d, self.device = D, torch.device(device)
    def extract_tokens(self, image):
        x = torch.as_tensor(image).float().permute(2, 0, 1)[None] / 255.0
        x = Fn.avg_pool2d(x, 8, ceil_mode=True)
        if BLUR > 1:
            x = Fn.avg_pool2d(x, BLUR, stride=1, padding=BLUR // 2, count_include_pad=False)
        x = x[0].permute(1, 2, 0)                                           # (h, w, 3)
        f = torch.zeros(*x.shape[:2], D)
        f[..., :3] = x
        f[..., 0] += RED
        return Fn.normalize(f + 1e-3, dim=-1)
    def extract(self, image, out_hw=None):
        from usplat.clip_dense import upsample_tokens
        return upsample_tokens(self.extract_tokens(image), out_hw or tuple(image.shape[:2]))
    def text_embedding(self, prompts, templates=("{}",)):
        rows = []
        for p in ([prompts] if isinstance(prompts, str) else prompts):
            t = torch.zeros(D)
            t[{"chair": 0, "table": 2, "sofa": 1}[p]] = 1.0
            rows.append(t)
        return torch.stack(rows)
def build(device="cpu", **kw):
    return Toy(device)
'''


def _fake_scene(root, n=3, split=320):
    """7-Scenes layout: left of `split` red with label 3 (chair), right blue with label 7
    (table); labels from row 40 on."""
    import cv2
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        base = root / f"frame-{i:06d}"
        img = np.zeros((480, 640, 3), np.uint8)
        img[:, :split, 2] = 255                                   # BGR: red left
        img[:, split:, 0] = 255                                   # blue right
        cv2.imwrite(str(base) + ".color.png", img)
        cv2.imwrite(str(base) + ".depth.png", np.full((480, 640), 2000, np.uint16))
        c2w = np.eye(4)
        c2w[0, 3] = 0.05 * i
        np.savetxt(str(base) + ".pose.txt", c2w)
        lab = np.zeros((480, 640), np.uint16)
        lab[40:, :split] = 3
        lab[40:, split:] = 7
        cv2.imwrite(str(base) + ".label.png", lab)


def _toy_world(tmp, root_for_paths=None, margin=0.01, inc="toyinc", cands=("toya", "toyb")):
    """Two toy datasets (toy1, toy2), each one validation + one test scene, a vocabulary, a
    toy weight file, Table I records (filled later), and a prereg in the real format."""
    pytest.importorskip("cv2")
    base = pathlib.Path(root_for_paths or tmp)
    vocab = dict(templates=["{}"], ignore=[0], classes={"3": {"name": "chair"}, "7": {"name": "table"},
                                                        "9": {"name": "sofa"}},
                 templates_by_extractor={x: ["{}"] for x in (inc, *cands)})
    (tmp / "vocab.json").write_text(json.dumps(vocab))
    wfile = tmp / "w.bin"
    wfile.write_bytes(b"toy weights")
    import hashlib
    datasets = {}
    for d, split in (("toy1", 320), ("toy2", 384)):   # splits on the 16-px label grid
        for sc in ("val", "test"):
            _fake_scene(tmp / d / sc, split=split)
        mk = lambda sc: [dict(name=f"{d}_{sc}", field=str(tmp / d / f"{sc}_field.pt"),
                              ply=str(tmp / d / f"{d}_{sc}.ply"), data=str(tmp / d / sc),
                              dataset="7scenes", stride=1,
                              label_pattern="{root}/frame-{idx:06d}.label.png")]
        (tmp / f"{d}_val.json").write_text(json.dumps(mk("val")))
        (tmp / f"{d}_test.json").write_text(json.dumps(mk("test")))
        for sc in ("val", "test"):
            (tmp / d / f"{d}_{sc}.ply").write_bytes(b"map")
        datasets[d] = dict(
            scenes=dict(validation=str(tmp / f"{d}_val.json"), test=str(tmp / f"{d}_test.json")),
            eval=dict(vocab=str(tmp / "vocab.json"), dataset="7scenes", stride=1, downscale=16, tau_cov=0.5),
            mapping=dict(dataset="7scenes", data=str(tmp / d / "val"), stride=1,
                         css_eval_data=str(tmp / d / "val"), css_eval_stride=1),
            maps=str(tmp / d / "{name}.ply"),
            table1=dict(val=str(tmp / f"t1_{d}_val.json"), test=str(tmp / f"t1_{d}_test.json"),
                        text_cache=str(tmp / "nocache.pt")),
            fields=str(base / "fields" / "extractor" / "{extractor}" / d),
            out=str(base / "results" / "extractor" / "{extractor}" / d),
            test_look=f"toy {d} look")
    p = json.loads(PREREG.read_text())
    p.update(registered="toy", incumbent=inc, order=list(cands),
             candidates={x: {"module": x} for x in cands},
             weights={str(wfile): dict(sha256=hashlib.sha256(b"toy weights").hexdigest(),
                                       source="toy", licence="toy", used_by=[inc, *cands])},
             third_party_code={}, datasets=datasets)
    p["stage_a"] = dict(p["stage_a"], protocol=dict(downscale=16, feat_dim=4, pca_frames=2, seed=0),
                        out=str(base / "results" / "extractor" / "select"))
    p["stage_b"] = dict(p["stage_b"], margin=margin)
    p["gpu"] = dict(p["gpu"], stage_a_need_mib={x: 100 for x in (inc, *cands)},
                    map_need_mib={x: 100 for x in cands}, distill_need_mib={x: 100 for x in cands},
                    eval_need_mib={x: 100 for x in cands})
    p["claim"] = dict(p["claim"], min_wins={"toy1": 1, "toy2": 1})
    return p


TOYS = {"toyinc": (15, 0.2), "toya": (1, 0.0), "toyb": (1, 2.0)}   # (blur, red): sharp toya wins


def _install_toys(pkg_dir=None):
    """Toy extractor modules: written into a scratch package dir, or registered in-process."""
    for name, (blur, red) in TOYS.items():
        src = TOY_EXTRACTOR.replace("{blur}", str(blur)).replace("{red}", str(red))
        if pkg_dir is not None:
            (pkg_dir / f"{name}.py").write_text(src)
        else:
            mod = types.ModuleType(f"usplat.extractors.{name}")
            exec(src, mod.__dict__)
            sys.modules[f"usplat.extractors.{name}"] = mod


def _x2d_args(prereg_path, dataset, x, out, cache):
    return ["--prereg", str(prereg_path), "--dataset", dataset, "--extractor", x, "--device", "cpu",
            "--downscale", "16", "--feat-dim", "4", "--pca-frames", "2", "--text-cache", str(cache),
            "--out", str(out)]


# ------------------------------------------------------------------------- extractor_2d
def test_extractor_2d_toy_and_refusals(tmp_path):
    p = _toy_world(tmp_path)
    pp = tmp_path / "prereg.json"
    pp.write_text(json.dumps(p))
    _install_toys()
    x2d = _load("extractor_2d_t", REPO / "scripts" / "extractor_2d.py")
    out = x2d.main(_x2d_args(pp, "toy1", "toya", tmp_path / "a.json", tmp_path / "c.pt"))
    r = out["scenes"]["toy1_val"]
    assert r["n_frames"] == 3 and r["n_classes"] == 2 and r["pca_fit"]["frames"] == 2
    assert r["pca"]["miou"] == pytest.approx(1.0) and r["full"]["miou"] == pytest.approx(1.0)
    assert r["pca"]["acc"] == pytest.approx(1.0)
    assert out["smoke"] is False and json.loads((tmp_path / "a.json").read_text())["status"] == "ok"
    bad = x2d.main(_x2d_args(pp, "toy1", "toyb", tmp_path / "b.json", tmp_path / "c2.pt"))
    assert bad["scenes"]["toy1_val"]["pca"]["miou"] < 0.6        # red bias: all 'chair'
    inc = x2d.main(_x2d_args(pp, "toy1", "toyinc", tmp_path / "i.json", tmp_path / "c3.pt"))
    assert 0.6 < inc["scenes"]["toy1_val"]["pca"]["miou"] < 0.99   # blurred boundary
    # determinism: the same record twice
    again = x2d.main(_x2d_args(pp, "toy1", "toya", tmp_path / "a2.json", tmp_path / "c.pt"))
    assert all(again["scenes"][s][k] == out["scenes"][s][k] for s in out["scenes"]
               for k in ("pca", "full", "pca_fit"))
    with pytest.raises(SystemExit, match="not in the prereg"):
        x2d.main(_x2d_args(pp, "toy1", "lseg", tmp_path / "z.json", tmp_path / "z.pt"))
    with pytest.raises(SystemExit, match="stage A protocol"):
        x2d.main(_x2d_args(pp, "toy1", "toya", tmp_path / "z.json", tmp_path / "z.pt")[:-2]
                 + ["--seed", "1", "--out", str(tmp_path / "z.json")])
    with pytest.raises(SystemExit, match="not a validation scene"):
        x2d.main(_x2d_args(pp, "toy1", "toya", tmp_path / "z.json", tmp_path / "z.pt")
                 + ["--scene", "toy1_test"])
    sm = x2d.main(_x2d_args(pp, "toy1", "toya", tmp_path / "s.json", tmp_path / "c.pt")
                  + ["--max-views", "1"])
    assert sm["smoke"] is True and sm["scenes"]["toy1_val"]["n_frames"] == 1
    print(f"PASS extractor_2d: toy mIoU 1.0 (sharp), {inc['scenes']['toy1_val']['pca']['miou']:.3f} "
          f"(blurred), {bad['scenes']['toy1_val']['pca']['miou']:.3f} (biased); deterministic; refusals")


# ------------------------------------------------------------------------- rules
def test_stage_b_rule_ties_margin():
    sel = _load("extractor_select_t", REPO / "scripts" / "extractor_select.py")
    p = dict(incumbent="inc", order=["a", "b", "c"], datasets={"r": {}, "s": {}},
             stage_b=dict(tie_tol=0.001, margin=0.01))
    tab = {"inc": {"r": 0.30, "s": 0.60}, "a": {"r": 0.32, "s": 0.605}, "b": {"r": 0.40, "s": 0.59},
           "c": {"r": 0.31, "s": 0.615}}
    w, deltas, score = sel.rule(p, tab)
    # min deltas: a +0.005, b -0.01, c +0.01 -> c wins; c's 0.01 >= margin
    assert w == "c" and score["c"] == pytest.approx(0.01) and score["b"] == pytest.approx(-0.01)
    tab["a"] = {"r": 0.3105, "s": 0.6105}                                 # a ties c within 0.001
    w, _, _ = sel.rule(p, tab)
    assert w == "a"                                                      # earlier in order
    tab["a"] = {"r": 0.50, "s": 0.55}                                     # big on one, worse on other
    w, _, score = sel.rule(p, tab)
    assert w == "c" and score["a"] < 0                                    # per-dataset min decides
    print("PASS stage B rule: max of the per-dataset minimum gain, ties to the earlier name")


def _write_a(path, dataset, x, scenes, prereg_path, commit="c" * 40, smoke=False, status="ok"):
    import hashlib
    body = dict(dataset=dataset, extractor=x, smoke=smoke,
                scenes={s: dict(pca=dict(miou=v, acc=v), full=dict(miou=v, acc=v),
                                ms_per_frame_extract=10.0) for s, v in scenes.items()},
                summary=dict(pca=dict(miou=sum(scenes.values()) / len(scenes))),
                prereg=dict(sha256=hashlib.sha256(open(prereg_path, "rb").read()).hexdigest()),
                peak_alloc_mib=100.0)
    path.write_text(json.dumps(dict(status=status, metrics=body,
                                    git=dict(commit=commit, dirty_tracked=False))))


def test_select_gates(tmp_path):
    sel = _load("extractor_select_t2", REPO / "scripts" / "extractor_select.py")
    p = _toy_world(tmp_path)
    pp = tmp_path / "prereg.json"
    pp.write_text(json.dumps(p))
    d = tmp_path / "A"
    d.mkdir()
    vals = {"toyinc": 0.50, "toya": 0.55, "toyb": 0.60}
    for ds in ("toy1", "toy2"):
        (tmp_path / f"t1_{ds}_val.json").write_text(json.dumps(dict(status="ok", scenes={
            f"{ds}_val": dict(rows={"2d_features": {"all": {"miou": 0.50}}})})))
        for x, v in vals.items():
            _write_a(d / f"a_{ds}_{x}.json", ds, x, {f"{ds}_val": v + (0.0 if ds == "toy1" or x != "toyb" else -0.2)}, pp)
    args = lambda: types.SimpleNamespace(prereg=str(pp), dir=str(d), out=str(d / "selection_a.json"))
    out = sel.select(args())
    assert out["reproduction_passed"] and out["winner"] == "toya" and out["gate_passed"]
    assert sel.winner(types.SimpleNamespace(prereg=str(pp), dir=str(d))) == "toya"
    # a stage-A record edited after the selection -> refused
    rec = json.loads((d / "a_toy1_toya.json").read_text())
    (d / "a_toy1_toya.json").write_text(json.dumps(rec) + " ")
    with pytest.raises(SystemExit, match="changed after the selection"):
        sel.winner(types.SimpleNamespace(prereg=str(pp), dir=str(d)))
    (d / "a_toy1_toya.json").write_text(json.dumps(rec))
    # R0: the incumbent does not reproduce Table I -> the gate fails whatever the gains
    (tmp_path / "t1_toy2_val.json").write_text(json.dumps(dict(status="ok", scenes={
        "toy2_val": dict(rows={"2d_features": {"all": {"miou": 0.52}}})})))
    out = sel.select(args())
    assert not out["reproduction_passed"] and not out["gate_passed"]
    with pytest.raises(SystemExit, match="gate not passed"):
        sel.winner(types.SimpleNamespace(prereg=str(pp), dir=str(d)))
    (tmp_path / "t1_toy2_val.json").write_text(json.dumps(dict(status="ok", scenes={
        "toy2_val": dict(rows={"2d_features": {"all": {"miou": 0.50}}})})))
    # margin not reached -> attempt ends
    p2 = dict(p, stage_b=dict(p["stage_b"], margin=0.10))
    pp.write_text(json.dumps(p2))
    for ds in ("toy1", "toy2"):
        for x, v in vals.items():
            _write_a(d / f"a_{ds}_{x}.json", ds, x, {f"{ds}_val": v}, pp)
    out = sel.select(args())
    assert out["winner"] == "toyb" and not out["gate_passed"] and "attempt ends" in out["decision"]
    # refusals: smoke record, other prereg, mixed commits, unfinished record
    _write_a(d / "a_toy1_toya.json", "toy1", "toya", {"toy1_val": 0.55}, pp, smoke=True)
    with pytest.raises(SystemExit, match="smoke"):
        sel.select(args())
    _write_a(d / "a_toy1_toya.json", "toy1", "toya", {"toy1_val": 0.55}, pp, commit="d" * 40)
    with pytest.raises(SystemExit, match="commits"):
        sel.select(args())
    _write_a(d / "a_toy1_toya.json", "toy1", "toya", {"toy1_val": 0.55}, pp, status="running")
    with pytest.raises(SystemExit, match="not finished"):
        sel.select(args())
    _write_a(d / "a_toy1_toya.json", "toy1", "toya", {"toy1_test": 0.55}, pp)
    with pytest.raises(SystemExit, match="validation scenes"):
        sel.select(args())
    print("PASS select: winner, R0 gate, margin gate, input hashes, refusals")


def test_claim_c1(tmp_path):
    sel = _load("extractor_select_t3", REPO / "scripts" / "extractor_select.py")
    p = _toy_world(tmp_path)
    pp = tmp_path / "prereg.json"
    pp.write_text(json.dumps(p))

    def rec(path, vals, avg):
        sc = {f"s{i}": dict(rows={"ours_P2": {"all": {m: v for m in ("miou", "acc", "ece", "nll", "brier", "aurc")}},
                                  "averaging": {"all": {m: a for m in ("miou", "acc", "ece", "nll", "brier", "aurc")}}})
              for i, (v, a) in enumerate(zip(vals, avg))}
        path.write_text(json.dumps(dict(status="ok", scenes=sc)))
    rec(tmp_path / "t1_toy1_test.json", [0.5, 0.5, 0.5], [0.5] * 3)
    rec(tmp_path / "t1_toy2_test.json", [0.5, 0.5, 0.5], [0.5] * 3)
    rec(tmp_path / "n1.json", [0.6, 0.4, 0.55], [0.7] * 3)       # mean up, 2/3 better
    rec(tmp_path / "n2.json", [0.5, 0.5, 0.5], [0.4] * 3)        # ties: no win
    a = types.SimpleNamespace(prereg=str(pp), datasets=["toy1", "toy2"],
                              tests=[str(tmp_path / "n1.json"), str(tmp_path / "n2.json")],
                              out=str(tmp_path / "claim.json"))
    out = sel.claim(a)
    assert out["datasets"]["toy1"]["C1_headline"] is True               # min_wins 1 in the toy
    assert out["datasets"]["toy2"]["C1_headline"] is False
    assert out["datasets"]["toy1"]["C1"]["miou"]["wins"] == 2
    assert out["datasets"]["toy1"]["C1"]["nll"]["wins"] == 1            # lower is better
    assert out["datasets"]["toy1"]["vs_averaging_as_is"]["miou"]["ours_better_in"] == 0
    print("PASS claim C1: mean + min wins per dataset, ties no win, lower-is-better metrics")


# ------------------------------------------------------------------------- drivers
def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True,
                          env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))


def test_drivers_end_to_end(tmp_path):
    """extractor_select.sh: every gate, then stage A + B on the toys -> winner toya; a second run
    refused. extractor_tab1.sh: refusals (gate failed, not registered, stage C before stage A,
    test before val, dirty tree) and the dry-run plan of a validation stage."""
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    pytest.importorskip("cv2")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    for f in ("extractor_2d.py", "extractor_select.py", "extractor_select.sh", "extractor_tab1.sh",
              "eval_seg.py", "make_tables.py", "disk_guard.py"):
        shutil.copy2(REPO / "scripts" / f, repo / "scripts" / f)
    (repo / "scripts" / "gpu_gate.sh").write_text("#!/usr/bin/env bash\nshift\nexec \"$@\"\n")
    (repo / "scripts" / "gpu_gate.sh").chmod(0o755)
    shutil.copytree(REPO / "usplat", repo / "usplat", ignore=shutil.ignore_patterns("__pycache__"))
    _install_toys(repo / "usplat" / "extractors")
    data = tmp_path / "data"
    data.mkdir()
    roots = tmp_path / "roots"                       # UAGS_FIELDS_ROOT / UAGS_RESULTS_ROOT
    p = _toy_world(data, root_for_paths=roots)
    p["stage_a"]["out"] = str(tmp_path / "out_select")
    pr = "configs/extractor_prereg.json"
    (repo / pr).write_text(json.dumps(p, indent=1))
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    env = dict(os.environ, PREREG=pr, DEVICE="cpu", PYTHONPATH="",
               UAGS_FIELDS_ROOT=str(roots / "fields"), UAGS_RESULTS_ROOT=str(roots / "results"))
    for k in list(env):
        if k.startswith(("UAGS_LSEG_", "UAGS_PROXYCLIP_")):
            env.pop(k)
    env.pop("OUT", None)

    def run(script, **kw):
        return subprocess.run(["bash", str(repo / "scripts" / script)], env=dict(env, **kw),
                              capture_output=True, text=True, timeout=900)

    def refused(r, msg):
        return r.returncode == 1 and msg in r.stdout + r.stderr

    # stage A/B gates
    r = run("extractor_select.sh", UAGS_LSEG_FLIP="1")
    assert refused(r, "UAGS_LSEG_FLIP is set"), r.stdout + r.stderr
    r = run("extractor_select.sh", OUT="/home/thanhnc/gauss_ws/data/uags_results/tab1/x")
    assert refused(r, "never touched"), r.stdout + r.stderr
    r = run("extractor_select.sh", OUT="/home/thanhnc/gauss_ws/data/uags_fields/x")
    assert refused(r, "OUT inside"), r.stdout + r.stderr
    (repo / pr).write_text((repo / pr).read_text().replace('"registered": "toy"', '"registered": ""'))
    r = run("extractor_select.sh")
    assert refused(r, "is not committed"), r.stdout + r.stderr
    _git(repo, "commit", "-qam", "unregistered")
    r = run("extractor_select.sh")
    assert refused(r, "not registered"), r.stdout + r.stderr
    _git(repo, "revert", "--no-edit", "HEAD")
    w = pathlib.Path(next(iter(p["weights"])))
    w.write_bytes(b"tampered")
    r = run("extractor_select.sh")
    assert refused(r, "sha256"), r.stdout + r.stderr
    w.write_bytes(b"toy weights")
    (repo / "usplat" / "extractors" / "toya.py").write_text(
        (repo / "usplat" / "extractors" / "toya.py").read_text() + "\n# local edit\n")
    r = run("extractor_select.sh")
    assert refused(r, "uncommitted changes to tracked files"), r.stdout + r.stderr
    _git(repo, "checkout", "--", "usplat/extractors/toya.py")
    r = run("extractor_select.sh", DRY_RUN="1")
    assert r.returncode == 0 and r.stdout.count("[dry-run]") == 6, r.stdout + r.stderr
    # stage C cannot start before stage B
    r = run("extractor_tab1.sh", DATASET="toy1", STAGE="val")
    assert refused(r, "no stage-A selection"), r.stdout + r.stderr
    # the Table I 2D records the incumbent must reproduce: computed by the same code
    _install_toys()
    x2d = _load("extractor_2d_t2", REPO / "scripts" / "extractor_2d.py")
    for ds in ("toy1", "toy2"):
        o = x2d.main(_x2d_args(repo / pr, ds, "toyinc", tmp_path / f"r_{ds}.json", tmp_path / f"r_{ds}.pt"))
        pathlib.Path(p["datasets"][ds]["table1"]["val"]).write_text(json.dumps(dict(status="ok", scenes={
            s: dict(rows={"2d_features": {"all": {"miou": v["pca"]["miou"]}}}) for s, v in o["scenes"].items()})))
    r = run("extractor_select.sh")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    sel = json.loads((tmp_path / "out_select" / "selection_a.json").read_text())
    assert sel["winner"] == "toya" and sel["reproduction_passed"] and sel["gate_passed"], sel
    for ds in ("toy1", "toy2"):
        a = json.loads((tmp_path / "out_select" / f"a_{ds}_toya.json").read_text())
        assert a["status"] == "ok" and a["git"]["dirty_tracked"] is False
    r = run("extractor_select.sh")
    assert refused(r, "the selection is final"), r.stdout + r.stderr
    # stage C: refusals, then the dry-run plan of the validation stage
    r = run("extractor_tab1.sh", STAGE="val")
    assert r.returncode == 2 and "set DATASET" in r.stdout, r.stdout + r.stderr
    r = run("extractor_tab1.sh", DATASET="toy1", STAGE="val", UAGS_PROXYCLIP_SHORT="448")
    assert refused(r, "UAGS_PROXYCLIP_SHORT is set"), r.stdout + r.stderr
    r = run("extractor_tab1.sh", DATASET="toy1", STAGE="val", UAGS_RESULTS_ROOT="/home/thanhnc/gauss_ws/data")
    assert refused(r, "is not under"), r.stdout + r.stderr
    r = run("extractor_tab1.sh", DATASET="toy1", STAGE="test")
    assert refused(r, "no finished stage C validation selection"), r.stdout + r.stderr
    r = run("extractor_tab1.sh", DATASET="toy1", STAGE="val", DRY_RUN="1")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = [l for l in r.stdout.splitlines() if l.startswith("[dry-run]")]
    assert len(plan) == 7, plan                      # map, f3dgs + 3 LEGaussians, eval, select-c
    assert "--extractor toya" in plan[0] and "_toya.pt" in plan[0] and "--vocab" in plan[0]
    assert "--with-2d" in plan[5] and "scenes_val.json" in plan[5]
    sc = json.loads((roots / "results" / "extractor" / "toya" / "toy1" / "scenes_val.json").read_text())
    assert sc[0]["field"].endswith("toy1_val_toya.pt") and sc[0]["ply"] == str(data / "toy1" / "toy1_val.ply")
    (repo / "scripts" / "eval_seg.py").write_text((repo / "scripts" / "eval_seg.py").read_text() + "\n# edit\n")
    r = run("extractor_tab1.sh", DATASET="toy1", STAGE="val", DRY_RUN="1")
    assert refused(r, "uncommitted changes to tracked files"), r.stdout + r.stderr
    _git(repo, "checkout", "--", "scripts/eval_seg.py")
    # a failed gate stops stage C: margin above every gain
    p2 = json.loads((repo / pr).read_text())
    p2["stage_b"]["margin"] = 0.9
    sel2 = _load("extractor_select_t4", REPO / "scripts" / "extractor_select.py")
    pp2 = tmp_path / "p2.json"
    pp2.write_text(json.dumps(p2))
    out2 = tmp_path / "out_select2"
    shutil.copytree(tmp_path / "out_select", out2)
    os.remove(out2 / "selection_a.json")
    with pytest.raises(SystemExit, match="another prereg"):          # records name the prereg
        sel2.select(types.SimpleNamespace(prereg=str(pp2), dir=str(out2), out=str(out2 / "s.json")))
    print("PASS drivers: gates refuse; stage A+B -> toya; second run refused; stage C gates")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
