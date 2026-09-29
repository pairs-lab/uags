"""K14/K15: run records, config hashes, seeds and section timers (CPU only).

 * a RunRecord writes every field (command, git state, versions, seed, config + hash,
   inputs, timestamps, wall time, memory, metrics, timing) as strict JSON, first with
   status "running", then "ok"; a CPU run creates no CUDA context;
 * numpy / torch scalars, paths and NaN in seed, name, tags, argv, timing_per, timer
   durations and metrics never break the write (no temp-file debris, no leaked watcher);
 * an exception still writes the record, with status "failed" and the traceback, and is
   re-raised; a failure inside __enter__ writes a "failed" record; when the record itself
   cannot be completed, a fallback record is written and the run's own exception wins;
 * a relative record path survives a chdir; the command keeps interpreter flags and -m;
 * the config hash does not depend on key order (nested too), changes with any value, and
   keeps int and str keys apart;
 * in a scratch git repo: dirty flag, diff hash = sha256 of `git diff HEAD --binary
   --full-index` (staged and unstaged edits, staged new files and deletions), untracked
   sources and configs (.json, .txt) hashed, excluded paths anchored at the top, run
   records skipped,
   and the saved patch turns a clean clone of the commit back into the working tree;
   non-UTF-8 names work and user git diff config does not change the hash;
 * Timer totals and statistics (numpy reference) on a fake clock, CUDA sync at section
   boundaries, warm-up handling, ms per keyframe; set_seed repeats and sets deterministic
   mode; nvidia-smi parsing;
 * peak memory (simulated CUDA counters): per-section peaks and the run peak survive the
   counter resets of nested sections.
"""
import hashlib
import json
import os
import pathlib
import random
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

import usplat.runlog as rl
from usplat.runlog import (RunRecord, Timer, set_seed, config_hash, canonical_json,
                           git_state, load_record)
from usplat.vmf_field import FieldConfig

HEX64 = set("0123456789abcdef")


def _strict_load(path):
    def bad(c):
        raise ValueError(f"non-standard JSON constant {c}")
    with open(path) as f:
        return json.load(f, parse_constant=bad)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           *args], check=True, capture_output=True).stdout


def _gpu_visible():
    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def test_record_fields(tmp_path):
    cuda_before = torch.cuda.is_initialized()
    path = tmp_path / "runs" / "demo.json"
    cfg = {"feat_dim": 64, "field": FieldConfig(), "lr": 0.1, "names": ("a", "b")}
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"classes": ["wall"]}\n')
    with RunRecord(path, config=cfg, seed=7, name="demo", argv=["run.py", "--x", "1"],
                   save_patch=False) as rec:
        first = _strict_load(path)                      # written on entry
        assert first["status"] == "running" and first["end"] is None
        rec.add_input(vocab)
        for _ in range(3):
            with rec.timer.section("keyframe"):
                time.sleep(0.01)
        rec.add("ap", torch.tensor(0.5))
        rec.add("undefined_ap", float("nan"))
        rec.add("per_class", np.array([1.0, 2.0]))
        rec.timing_per = 3
    r = _strict_load(path)
    for k in ("schema", "name", "run_id", "status", "argv", "command", "orig_argv", "cwd",
              "start", "end", "wall_s", "cpu_s", "seed", "config", "config_sha256", "git",
              "env", "memory", "metrics", "inputs", "outputs", "timing", "error",
              "record_error", "deterministic_algorithms"):
        assert k in r, k
    assert r["status"] == "ok" and r["error"] is None and r["record_error"] is None
    assert r["seed"] == 7
    assert r["argv"] == ["run.py", "--x", "1"] and r["command"].endswith("run.py --x 1")
    assert r["config_sha256"] == config_hash(cfg) and set(r["config_sha256"]) <= HEX64
    assert r["config"]["field"]["change_thr"] == "inf"  # FieldConfig default, strict JSON
    assert r["inputs"] == [{"path": str(vocab.resolve()), "bytes": vocab.stat().st_size,
                            "sha256": hashlib.sha256(vocab.read_bytes()).hexdigest()}]
    g = r["git"]
    assert "error" not in g, g
    import usplat
    top = subprocess.run(["git", "-C", usplat.__path__[0], "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert g["commit"] == top and isinstance(g["dirty"], bool)
    assert len(g["diff_sha256"]) == 64 and set(g["diff_sha256"]) <= HEX64
    assert len(g["code_sha256"]) == 64
    assert all(not u["path"].startswith(("docs/", "papers/", "outputs/"))
               for u in g["untracked"])
    e = r["env"]
    assert e["python"] and e["torch"] == torch.__version__ and e["numpy"] == np.__version__
    assert "gsplat" in e and "gpus" in e
    if _gpu_visible():                                  # nvidia-smi works -> GPUs listed
        assert isinstance(e["gpus"], list) and e["gpus"] and e["gpus"][0]["name"]
    if "PYTHONPATH" in os.environ:
        assert e["env"]["PYTHONPATH"] == os.environ["PYTHONPATH"]
    assert r["wall_s"] >= 0.03 and r["start"] < r["end"]
    assert r["metrics"] == {"ap": 0.5, "undefined_ap": "nan", "per_class": [1.0, 2.0]}
    t = r["timing"]["keyframe"]
    assert t["count"] == 3 and t["total_s"] >= 0.03
    assert abs(t["ms_per_keyframe"] - t["total_s"] * 1e3 / 3) < 1e-9
    assert r["memory"]["peak_rss_bytes"] > 0
    if not cuda_before:                                 # CPU run: no GPU context created
        assert not torch.cuda.is_initialized()
        assert r["memory"]["cuda_initialized"] is False
        assert r["memory"]["peak_gpu_alloc_bytes"] is None
    assert load_record(path)["name"] == "demo"
    print(f"PASS record: {len(r)} fields, commit {g['commit'][:8]} dirty={g['dirty']}, "
          f"config {r['config_sha256'][:12]}, wall {r['wall_s']:.3f} s")


def test_record_json_robust(tmp_path):
    """numpy / torch scalars, paths and NaN anywhere in the record (review R1-R3, R8)."""
    path = tmp_path / "robust.json"
    tm = Timer()
    tm.add("external", float("nan"))                    # e.g. a failed timing
    with RunRecord(path, seed=np.int64(3), name=pathlib.Path("demo"), timer=tm,
                   tags=[pathlib.Path("tab2"), np.int64(2)],
                   argv=["run.py", pathlib.Path("cfg.yaml")], save_patch=False) as rec:
        with rec.timer.section("keyframe"):
            pass
        rec.add("out_dir", pathlib.Path("outputs/x"))
        rec.timing_per = np.int64(4)                    # e.g. n_kf = mask.sum()
    r = _strict_load(path)
    assert r["status"] == "ok" and r["seed"] == 3 and type(r["seed"]) is int
    assert r["name"] == "demo" and r["tags"] == ["tab2", 2]
    assert r["argv"] == ["run.py", "cfg.yaml"] and r["command"].endswith("run.py cfg.yaml")
    assert r["metrics"]["out_dir"] == "outputs/x"
    assert r["timing_per"] == {"unit": "keyframe", "count": 4}
    kf = r["timing"]["keyframe"]
    assert abs(kf["ms_per_keyframe"] - kf["total_s"] * 1e3 / 4) < 1e-9
    assert r["timing"]["external"]["total_s"] == "nan"
    # a torch scalar for timing_per gives the same numbers
    s_np, s_t = tm.summary(per=np.int64(4)), tm.summary(per=torch.tensor(4))
    assert s_np["keyframe"]["ms_per_keyframe"] == s_t["keyframe"]["ms_per_keyframe"]
    assert not list(tmp_path.glob(".*.tmp")) and rl._WATCHES == []
    with pytest.raises(TypeError):
        RunRecord(tmp_path / "bad.json", seed=3.5)      # a float seed is a bug, not a seed
    print("PASS robust: numpy/torch/path/NaN values written, no debris")


def test_failed_status(tmp_path):
    path = tmp_path / "fail.json"
    with pytest.raises(ValueError, match="boom"):
        with RunRecord(path, config={"a": 1}, seed=0, save_patch=False) as rec:
            rec.add("partial", 3)
            rec.timing_per = torch.tensor(16)           # review R2b: must not mask the error
            with rec.timer.section("keyframe"):
                raise ValueError("boom at keyframe 0")
    r = _strict_load(path)
    assert r["status"] == "failed" and r["end"] is not None
    assert r["error"]["type"] == "ValueError" and "boom" in r["error"]["message"]
    assert "Traceback" in r["error"]["traceback"] and "boom" in r["error"]["traceback"]
    assert r["metrics"] == {"partial": 3} and r["record_error"] is None
    assert r["timing"]["keyframe"]["count"] == 1        # a raising section is still timed
    # SystemExit(0) is a normal end, SystemExit(2) is not; Ctrl-C is a failure
    with pytest.raises(SystemExit):
        with RunRecord(tmp_path / "exit0.json", save_patch=False):
            raise SystemExit(0)
    assert _strict_load(tmp_path / "exit0.json")["status"] == "ok"
    with pytest.raises(SystemExit):
        with RunRecord(tmp_path / "exit2.json", save_patch=False):
            raise SystemExit(2)
    assert _strict_load(tmp_path / "exit2.json")["status"] == "failed"
    with pytest.raises(KeyboardInterrupt):
        with RunRecord(tmp_path / "ctrlc.json", save_patch=False):
            raise KeyboardInterrupt
    assert _strict_load(tmp_path / "ctrlc.json")["error"]["type"] == "KeyboardInterrupt"
    assert rl._WATCHES == []
    print("PASS failed: status, error, traceback and partial metrics written; exit codes")


def test_record_failure_paths(tmp_path):
    """The record's own failures: in __enter__ (review R7) and while finishing (R2b)."""
    # 1) bad config: __exit__ never runs, so __enter__ writes the failed record itself
    p = tmp_path / "enter.json"
    with pytest.raises(TypeError):
        with RunRecord(p, config={"f": object()}, save_patch=False):
            pass
    r = _strict_load(p)
    assert r["status"] == "failed" and r["error"]["type"] == "TypeError"
    assert r["error"]["stage"] == "enter" and rl._WATCHES == []

    def boom():
        raise RuntimeError("cuda query failed after a device error")

    # 2) the run raised and the snapshot fails too: the run's exception reaches the caller
    p = tmp_path / "both.json"
    with pytest.warns(RuntimeWarning, match="incomplete"):
        with pytest.raises(ValueError, match="real failure"):
            with RunRecord(p, save_patch=False) as rec:
                rec.add("partial", np.float32(0.25))
                rec._snapshot = boom
                raise ValueError("real failure")
    r = _strict_load(p)
    assert r["status"] == "failed" and r["error"]["type"] == "ValueError"
    assert r["record_error"]["type"] == "RuntimeError" and r["metrics"] == {"partial": 0.25}
    assert rl._WATCHES == [] and not list(tmp_path.glob(".*.tmp"))

    # 3) the run finished but the record cannot be completed: status "incomplete", raised
    p = tmp_path / "rec_only.json"
    with pytest.raises(RuntimeError, match="cuda query"):
        with RunRecord(p, save_patch=False) as rec:
            rec.add("ap", 0.5)
            rec._snapshot = boom
    r = _strict_load(p)
    assert r["status"] == "incomplete" and r["error"] is None
    assert r["record_error"]["type"] == "RuntimeError" and r["metrics"] == {"ap": 0.5}
    assert rl._WATCHES == []

    # 4) a Timer section whose closing sync fails keeps the body's exception
    tm = Timer(memory=True)

    def bad_sync():
        raise RuntimeError("sync failed")
    with pytest.raises(ValueError, match="kernel"):
        with tm.section("keyframe"):
            tm._sync = bad_sync
            raise ValueError("kernel error")
    assert tm._open == {} and rl._WATCHES == []
    print("PASS failure paths: enter failure recorded, run exception never masked")


def test_relative_path_and_command(tmp_path, monkeypatch):
    # a relative record path is fixed at construction: a chdir in the run changes nothing
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    with RunRecord("rel/run.json", save_patch=False) as rec:
        monkeypatch.chdir(tmp_path / "sub")
        rec.add("x", 1)
    r = _strict_load(tmp_path / "rel" / "run.json")
    assert r["status"] == "ok" and r["metrics"] == {"x": 1}
    assert not (tmp_path / "sub" / "rel").exists()

    # the command is the interpreter command line (flags, -m), argv the script's arguments
    monkeypatch.setenv("PYTHONPATH", ".")
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-test")
    for argv in (None, list(sys.argv)):
        p = tmp_path / "cmd.json"
        with RunRecord(p, argv=argv, save_patch=False):
            pass
        r = _strict_load(p)
        assert r["command"] == shlex.join(sys.orig_argv) and r["argv"] == sys.argv
        assert r["orig_argv"] == sys.orig_argv
        assert r["env"]["env"]["PYTHONPATH"] == "."
        assert r["env"]["env"]["CUDA_HOME"] == "/opt/cuda-test"
    with RunRecord(p, argv=["run.py", "--a"], save_patch=False):
        pass
    assert _strict_load(p)["command"] == shlex.join([os.path.basename(sys.executable),
                                                     "run.py", "--a"])
    print(f"PASS path/command: {shlex.join(sys.orig_argv)[:60]}")


def test_config_hash_stable():
    a = {"b": {"y": [1, 2.5], "x": None}, "a": 1, "c": float("inf")}
    b = {"c": float("inf"), "a": 1, "b": {"x": None, "y": [1, 2.5]}}
    assert canonical_json(a) == canonical_json(b)
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({**a, "a": 2})
    assert config_hash({"a": 1}) != config_hash({"a": 1.0})     # int and float differ
    import argparse
    import dataclasses
    ns = argparse.Namespace(lr=0.1, dim=64)
    assert config_hash(ns) == config_hash({"dim": 64, "lr": 0.1})

    @dataclasses.dataclass
    class C:
        dim: int = 64
        lr: float = 0.1
    assert config_hash(C()) == config_hash({"lr": 0.1, "dim": 64})
    assert config_hash([C(), {"k": C()}]) == config_hash([{"dim": 64, "lr": 0.1},
                                                          {"k": {"dim": 64, "lr": 0.1}}])
    assert config_hash(FieldConfig()) == config_hash(FieldConfig().to_dict())
    with pytest.raises(TypeError):
        config_hash({"f": object()})                    # no stable text form
    # keys (review R5/R6): int and str keys stay apart, key order still does not matter
    assert config_hash({1: "a"}) != config_hash({"1": "a"})
    assert config_hash({True: "a"}) != config_hash({1: "a"})
    assert config_hash({1: "a", 2: "b"}) == config_hash({2: "b", 1: "a"})
    assert config_hash({np.int64(1): "a"}) == config_hash({1: "a"})
    assert config_hash({1: "a", "1": "b"}) != config_hash({"1": "a", 1: "b"})

    class K:
        pass
    for bad in ({K(): 1}, {1.5: "a"}, {(1, 2): "a"}, {rl._ITEMS: []}):
        with pytest.raises(TypeError):
            config_hash(bad)                            # repr would embed an address
    print(f"PASS config hash: key-order invariant ({config_hash(a)[:12]}), keys typed")


def test_git_state_and_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    (repo / "b.py").write_text("gone = 1\n")
    (repo / "m.pyc").write_bytes(b"\x00\x01")
    (repo / ".gitignore").write_text("ignored.py\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    clean = git_state(repo)
    assert clean["dirty"] is False and clean["untracked"] == []
    assert clean["diff_sha256"] == hashlib.sha256(b"").hexdigest()

    (repo / "a.py").write_text("x = 2\n")               # staged edit ...
    _git(repo, "add", "a.py")
    (repo / "a.py").write_text("x = 3\n")               # ... then edited again
    (repo / "s.py").write_text("s = 1\n")               # staged new file
    _git(repo, "add", "s.py")
    _git(repo, "rm", "-q", "b.py")                      # staged deletion
    (repo / "m.pyc").write_bytes(b"\x00\x02")           # excluded (bytecode)
    (repo / "new.py").write_text("y = 3\n")             # untracked source
    (repo / "configs").mkdir()
    (repo / "configs" / "vocab.json").write_text('{"classes": ["wall", "floor"]}\n')
    (repo / "notes.txt").write_text("n\n")              # untracked text config
    (repo / "ignored.py").write_text("z\n")             # git-ignored
    for d in ("docs", "outputs", "papers"):             # excluded top-level directories
        (repo / d).mkdir()
        (repo / d / "gen.py").write_text("r\n")
    for d in ("scripts/outputs", "usplat/docs"):        # same names, nested: code
        (repo / d).mkdir(parents=True)
        (repo / d / "gen.py").write_text("w\n")
    with RunRecord(repo / "results" / "prev.json", repo=repo, save_patch=False):
        pass                                            # a run record inside the repo
    head = _git(repo, "rev-parse", "HEAD").decode().strip()

    path = tmp_path / "runs" / "r.json"
    with RunRecord(path, config={}, repo=repo, save_patch=True):
        pass
    g = _strict_load(path)["git"]
    ref = _git(repo, "diff", "--binary", "--full-index", "HEAD", "--", "a.py", "b.py", "s.py")
    assert g["commit"] == head and g["dirty"] and g["dirty_tracked"]
    assert g["diff_sha256"] == hashlib.sha256(ref).hexdigest()
    assert [u["path"] for u in g["untracked"]] == [
        "configs/vocab.json", "new.py", "notes.txt", "scripts/outputs/gen.py",
        "usplat/docs/gen.py"]
    assert g["untracked"][1]["sha256"] == hashlib.sha256(b"y = 3\n").hexdigest()

    # the patch turns a clean checkout of the commit into the code that ran
    pf = path.parent / "patches" / f"{g['patch_sha256']}.patch"
    assert pf.exists() and str(pf) == g["patch_file"] and g["patch_complete"] is True
    assert hashlib.sha256(pf.read_bytes()).hexdigest() == g["patch_sha256"]
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True)
    _git(clone, "apply", str(pf))
    assert (clone / "a.py").read_text() == "x = 3\n" and (clone / "s.py").read_text() == "s = 1\n"
    assert not (clone / "b.py").exists() and (clone / "new.py").read_text() == "y = 3\n"
    assert (clone / "configs" / "vocab.json").read_text() == \
        (repo / "configs" / "vocab.json").read_text()
    assert (clone / "notes.txt").exists() and (clone / "scripts/outputs/gen.py").exists()
    assert (clone / "usplat/docs/gen.py").exists()
    for gone in ("ignored.py", "docs", "outputs", "papers", "results"):
        assert not (clone / gone).exists(), gone

    # same code state -> same patch file (content-addressed, written once)
    with RunRecord(tmp_path / "runs" / "r2.json", repo=repo):
        pass
    g2 = _strict_load(tmp_path / "runs" / "r2.json")["git"]
    assert g2["patch_sha256"] == g["patch_sha256"] and g2["code_sha256"] == g["code_sha256"]
    assert len(list((path.parent / "patches").glob("*.patch"))) == 1
    # an edited untracked config changes the code state, not the tracked diff
    (repo / "configs" / "vocab.json").write_text('{"classes": ["wall", "chair"]}\n')
    g3 = git_state(repo)
    assert g3["diff_sha256"] == g["diff_sha256"] and g3["code_sha256"] != g["code_sha256"]
    assert "error" in git_state(tmp_path / "runs")      # not a repo: error, no crash
    print(f"PASS git: diff {g['diff_sha256'][:12]}, untracked {len(g['untracked'])}, "
          f"patch applies to a clean clone")


def test_git_state_edge_cases(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    lines = [f"v{i} = {i}\n" if i % 7 else "\n" for i in range(60)]
    (repo / "a.py").write_text("".join(lines))
    (repo / "z.py").write_text("z = 0\n")
    (repo / "é.py").write_text("e = 0\n")               # non-ASCII (quoted by git)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    lines[20], lines[26], lines[45] = "v20 = -1\n", "v26 = -1\n", "\n\nextra = 1\n"
    (repo / "a.py").write_text("".join(lines))
    (repo / "z.py").write_text("z = 1\n")
    (repo / "é.py").write_text("e = 1\n")

    # user git config must not change the hashed diff (review P5)
    before = git_state(repo)
    raw0 = _git(repo, "diff", "HEAD")
    order = tmp_path / "order.txt"
    order.write_text("z.py\n")
    for k, v in (("diff.context", "8"), ("diff.interHunkContext", "10"),
                 ("diff.algorithm", "patience"), ("diff.indentHeuristic", "false"),
                 ("diff.orderFile", str(order)), ("diff.noprefix", "true"),
                 ("diff.mnemonicPrefix", "true"), ("diff.suppressBlankEmpty", "true"),
                 ("core.quotePath", "false"), ("core.abbrev", "12")):
        _git(repo, "config", k, v)
    assert _git(repo, "diff", "HEAD") != raw0           # the config does change plain diffs
    after = git_state(repo)
    assert after["diff_sha256"] == before["diff_sha256"]
    assert after["code_sha256"] == before["code_sha256"]

    # a non-UTF-8 untracked name (review P4): listed, recorded, and in the patch
    raw_name = b"caf\xe9.py"
    with open(os.path.join(os.fsencode(str(repo)), raw_name), "w") as f:
        f.write("c = 1\n")
    st = git_state(repo, patch_dir=tmp_path / "patches")
    assert "error" not in st, st
    assert [u["path"] for u in st["untracked"]] == [os.fsdecode(raw_name)]
    assert st["patch_complete"] is True
    with RunRecord(tmp_path / "runs" / "nonutf8.json", repo=repo, save_patch=False):
        pass
    r = _strict_load(tmp_path / "runs" / "nonutf8.json")
    assert r["status"] == "ok" and r["git"]["untracked"][0]["path"] == os.fsdecode(raw_name)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True)
    _git(clone, "apply", st["patch_file"])
    with open(os.path.join(os.fsencode(str(clone)), raw_name)) as f:
        assert f.read() == "c = 1\n"

    # an untracked file above the patch limit: hashed and listed, patch marked incomplete
    monkeypatch.setattr(rl, "_MAX_PATCH_BYTES", 4)
    st = git_state(repo, patch_dir=tmp_path / "patches")
    assert st["untracked"][0]["sha256"] and st["patch_complete"] is False
    print("PASS git edge cases: config-independent hash, non-UTF-8 names, size limit")


def test_timer_sums():
    now = [0.0]
    clock = lambda: now[0]
    tm = Timer(clock=clock, warmup=1)
    durs = [0.5, 0.1, 0.2, 0.6]                         # first one = warm-up (JIT)
    for d in durs:
        with tm.section("keyframe"):
            with tm.section("render"):
                now[0] += d / 2
            now[0] += d / 2
    tm.add("clip", 0.25)
    s = tm.summary(per=len(durs))
    kf, rd = s["keyframe"], s["render"]
    assert kf["count"] == 4 and abs(kf["total_s"] - sum(durs)) < 1e-12
    assert abs(rd["total_s"] - sum(durs) / 2) < 1e-12
    post = np.array(durs[1:]) * 1e3                     # 100, 200, 600 ms: mean != median
    ref = {"mean_ms": post.mean(), "std_ms": post.std(), "median_ms": np.median(post),
           "p90_ms": np.percentile(post, 90), "min_ms": post.min(), "max_ms": post.max()}
    assert kf["n_warmup"] == 1
    for k, v in ref.items():
        assert abs(kf[k] - v) < 1e-9, (k, kf[k], v)
    assert abs(kf["median_ms"] - 200.0) < 1e-9 and abs(kf["mean_ms"] - 300.0) < 1e-9
    assert abs(kf["p90_ms"] - 520.0) < 1e-9
    assert abs(kf["ms_per_keyframe"] - 1400.0 / 4) < 1e-9     # warm-up included
    assert abs(rd["median_ms"] - 100.0) < 1e-9
    assert s["clip"]["count"] == 1 and s["clip"]["mean_ms"] is None      # only warm-up
    assert abs(tm.total("keyframe") - 1.4) < 1e-12 and tm.count("render") == 4
    with pytest.raises(RuntimeError):
        with tm.section("a"):
            with tm.section("a"):
                pass
    with pytest.raises(RuntimeError):
        tm.stop("never_started")
    assert "keyframe" in tm.report(per=4)
    tm.reset()
    assert tm.summary() == {}
    # real clock: totals cover the sleeps
    rt = Timer()
    for _ in range(3):
        with rt.section("sleep"):
            time.sleep(0.005)
    assert rt.total("sleep") >= 0.015
    print(f"PASS timer: totals {kf['total_s']:.3f} s, mean after warm-up {kf['mean_ms']:.1f} ms")


def test_timer_cuda_sync(monkeypatch):
    """With CUDA up, every section boundary synchronises (simulated: no GPU touched)."""
    calls = []
    monkeypatch.setattr(rl, "_cuda_on", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: calls.append(1))
    tm = Timer()
    for _ in range(3):
        with tm.section("keyframe"):
            with tm.section("clip"):
                pass
    assert len(calls) == 3 * 4                          # start + stop of both sections
    calls.clear()
    with Timer(sync=False).section("x"):
        pass
    assert calls == []
    monkeypatch.setattr(rl, "_cuda_on", lambda: False)
    with Timer().section("y"):
        pass
    assert calls == []                                  # CPU run: never touches CUDA
    print("PASS timer sync: 2 syncs per section with CUDA, none without")


def test_set_seed(monkeypatch, tmp_path):
    def draw():
        return (random.random(), np.random.rand(), torch.rand(3).tolist())
    set_seed(123)
    a = draw()
    set_seed(np.int64(123))
    assert draw() == a
    set_seed(124)
    assert draw() != a
    prev = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        torch.use_deterministic_algorithms(False)
        set_seed(1, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        torch.use_deterministic_algorithms(False)
        with RunRecord(tmp_path / "det.json", seed=0, deterministic=True, save_patch=False):
            pass
        r = _strict_load(tmp_path / "det.json")
        assert r["deterministic"] is True and r["deterministic_algorithms"] is True
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev[2:]
    print("PASS set_seed: python, numpy and torch streams repeat; deterministic mode set")


def test_nvidia_smi_parse(monkeypatch):
    out = "NVIDIA GeForce RTX 5070 Ti, 16303, 1234, 575.64\nbroken line\n"
    with monkeypatch.context() as m:
        m.setattr(rl.subprocess, "run",
                  lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""))
        assert rl._nvidia_smi() == [{"name": "NVIDIA GeForce RTX 5070 Ti",
                                     "memory_total_mib": 16303, "memory_used_mib": 1234,
                                     "driver": "575.64"}]
        m.setattr(rl.subprocess, "run",
                  lambda *a, **k: subprocess.CompletedProcess(a, 9, stdout="", stderr="x"))
        assert rl._nvidia_smi() is None

        def missing(*a, **k):
            raise FileNotFoundError("nvidia-smi")
        m.setattr(rl.subprocess, "run", missing)
        assert rl._nvidia_smi() is None
    if _gpu_visible():
        gpus = rl.environment()["gpus"]
        assert isinstance(gpus, list) and gpus and gpus[0]["memory_total_mib"] > 0
    print("PASS nvidia-smi: parsed, missing and failing tool give None")


def test_peak_memory_folding(tmp_path, monkeypatch):
    """CUDA peak counters simulated on CPU: sections reset the device counter, and the
    run peak and the enclosing sections still see every earlier peak."""
    class FakeCuda:                                     # torch's peak-counter semantics
        cur = peak = 0

        def alloc(self, n):
            self.cur += n
            self.peak = max(self.peak, self.cur)

        def free(self, n):
            self.cur -= n

        def reset(self, *a):
            self.peak = self.cur
    fc = FakeCuda()
    monkeypatch.setattr(rl, "_cuda_on", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: fc.peak)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: fc.peak)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fc.reset)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "fake")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 0)

    tm = Timer(memory=True)
    with RunRecord(tmp_path / "mem.json", timer=tm, save_patch=False):
        fc.alloc(256)                                   # resident for the whole run
        with tm.section("big"):
            fc.alloc(512); fc.free(512)
        with tm.section("small"):
            fc.alloc(64); fc.free(64)
        with tm.section("outer"):
            with tm.section("inner"):
                fc.alloc(100); fc.free(100)
            fc.alloc(500); fc.free(500)                 # after the inner reset
    r = _strict_load(tmp_path / "mem.json")
    pk = {k: v["peak_alloc_bytes"] for k, v in r["timing"].items()}
    assert pk == {"big": 768, "small": 320, "inner": 356, "outer": 756}, pk
    assert r["memory"]["peak_gpu_alloc_bytes"] == 768 and r["memory"]["cuda_initialized"]
    assert r["memory"]["peak_gpu_reserved_bytes"] == 768
    assert r["memory"]["cuda_device"] == "fake" and r["env"]["cudnn"] == 0
    assert rl._WATCHES == []                            # every watcher closed
    print(f"PASS peak folding: sections {pk}, run {r['memory']['peak_gpu_alloc_bytes']}")
