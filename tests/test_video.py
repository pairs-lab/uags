"""CPU tests of the supplementary-video scripts (mock backend, no GPU).

* scripts/video_nor_timeline.py on the toy NeRF-Object-Removal scene of tests/test_eval_change.py:
  the dump of scripts/dump_nor_qual.py is made first (as the real NOR dumps), then the timeline
  must reproduce the toy's Table III record, equal the dump's end state bit for bit, give the
  dump's curves.json exactly, keep the reset bits monotone, and leave the patches undone.
* scripts/make_video.py: the main.tex number parser on the real paper (every caption number
  comes from it), caption templates refuse unknown keys, frame layout and a tiny encode.
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


TDQ = _load("test_dump_nor_qual_for_video", REPO / "tests" / "test_dump_nor_qual.py")
DQ = TDQ.DQ
EC = DQ.EC
VT = _load("video_nor_timeline", REPO / "scripts" / "video_nor_timeline.py")


def _mv():
    return _load("make_video", REPO / "scripts" / "make_video.py")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    t = TDQ.toy_inputs(tmp_path_factory.mktemp("video_nor"))
    dump = t["tmp"] / "dump"
    DQ.main(TDQ.argv(t, dump, extra=["--curves"]))
    out = t["tmp"] / "timeline"
    sha = DQ.sha256_file(t["cache"])
    res = VT.main(["--table", str(t["rec"]), "--dump", str(dump), "--tolerance", str(t["tol"]),
                   "--vocab", str(t["vocab"]), "--text-cache", str(t["cache"]),
                   "--betas", str(t["betas"]), "--curves-ref", str(dump / "curves.json"),
                   "--out", str(out), "--backend", "mock", "--device", "cpu"])
    t.update(dump=dump, out=out, res=res, cache_sha=sha,
             meta=json.loads((out / "meta.json").read_text()))
    return t


def test_timeline_reproduces_and_equals_the_dump(toy):
    meta = toy["meta"]
    assert toy["res"]["exit"] == 0 and meta["status"] == "done"
    rep = meta["reproduction"]
    assert rep["ok"], rep["failures"]
    assert all(d["delta"] == 0.0 for d in rep["operating_point"].values())
    assert meta["gaussian_max_delta"] == 0.0
    assert all(meta["dump_equal"].values()), meta["dump_equal"]
    assert all(v <= VT.PQ_TOL for v in meta["render_vs_dump"].values()), meta["render_vs_dump"]
    assert meta["curves_equal_ref"] is True
    assert EC.DirectionalSemanticField is DQ.DirectionalSemanticField     # patches undone
    assert EC.residuals_view.__module__ == EC.__name__
    assert DQ.sha256_file(toy["cache"]) == toy["cache_sha"]


def test_timeline_arrays(toy):
    z = np.load(toy["out"] / "timeline.npz")
    n = toy["meta"]["n_keyframes"]
    assert z["pq_ours"].shape[0] == n == len(z["update_index"]) == z["photos"].shape[0]
    for k in ("pq_ours", "pq_avg"):
        v = z[k].astype(float)
        assert np.nanmin(v) >= 0.0 and np.nanmax(v) <= 1.0 + 1e-3
    c = json.loads((toy["out"] / "curves.json").read_text())
    rs = np.array(c["reset_stale"])
    assert np.all(np.diff(rs) >= 0) and np.all(np.diff(c["reset_static"]) >= 0)  # ever reset
    # the toy object is removed: ours answers its class less often at the end than before
    assert c["ours"]["stale_is_query"][-1] < c["ours"]["stale_is_query"][0]


# ------------------------------------------------------------------- make_video (CPU)
def test_paper_numbers_parse():
    MV = _mv()
    nums = MV.paper_numbers(REPO / "papers" / "main.tex")
    # spot values: cross-checked against the table rows in the test itself (regex-free path)
    tex = (REPO / "papers" / "main.tex").read_text()
    for key, val in nums.items():
        assert isinstance(val, str) and val, key
    assert nums["fused.feat.gap"] in tex and nums["fused.ours.gap"] in tex
    assert float(nums["fused.feat.gap"]) > float(nums["fused.ours.gap"])   # the paper's finding
    assert float(nums["stale.ours"]) < float(nums["stale.avg"]) < float(nums["stale.before"])
    assert float(nums["kept.full"]) < float(nums["kept.nodepth"])


def test_caption_template_refuses_unknown_keys():
    MV = _mv()
    nums = {"a": "1.0"}
    assert MV.fill("x {a} y", nums) == "x 1.0 y"
    with pytest.raises(KeyError):
        MV.fill("x {b}", nums)


def test_layout_and_tiny_encode(tmp_path):
    MV = _mv()
    frame = MV.blank()
    assert frame.size == (MV.W, MV.H)
    MV.caption(frame, "A caption that must fit inside the safe area of the frame.")
    path = tmp_path / "c.mp4"
    MV.write_clip(path, [frame] * 15)
    info = MV.probe(path)
    assert info["width"] == MV.W and info["height"] == MV.H and info["frames"] == 15


def test_caption_numbers_must_come_from_the_paper(tmp_path):
    MV = _mv()
    tex = tmp_path / "t.tex"
    tex.write_text("AP 53.4 and 75.0")
    ok = MV.Seg("A", "a", 1, None, ["AP 53.4 vs 75.0, frame 373"], [], local=["373"])
    MV.check_caption_numbers([ok], tex)
    bad = MV.Seg("B", "b", 1, None, ["AP 53.5"], [])
    with pytest.raises(AssertionError):
        MV.check_caption_numbers([bad], tex)
