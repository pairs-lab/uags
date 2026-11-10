"""scripts/stale_baselines_nor.py and scripts/distill_update_nor.py (configs/stale_baselines_prereg.json)
on the toy NeRF-Object-Removal scene of tests/test_dump_nor_qual.py (CPU, mock backend):
 * the control row reproduces the dump's 'before' set exactly; a distilled file that holds the
   mapping mu* gives the control's counts; every row counts the same pixels;
 * the 2D row reads the update frames, which no longer show the object: no stale answer;
 * a dump of another stage, a control that does not reproduce the dump (exit 2) and a finished
   test record are refused or flagged;
 * distill_update_nor.py trains on the mapping and the update keyframes of the record, and its
   files (Feature 3DGS and LEGaussians) are read by the row loader.
"""
import importlib.util
import json
import pathlib
import shutil

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TDQ = _load("toy_dump_nor_qual", REPO / "tests" / "test_dump_nor_qual.py")
SB = _load("stale_baselines_nor", REPO / "scripts" / "stale_baselines_nor.py")
DU = _load("distill_update_nor", REPO / "scripts" / "distill_update_nor.py")


def _argv(t, out, extra=(), stage="toy", dump=None):
    return ["--table", str(t["rec"]), "--dump", str(dump or t["dump"]), "--vocab", str(t["vocab"]),
            "--text-cache", str(t["cache"]), "--stage", stage, "--out", str(out),
            "--backend", "mock", "--device", "cpu", *extra]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    t = TDQ.toy_inputs(tmp_path_factory.mktemp("stale_baselines"))
    t["dump"] = t["tmp"] / "dump"
    TDQ.DQ.main(TDQ.argv(t, t["dump"]))
    ck = torch.load(t["tmp"] / "field.pt", map_location="cpu", weights_only=False)
    from usplat import distill as D
    from usplat.vmf_field import DirectionalSemanticField
    mu = DirectionalSemanticField.from_state(ck["field"], device="cpu").mu_post().float()
    t["frozen"] = t["tmp"] / "mu_f3dgs.pt"
    torch.save(dict(name="f3dgs", features=mu, decoder=None,
                    pca_basis_sha256=D.basis_sha256(ck["pca"])), t["frozen"])
    t["con"] = json.loads((t["dump"] / "contrast.json").read_text())
    return t


def test_control_2d_and_a_distilled_row(toy):
    out = toy["tmp"] / "light.json"
    res = SB.main(_argv(toy, out, ["--distilled", str(toy["frozen"])]))
    assert res["exit"] == 0 and res["status"] == "ok"
    rows, before = res["rows"], toy["con"]["sets"]["before"]
    assert rows["before"]["n_mask_cov"] == before["n_mask_cov"] > 0
    assert rows["before"]["n_query_in_mask"] == before["n_query_in_mask"] > 0
    assert res["control"]["abs_diff"] == 0.0 and res["control"]["ok"]
    assert rows["f3dgs"] == rows["before"]                    # the same features, rendered
    assert {r["n_mask_cov"] for r in rows.values()} == {before["n_mask_cov"]}
    assert rows["2d"]["stale_answer_rate"] == 0.0             # the frames show the wall
    assert set(rows) == {"before", "2d", "f3dgs"} and set(res["distilled"]) == {"f3dgs"}
    saved = json.loads(out.read_text())
    assert saved["status"] == "ok" and saved["query"]["name"] == "object"
    assert saved["n_frames_no_mask"] == 0 and saved["n_update_keyframes"] > 0
    assert "scripts/stale_baselines_nor.py" in saved["provenance"]["code"]


def test_refusals_and_the_stop_rule(toy, tmp_path):
    with pytest.raises(SystemExit, match="toy dump"):         # the dump's stage
        SB.main(_argv(toy, tmp_path / "a.json", stage="val"))
    bad = tmp_path / "dump"
    shutil.copytree(toy["dump"], bad)
    con = json.loads((bad / "contrast.json").read_text())
    con["sets"]["before"]["stale_answer_rate"] -= 0.2         # a dump this run does not reproduce
    (bad / "contrast.json").write_text(json.dumps(con))
    res = SB.main(_argv(toy, tmp_path / "b.json", ["--no-2d"], dump=bad))
    assert res["exit"] == 2 and res["status"] == "control failed"
    assert "2d" not in res["rows"]
    done = tmp_path / "c.json"
    done.write_text(json.dumps(dict(status="ok", stage="test")))
    with pytest.raises(SystemExit, match="run once"):
        SB.main(_argv(toy, done))
    con = json.loads((toy["dump"] / "contrast.json").read_text())
    con["query"]["prompt"] = "another prompt"
    (bad / "contrast.json").write_text(json.dumps(con))
    with pytest.raises(SystemExit, match="prompt"):
        SB.main(_argv(toy, tmp_path / "d.json", dump=bad))
    assert SB.control_check(dict(stale_answer_rate=0.50, n_mask_cov=1000),
                            dict(stale_answer_rate=0.504, n_mask_cov=1004))["ok"]
    assert not SB.control_check(dict(stale_answer_rate=0.50, n_mask_cov=1000),
                                dict(stale_answer_rate=0.506, n_mask_cov=1000))["ok"]
    assert not SB.control_check(dict(stale_answer_rate=0.50, n_mask_cov=1000),
                                dict(stale_answer_rate=0.50, n_mask_cov=1010))["ok"]


def test_retraining_uses_both_parts_and_feeds_the_rows(toy, tmp_path):
    views = json.loads(toy["rec"].read_text())
    views = (views.get("metrics") or views)["views"]
    files = []
    for method, extra in (("f3dgs", []), ("legs", ["--codes", "8"])):
        out = tmp_path / f"{method}_upd.pt"
        rec = DU.main(["--table", str(toy["rec"]), "--method", method, "--iters", "300",
                       "--min-iters", "1000000", "--lr", "0.02", "--backend", "mock",
                       "--device", "cpu", "--out", str(out), *extra])
        assert rec["n_map_frames"] == len(views["mapping"])
        assert rec["n_update_frames"] == len(views["update"]) == len(rec["update_index"])
        assert rec["n_frames"] == rec["n_map_frames"] + rec["n_update_frames"]
        assert rec["name"].endswith("_upd") and rec["config"]["frames"] == "mapping+update"
        assert rec["iters_run"] == 300 and not (tmp_path / f"{method}_upd.pt.ckpt").exists()
        files.append(str(out))
    res = SB.main(_argv(toy, tmp_path / "trained.json", ["--no-2d", "--distilled", ",".join(files)]))
    assert res["exit"] == 0
    assert set(res["rows"]) == {"before", "f3dgs_upd", "legs_l1_upd"}
    n = res["rows"]["before"]["n_mask_cov"]
    for r in ("f3dgs_upd", "legs_l1_upd"):
        assert res["rows"][r]["n_mask_cov"] == n
        assert 0.0 <= res["rows"][r]["stale_answer_rate"] <= 1.0
        assert res["distilled"][r]["name"] == r
    assert DU.code_sha256() != DU.DF.code_sha256()
