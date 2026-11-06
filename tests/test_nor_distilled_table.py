"""scripts/nor_distilled_table.py: lambda selection (highest mean validation Gaussian AP, ties ->
smaller lambda), merged mean / population std cells, refusal of mixed settings."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nor_distilled_table", REPO / "scripts" / "nor_distilled_table.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
SET = dict(change_quantile=0.999, k=2, tau_cov=0.5, undef_frac=0.5, inflate_m=0.02, near_m=0.02,
           update_stride=1, backend="gsplat")


def _rec(d, scene, summ, settings=SET):
    rows = {r: dict(zip(T.METRICS, v)) for r, v in summ.items()}
    (d / f"{scene}.json").write_text(json.dumps(dict(status="ok", metrics=dict(
        scene=scene, settings=settings, summary=rows, control_check=None))))


def test_select_and_merge(tmp_path):
    for s, a, b, c in (("001", 0.5, 0.6, 0.6), ("002", 0.7, 0.6, 0.6)):
        _rec(tmp_path, s, {"legs_l1_feat": (a,) * 6, "legs_l2_feat": (b,) * 6, "legs_l4_feat": (c,) * 6})
    out = T.main(["select", "--dir", str(tmp_path), "--scenes", "001,002", "--rows",
                  "legs_l4_feat,legs_l2_feat,legs_l1_feat", "--out", str(tmp_path / "s.json")])
    assert out["chosen"] == "legs_l1_feat" and out["lambda_u"] == 1.0      # 0.6 = 0.6 = 0.6: smallest
    m = T.main(["merge", "--dir", str(tmp_path), "--scenes", "001,002", "--rows", "legs_l1_feat",
                "--labels", "LEGaussians, feature residual", "--out", str(tmp_path / "m.json"),
                "--tex", str(tmp_path / "r.tex")])
    line = (tmp_path / "r.tex").read_text().splitlines()[1]
    assert line.startswith("LEGaussians, feature residual & F & 60.0{\\scriptsize$\\,\\pm\\,$10.0}")
    assert m["table"]["legs_l1_feat"]["g_ap"]["n"] == 2


def test_mixed_settings_are_refused(tmp_path):
    _rec(tmp_path, "001", {"f3dgs_feat": (0.5,) * 6})
    _rec(tmp_path, "002", {"f3dgs_feat": (0.5,) * 6}, dict(SET, k=1))
    with pytest.raises(SystemExit):
        T.main(["merge", "--dir", str(tmp_path), "--scenes", "001,002", "--rows", "f3dgs_feat",
                "--out", str(tmp_path / "m.json")])


def test_update_stride_is_per_scene_and_must_match_its_record(tmp_path):
    """Scenes may differ in update stride (the Table III records use 1-3), but each must equal its
    own Table III record's (2026-11-07)."""
    for s, stride in (("001", 2), ("003", 1)):
        ref = tmp_path / f"ref_{s}.json"
        ref.write_text(json.dumps(dict(config=dict(update_stride=stride))))
        rows = {"f3dgs_feat": dict(zip(T.METRICS, (0.5,) * 6))}
        (tmp_path / f"{s}.json").write_text(json.dumps(dict(status="ok", metrics=dict(
            scene=s, settings=dict(SET, update_stride=stride), summary=rows,
            control_check=dict(reference=dict(path=str(ref)), all_exact=True, max_abs_diff=0.0,
                               **{k: dict(abs_diff=0.0) for k in ("g_ap", "p_ap", "fa", "rec", "threshold", "n_observed")})))))
    T.main(["merge", "--dir", str(tmp_path), "--scenes", "001,003", "--rows", "f3dgs_feat",
            "--out", str(tmp_path / "m.json")])
    (tmp_path / "ref_003.json").write_text(json.dumps(dict(config=dict(update_stride=2))))
    with pytest.raises(SystemExit):
        T.main(["merge", "--dir", str(tmp_path), "--scenes", "001,003", "--rows", "f3dgs_feat",
                "--out", str(tmp_path / "m.json")])
