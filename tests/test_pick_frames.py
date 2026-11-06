"""papers/figure_tools/pick_frames.py static (CPU): the frame rule counts the classes with at least 0.5% of
a frame's labelled pixels (ignored pixels left out) and breaks ties by the earliest frame; the
scene rule takes the lower median of the published per-scene mIoU of ours_P2; --check exits 2
when an outcome differs from the review's, and passes when it matches."""
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


PF = _load("pick_frames", REPO / "papers" / "figure_tools" / "pick_frames.py")


def _scene(tmp, labels):
    """A 7-Scenes-style folder, one frame per label map (raw ids; 0 ignored)."""
    import cv2
    root = tmp / "data"
    root.mkdir(parents=True, exist_ok=True)
    for i, lab in enumerate(labels):
        base = root / f"frame-{i:06d}"
        cv2.imwrite(str(base) + ".color.png", np.zeros((480, 640, 3), np.uint8))
        cv2.imwrite(str(base) + ".depth.png", np.full((480, 640), 2000, np.uint16))
        c2w = np.eye(4)
        c2w[0, 3] = 0.05 * i
        np.savetxt(str(base) + ".pose.txt", c2w)
        cv2.imwrite(str(base) + ".label.png", lab.astype(np.uint16))
    (tmp / "vocab.json").write_text(json.dumps(dict(
        templates=["a photo of a {}."], ignore=[0], classes={3: "chair", 7: "table", 9: "sofa"})))
    sc = [dict(name="toy", field="", ply="", data=str(root), dataset="7scenes", stride=1,
               label_pattern="{root}/frame-{idx:06d}.label.png")]
    (tmp / "scenes.json").write_text(json.dumps(sc))


def _labels():
    a = np.zeros((480, 640), np.int64)
    a[:, :320] = 3                     # frame 0: chair, table
    a[:, 320:] = 7
    b = a.copy()                       # frame 1: right half ignored (0), chair on the left
    b[:, 320:] = 0
    b[:2, :320] = 9                    # sofa: 640 of 153600 labelled px = 0.42% < 0.5%
    c = a.copy()                       # frame 2: chair, table, sofa (1 %)
    c[:5, :] = 9
    d = c.copy()                       # frame 3: same classes as frame 2 (a tie)
    return [a, b, c, d]


def test_frame_rule_min_share_and_ties(tmp_path):
    _scene(tmp_path, _labels())
    out = PF.main(["static", "--set", f"toy={tmp_path / 'scenes.json'},{tmp_path / 'vocab.json'}",
                   "--downscale", "1", "--out", str(tmp_path / "sel.json")])
    fr = out["sets"]["toy"]["frames"]["toy"]
    n = {t["frame"]: t["n_classes"] for t in fr["table"]}
    assert n == {0: 2, 1: 1, 2: 3, 3: 3}, n            # frame 1: sofa below 0.5%, table ignored
    assert fr["pick"]["frame"] == 2 and fr["pick"]["classes"] == ["chair", "table", "sofa"]
    assert PF.frame_classes(torch.full((4, 4), -1)) == ([], 0)
    saved = json.loads((tmp_path / "sel.json").read_text())
    assert saved["sets"]["toy"]["frames"]["toy"]["pick"]["frame"] == 2
    print(f"PASS frame rule: classes per frame {n}, tie -> frame 2")


def test_scene_rule_lower_median_and_check(tmp_path):
    def table(vals):
        return dict(scenes={s: dict(rows=dict(ours_P2=dict(all=dict(miou=v))))
                            for s, v in vals.items()})
    pick, rank = PF.lower_median_scene(table(dict(a=0.3, b=0.1, c=0.2, d=0.4)))
    assert pick == "c" and [s for s, _ in rank] == ["b", "c", "a", "d"]   # 2nd of 4
    assert PF.lower_median_scene(table(dict(a=0.3, b=0.1, c=0.2)))[0] == "c"   # middle of 3
    assert PF.lower_median_scene(table(dict(a=0.2, b=0.2)))[0] == "a"   # ties by name
    # --check: the review's outcomes; the toy set "replica" with scene "room_1" ...
    _scene(tmp_path, _labels())
    sc = json.loads((tmp_path / "scenes.json").read_text())
    sc[0]["name"] = "room_1"
    (tmp_path / "scenes.json").write_text(json.dumps(sc))
    (tmp_path / "t.json").write_text(json.dumps(dict(metrics=table(dict(room_1=0.3, x=0.2,
                                                                         y=0.4)))))
    argv = ["static", "--check", "--downscale", "1", "--set",
            f"replica={tmp_path / 'scenes.json'},{tmp_path / 'vocab.json'},{tmp_path / 't.json'}",
            "--out", str(tmp_path / "sel.json")]
    with pytest.raises(SystemExit) as e:               # frame 2 is not the review's 66
        PF.main(argv)
    assert e.value.code == 2
    c = json.loads((tmp_path / "sel.json").read_text())["check"]
    assert c["status"] == "FAIL" and c["mismatches"] == ["room_1: frame 2 (expected 66)"]
    old = PF.EXPECTED["frames"]["room_1"]
    PF.EXPECTED["frames"]["room_1"] = 2
    try:
        out = PF.main(argv)
    finally:
        PF.EXPECTED["frames"]["room_1"] = old
    assert out["check"]["status"] == "PASS" and out["check"]["n_checked"] == 2
    print("PASS scene rule (lower median, ties by name) and --check (exit 2 on a mismatch)")
