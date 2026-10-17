"""scripts/plot_views.py (the views figure, fig:views) on a tiny fake views JSON and Table I JSON:
 * the printed table holds the input numbers (AUSE x100, mean +- std) of every curve, M and
   target; e_red has no M = 180 point; the PDF (3.5 x 2.6 in, no Type 3 fonts) and the PNG exist;
 * M = 180 comes from the Table I summary when both files hold the same scenes, and is
   recomputed (population std) from the per-scene values of the views scenes otherwise;
 * the full-map check prints 0 when both files score the same full map, a warning otherwise;
   the covered-pixel and temperature checks pass on equal counts / betas and fail otherwise,
   and a failed check stops --stage test before anything is written (--allow-mismatch);
 * the random-ranking level equals usplat.evaluation.sparsification_curve with a constant u,
   from the per-scene counts, also at M = 180 from Table I's covered n_pixels and acc;
 * what is drawn: per panel, every curve's y = 100 x mean, x = M x dodge, error-bar ends,
   colour and marker, the random line, hollow points, the legend labels and the x ticks;
 * a per-scene Table I value written as the string "nan" counts as missing, not a crash;
 * --defined plots summary_on_ale_defined (no M = 180 point, the "share" entry is not a row);
 * missing scenes, rows or signals and unfinished Table I records stop with a clear message;
 * scene_stats equals usplat.seg_eval.scene_stats; the CLI runs as a script.
"""
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plot_views", REPO / "scripts" / "plot_views.py")
pv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv)

KS = (10, 30, 90)
CURVES = [("ours_P0", "u_epi"), ("ours_P0", "u_ale"), ("averaging", "msp"),
          ("normlift", "normlift_epi"), ("normlift", "normlift_1-Cinter"),
          ("normlift", "normlift_1-R"), ("viewcount", "viewcount")]


def _stat(mean, std, n):
    return dict(mean=mean, std=std, n=n, n_nan=0)


def _summary(n, offset, share=False):
    """{k: {row: {signal: {target: stats}}}}, a different value in every cell (x100: 3
    decimals exactly); the ours row also has an unused signal, as eval_views.py writes."""
    summ, i = {}, 0
    for k in KS:
        unused = {"e_red": _stat(0.9, 0.0, n), "e_irr": _stat(0.9, 0.0, n)}
        summ[str(k)] = {"ours_P0": {"msp": unused}}
        for row, sig in CURVES:
            for t in ("e_red", "e_irr"):
                i += 1
                summ[str(k)].setdefault(row, {}).setdefault(sig, {})[t] = \
                    _stat((offset + 7 * i) / 1e5, i / 1e5, n)
        if share:
            summ[str(k)]["share"] = _stat(0.25, 0.01, n)
    return summ


def _per_scene(j, s):
    return 0.05 + 0.02 * int(s[1:]) + 0.001 * j     # covered AUSE of curve j in scene "s<i>"


def _miou(s):
    return 0.3 + 0.01 * int(s[1:])                  # full-map mIoU of scene "s<i>"


ROWS = ("ours_P0", "averaging", "normlift", "viewcount")


def _counts(s, k, row):
    """(n_pixels, n_e_red, n_e_irr) of scene "s<i>" at M = k (ours and the references differ)."""
    i, r = int(s[1:]), int(row != "ours_P0")       # the references share one predictive
    n = 5000 + 100 * i
    return n, (300 + 7 * i + r) // (1 + KS.index(k)), 800 + 11 * i + 13 * KS.index(k) + r


def _scene_ause(s, k, j, t):
    """Per-scene AUSE of curve j at M = k (a fraction below the random level)."""
    return (0.002 + 0.0003 * j + 0.0005 * int(s[1:])) * (3 if t == "e_irr" else 1) / \
        (1 + KS.index(k))


def _acc(s, row):
    """Covered accuracy of the full map: 850 + 9 i + r errors among 5000 + 100 i pixels."""
    n = 5000 + 100 * int(s[1:])
    return (n - (850 + 9 * int(s[1:]) + int(row != "ours_P0"))) / n


def _write(tmp_path, views_scenes, tab1_scenes, miou_shift=0.0, counts=False, betas=None):
    """A views JSON (as scripts/replica_views.sh merges it) and a Table I runlog record;
    scenes are named s0, s1, ... counts=True adds the per-scene pixel and target counts (views)
    and covered n_pixels / acc (Table I); betas = (views, Table I) writes part files with the
    first and Table I betas with the second."""
    n = len(views_scenes)
    views = dict(ks=list(KS),
                 scenes={s: dict(full=dict(miou={r: _miou(s) + miou_shift
                                                 for r in ("ours_P0", "averaging")}))
                         for s in views_scenes},
                 summary=_summary(n, 100), summary_on_ale_defined=_summary(n, 50000, share=True))
    if counts:
        for s in views_scenes:
            views["scenes"][s]["k"] = {str(k): dict(rows={r: dict(zip(
                ("n_pixels", "n_e_red", "n_e_irr"), _counts(s, k, r)),
                ause={sig: {t: _scene_ause(s, k, j, t) for t in ("e_red", "e_irr")}
                      for j, (row, sig) in enumerate(CURVES) if row == r})
                for r in ROWS}) for k in KS}
    if betas is not None:
        views["parts"] = {}
        for s in views_scenes:
            part = tmp_path / f"views_part_{s}.json"
            part.write_text(json.dumps(dict(betas=betas[0], provenance=dict(
                config=dict(ause_bins=20)))))
            views["parts"][s] = str(part)
    rows = {}
    for j, (row, sig) in enumerate(CURVES):
        rows.setdefault(row, {})[sig] = j
    cov = (lambda s, r: dict(n_pixels=5000 + 100 * int(s[1:]), acc=_acc(s, r))) if counts else \
        (lambda s, r: {})
    tab_scenes = {s: dict(rows={r: dict(all=dict(miou=_miou(s)),
                                        covered=dict(ranking={sig: dict(ause=_per_scene(j, s))
                                                              for sig, j in sigs.items()},
                                                     **cov(s, r)))
                                for r, sigs in rows.items()})
                  for s in tab1_scenes}
    summary = {r: dict(covered=dict(ause={sig: _stat(0.0888 + j / 1e4, 0.0111, len(tab1_scenes))
                                          for sig, j in sigs.items()}))
               for r, sigs in rows.items()}       # NOT the per-scene mean: shows which is used
    tab1 = dict(status="ok", metrics=dict(scenes=tab_scenes, summary=summary))
    if betas is not None:
        tab1["metrics"]["betas"] = betas[1]
    vp, tp = tmp_path / "views.json", tmp_path / "tab1.json"
    vp.write_text(json.dumps(views))
    tp.write_text(json.dumps(tab1))
    return views, tab1, str(vp), str(tp)


def _parse(text):
    """{(target, 'row/signal'): {M: cell text}} from the printed Markdown table."""
    block = text[text.index("| target | curve |"):].split("\n\n")[0]   # the curves table
    lines = [ln for ln in block.splitlines() if ln.startswith("| ")]
    ms = [int(c.strip()[2:]) for c in lines[0].strip("|").split("|")[3:]]
    out = {}
    for ln in lines[1:]:
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        out[(c[0], c[2])] = dict(zip(ms, c[3:]))
    return out


def _cell(st):
    return f"{100 * st['mean']:.3f} +- {100 * st['std']:.3f}"


def _run(capsys, argv):
    res = pv.main(argv)
    return res, capsys.readouterr().out


def test_table_matches_inputs_and_pdf(tmp_path, capsys):
    views, tab1, vp, tp = _write(tmp_path, ["s0", "s1"], ["s1", "s0"])     # same set, any order
    pdf = tmp_path / "fig" / "views.pdf"
    res, out = _run(capsys, ["--stage", "val", "--views", vp, "--tab1", tp, "--out", str(pdf),
                             "--scale", "raw"])
    got = _parse(out)
    assert len(got) == 2 * len(CURVES)
    for t in ("e_red", "e_irr"):
        for row, sig in CURVES:
            cells = got[(t, f"{row}/{sig}")]
            for k in KS:                                    # every cell is the input number
                assert cells[k] == _cell(views["summary"][str(k)][row][sig][t])
            want_full = "--" if t == "e_red" else _cell(tab1["metrics"]["summary"][row]
                                                        ["covered"]["ause"][sig])
            assert cells[180] == want_full                  # same scenes: Table I summary
    assert "its summary (same scenes)" in out
    assert "check full-map mIoU: PASS: max |mIoU views full map - Table I| = 0.00e+00 over 4" \
        in out
    assert "WARNING" not in out and "N = 2 scenes: s0, s1" in out
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and b"/Type3" not in data
    assert re.search(rb"/MediaBox \[ 0 0 252 187\.2 \]", data)      # 3.5 x 2.6 in
    assert (tmp_path / "fig" / "views.png").stat().st_size > 10_000
    assert res["pdf"] == str(pdf) and res["png"] == str(tmp_path / "fig" / "views.png")
    print("PASS views table = inputs; M=180 from the Table I summary; PDF", len(data), "bytes")


def test_full_point_recomputed_on_the_views_scenes(tmp_path, capsys):
    views, tab1, vp, tp = _write(tmp_path, ["s0", "s1"], ["s0", "s1", "s2"], miou_shift=0.01)
    _, out = _run(capsys, ["--stage", "val", "--views", vp, "--tab1", tp,
                           "--out", str(tmp_path / "v.pdf"), "--band", "--scale", "raw"])
    got = _parse(out)
    for j, (row, sig) in enumerate(CURVES):     # scenes s0 and s1 only: population mean / std
        vals = [_per_scene(j, "s0"), _per_scene(j, "s1")]
        mean = sum(vals) / 2
        want = _stat(mean, abs(vals[0] - mean), 2)
        assert got[("e_irr", f"{row}/{sig}")][180] == _cell(want)
        assert got[("e_red", f"{row}/{sig}")][180] == "--"
    assert "recomputed over the 2 views scenes (the file holds 3)" in out
    assert "check full-map mIoU: FAIL" in out and "comes from another evaluation" in out
    assert (tmp_path / "v.pdf").exists() and (tmp_path / "v.png").exists()
    print("PASS M=180 recomputed over the views scenes; full-map mismatch flagged")


def test_defined_analysis(tmp_path, capsys):
    views, _, vp, tp = _write(tmp_path, ["s0", "s1"], ["s0", "s1"])
    _, out = _run(capsys, ["--stage", "val", "--views", vp, "--tab1", tp, "--defined",
                           "--out", str(tmp_path / "d.pdf"), "--scale", "raw"])
    got = _parse(out)
    summ = views["summary_on_ale_defined"]
    for (t, rs), cells in got.items():
        row, sig = rs.split("/")
        assert sorted(cells) == list(KS)                    # no M = 180 column
        for k in KS:
            assert cells[k] == _cell(summ[str(k)][row][sig][t])
    assert "SECONDARY analysis" in out and "M=10 25.0%" in out
    assert (tmp_path / "d.pdf").exists()
    print("PASS --defined: summary_on_ale_defined, no M=180 point")


def test_refusals(tmp_path):
    views, tab1, vp, tp = _write(tmp_path, ["s0", "s1"], ["s0", "s2"])
    out = ["--out", str(tmp_path / "x.pdf"), "--scale", "raw"]
    with pytest.raises(SystemExit, match=r"no scenes \['s1'\]"):        # not the same scenes
        pv.main(["--views", vp, "--tab1", tp] + out)
    del views["summary"]["30"]["normlift"]["normlift_1-R"]
    (tmp_path / "views.json").write_text(json.dumps(views))
    with pytest.raises(SystemExit, match=r"M=30, row 'normlift', signal 'normlift_1-R'"):
        pv.main(["--views", vp, "--tab1", ""] + out)                 # no M = 180: views only
    tab1["status"] = "running"
    (tmp_path / "tab1.json").write_text(json.dumps(tab1))
    with pytest.raises(SystemExit, match="not a finished evaluation"):
        pv.main(["--views", vp, "--tab1", tp] + out)
    with pytest.raises(SystemExit, match="no such file"):
        pv.main(["--views", str(tmp_path / "views_test.json"), "--tab1", tp] + out)
    assert not (tmp_path / "x.pdf").exists()
    print("PASS refusals: other scenes, missing signal, unfinished record, missing file")


def test_scene_stats_matches_seg_eval():
    from usplat.seg_eval import scene_stats
    for vals in ([0.1], [0.05, 0.07], [0.3, float("nan"), 0.1, 0.25], [float("nan")]):
        a, b = pv.scene_stats(vals), scene_stats(vals)
        assert a["n"] == b["n"] and a["n_nan"] == b["n_nan"]
        for key in ("mean", "std"):
            assert (a[key] != a[key] and b[key] != b[key]) or abs(a[key] - b[key]) < 1e-15
    print("PASS scene_stats = usplat.seg_eval.scene_stats")


def test_cli_runs_as_a_script(tmp_path):
    views, _, vp, tp = _write(tmp_path, ["s0", "s1"], ["s0", "s1"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "plot_views.py"), "--stage", "val",
                        "--views", vp, "--tab1", tp, "--out", str(tmp_path / "cli.pdf"),
                        "--scale", "raw"],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=300)
    assert r.returncode == 0, r.stderr
    k, (row, sig) = KS[0], CURVES[0]
    assert _parse(r.stdout)[("e_red", f"{row}/{sig}")][k] == \
        _cell(views["summary"][str(k)][row][sig]["e_red"])
    assert (tmp_path / "cli.pdf").exists() and (tmp_path / "cli.png").exists()
    print("PASS CLI end to end")


def _rand(n, n_t, bins=20):
    """AUSE of a constant uncertainty on a 0/1 error with n_t ones: the reference value."""
    import torch
    from usplat.evaluation import sparsification_curve
    e = torch.zeros(n, dtype=torch.float64)
    e[:n_t] = 1
    return sparsification_curve(e, torch.zeros(n), num_bins=bins)["ause"]


def test_random_level_and_target_sizes(tmp_path, capsys):
    views, tab1, vp, tp = _write(tmp_path, ["s0", "s1"], ["s0", "s1"], counts=True)
    res, out = _run(capsys, ["--stage", "val", "--views", vp, "--tab1", tp,
                             "--out", str(tmp_path / "r.pdf"), "--scale", "raw"])
    sizes = res["sizes"]
    for t in ("e_red", "e_irr"):
        for row in ROWS:
            ms = [k for k, _ in sizes[t][row]]
            assert ms == (list(KS) + [180] if t == "e_irr" else list(KS))
            for k, c in sizes[t][row]:
                if k == 180:                          # Table I covered n_pixels and acc
                    per = [(5000 + 100 * i, 850 + 9 * i + int(row != "ours_P0")) for i in (0, 1)]
                else:
                    per = [(n, red if t == "e_red" else irr)
                           for n, red, irr in (_counts(f"s{i}", k, row) for i in (0, 1))]
                want = [_rand(n, x) for n, x in per]
                assert abs(c["random"]["mean"] - sum(want) / 2) < 1e-12
                assert abs(c["random"]["std"] - abs(want[0] - want[1]) / 2) < 1e-12
                assert abs(c["share"]["mean"] - sum(x / n for n, x in per) / 2) < 1e-12
                assert c["min_count"] == min(x for _, x in per)
    assert "| e_red | ours_P0 | random AUSE x100 |" in out
    assert "| e_irr | averaging = normlift = viewcount | smallest target count |" in out
    print("PASS random level = sparsification_curve(constant u); shares and counts")


def _container(ax, gid):
    """(data line, error-bar segments) of the curve drawn with this gid."""
    for c in ax.containers:
        if c.lines[0].get_gid() == gid:
            return c.lines[0], c.lines[2][0].get_segments()
    raise AssertionError(f"no curve {gid}")


def test_drawn_figure_matches_the_data(tmp_path):
    import math
    views, tab1, vp, tp = _write(tmp_path, ["s0", "s1"], ["s0", "s1"], counts=True)
    v, t1 = pv.load(vp), pv.load(tp)
    data, sizes = pv.collect(v, vp, False, t1, tp)
    sizes["e_red"]["ours_P0"][2][1]["min_count"] = 5               # M=90: a noisy point
    data["e_irr"][("normlift", "normlift_1-R")][0][1]["n"] = 1        # M=10: one scene only
    _, _, fig = pv.plot(data, str(tmp_path / "d.pdf"), sizes=sizes, n_scenes=2, keep=True)
    axes = fig.axes[:4]
    n_curves = 1 + len(pv.REFS)
    for i, (ch, _) in enumerate(pv.CHANNELS):
        for j, (t, _) in enumerate(pv.TARGETS):
            ax = axes[2 * i + j]
            curves = [(pv.OURS, ("ours_P0", ch))] + [(r, (r[2], r[3])) for r in pv.REFS]
            for c, (style, key) in enumerate(curves):
                line, segs = _container(ax, "/".join(key))
                f = math.exp((c - (n_curves - 1) / 2) * pv.DODGE)
                pts = data[t][key]
                assert list(line.get_xdata()) == pytest.approx([m * f for m, _ in pts])
                ys = [100 * st["mean"] for _, st in pts]
                assert list(line.get_ydata()) == pytest.approx(ys)
                assert line.get_color() == style[4] and line.get_marker() == style[5]
                for (m, st), seg, y in zip(pts, segs, ys):
                    d = 100 * st["std"]
                    assert seg[0][1] == pytest.approx(y - min(d, y))
                    assert seg[1][1] == pytest.approx(y + d)
                hollow = [ln for ln in ax.lines if ln.get_gid() == "/".join(key) + "/hollow"]
                want = [(m, 100 * st["mean"]) for m, st in pts
                        if (t, key, m) in {("e_red", ("ours_P0", "u_epi"), 90),
                                           ("e_red", ("ours_P0", "u_ale"), 90),
                                           ("e_irr", ("normlift", "normlift_1-R"), 10)}]
                assert len(hollow) == (1 if want else 0)
                if want:
                    assert list(hollow[0].get_ydata()) == pytest.approx([y for _, y in want])
                    assert hollow[0].get_markerfacecolor() == "white"
            rnd = [ln for ln in ax.lines if ln.get_gid() == "random/ours_P0"]
            assert len(rnd) == 1
            assert list(rnd[0].get_xdata()) == [m for m, _ in sizes[t]["ours_P0"]]
            assert list(rnd[0].get_ydata()) == pytest.approx(
                [100 * c["random"]["mean"] for _, c in sizes[t]["ours_P0"]])
            ticks = [int(x) for x in ax.get_xticks()]
            assert ticks == (list(KS) if t == "e_red" else list(KS) + [180])
    labels = [x.get_text() for x in fig.legends[0].get_texts()]
    assert labels == [pv.OURS[1]] + [r[1] for r in pv.REFS]
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("PASS drawn curves, error bars, colours, markers, hollow points, random line, ticks")


def test_checks_and_the_test_stage_refusal(tmp_path, capsys):
    b = {"ours_P0": 30.0, "averaging": 31.0, "css": 2.0}
    views, tab1, vp, tp = _write(tmp_path, ["s0", "s1"], ["s0", "s1"], counts=True,
                                 betas=(b, b))
    ok = tmp_path / "ok.pdf"
    res, out = _run(capsys, ["--stage", "test", "--views", vp, "--tab1", tp, "--out", str(ok)])
    assert "check covered pixels: PASS: views n_pixels = Table I covered n_pixels in 24 of 24" \
        in out
    assert "check temperatures: PASS: betas of 2 part files = Table I betas in 6 of 6" in out
    assert ok.exists() and [r[1] for r in res["checks"]] == ["PASS", "PASS", "PASS"]
    # other temperatures: --stage test refuses before writing, --allow-mismatch writes
    _write(tmp_path, ["s0", "s1"], ["s0", "s1"], counts=True, betas=(b, dict(b, averaging=9.0)))
    bad = tmp_path / "bad.pdf"
    with pytest.raises(SystemExit, match=r"failed checks \['temperatures'\]"):
        pv.main(["--stage", "test", "--views", vp, "--tab1", tp, "--out", str(bad)])
    assert not bad.exists()
    res, out = _run(capsys, ["--stage", "test", "--views", vp, "--tab1", tp, "--out", str(bad),
                             "--allow-mismatch"])
    assert bad.exists() and "check temperatures: FAIL" in out and "WARNING" in out
    # other covered pixels in one scene
    tab1["metrics"]["scenes"]["s1"]["rows"]["averaging"]["covered"]["n_pixels"] = 1
    (tmp_path / "tab1.json").write_text(json.dumps(tab1))
    with pytest.raises(SystemExit, match="covered pixels"):
        pv.main(["--stage", "test", "--views", vp, "--tab1", tp, "--out", str(bad)])
    _, out = _run(capsys, ["--stage", "val", "--views", vp, "--tab1", tp, "--out", str(bad)])
    assert "check covered pixels: FAIL: views n_pixels = Table I covered n_pixels in 21 of 24" \
        in out
    print("PASS checks: pixels and temperatures; test stage refuses on a failed check")


def test_nan_strings_count_as_missing(tmp_path, capsys):
    views, tab1, vp, tp = _write(tmp_path, ["s0", "s1"], ["s0", "s1", "s2"])
    j = CURVES.index(("normlift", "normlift_1-R"))
    tab1["metrics"]["scenes"]["s0"]["rows"]["normlift"]["covered"]["ranking"]["normlift_1-R"][
        "ause"] = "nan"                                   # as usplat.runlog writes NaN
    (tmp_path / "tab1.json").write_text(json.dumps(tab1))
    _, out = _run(capsys, ["--stage", "val", "--views", vp, "--tab1", tp,
                           "--out", str(tmp_path / "n.pdf"), "--scale", "raw"])
    cell = _parse(out)[("e_irr", "normlift/normlift_1-R")][180]
    assert cell == f"{100 * _per_scene(j, 's1'):.3f} +- 0.000 (n=1)"
    assert pv.scene_stats(["nan", 0.1, None])["n"] == 1
    with pytest.raises(SystemExit, match="not a number"):
        pv.num("abc")
    print("PASS 'nan' strings are missing values, not a crash")


def test_relative_scale(tmp_path, capsys):
    """The default scale: per scene AUSE / random AUSE of the same errors, then mean +- std;
    M = 180 from Table I's per-scene covered AUSE, n_pixels and acc; the random line at 1."""
    import math
    views, tab1, vp, tp = _write(tmp_path, ["s0", "s1"], ["s0", "s1"], counts=True)
    res, out = _run(capsys, ["--stage", "val", "--views", vp, "--tab1", tp,
                             "--out", str(tmp_path / "rel.pdf")])
    data = res["data"]
    for t in ("e_red", "e_irr"):
        for j, (row, sig) in enumerate(CURVES):
            for k, st in data[t][(row, sig)]:
                if k == 180:
                    per = [(_per_scene(j, f"s{i}"), 5000 + 100 * i,
                            850 + 9 * i + int(row != "ours_P0")) for i in (0, 1)]
                else:
                    per = [(_scene_ause(f"s{i}", k, j, t),) + (lambda c: (c[0], c[1] if t ==
                           "e_red" else c[2]))(_counts(f"s{i}", k, row)) for i in (0, 1)]
                r = [a / _rand(n, x) for a, n, x in per]
                assert st["n"] == 2 and abs(st["mean"] - sum(r) / 2) < 1e-12
                assert abs(st["std"] - abs(r[0] - r[1]) / 2) < 1e-12
    assert "plotted: AUSE / random-ranking AUSE per scene" in out
    assert "| target | curve (AUSE / random) |" in out and "| target | curve (raw AUSE x100) |" \
        in out
    _, _, fig = pv.plot(data, str(tmp_path / "rel2.pdf"), sizes=res["sizes"], n_scenes=2,
                        keep=True, relative=True)
    for p, ax in enumerate(fig.axes[:4]):
        rnd = [ln for ln in ax.lines if ln.get_gid() == "random/ours_P0"]
        assert len(rnd) == 1 and set(rnd[0].get_ydata()) == {1.0}
        line, _ = _container(ax, f"ours_P0/{pv.CHANNELS[p // 2][0]}")
        assert all(math.isfinite(y) and 0 < y < 1 for y in line.get_ydata())
    import matplotlib.pyplot as plt
    plt.close(fig)
    # a per-scene "nan" in Table I is a missing scene at M = 180, not a crash
    tab1["metrics"]["scenes"]["s0"]["rows"]["normlift"]["covered"]["ranking"]["normlift_1-R"][
        "ause"] = "nan"
    (tmp_path / "tab1.json").write_text(json.dumps(tab1))
    res, _ = _run(capsys, ["--stage", "val", "--views", vp, "--tab1", tp,
                           "--out", str(tmp_path / "rel3.pdf")])
    assert dict(res["data"]["e_irr"][("normlift", "normlift_1-R")])[180]["n"] == 1
    print("PASS relative scale: per-scene AUSE / random AUSE; random line at 1")
