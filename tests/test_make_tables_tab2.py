"""scripts/make_tables.py tab2: the rows of Table II (tab:change, stale content on
NeRF-Object-Removal) from a scripts/nor_tab2.py merge record (a fake one here; no GPU):
 * rows, inputs and the \\midrule follow papers/main.tex; cells are mean {+- population std}
   in %, AP and AUROC with 1 decimal, FA and Rec. with --digits (default 1);
 * marks on the printed means of every column, all rows ranked: bold best, underline second
   best, equal printed values share the mark; FA is better when lower; the fused row has no
   operating point ("--", never marked);
 * the merge's "nan" strings (usplat.runlog) read back as NaN; a missing row becomes a
   comment, a cell over fewer scenes is flagged; an unfinished or foreign file is refused;
 * the comment lines after the rows: unobserved Gaussians, stale and box counts, never-scored
   shares and the common-set AP / AUROC per row, Kept, the per-scene wins of ours, the
   settings, and warnings (k of the runs, mock backend, an inexact mapping replay);
 * the CLI prints the same text.
"""
import importlib.util
import json
import re
import math
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ROWS = ("ncc", "depth", "feat", "normlift", "dispersion", "ours", "fused")
LABELS = ["NCC photometric residual", "Depth residual",
          "Feature residual, $k=2$~\\cite{ackermann2025clsplats, jiang2026mif}\\reimpl",
          "NormLift reliability drop~\\cite{zang2026normlift}\\reimpl",
          "Dispersion $1-\\hat\\rho$ (ours)", "Ours: test, reset, re-admission",
          "Ours fused with depth residual"]
INPUTS = ["I", "D", "F", "F", "F", "F+D", "F+D"]
NAN = float("nan")


def _mt():
    spec = importlib.util.spec_from_file_location("make_tables", REPO / "scripts" / "make_tables.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    return mt


def _st(mean, std=0.01, n=12):
    return dict(mean=mean, std=std if n else NAN, n=n, n_nan=12 - n)


# column values per row: each row and column has its own value; FA lower is better
VALUES = dict(
    g_ap=[0.10, 0.30, 0.20, 0.05, 0.25, 0.45, 0.40],          # best ours, second fused
    g_auroc=[0.70, 0.85, 0.80, 0.60, 0.75, 0.90, 0.90],       # ours = fused (tie), then depth
    p_ap=[0.12, 0.33, 0.21, 0.06, 0.24, 0.40, 0.42],          # best fused, second ours
    p_auroc=[0.71, 0.86, 0.81, 0.61, 0.76, 0.91, 0.93],
    fa=[0.050, 0.030, 0.040, 0.001, 0.0021, 0.002, None],     # 0.1 best; 0.2 (ours, disp.) tie
    rec=[0.60, 0.70, 0.65, 0.01, 0.02, 0.55, None])


def _body(n_scenes=12, k=2, backend="gsplat", replay_ok=True):
    """A merge record's metrics as scripts/nor_tab2.py writes them (before to_jsonable)."""
    names = [f"{i:03d}" for i in range(5, 5 + n_scenes)]
    summary = {}
    for j, r in enumerate(ROWS):
        s = {k_: (_st(v[j]) if v[j] is not None else _st(NAN, n=0)) for k_, v in VALUES.items()}
        s.update(g_ap_common=_st(0.5 + 0.01 * j), g_auroc_common=_st(0.8 + 0.01 * j),
                 p_ap_common=_st(0.4 + 0.01 * j), p_auroc_common=_st(0.7 + 0.01 * j),
                 g_unscored=_st(0.001 * j), p_unscored=_st(0.002 * j),
                 threshold=_st(0.5) if r != "fused" else _st(NAN, n=0))
        summary[r] = s
    per = {}
    for i, s in enumerate(names):                # ours beats NCC everywhere, ties nothing
        rows = {r: {k_: (VALUES[k_][j] + 0.001 * i if VALUES[k_][j] is not None else None)
                    for k_ in VALUES} for j, r in enumerate(ROWS)}
        if i % 3 == 0:                           # in a third of the scenes depth beats ours
            rows["depth"]["g_ap"] = 0.9
        per[s] = dict(rows=rows)
    counts = dict(unobserved_share=_st(0.123, 0.02), n_stale_observed=_st(812.0, 100),
                  stale_share=_st(0.0456, 0.01), box_static_share=_st(0.25), p_pos_share=_st(0.08),
                  n_mapping_views=_st(45.0), n_update_views=_st(43.5), g_common_share=_st(0.9),
                  p_common_share=_st(0.95), pooled_unobserved_share=0.111)
    return dict(split="test", scenes_order=names, rows=list(ROWS),
                inputs=dict(zip(ROWS, INPUTS)), summary=summary,
                ours=dict(kept=_st(0.07, 0.03), kept_after_mapping=_st(0.20, 0.05),
                          changed_end_fa=_st(0.0015, 0.0005), changed_end_rec=_st(0.6, 0.1),
                          n_resets=_st(1234.0)),
                counts=counts, maps=dict(n_gaussians=_st(456789.0), psnr=_st(27.5), ssim=_st(0.9)),
                scenes=per, settings=dict(change_quantile=0.999, k=k, tau_cov=0.5, inflate_m=0.02,
                                          near_m=0.02, kf_trans=0.05, kf_rot=5.0, downscale=2,
                                          extractor="lseg", backend=backend),
                pipeline="iters=30000", fit_config=dict(iters=30000, max_gaussians=1000000,
                                                        max_res=720),
                replay=dict(all_ok=replay_ok, not_ok=[] if replay_ok else ["007"],
                            max_rel_dW1=0.0 if replay_ok else 2e-3))


def _write(tmp_path, body=None, status="ok", name="change_test.json"):
    sys.path.insert(0, str(REPO))
    from usplat.runlog import to_jsonable
    p = tmp_path / name
    p.write_text(json.dumps(to_jsonable(dict(status=status, metrics=body or _body()))))
    return str(p)


def _rows(tex):
    """[(label, input, [cells])] of the table lines, and the index of each \\midrule."""
    rows, mids = [], []
    for line in tex.splitlines():
        if line == "\\midrule":
            mids.append(len(rows))
        elif not line.startswith("%"):
            assert line.endswith("\\\\"), line
            label, inp, *cells = [c.strip() for c in line[:-2].split(" & ")]
            rows.append((label, inp, cells))
    return rows, mids


def test_rows_cells_and_marks(tmp_path):
    mt = _mt()
    tex = mt.tab2(_write(tmp_path))
    rows, mids = _rows(tex)
    assert [r[0] for r in rows] == LABELS and [r[1] for r in rows] == INPUTS and mids == [5]
    cells = {label: c for label, _, c in rows}
    pm = "{\\scriptsize$\\,\\pm\\,$1.0}"
    assert cells[LABELS[0]] == ["10.0" + pm, "70.0" + pm, "12.0" + pm, "71.0" + pm, "5.0" + pm,
                                "60.0" + pm]
    ours, fused, disp = cells[LABELS[5]], cells[LABELS[6]], cells[LABELS[4]]
    assert ours[0] == "\\textbf{45.0}" + pm and fused[0] == "\\underline{40.0}" + pm  # G AP
    assert ours[1].startswith("\\textbf{90.0}") and fused[1].startswith("\\textbf{90.0}")  # tie
    assert cells[LABELS[1]][1].startswith("\\underline{85.0}")         # next distinct value
    assert fused[2].startswith("\\textbf{42.0}") and ours[2].startswith("\\underline{40.0}")
    assert cells[LABELS[3]][4].startswith("\\textbf{0.1}")             # FA: lower is better
    assert ours[4].startswith("\\underline{0.2}") and disp[4].startswith("\\underline{0.2}")
    assert fused[4] == "--" and fused[5] == "--"                        # no operating point
    assert cells[LABELS[1]][5].startswith("\\textbf{70.0}")             # Rec: higher is better
    assert cells[LABELS[2]][5].startswith("\\underline{65.0}")
    assert sum(x.startswith("\\textbf") for _, _, c in rows for x in c) == 7  # 6 + 1 tie
    head = tex.splitlines()[:2]
    assert head[0].startswith("% generated by scripts/make_tables.py tab2 ")
    assert "split test, scenes 005, 006" in head[0] and head[0].endswith("N = [12]")
    assert "FA: lower is better" in head[1]
    print("PASS tab2: rows, inputs, midrule, cells, marks (ties, FA lower, fused unranked)")


def test_digits_split_a_tie(tmp_path):
    mt = _mt()
    rows, _ = _rows(mt.tab2(_write(tmp_path), digits=2))
    cells = {label: c for label, _, c in rows}
    assert cells[LABELS[5]][4].startswith("\\underline{0.20}")         # ours 0.20
    assert cells[LABELS[4]][4].startswith("0.21")                     # dispersion: third now
    assert cells[LABELS[0]][0].startswith("10.0{")                    # AP keeps 1 decimal
    assert cells[LABELS[0]][5].startswith("60.00{")                   # Rec takes --digits
    print("PASS tab2: --digits sets FA / Rec decimals only")


def test_notes(tmp_path):
    mt = _mt()
    tex = mt.tab2(_write(tmp_path))
    notes = [ln for ln in tex.splitlines() if ln.startswith("%")][2:]
    joined = "\n".join(notes)
    assert ("% unobserved Gaussians (no update view; left out of every Gaussian-level cell): "
            "12.3 +- 2.0% per scene, 11.1% of all Gaussians of the 12 scenes") in joined
    assert "812 per scene (4.56 +- 1.00% of the observed); 25.0% of the observed" in joined
    assert ("% Depth residual: never scored (ranked last) G 0.1% / P 0.2%; on the set every row "
            "defines (G 90.0%, P 95.0% of the population): G AP 51.0, AUROC 81.0; P AP 41.0, "
            "AUROC 71.0") in joined
    assert "Kept (stale observed Gaussians in the robust set after the update) 7.0 +- 3.0%" in joined
    assert "% ours vs Depth residual: 8/12, 12/12, 12/12, 12/12" in joined
    assert "% ours vs NCC photometric residual: 12/12, 12/12, 12/12, 12/12" in joined
    assert "% ours vs Ours fused with depth residual: 12/12, 0/12, 0/12, 0/12" in joined
    assert "% settings: q 0.999, k 2, tau_cov 0.5" in joined and "WARNING" not in joined
    assert "% maps: 456789 Gaussians, held-out PSNR 27.50 dB" in joined
    print("PASS tab2: comment lines")


def test_warnings_missing_row_and_refusals(tmp_path):
    mt = _mt()
    b = _body(k=1, backend="mock", replay_ok=False)
    del b["summary"]["normlift"]
    b["summary"]["depth"]["p_ap"] = _st(0.3, n=11)
    tex = mt.tab2(_write(tmp_path, b))
    assert f"% {LABELS[3]}: row 'normlift' not in " in tex
    assert f"% {LABELS[1]}: fewer than the 12 scenes in P AP" in tex
    assert "% WARNING: the feature-residual label says k=2, the runs used k = 1" in tex
    assert "% WARNING: backend 'mock', not a protocol run" in tex
    assert "% WARNING: the mapping replay was not exact in scenes ['007']" in tex
    rows, _ = _rows(tex)
    assert len(rows) == 6 and "N = [11, 12]" in tex.splitlines()[0]
    with pytest.raises(SystemExit, match="not a finished merge"):
        mt.tab2(_write(tmp_path, status="failed", name="f.json"))
    p = tmp_path / "foreign.json"
    p.write_text(json.dumps(dict(status="ok", metrics=dict(summary={}))))
    with pytest.raises(SystemExit, match="not a scripts/nor_tab2.py merge"):
        mt.tab2(str(p))
    print("PASS tab2: warnings, missing row, fewer scenes, refusals")


FUSED_ROWS = REPO / "docs" / "reports" / "2026-10-20-figures-video" / "logs" / "nor_fused" / "nor_fused_test_rows.tex"


def _paper_rows():
    """[(label, input)] of the tab:change rows of papers/main.tex, and its \\midrule indices."""
    tex = (REPO / "papers" / "main.tex").read_text()
    block = tex[tex.index("\\label{tab:change}"):]
    block = block[block.index("Method & Input"):block.index("\\bottomrule")]
    rows, mids = [], []
    for chunk in block.split("\\\\")[1:]:                    # after the header row
        lines = [ln.strip() for ln in chunk.splitlines()
                 if ln.strip() and not ln.strip().startswith("%")]
        if lines and lines[0] == "\\midrule":
            if rows:
                mids.append(len(rows))
            lines = lines[1:]
        if lines:
            cells = [c.strip() for c in " ".join(lines).split("&")]
            rows.append((cells[0], cells[1]))
    return rows, mids


WU_TABLES = REPO / "docs" / "reports" / "2026-11-06-whole-update-score" / "logs" / "tables"


def _wut():
    spec = importlib.util.spec_from_file_location("whole_update_tables",
                                                  REPO / "scripts" / "whole_update_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row_lines(text):
    """The non-comment rows and rules of a tabular body, stripped."""
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("%")
            and ("&" in ln or ln.strip() == "\\midrule")]


def test_rows_match_the_paper(tmp_path):
    if not (REPO / "papers" / "main.tex").exists():
        pytest.skip("papers/main.tex not found")
    if not (FUSED_ROWS.exists() and WU_TABLES.exists()):
        pytest.skip("docs/reports is not in this checkout (git-ignored since 2026-11-06)")
    paper, pmids = _paper_rows()
    rows, mids = _rows(_mt().tab2(_write(tmp_path)))
    tab2_rows = [(label, inp) for label, inp, _ in rows]
    assert tab2_rows == [(label, inp) for (label, inp, _) in [e for e in _mt().TAB2 if e]]
    # since 2026-11-06 (attempt 07, adopted by configs/whole_update_prereg.json) the layout is
    # scripts/whole_update_tables.TAB3: the tab2 rows in their order, the update-mean and
    # whole-update rows of its tracked rows file, the fused baselines of their own file
    layout = _wut().TAB3
    want, want_mids = [], []
    for e in layout:
        if e is None:
            want_mids.append(len(want))
        else:
            want.append((e[0], e[1]))
    assert paper == want and pmids == want_mids, (paper, pmids)
    assert [r for r in paper if r in tab2_rows] == tab2_rows
    # tab2's \midrule before 'Ours' and the one that opens the fused block, before 'Ours fused'
    assert paper.index(tab2_rows[mids[0]]) in pmids and paper.index(tab2_rows[-1]) in pmids
    fused = [ln.split("&") for ln in FUSED_ROWS.read_text().splitlines()
             if "&" in ln and not ln.lstrip().startswith("%")]
    tex = (REPO / "papers" / "main.tex").read_text()
    block = tex[tex.index("\\label{tab:change}"):tex.index("\\end{tabular}", tex.index("\\label{tab:change}"))]
    cells = {c[0].strip(): [x.strip() for x in c] for c in
             (ln.rstrip("\\ ").split("&") for ln in block.splitlines() if ln.count("&") >= 7)}
    unmark = lambda s: re.sub(r"\\(textbf|underline)\{([^{}]*)\}", r"\2", s)
    # the fused baselines (pre-registered, plot_nor_qual.py fused): their file's inputs and cells
    for e in [e for e in layout if e and e[2] == "fusedfile"]:
        f = fused[e[3]]
        assert e[1] == f[1].strip(), e
        assert [unmark(x) for x in cells[e[0]][2:6]] == [x.strip() for x in f[2:6]], e[0]
    # the attempt-07 rows: their tracked file's inputs and all six cells
    new = {c[0].strip(): [x.strip() for x in c] for c in
           (ln.rstrip("\\ ").split("&") for ln in (WU_TABLES / "tab3_new_rows.tex").read_text().splitlines()
            if "&" in ln and not ln.lstrip().startswith("%"))}
    news = [e for e in layout if e and e[2] == "new"]
    assert sorted(new) == sorted(e[0] for e in news)
    for e in news:
        assert [unmark(x) for x in cells[e[0]][1:]] == new[e[0]][1:], e[0]
    # and the whole marked body (after its source comment) is the generated one
    start = block.index("%% generated by scripts/whole_update_tables.py")
    assert _row_lines(block[start:]) == _row_lines((WU_TABLES / "tab3_body.tex").read_text())


def test_fused_table_matches_the_generated_rows():
    if not (REPO / "papers" / "main.tex").exists():
        pytest.skip("papers/main.tex not found")
    if not WU_TABLES.exists():
        pytest.skip("docs/reports is not in this checkout (git-ignored since 2026-11-06)")
    tex = (REPO / "papers" / "main.tex").read_text()
    i = tex.index("\\label{tab:fused}")
    block = tex[i:tex.index("\\end{table*}", i)]
    body = [ln for ln in _row_lines(block)
            if ln != "\\midrule" and not ln.startswith(("Method &", "& &", "Fused baseline &"))]
    assert body == _row_lines((WU_TABLES / "tab4_body.tex").read_text())
    assert "Holm over the 24 tests" in block and len(_wut().OURS_FUSED) * len(_wut().BASE_FUSED) * 2 == 24


def test_cli_prints_tab2(tmp_path):
    mt = _mt()
    p = _write(tmp_path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "make_tables.py"), "tab2", p,
                          "--digits", "2"], capture_output=True, text=True, env=env,
                         check=True).stdout
    assert out == mt.tab2(p, digits=2) + "\n"
