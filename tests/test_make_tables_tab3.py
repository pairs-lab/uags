"""scripts/make_tables.py tab3: the Replica block of Table III (tab:ablation) from a tiny fake
eval_seg.py JSON (no GPU, no torch).
 * the full method is the ours predictive (P0, P1 or P2) with the lowest VALIDATION NLL, never
   another ours row (P3, scalar) even when its NLL is lower; a comment line says which and why,
   and says "in-sample" when the val run fit its betas on the scenes it scores; the numbers
   come from the evaluated file, not from val;
 * each variant row reads abl_<variant>_P2 when the full method is P2, else abl_<variant>_P0;
 * columns mIoU, ECE, NLL, AUSE epi (u_epi), AUSE ale (u_ale): means only, all in % but NLL;
   --digits sets the decimals of ECE and AUSE;
 * the two variants hold different numbers in the fake files: the default reads "all" (choice
   and cells), --variant covered reads "covered";
 * a missing row becomes a comment, a row or a single cell over fewer scenes is flagged, no
   --val gives ours_P2, and a variant that prints the full method's cells is flagged;
 * --tab1 compares the full method's printed cells with the same row of a Table I file;
 * per scene: each variant's sign against the full method and each row's u_ale undefined share;
 * cell() keeps its +- std form for tab1 (std=True by default);
 * the labels match the Replica block of papers/main.tex, and the CLI prints the same text.
"""
import copy
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
VARIANTS = ("pixels", "est_rbar", "est_joint", "est_legacy", "centre", "legacy")  # paper order
OTHER_OURS = ("ours_P3_b20", "ours_scalar")        # ours rows that are never the full method
ROWS = (["ours_P0", "ours_P1", "ours_P2", *OTHER_OURS]
        + [f"abl_{v}_{p}" for v in VARIANTS for p in ("P0", "P2")])
LABELS = ["\\textbf{Full method}", "Pixels instead of views", "Uncorrected mean length $\\bar r$",
          "Prior-shrunk agreement", "Shrunk mean length $\\tilde r$", "Centered features",
          "Pixel-level variant"]
COV = 100                      # the covered variant of row i holds the numbers of row i + COV


def _mt():
    spec = importlib.util.spec_from_file_location("make_tables",
                                                  REPO / "scripts" / "make_tables.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    return mt


def _stat(mean, n=6):
    return dict(mean=mean, std=0.01, n=n, n_nan=0)


def _metrics(i, n=6):
    """Row number i: each column has its own range and each row its own value, so a printed
    cell tells which row and column it came from."""
    return dict(miou=_stat(0.20 + 0.01 * i, n), acc=_stat(0.70, n), ece=_stat(0.10 + 0.01 * i, n),
                nll=_stat(1.00 + 0.01 * i, n),
                ause=dict(u_epi=_stat(0.40 + 0.01 * i, n), u_ale=_stat(0.60 + 0.01 * i, n),
                          msp=_stat(0.90, n), entropy=_stat(0.95, n)))


def _row(i, n=6):
    """Both variants of row i: "covered" holds the numbers of row i + COV, so a printed cell
    also tells which variant it came from."""
    return {"all": _metrics(i, n), "covered": _metrics(i + COV, n)}


def _summary(offset=0):
    return {r: _row(i + offset) for i, r in enumerate(ROWS)}


def _expected(i, digits=1):
    """The cells of row number i (i + COV: its covered variant), written out without the code
    under test; digits = the decimals of ECE and AUSE."""
    z = "0" * digits
    return [f"{20 + i}.0", f"{10 + i}.{z}", f"{1 + i // 100}.{i % 100:02d}", f"{40 + i}.{z}",
            f"{60 + i}.{z}"]


def _write(path, summary, record=True, scenes=None, config=None):
    """A run record ({status, config, metrics: {summary, scenes}}) like the real files, or a
    plain body."""
    b = dict(summary=summary)
    if scenes is not None:
        b["scenes"] = scenes
    d = dict(status="ok", config=config or {}, metrics=b) if record else b
    path.write_text(json.dumps(d))
    return str(path)


def _files(tmp_path, val_nll, cov_nll=None, config=None):
    """The evaluated file (its own NLL order would pick P0) and a validation file with other
    numbers everywhere: the given NLL of ours_P0 / P1 / P2 (cov_nll: of their covered
    variant), and the lowest NLL of all on the ours rows that are never the full method.
    With config, the val file is a run record with that eval_seg.py config."""
    test = _write(tmp_path / "eval_test.json", _summary())
    vs = _summary(offset=30)
    for variant, nll in (("all", val_nll), ("covered", cov_nll or {})):
        for r, v in nll.items():
            vs[r][variant]["nll"]["mean"] = v
        for r in OTHER_OURS:
            vs[r][variant]["nll"]["mean"] = 0.10
    return test, _write(tmp_path / "eval_val.json", vs, record=config is not None, config=config)


def _data(tex):
    """{label: [cells]} of the table lines (comments left out)."""
    out = {}
    for line in tex.splitlines():
        if line.startswith("%"):
            continue
        assert line.endswith("\\\\"), line
        label, *cells = [c.strip() for c in line[:-2].split("&")]
        out[label] = cells
    return out


@pytest.mark.parametrize("val_nll, full, pred", [
    (dict(ours_P0=0.90, ours_P1=0.95, ours_P2=0.80), "ours_P2", "P2"),
    (dict(ours_P0=0.70, ours_P1=0.95, ours_P2=0.80), "ours_P0", "P0"),
    (dict(ours_P0=0.90, ours_P1=0.60, ours_P2=0.80), "ours_P1", "P0"),   # no P1 ablation rows
])
def test_full_method_chosen_on_val(tmp_path, val_nll, full, pred):
    mt = _mt()
    test, val = _files(tmp_path, val_nll)
    tex = mt.tab3(test, val=val)
    head = tex.splitlines()[:3]
    assert head[0].startswith(f"% generated by scripts/make_tables.py tab3 {test} --variant all "
                              f"--val {val}") and head[0].endswith("N = [6]"), head[0]
    assert head[1].startswith(f"% full method = {full} (lowest validation NLL among "), head[1]
    assert "ours_P0 0.9000" in head[1] or "ours_P0 0.7000" in head[1], head[1]
    assert full in mt.FULL and not any(r in head[1] for r in OTHER_OURS), head[1]  # NLL 0.10
    assert "in-sample" not in head[1]                    # a plain body: no eval_seg.py config
    assert f"variants: abl_<variant>_{pred}" in head[1]
    assert ("no P1 row" in head[1]) == (full == "ours_P1")
    assert "coverage rule of Table I" in head[2] and "M=180 point is not this cell" in head[2]
    data = _data(tex)
    assert list(data) == LABELS                                  # every row, in paper order
    assert data[LABELS[0]] == _expected(ROWS.index(full))       # test numbers, not val's
    for label, v in zip(LABELS[1:], VARIANTS):
        assert data[label] == _expected(ROWS.index(f"abl_{v}_{pred}")), (label, data[label])
    assert "\\pm" not in tex                                     # means only
    assert not [l for l in tex.splitlines()[1:] if ": N = " in l or "same printed" in l]
    assert tex.splitlines()[-1] == "% no per-scene results in the file: no per-scene signs"
    print(f"PASS tab3: full method {full}, variants abl_*_{pred}")


def test_variant_all_is_the_default_and_covered_reads_covered(tmp_path):
    """The two variants hold different cells and rank the val NLLs differently: the default
    takes the choice and every cell from "all", --variant covered from "covered"."""
    mt = _mt()
    test, val = _files(tmp_path, dict(ours_P0=0.90, ours_P1=0.95, ours_P2=0.80),
                       cov_nll=dict(ours_P0=0.70, ours_P1=0.95, ours_P2=0.80))
    for tex, variant, full, pred, off in (
            (mt.tab3(test, val=val), "all", "ours_P2", "P2", 0),
            (mt.tab3(test, "covered", val), "covered", "ours_P0", "P0", COV)):
        head = tex.splitlines()
        assert f"--variant {variant} " in head[0], head[0]
        assert head[1].startswith(f"% full method = {full} (lowest validation NLL"), head[1]
        assert ("coverage rule of Table I" in head[2]) == (variant == "all"), head[2]
        assert ("on covered pixels only (as Fig. views)" in head[2]) == (variant == "covered")
        data = _data(tex)
        assert data[LABELS[0]] == _expected(ROWS.index(full) + off), (variant, data[LABELS[0]])
        for label, v in zip(LABELS[1:], VARIANTS):
            assert data[label] == _expected(ROWS.index(f"abl_{v}_{pred}") + off), (variant, label)
    print("PASS tab3: default reads 'all', --variant covered reads 'covered'")


def test_missing_row_fewer_scenes_and_no_val(tmp_path):
    mt = _mt()
    summ = _summary()
    del summ["abl_centre_P2"]
    summ["abl_legacy_P2"] = _row(ROWS.index("abl_legacy_P2"), n=5)
    summ["abl_pixels_P2"]["all"]["ause"]["u_ale"].update(n=1, n_nan=5)  # one cell, 1 of 6 scenes
    summ["ours_P2"]["all"]["ause"]["u_epi"].update(n=5, n_nan=1)        # also in the full method
    test = _write(tmp_path / "eval_test.json", summ)
    tex = mt.tab3(test)                                          # no --val: ours_P2
    assert "% full method = ours_P2 (no --val: default, not chosen on validation)" in tex
    assert f"% Centered features: row 'abl_centre_P2' not in {test}" in tex
    assert "% Pixel-level variant: N = 5 scenes, the full method has 6" in tex
    assert "% Pixels instead of views: N = 1 scenes for AUSE ale, the full method has 6" in tex
    assert "% \\textbf{Full method}: N = 5 scenes for AUSE epi, the full method has 6" in tex
    assert "N = [1, 5, 6]" in tex.splitlines()[0]
    data = _data(tex)
    assert "Centered features" not in data and len(data) == 6
    assert data["Pixel-level variant"] == _expected(ROWS.index("abl_legacy_P2"))
    assert data["Pixels instead of views"] == _expected(ROWS.index("abl_pixels_P2"))  # flagged
    nan = dict(summ, abl_pixels_P2=_row(4))
    nan["abl_pixels_P2"]["all"]["ause"].pop("u_epi")              # a channel not evaluated
    tex = mt.tab3(_write(tmp_path / "nan.json", nan))
    assert _data(tex)["Pixels instead of views"][3] == "--"
    print("PASS tab3: missing row, fewer scenes (row and cell) and missing channel are reported")


def test_in_sample_note(tmp_path):
    """The full-method comment says "in-sample" when the val run fit its betas on the scenes
    it scores (--val-scenes = --scenes, no --betas), and not otherwise."""
    mt = _mt()
    nll = dict(ours_P0=0.90, ours_P1=0.95, ours_P2=0.80)
    same = dict(val_scenes="configs/replica_val.json", scenes="configs/replica_val.json",
                betas="")
    test, val = _files(tmp_path, nll, config=same)
    head = mt.tab3(test, val=val).splitlines()[1]
    assert head.startswith("% full method = ours_P2 (lowest validation NLL among ")
    assert head.endswith("; in-sample: the betas were fit on these same scenes); variants: "
                         "abl_<variant>_P2"), head
    other = dict(val_scenes="", scenes="configs/replica_test.json", betas="eval_val.json")
    test, val = _files(tmp_path, nll, config=other)
    assert "in-sample" not in mt.tab3(test, val=val)
    print("PASS tab3: in-sample note")


def test_digits_and_same_printed_cells(tmp_path):
    """--digits sets the decimals of ECE and AUSE only (mIoU keeps 1, NLL 2). A variant that
    prints the full method's cells gets a comment, which goes away when more decimals tell
    the two rows apart."""
    mt = _mt()
    summ = _summary()
    twin = copy.deepcopy(summ["ours_P2"])
    twin["all"]["ece"]["mean"] += 0.0001             # 12.01 %: "12.0" at 1 decimal, "12.01" at 2
    summ["abl_est_rbar_P2"] = twin
    test = _write(tmp_path / "eval_test.json", summ)
    tex = mt.tab3(test)
    assert tex.count("same printed cells") == 1
    assert "% Uncorrected mean length $\\bar r$: same printed cells as the full method" in tex
    tex2 = mt.tab3(test, digits=2)
    assert "same printed cells" not in tex2 and "--digits 2;" in tex2.splitlines()[0]
    data, i = _data(tex2), ROWS.index("ours_P2")
    assert data[LABELS[0]] == _expected(i, digits=2), data[LABELS[0]]
    assert data["Uncorrected mean length $\\bar r$"][1] == f"{10 + i}.01"
    assert data["Pixel-level variant"] == _expected(ROWS.index("abl_legacy_P2"), digits=2)
    print("PASS tab3: --digits and same printed cells")


def test_tab1_comparison(tmp_path):
    """--tab1: a comment compares the full method's printed cells with the same row of a
    Table I file and names each cell that prints differently."""
    mt = _mt()
    test, val = _files(tmp_path, dict(ours_P0=0.90, ours_P1=0.95, ours_P2=0.80))
    t1 = _summary()
    same = _write(tmp_path / "tab1_same.json", t1)
    tex = mt.tab3(test, val=val, tab1=same)
    assert f"--val {val} --tab1 {same};" in tex.splitlines()[0]
    assert tex.splitlines()[3] == (f"% full method vs Table I ({same}, row ours_P2): "
                                   "same printed cells")
    t1["ours_P2"]["all"]["ece"]["mean"] += 0.001      # another beta fit: ECE 12.0 -> 12.1
    t1["ours_P2"]["all"]["nll"]["mean"] += 0.0001     # below the printed digits of NLL
    other = _write(tmp_path / "tab1_other.json", t1)
    note = mt.tab3(test, val=val, tab1=other).splitlines()[3]
    assert note == (f"% full method vs Table I ({other}, row ours_P2): "
                    "differs in ECE 12.0 here, 12.1 in Table I"), note
    note = mt.tab3(test, val=val, tab1=other, digits=2).splitlines()[3]
    assert note.endswith("differs in ECE 12.00 here, 12.10 in Table I"), note
    del t1["ours_P2"]
    gone = _write(tmp_path / "tab1_gone.json", t1)
    assert mt.tab3(test, val=val, tab1=gone).splitlines()[3] == (
        f"% full method vs Table I: row 'ours_P2' not in {gone}")
    assert _data(mt.tab3(test, val=val, tab1=other)) == _data(mt.tab3(test, val=val))
    print("PASS tab3: --tab1 comparison")


def _scene(miou, ece, nll, epi, ale, undef, n_pixels=1000):
    """One scene's "all" result of one row, as eval_seg.py writes it (AUSE under "ranking")."""
    return {"all": dict(miou=miou, ece=ece, nll=nll, n_pixels=n_pixels,
                        ranking=dict(u_epi=dict(ause=epi),
                                     u_ale=dict(ause=ale, n_undefined=undef)))}


def test_per_scene_signs_and_undefined_share(tmp_path):
    """Per scene, in file order: + / - / = against the full method (mIoU higher is better, the
    rest lower; equal up to float noise is "="; a missing value is "?"), and each row's
    share of pixels with an undefined u_ale."""
    mt = _mt()
    per = {"ours_P2": [_scene(0.30, 0.07, 1.05, 0.10, 0.06, 100),
                       _scene(0.33, 0.03, 0.59, 0.07, 0.04, 150)],
           "abl_pixels_P2": [_scene(0.29, 0.08, 1.06, 0.09, 0.05, 2),
                             _scene(0.32, 0.02, None, 0.08, 0.05, 8)],          # NLL missing
           "abl_centre_P2": [_scene(0.30, 0.06, 1.04, 0.10 * (1 + 1e-12), 0.05, 100),  # noise
                             _scene(0.33, 0.02, 0.58, 0.07, 0.05, 150)]}
    scenes = {s: {"rows": {r: v[k] for r, v in per.items()}} for k, s in enumerate(("sb", "sa"))}
    tex = mt.tab3(_write(tmp_path / "eval_test.json", _summary(), scenes=scenes)).splitlines()
    k = [i for i, l in enumerate(tex) if l.startswith("% per scene (sb, sa): + better, ")]
    assert len(k) == 1 and "u_ale undefined = % of the pixels" in tex[k[0]], tex
    assert tex[k[0] + 1:] == [
        "% \\textbf{Full method}: u_ale undefined 10.0 15.0",
        "% Pixels instead of views: mIoU --, ECE -+, NLL -?, AUSE epi +-, AUSE ale +-; "
        "u_ale undefined 0.2 0.8",
        "% Uncorrected mean length $\\bar r$: mIoU ??, ECE ??, NLL ??, AUSE epi ??, "
        "AUSE ale ??; u_ale undefined ? ?",                       # no per-scene results
        "% Prior-shrunk agreement: mIoU ??, ECE ??, NLL ??, AUSE epi ??, AUSE ale ??; "
        "u_ale undefined ? ?",
        "% Shrunk mean length $\\tilde r$: mIoU ??, ECE ??, NLL ??, AUSE epi ??, AUSE ale ??; "
        "u_ale undefined ? ?",
        "% Centered features: mIoU ==, ECE ++, NLL ++, AUSE epi ==, AUSE ale +-; "
        "u_ale undefined 10.0 15.0",
        "% Pixel-level variant: mIoU ??, ECE ??, NLL ??, AUSE epi ??, AUSE ale ??; "
        "u_ale undefined ? ?"], tex[k[0] + 1:]
    print("PASS tab3: per-scene signs and u_ale undefined share")


def test_cell_keeps_std_for_tab1():
    mt = _mt()
    assert mt.cell(_stat(0.5), 100.0) == "50.0{\\scriptsize$\\,\\pm\\,$1.0}"
    assert mt.cell(_stat(0.5), 100.0, std=False) == "50.0"
    assert mt.cell(_stat(0.5, n=1), 100.0) == "50.0" and mt.cell(None, 100.0) == "--"


def test_labels_match_the_paper():
    """The row labels of TAB3 are the Replica block of tab:ablation in papers/main.tex."""
    p = REPO / "papers" / "main.tex"
    if not p.exists():
        pytest.skip("papers/main.tex not found")
    tex = p.read_text()
    block = tex[tex.index("\\label{tab:ablation}"):]
    block = block[block.index("Variant & mIoU"):block.index("\\emph{Object removal, test}")]
    paper = [l.split("&")[0].strip() for l in block.splitlines()[1:]
             if "&" in l and not l.lstrip().startswith("%")]
    assert paper == [label for label, _ in _mt().TAB3] == LABELS, paper


def test_cli_prints_tab3(tmp_path):
    mt = _mt()
    test, val = _files(tmp_path, dict(ours_P0=0.90, ours_P1=0.95, ours_P2=0.80),
                       cov_nll=dict(ours_P0=0.70, ours_P1=0.95, ours_P2=0.80))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")

    def cli(*extra):
        return subprocess.run([sys.executable, str(REPO / "scripts" / "make_tables.py"), "tab3",
                               test, "--val", val, *extra], capture_output=True, text=True,
                              env=env, check=True).stdout
    assert cli() == mt.tab3(test, val=val) + "\n"
    cov = cli("--variant", "covered")
    assert cov == mt.tab3(test, "covered", val) + "\n" and "--variant covered" in cov
    assert cli("--tab1", test, "--digits", "2") == mt.tab3(test, val=val, tab1=test,
                                                           digits=2) + "\n"
    print("PASS tab3 CLI")
