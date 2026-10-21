# NOR figures and fused baselines: summary (2026-10-21)

Verdict: DONE, with one negative result. All 14 dumps reproduce Table III. The figures are written. The pre-registered claim rule for the fused row FAILS: the feature residual fused with depth ranks stale Gaussians better than ours fused with depth.

D = /home/thanhnc/gauss_ws/data, Q = D/uags_results/qual/nor, T = D/uags_results/nor/tab2_relmax, L = this logs/ folder.
Environment: as in the task prompt. GPU jobs ran through `scripts/gpu_gate.sh 6500` with headroom 1024 and poll 5 s. Nothing was committed.

## 1 Key numbers

| What | Value | Source |
|---|---|---|
| Run-to-run noise, Table III repeat of val 001 | max cell difference 4.5e-8 (pixel/dispersion/AP); tolerance = 0.002 | Q/noise/tolerance.json, Q/noise/001_repeat.json |
| Reproduction, 14 dumps | max AP/AUROC delta 4.6e-7 (009); FA/Rec delta 0; keyframes and per-frame n_covered/n_pos identical everywhere | Q/<scene>/meta.json `reproduction` |
| Fused rows, test mean G AP (N=10) | ours+depth 75.0, **feat+depth 81.1**, normlift+depth 79.9, ncc+depth 61.2, dispersion+depth 34.2 | L/nor_fused/nor_fused_test.json |
| Fused rows, test mean P AP | ours+depth 61.8, feat+depth 64.3, **normlift+depth 66.9**, ncc+depth 56.8, dispersion+depth 35.8 | same |
| F2: ours+depth higher (G AP / P AP) | vs ncc 9/10 / 8/10; vs feat **0/10** / 3/10; vs NormLift 1/10 / 2/10; vs dispersion 10/10 / 10/10 | L/nor_fused/nor_fused_test_f2.tex |
| F2 Holm p (G AP / P AP) | ncc 0.107 / 0.328; feat **0.016** / 0.344; NormLift 0.107 / 0.328; dispersion 0.016 / 0.016 | same |
| Claim-rule verdict | **May NOT stay.** feat+depth has the highest test mean G AP (81.1 vs 75.0). The abstract, introduction and results must state the measured comparison. | L/nor_fused/nor_fused_verdict.md |
| N2 stale-answer rate, test (N=10, new measurement) | before 48.3±20.6, change test off 34.2±37.3, ours 20.7±33.9, averaging 34.3±37.2 (%); ours below change test off in 9/10 scenes (delta cos 10/10) | L/nor/n2_numbers_test.json, .tex |
| N2 queries (rule, never re-prompted) | 005 book, 006 box, 007 handbag, 008 backpack, 009 wall, 010 phone, 011 handbag, 013 wall, 014 bike, 015 bike; val 001 wall, 002 chair, 003 book, 004 chair | Q/<scene>/meta.json `query` |

The secondary F2 analysis uses the published fused cells instead of the re-run ones. It gives identical x/10, medians and p-values, because the re-run fused row equals the record to 1e-7 (`f2_published_fused` in the same JSON).

## 2 What I built (new files only)

| File | Role |
|---|---|
| scripts/dump_nor_qual.py | GPU dump. It replays the Table III evaluation by importing `eval_change_nor.py`, with every setting taken from the record. It captures the two update fields with a `from_state` subclass and identity asserts, and captures the pooled pixel population by wrapping `ranking_table`. Reproduction stop rule: exit 2. Also: the 4 fused baselines with separate `ranking_table`/`pixel_level` calls, the N2 query, the rule-frame renders (720x540 and 360x270), contrast.json, fa_frames.json, curves.json and meta.json. `--text-only` fills the text-cache copy on the CPU. |
| scripts/plot_nor_qual.py | CPU. Subcommands: `noise`, `select` (frame and turn rule vs the fixed outcomes), `change` (N1), `query` (N2), `teaser`, `grid`, `numbers` (N2 table), `fused` (F2 statistics, LaTeX and verdict). Captions come from the records; a PDF check covers fonts, metadata and paths. |
| scripts/nor_qual.sh | The val/test dump loop through the gate; it stops on the first failure. |
| tests/test_dump_nor_qual.py | 6 tests on the toy NOR scene (mock backend). The dump reproduces the toy record exactly and the patches are restored. Scores equal those of an unpatched `run_update`. The fused rows equal `fuse_rank`, and Table III's own cells and common sets are untouched. Also covered: midrank, the frame and turn rules, the query, contrast, curves, and the refusals (missing prompt, finished test dump, non-reproducing record -> exit 2). |
| tests/test_plot_nor_qual.py | 5 tests. The sign test equals scipy's binomtest; Holm on a hand example; the noise rule; the F2 report and verdict in both directions; N1, N2 and the teaser on the toy dump (TrueType only, no paths, caption words follow C1-C3). |
| L/nor/pretest_decisions.md | Decisions D1-D10, fixed before any test dump (D10, the colour change, was added after validation). |

## 3 Commands and outputs

| Step | Command (from the repo root, env as in the prompt) | Output |
|---|---|---|
| Noise repeat | `gpu_gate.sh 6500 python3 scripts/eval_change_nor.py --field D/uags_fields/nor_001_lseg_relmax.pt --align none --update-stride 2 --tau-cov 0.5 --out Q/noise/001_repeat.json` | Q/noise/001_repeat.{json,log} |
| Tolerance | `python3 scripts/plot_nor_qual.py noise --a T/val/001.json --b Q/noise/001_repeat.json --out Q/noise/tolerance.json` | 0.002 |
| Frame rule check | `python3 scripts/plot_nor_qual.py select --records T --root D/nerf_object_removal/nerf-object-removal --out Q/select.json` | PASS, 14/14 frames and turns as fixed |
| Text cache | `cp D/uags_fields/text_cache_lseg.pt D/uags_results/qual/text_cache_lseg_nor.pt`; `python3 scripts/dump_nor_qual.py --text-only ...` (CPU) | 50 prompts added; the 51 originals unchanged; Q/logs/text_only.log |
| Val dumps | 001: direct call (Q/logs/dump_001.log); 002-004: `STAGE=val SCENES="002 003 004" scripts/nor_qual.sh` | Q/001..004 |
| Test dumps | `STAGE=test scripts/nor_qual.sh` | Q/005..015; Q/logs/nor_qual_test.20261021-150821.log |
| Fused stats | `python3 scripts/plot_nor_qual.py fused --dumps Q --records T [--test 005,...,015] --out-dir L/nor_fused` | nor_fused_{val,test}.json, *_scenes.md, nor_fused_test_rows.tex, nor_fused_test_f2.tex, nor_fused_verdict.md |
| N2 numbers | `python3 scripts/plot_nor_qual.py numbers --dumps Q --scenes <list> --stage {val,test} --json L/nor/n2_numbers_{val,test}.json --tex ....tex` | L/nor/ |
| Teaser | `python3 scripts/plot_nor_qual.py teaser --dump Q/002 --record T/val/002.json --out papers/figs/teaser.pdf --caption papers/figs/teaser.txt --meta Q/fig_teaser.json` | papers/figs/teaser.{pdf,png,txt} |
| N1 | `python3 scripts/plot_nor_qual.py change --dumps Q --records T --scenes rule --out papers/figs/qual_change.pdf --caption papers/figs/qual_change.txt --meta Q/fig_qual_change.json` | papers/figs/qual_change.{pdf,png,txt} (rule gives 015, 013, 006) |
| N2 | the same with `query` and `qual_query` | papers/figs/qual_query.{pdf,png,txt} |
| Supplementary (optional) | `... grid --scenes 005,...,015 --out Q/supp/supp_change_test.pdf` | Q/supp/ (test only; no validation grid) |

## 4 Per-scene dumps

| Scene | Stage | Frame | Turns | Query (share %) | Max AP/AUROC delta | FA/Rec delta | Frames equal | Capture asserts | Wall s | Peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 001 | val | 450 | 1 | wall 63.7 | 7.9e-08 | 0 | yes | pass | 968 | 4806 |
| 002 | val | 373 | 1 | chair 78.2 | 5.5e-09 | 0 | yes | pass | 649 | 4807 |
| 003 | val | 307 | 1 | book 86.8 | 1.2e-07 | 0 | yes | pass | 113 | 4803 |
| 004 | val | 266 | 1 | chair 72.1 | 1.2e-08 | 0 | yes | pass | 149 | 4806 |
| 005 | test | 201 | 1 | book 99.0 | 4.1e-08 | 0 | yes | pass | 118 | 4805 |
| 006 | test | 178 | 1 | box 57.8 | 1.3e-08 | 0 | yes | pass | 162 | 4805 |
| 007 | test | 270 | 0 | handbag 81.0 | 2.2e-08 | 0 | yes | pass | 343 | 4806 |
| 008 | test | 424 | 0 | backpack 97.4 | 1.9e-08 | 0 | yes | pass | 219 | 4807 |
| 009 | test | 354 | 1 | wall 49.4 | 4.6e-07 | 0 | yes | pass | 122 | 1993 |
| 010 | test | 519 | 1 | phone 49.5 | 3.2e-08 | 0 | yes | pass | 99 | 2041 |
| 011 | test | 496 | 1 | handbag 51.5 | 5.0e-08 | 0 | yes | pass | 259 | 1988 |
| 013 | test | 556 | 1 | wall 71.8 | 2.6e-07 | 0 | yes | pass | 71 | 1981 |
| 014 | test | 748 | 0 | bike 93.1 | 9.0e-08 | 0 | yes | pass | 194 | 4806 |
| 015 | test | 675 | 0 | bike 94.4 | 9.3e-09 | 0 | yes | pass | 179 | 4806 |

The val dumps ran with dump script sha c9526356... and the test dumps with 053d66ad.... The only difference is the TINT colour constant (D10). The frozen code hashes equal the records' (`code.eval_code_equal` and `mapping_code_equal` are true in every meta).

## 5 Test-look ledger

| # | Scene(s) | Script | Git | Purpose |
|---|---|---|---|---|
| 0 | 005 | dump_nor_qual.py (053d66ad) | af70b11 | Crashed 15:06-15:08 with CUDA OOM during run_update: another process took 2.2 GB after the gate check. It wrote no metric or output (meta status "running"). |
| 1-10 | 005, 006, 007, 008, 009, 010, 011, 013, 014, 015 (one dump each, 15:08-15:40) | dump_nor_qual.py (053d66ad) via nor_qual.sh | 8360699 | Figures N1/N2, fused baselines F2, N2 numbers, all from one pass per scene |
| CPU reads | the same dumps | plot_nor_qual.py fused / numbers / change / query / grid | 8360699 | Statistics and figures. No display parameter changed after the test renders. |

## 6 Checks

| ID | Check | Result |
|---|---|---|
| V1 | Every dump reproduces its record: keyframes, per-frame counts, G/P AP+AUROC within 0.002, ours FA/Rec within 0.5 points | PASS 14/14 (section 4) |
| V2 | Capture identity asserts (ours score, robust_end, changed_end, dispersion, NormLift drop, flags = resets); curve label shortcut agrees with mu* at >= 99.999% (val) | PASS 14/14 |
| V3 | Printed numbers come from the records: N1 AP, FA/Rec; teaser Rec/FA and 48 keyframes; captions generated from the records | PASS (by construction; tested in test_plot_nor_qual) |
| V4 | New tests 11/11 pass (7 min); existing tests/test_eval_change.py and tests/test_render_qual.py 13/13 pass | PASS |
| V5 | pdffonts: CID TrueType only, no Type 3; pdfinfo: Matplotlib creator only; no user name or /home in the PDF bytes or the captions | PASS for teaser, qual_change and qual_query. Visual check: no faces or plates seen. Third-party marks: the 006 box print, the 002 monitor and mouse-pad, the 015 bike lock. |
| V6 | No frozen file touched (`git status`: only untracked new files) | PASS |
| V7 | N2 uses one colour scale per row (asserted); N1 rank population = n_pix - n_nan of each row (asserted per dump, equal to the record) | PASS |
| V8 | Test looks logged (section 5); display changes after a test render: none | PASS |
| Rule | `select`: frames and turns equal the outcomes fixed on 2026-10-21 for all 14 scenes | PASS |

## 7 GPU time

- Dumps: 3645 s of wall under the gate in total. Validation took 1879 s (001 alone took 968 s; the CPU load average was 37-44 from other users). Test took 1766 s.
- Noise repeat: 328 s. Crashed 005 attempt: about 1.5 min.
- Waiting at the gate: about 55 min before 002 and about 25 min before 005. The usage-limit pause was 11:02-14:39.
- Peak allocation was 4.8 GB on the 1M-Gaussian scenes, under the 6500 MiB cap.

## 8 Open issues and decisions for the lead

1. **Fused claim (needs a paper change).** By the pre-registered rule, "ours fused with depth ranks stale Gaussians best" may not stay. The feature residual fused with depth is higher in 10/10 scenes at Gaussian AP (median +1.2 points, Holm p 0.016). NormLift fused with depth is higher in 9/10 (Holm p 0.107). NormLift+depth also has the best pixel AP (66.9). Ours+depth still beats NCC+depth and dispersion+depth. Ready-made LaTeX rows: L/nor_fused/nor_fused_test_rows.tex and nor_fused_test_f2.tex. The disclosure that this is an extra evaluation on the NOR test scenes is required (prereg).
2. **Generic N2 queries.** The query rule gives generic or wrong classes in several scenes: 009 and 013 'wall', 010 'phone' (a heater), 007 and 011 'handbag' (a jacket and a backpack). It is reported as computed. For 'wall' scenes the "after" maps show the object region becoming more wall-like, the opposite of a stale answer. The N1/N2 row 013 shows exactly this. The stale-answer numbers are a new test measurement and must be disclosed as such.
3. **Teaser rule C1 failed.** At frame 373, averaging's mean p(chair) over the ground-truth region after the update is 0.06 (< 0.5), so the caption is neutral and the headline sentence is downgraded automatically. The curve (k) shows that ours drops the chair's top class to 15% against 44% for averaging, but also changes the top class of 33% of the static Gaussians against 20% for averaging. These values are plotted, not printed. Teaser deviations from the design: it uses full upright frames (no crop, per the review) and the 101-class Replica vocabulary with the Replica temperatures (decision 2).
4. **Colours.** Flagged and reset Gaussians are orange #E69F00, as the lead asked. The averaging curve in the teaser uses vermillion #D55E00, which is close to it; recolour it if the lead prefers.
5. **Not done.** Fig. F, the false-alarm and miss rows: fa_frames.json is dumped for every scene, but no figure was made (optional). The supplementary validation grid was not made: the validation dumps carry the old tint and would need a rerun (about 4 x 3-16 min of GPU). R4 theta_fix is not in this task.
6. **PDF sizes.** qual_change 6.7 MB, teaser 5.2 MB, qual_query 3.5 MB (images at 600 dpi). Lower `dpi` in `RQ.save` if the venue limits size.
7. **Anonymity of data files.** The meta and record JSONs in Q hold absolute paths and the host name. Never ship them.
