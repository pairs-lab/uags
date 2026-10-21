# Figures, video and stronger results for the UAGS paper
2026-10-20 | root: /home/thanhnc/gauss_ws/src/uags | git: master@c95a871 | [all reports](../INDEX.md)
Follows: [Complete the paper](../2026-10-17-paper-completion/README.md)
Verdict: IN PROGRESS -- designs done; 4 implementation agents running; video next
Key result: n/a
Attempts: n/a | Time: started 2026-10-20 14:50 JST

## 0 Context
- Ask (user, 2026-10-20): "in this step, we do not need to keep this in 8 pages, just try to improve results, qualitative
  results, teaser figure, pipeline figure, video, quantitative results".
- Rules kept: no tuning on test scenes (Replica, ScanNet and NOR test results are already seen; a method change must be chosen
  on validation and every extra test look disclosed); figure frames/scenes chosen by a rule written before rendering, or from
  validation scenes; double-anonymous (no names or lab marks in figures or video); GPU only through scripts/gpu_gate.sh.
- Machine: RTX 4070 Ti 12 GB (shared), i7-13700KF. No system ffmpeg: installed imageio-ffmpeg 0.6.0 (bundled ffmpeg 7.0.2)
  into ~/venvs/uags with `uv pip install --no-deps` (the venv has no pip); torch stays 2.11.0+cu128; H.264 mp4 write/read test PASS.

## 1 Plan
| Step | What | Status |
|---|---|---|
| 1 | Design workflow (wf_7001c8b3-071): code + data inventory; designs for teaser, qualitative, video, quantitative; critics for quantitative and qualitative ([designs](logs/design/)) | done 2026-10-21 08:47; pipeline design and the teaser/video critiques failed (usage limit) |
| 2 | Pre-registration of the fused baselines for Table III ([configs/nor_fused_prereg.json](../../../configs/nor_fused_prereg.json)), committed before any fused score (dae4b10) | done |
| 3 | Paper integrity edits from the quantitative review: "we run them once" replaced by the list of test evaluations; ScanNet P2 coefficients disclosed (gamma_e -0.80, gamma_a -2.06 vs +0.36, +11.8 on Replica; eval_val.json betas) | done (6029f04) |
| 4 | Four implementation agents (background): NOR dump + teaser + change/query figures + fused baselines; static comparison figure; CPU statistics (quant_stats, calibration plot); pipeline figure v2 | running |
| 4a | Statistics agent (CPU): scripts/quant_stats.py, plot_calibration.py, 26 tests, configs/quant_prereg.json ([summary](logs/quant_summary.md)) | done, committed 6032669 |
| 4b | Paper: calibration claims restated by scene tests (ours ties averaging; training-free lifting beats the trained maps); pooled-ECE caveat; ScanNet full-map ranking; measured r_static; P2 sign in Limitations; ScanNet build cost; appendix with Tables V-IX and Fig. 5 (all cells generated, not typed) | done (85adf2c, d258306); 12 pages, 0 warnings |
| 5 | Video (after 4: reuses the NOR dump and static renders) | to do |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Statistics tests | `python3 -m pytest tests/test_quant_stats.py -q` | 26 passed (sign test = scipy binomtest; ECE from bins = stored ECE, max err 5.6e-17) | PASS |
| Paper numbers vs JSON (lead, spot check) | python read of logs/quant/{seg,nor,views,cost}.json | F1 wins/Holm, both ECE aggregations, r_static, ScanNet views, cost equal to the text | PASS |
| Paper build | pdflatex + bibtex x2 on a copy of papers/ | 12 pages, 0 warnings, no overfull boxes | PASS |

## 7 Open issues
- none yet

## 8 Files
- this folder
