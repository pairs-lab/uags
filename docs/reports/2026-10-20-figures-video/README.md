# Figures, video and stronger results for the UAGS paper
2026-10-20 | root: /home/thanhnc/gauss_ws/src/uags | git: master@c95a871 | [all reports](../INDEX.md)
Follows: [Complete the paper](../2026-10-17-paper-completion/README.md)
Verdict: PARTIAL -- teaser, pipeline, qualitative, calibration and video done; fused-baseline check NEGATIVE for our "fused best" claim (paper corrected); paper 17 pages / 17.9 MB to trim
Key result: stale-answer rate after the update 20.7% (ours) vs 34.3% (averaging), test N=10, new measurement ([numbers](logs/nor/n2_numbers_test.tex)); fused baselines beat ours+depth (81.1, 79.9 vs 75.0, [verdict](logs/nor_fused/nor_fused_verdict.md))
Attempts: 4 (0 Good / 4 Bad) | Time: started 2026-10-20 14:50 JST

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
| 4c | Pipeline figure v2 (agent): real validation thumbnails (Replica office_0 frame 825; NOR 002 frame 335 by "largest masks_real area", not the later frame-373 rule), loop over mapping and update passes, robust set; replay of NOR 002 reproduces the val record (resets 117,170, Rec 0.536, FA 0.140) ([summary](logs/pipeline_summary.md)) | done, in the paper (Fig. 1) |
| 4d | Paper error found by the pipeline agent: Sec. III-F wrote the change score with omega; the code weights the test stream by nu (`usplat/vmf_field.py` _ingest: Rf += wt*u, wt = change weights). Fixed, and stated that Replica/ScanNet fields use nu = omega (field configs: change_weight view/None, fast_min 1) and NOR fields relmax (fast_min 0.5) | done |
| 4e | Static comparison figure (agent): room_1 f66 + scene0140_00 f57 (rules fixed before rendering, CPU-reproduced); code frozen on validation scenes before any test render; every render re-scores mIoU/AUSE vs the published JSON (max diff 2.2e-6); 41 tests pass ([summary](logs/static_summary.md)) | done, in the paper (Fig. qual-compare) + supp_compare_all.pdf (14 scenes) |
| 4f | NOR agent: dump of the Table III update for all 14 scenes, each reproducing its record (max AP diff 4.6e-7; noise repeat tolerance 0.002); teaser (val 002, frame 373), stale-content figure (test 015/013/006 by rule), query figure; 10 test looks logged ([summary](logs/nor_summary.md)) | done, in the paper |
| 4g | Pre-registered fused baselines (F2, [verdict](logs/nor_fused/nor_fused_verdict.md)): feature+depth 81.1, NormLift+depth 79.9, ours+depth 75.0, NCC+depth 61.2, dispersion+depth 34.2 (G-AP, test N=10); ours+depth higher than feature+depth in 0/10 (Holm p 0.016), than NormLift+depth in 1/10 (Holm 0.107) | **NEGATIVE for the claim**: "fused with depth ranks best" removed from abstract, intro, results, conclusion; new Table (tab:fused) |
| 4h | New measurement (test, disclosed): stale-answer rate (share of the removed object's covered pixels whose top class is the query) 48.3 before, 34.3 averaging, 34.2 change test off, 20.7 ours; ours < test-off in 9/10 ([numbers](logs/nor/n2_numbers_test.tex)) | in abstract, intro, results, conclusion |
| 4i | Teaser caption rule C1 failed (averaging's p(chair) after the update 0.06 < 0.5): caption neutral, as the rule requires | noted |
| 5 | Video (agent): 2:24, 1920x1080, 30 fps, H.264, 37.9 MB zip (limit 50 MB); replays of 002 and 014 equal their Table III records (FA, Rec, Kept exact; AP within 2e-8); every caption number parsed from main.tex; anonymous; 6 tests pass ([summary](logs/video_summary.md)). Files: /home/thanhnc/gauss_ws/data/uags_results/video/supplementary_video.zip (37.9 MB), master.mp4, stills/ | done (9b6002b) |

## 3 Attempts (method)
| # | Idea | Result | Critic |
|---|---|---|---|
| 01 | Spatial smoothing of the per-Gaussian sums over 3D neighbours (BKI kernel), 24 kernels, pre-registered (configs/smooth_prereg.json, cafb61d; underflow amendment b4d6205), chosen by pooled val NLL | Replica: no kernel beats P2 on val (more smoothing = worse). ScanNet: k32_s4_g0.8 chosen (val NLL 0.4031 vs 0.4104); test NLL 0.535 vs P2 0.509, averaging 0.512; ECE 6.6 vs 6.1, 6.2; AURC 5.4 vs 5.0, 4.4; C1 fails on all metrics ([note](logs/smooth_stage1.md)) | **Bad**: neighbours carry the same LSeg feature; the val gain was the P2 refit (gamma_e -3.04) overfitting 2 val scans. Disclosed in the appendix; Table I unchanged. |
| 02 | Fused detectors fire our reset (V1 feat+depth, V2 ours+feat+depth, V3 NormLift+depth), pre-registered (configs/reset_fused_prereg.json, afd693b; memory fix f0fc8ee) | val: V2 chosen (stale-answer 24.14 vs ours 24.77; G-AP 85.7 vs 60.3). Test: stale-answer 26.0 vs ours 20.7, lower in 3/10; G-AP 79.3 vs 53.4, FA 3.7 vs 6.2 ([note](logs/reset_fused_stage1.md)) | **Bad** for the claim: better detector, worse forgetting (misses thin objects, e.g. book 005). Disclosed in Sec. V-D; ours stays. |
| 03 | Uncertainty-guided fusion of the map with the current frame's 2D prediction (w = sigmoid of u_epi, u_ale, undefined, 1-alpha), pre-registered (configs/fusion_prereg.json, 9671f02) vs constant / view-count / NormLift weights | val Replica: +1.4 mIoU for every fused row, OURS-UG best NLL; test: fused rows move mIoU by <= 0.4; OURS-UG worse than its constant-weight version in 11/14 scenes (mIoU, NLL); C1 fails on all metrics, all Holm p = 1 ([note](logs/fusion_stage1.md)) | **Bad**: the 5-parameter weight overfits 2 val scenes per dataset. Disclosed in the appendix; Table I unchanged. |
| 04 | Stronger extractor for every row (RADSeg, ProxyCLIP-H, Trident vs LSeg), pre-registered (configs/extractor_prereg.json, dc879b6): winner must gain >= 1.0 val 2D mIoU on both datasets | val 2D mIoU Replica / ScanNet: LSeg 30.8 / 67.5, RADSeg 34.0 / 43.1, ProxyCLIP-H 26.1 / 38.0, Trident 29.4 / 35.9; reproduction gate PASS ([note](logs/extractor_stage1.md)) | **Bad**: training-free extractors lose indoor classes on real scans (refrigerator, window, cabinet). Stopped on validation; no test look. LSeg stays. |

## 4 Paper framing (user decision 2026-10-22: option 1)
Table I is presented as a parity check (same LSeg features for every map row). The abstract, introduction, experiment
questions and conclusion now lead with the two capabilities only our method gives: the u_epi/u_ale split (Replica 6/6;
ScanNet full map u_ale beats u_epi 8/8) and forgetting after a real removal (stale-answer rate 21% vs 34%). Build-time
claim limited to the semantic layer (130-225x; end to end 2.9-4x). Commit 60a7e56.

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Statistics tests | `python3 -m pytest tests/test_quant_stats.py -q` | 26 passed (sign test = scipy binomtest; ECE from bins = stored ECE, max err 5.6e-17) | PASS |
| Paper numbers vs JSON (lead, spot check) | python read of logs/quant/{seg,nor,views,cost}.json | F1 wins/Holm, both ECE aggregations, r_static, ScanNet views, cost equal to the text | PASS |
| Paper build | pdflatex + bibtex x2 on a copy of papers/ | 12 pages, 0 warnings, no overfull boxes | PASS |

## 7 Open issues
- Fused baselines beat ours+depth on stale-content ranking (4g); the paper now says so. The remaining positive NOR result is the stale-answer rate (4h), a new test measurement.
- Calibration: ours ties averaging (no Holm-significant difference); the lead over trained maps is shared by averaging (training-free lifting).
- Paper is 17 pages and 17.9 MB (600 dpi images in teaser/qual_change/qual_query); trim both before submission.
- Teaser sits on page 2 (figure* cannot float to page 1 in IEEEtran without extra work).
- Query rule gives generic classes in some scenes (009, 013 'wall'; 010 'phone'); reported as computed.
- Video 002 caption names only the upside (chair top class 44% averaging vs 15% ours); the static cost (33% vs 20% of static Gaussians relabelled) is only in the curves. Consider neutral wording.
- Dataset licences for showing images in the video were not checked.
- Not done: Fig. F (failure cases), fixed-threshold analysis (R4), PR curves, B7 reducibility check, ScanNet views experiment (B4).

## 8 Files
- this folder
