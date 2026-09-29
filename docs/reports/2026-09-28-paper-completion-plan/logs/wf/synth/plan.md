# UAGS final plan (synth, 2026-09-28)

Base: plan B's evidence design (highest acceptance from both judges). Scope and dates from A; estimator and GPU gate from C. **[A]** = assumption with no report behind it. WF = `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/wf`. I changed no project file.

**Disagreements settled:**
- **Date:** Nov 15, not B's Nov 27, which leaves 3 days of slack. Nov 27 becomes the slip date, decided Oct 9.
- **Estimator:** build both the joint prior and Kish; pick on val. On a tie, use the joint prior: it is one posterior with (R, n) sufficient.
- **Optimised field:** required (B), not a stretch goal (A).
- **RIO10 test:** 7 scenes. If disk is short, the 4 smallest by size, fixed before any result.
- **Localization row (C):** cut; it collides with USGLoc.
- **False-alarm control:** on Replica, not USGLoc's 7-Scenes maps.
- **USGLoc code:** copy at a pinned commit, not PYTHONPATH.
- **G1:** kept though weak. It costs under 1 CPU-day; G3 is the real test.

## 1 Scope decision

| Keep (narrowed) | Cut or demote |
|---|---|
| **C1:** a vMF conjugate posterior per Gaussian, with evidence counted in views. Word it as "a directional generalization; shares the sufficient statistic and MAP label of categorical BKI", and cite LatentBKI and Mardia & El-Atoum. | tab:loc (main.tex:456-480), the ATE column, and the keyword "localization" |
| **C2:** an epistemic channel (view count) and an aleatoric channel (dispersion), with a defined per-pixel predictive | The EFE section (main.tex:371-388) |
| **C4:** a temporal change test plus consolidation, tested on real rearrangements in RIO10 | The gradient trick (main.tex:278-307) becomes one paragraph that cites Joseph et al.; drop Fig. 3 |

**State openly:** μ* is the T·α-weighted mean, so the claim is **same mIoU, better uncertainty**. With m0 = 0 and c0 = β, u_ale = 1 − NormLift reliability (max difference 0.0). NormLift also splits intra- and inter-view terms. **Left to prove:** view-level evidence with its own epistemic channel (G2), and change handling on real data (G3); NormLift is static only.

**Title:** "Uncertainty-Aware Open-Vocabulary Gaussian Splatting for Semantic Mapping in Changing Scenes". Fallback titles are in §7.

**Abstract, in order:** the problem (no trust signal, stale semantics); the method; Replica (equal mIoU, lower ECE and AUSE, the two channels separate); RIO10 stale-content AP. Numbers go in only after the test runs.

## 2 Technical fixes before any experiment

**Symbols:** a0 = view-mass floor; z̄ = scene mean feature; (m0, c0) = the prior. This fixes the plans' double use of "m0".

| # | Fix | Why | Validation |
|---|---|---|---|
| F1 | Store (R_v, m_v) per keyframe. Set R = Σ w_v R_v/m_v and n = Σ w_v, with w_v = min(1, m_v/(a0·H·W)). **a0 is a fraction of image area**, so n does not depend on resolution. Drop near-duplicate keyframes. Set τ_min in views. | n_eff counts pixels (~50× over-count). | n changes <1% between ds4 and full resolution [A]. The share of Gaussians stable after 1 view goes from 1.00 to 0. |
| F2 | Two estimators. Joint prior: A_d(κ̂) = ‖R+m0‖/(n+c0). Kish: ρ̂² = (n_K r̄²−1)/(n_K−1), undefined when n_K ≤ 1. Set u_ale = 1 − estimate. | r̃ has no effect when r̄ = 1. | One view gives 1/(1+c0). Uniform directions give ρ̂ ≈ 0 (0.062/0.021). C6 check: 0.3330 vs 0.3333. |
| F3 | Centre at read-out: R_c = R − n·z̄. This is exact because R is linear in unit z, and it needs no extra memory. **Queries use the raw R.** Dispersion and change use R_c. | Random CLIP pairs have cos ≈ 0.65. Centring the query would shift class scores by n·z̄ᵀt_c (a gap in all plans). | R_c equals the directly accumulated value within 1e-6. The query argmax does not change. |
| F4 | Change score c = 1 − cos(μ_slow, μ_fast) from two centred EWMA resultants. Split r̄ into r_within and r_across. | Changed content has r̄ 0.86–0.95, and r̄ ignores order. | Gates G1 and G3 |
| F5 | When c fires: reset (R, n) and set a persistent `changed` flag. Re-admit a Gaussian after τ_min new views **and** a small depth residual. λ per keyframe. One config file. | Decay leaves r̄ unchanged; forgotten Gaussians return. | Unit test on an A→B sequence |
| F6 | Composite μ*, u_epi and u_ale with T·α. Define p(c) = softmax(β μ_pᵀt_c), with β fit on val. **Every row uses this**, so no ECE cell is "--". | No render; "calibrated" undefined. | Mock vs brute force |
| F7 | Decode-first query μ*ᵀ(Bᵀt). Fit PCA on images only, so no evaluation vocabulary leaks in. | Text keeps only 11% of its energy. 13–29% of labels flip. | Argmax agrees with 512-d on ≥ 0.95 of pixels [A]; now 0.707. |
| F8 | Extract features at full 640×480. Make laion2b the default. Cache fp16 PCA-64. | At ds4, labels agree with full resolution on only 55–71% of pixels. | Agreement on val |
| F9 | Per-Gaussian gsplat identity test (one-hot, N ≈ 300). One `rasterization()` call with `channel_chunk`. Drop `.clone()`. | The test checks only linearity and mass. | Error < 1e-4 [A]. Under 5 min on the GPU. |
| F10 | Text fixes: κ only scales κ* (report the κ̂ MLE); exact variance 1 − A_d; "rasterizer affine in features"; one AUSE definition. | math E3–E5, E11 | Supervisor read-through |

## 3 Experiments

**Rules:** same 3DGS geometry and same MaskCLIP ViT-B/16 laion2b features for every method. All parameters are set on val; the test runs once from a frozen commit. Report mean ± std with N; keep any row where a baseline wins.

**Splits:**
- **Replica (Semantic-NeRF).** Val: office_0, room_2. Test: room_0, room_1, office_1–4 (N = 6). Map on Seq_1, every 5th frame (180 frames). Evaluate on Seq_2, every 10th frame (90 frames).
- **RIO10.** Val: 01, 04, 06. Test: 02, 03, 05, 07, 08, 09, 10. Map on seqXX_01, every 10th frame [A]. Update with the first half of seqXX_02, every 5th frame [A], with ground-truth poses.

**Stale ground truth:** stale = more than 5 cm from the seq02 surface and less than 5 cm [A] from the seq01 surface. Surfaces come from **rendered depth back-projected with the aligned poses**; meshes only cross-check (alignment unchecked). **Hidden** = behind the seq02 depth in every update view; left out of AP and reported separately.

| Item | Question | Rows (fair) | Metrics | Success | GPU-h [A] | Needs |
|---|---|---|---|---|---|---|
| **Tab I** (Replica test) | At equal mIoU, is our uncertainty better ranked and better calibrated? | 2D MaskCLIP; pixel-weighted averaging (Joseph/Occam); **optimised gsplat field**; NormLift (pixel Kish, β = 1) on our μ*; CSS Dirichlet with variance and MI/entropy; Ours | mIoU↑, Acc↑, ECE-15↓, NLL↓, AUSE_epi↓, AUSE_ale↓ | mIoU within 1 point of averaging [A]. ECE below the best deterministic row. AUSE_epi below NormLift and CSS on ≥ 5/6 scenes [A]. | ≤ 10 | F1–F9 |
| **Fig. views** | Does u_epi track error that more views remove, and u_ale the error that stays? | Ours, NormLift, CSS variance; maps from 10/30/90/180 views | 2×2 cross-AUSE↓ | For Ours, diagonal < off-diagonal | 1 | Tab I |
| **Tab II** (RIO10 test) | Does the change test find stale semantics, and does consolidation clean them out? | Photometric residual; **depth residual**; scalar variance; CSS variance/MI; NormLift drop; raw r̄; Ours u_ale; Ours change score; consolidation on/off | Stale AP↑, AUROC↑; stale mass left↓; static mass wrongly forgotten↓; results by SemanticChange | AP above both residuals on ≥ 5/7 scenes [A] | ≤ 13 | RIO10, F3–F5 |
| Tab II static row | How many false alarms? | Same rows. Replica: map on Seq_1, update on Seq_2 | % flagged↓ | ≤ photometric residual | 0.5 | Tab I maps |
| **Tab III** | Which parts matter? | Remove one at a time: (a) views vs pixels; (b) joint prior vs Kish vs r̃; (c) no centring; (d) change score vs r̄; (e) no depth re-admission; (f) vMF vs categorical; (g) query fix | As in Tab I/II | Drop parts that do not help on val, and rename the method to what remains | 3 | I, II |
| Fig. sensitivity (val) | Are the thresholds robust? | none | AP and ECE vs τ_min, c0, λ, a0, δ ∈ {3, 5, 10} cm | A stated working range | 1 | II |
| Fig. qualitative | Illustrate | One RIO10 val frame, one Replica frame | RGB, query, u_epi, u_ale, c | none | 0.2 | I, II |
| Timing | What does it cost? | Ours vs averaging, idle GPU | ms per keyframe, peak GB | ≤ 3 GB at d = 64 | 0.2 | F9 |
| G5 re-run | Reproducible? | One Replica scene and one RIO10 scene, re-run from the saved commands | Same numbers | Match | 0.5 | all |

**Compute** [A, not measured]: ≤ 30 GPU-h; free memory, not hours, is the limit. CPU extraction: Replica ≈ 1.1 h (2,160 frames × 1.81 s); RIO10 updates ≈ 1.7 h, plus seq01 frames (not counted). **Out of scope:** 7-/12-Scenes, ScanNet, ACE/ACE-G/Marepo, Chu & Kim, active mapping.

## 4 Code tasks (in order)

| # | Task | Files | pd |
|---|---|---|---|
| K1 | `gpu_gate.sh`: start only if free memory ≥ peak + 4 GB (+ 7 GB while MASt3R is queued), `pgrep -f 'run_7scenes\|exp_floor_matcher\|train_map\|train_encoders'` finds nothing, and the `flock` lock is taken. Then call `set_per_process_memory_fraction`. Also install plyfile and make laion2b the default. | scripts/, run_mapping.py:194 | 0.5 |
| K2 | F1 with tests | bki_accumulate.py, vmf_field.py, tests/test_view_count.py | 1 |
| K3 | F2 and κ̂ | vmf_field.py, tests/test_estimators.py | 1 |
| K4 | F3, F4, F5 and the config file | usplat/change.py, consolidation.py, tests/ | 2 |
| K5 | G1 replay using the math C3b protocol: 7-Scenes chess seq-01, frames 0–190, stride 10, CPU only, internal (not in the paper) | scripts/cpu_check_change.py | 0.75 |
| K6 | F7, and the F8 cache | feature_pca.py, scripts/extract_clip.py | 1.25 |
| K7 | F6 and F9 | gsplat_adapter.py, tests/test_gsplat_gpu.py | 1.75 |
| K8 | Replica Semantic-NeRF loader and splits | datasets.py | 1 |
| K9 | fit_gs.py from `git -C ../usgloc show adea106:usgloc/mapping.py`; record the hash and credit it | scripts/fit_gs.py | 1.5 |
| K10 | Evaluation harness: mIoU/Acc, ECE + β, NLL, cross-AUSE. One AUSE definition, copied from usgloc/metrics.py:79 at the same commit. | evaluation.py, scripts/eval_seg.py | 2 |
| K11 | Baselines: averaging and NormLift 0.5, CSS 1.5, optimised field 1.5 | usplat/baselines/ | 3.5 |
| K12 | RIO10 loader (540×960 portrait), stale and hidden ground truth, update protocol | datasets.py, scripts/rio10_stale_gt.py | 2.5 |
| K13 | Photometric and depth residuals; Replica static run | scripts/eval_change.py | 0.75 |
| K14 | Run scripts that write JSON with the command and git hash; sweeps; figures | scripts/ | 1.5 |
| K15 | Timing and memory measurement | run_mapping.py | 0.5 |
| | **Total** | | **21.5** |

## 5 Writing tasks (7.5 pd)

| Task | main.tex lines | pd |
|---|---|---|
| Port to ieeeconf; check `\IEEEPARstart`, `IEEEkeywords` and `\markboth`; anonymise; build on Overleaf (tectonic is missing) | 2, 13, 39–45, 73–81, 522–523 | 0.5 |
| Abstract, intro and contributions; remove "principled", "without heuristics", "any rasterizer", "continuous generalization" and "localization" | 53–133 | 1 |
| Related work and bib [supervisor, A]: add the 12 must-cites of sota §3 plus RIO10; fix the sota §4 mismatches and l.150–152, 166, 175, 185; cite USGLoc in the third person; check MaskCLIP, AUSE, ECE and Kish by DOI. | 136–191, 526–624 | 1.5 |
| Method: a preliminaries equation (defines T); F1–F6; one paragraph on the gradient trick; delete EFE; add an algorithm box | 192–388 | 2 |
| Experiments, implementation table, limitations (fixed geometry, the CLIP common component, Tango image quality) | 389–520 | 1.75 |
| Figures: merge Fig 2 and Fig 4; drop Fig 3; add the views/sensitivity panel and fig:qual | figs/ | 0.75 |

**Page budget:** 6 pages including references. Intro 0.8, related work 0.6, method 1.7, experiments 2.2, limitations 0.2, references 0.5. A paid 7th page is allowed (D4).

## 6 Timeline (target Sun 2026-11-15; slip date Fri 11-27; hard stop Mon 11-30)

**Why:** the IROS 2027 transfer window closes Apr 30, 2027 and RA-L decides within 5 months, so Nov 30 is the last safe date; ICRA 2027 is out. Load is 29 pd for 34 workdays to Nov 13, which holds only if the student is mostly on UAGS (the same author committed to USGLoc today; 15 files uncommitted).

| Week | Work | Gate (Friday) |
|---|---|---|
| W1 Sep 29–Oct 4 | **Day 1:** submit the RIO10 form; download Replica (6.76 GB, about 11 min at 10.3 MB/s); ask when USGLoc will be submitted. Then K1–K3, the paper port and the bibliography. | **G0 Oct 2:** form sent; Replica on disk; 6/6 tests pass |
| W2 Oct 5–11 | K4–K8; CPU extraction. Night jobs only if the gate passes: F9, and peak memory on office_0. | **G1 Oct 9:** replay AUROC > 0.633, and n does not change with resolution. If it fails: rename u_ale "ambiguity" and consolidate on c only. **Capacity:** if USGLoc is not submitted, move the target to Nov 27. |
| W3 Oct 12–18 | K9–K11; fit 8 Replica maps at night. The RIO10 val scenes must be on disk by **Wed Oct 14**; if not, switch to the Bonn box sequences. | none |
| W4 Oct 19–25 | Tab I and Fig. views on val; tune a0, c0, β and τ_min; choose the F2 estimator | **G2 Oct 23:** AUSE_epi below NormLift and CSS on both val scenes, and ECE ≤ the best deterministic row. If only AUSE_ale ties NormLift: say so, and move the claim to the epistemic channel and change handling. If AUSE_epi fails: use F-B. |
| W5 Oct 26–Nov 1 | K12–K13; Tab II and the static row on val; sensitivity; ablations on val | **G3 Oct 30:** AP above the photometric and depth residuals on ≥ 2/3 val scenes. If not: use F-A. |
| W6 Nov 2–8 | Freeze the commit. Run the test once: Replica 6 scenes, RIO10 7 (or 4). Ablations; timing on an idle GPU; figures. | **G4 Nov 6:** test runs started, and every table cell comes from a log. If not, move to Nov 27. |
| W7 Nov 9–15 | Write the results; supervisor review Nov 11–12; pdfinfo and anonymity check; note to the editor about USGLoc | **G5 Nov 13:** the re-run matches, and the paper is ≤ 7 pages. Submit Nov 15. |

**Never submit val-only results.** If G2 and G3 both fail, do not submit in November. My estimate (window close minus 5 months) is about Dec 31 for AIM 2027 or Jan 29 for Humanoids 2027.

## 7 Risks

| Risk | Evidence | Mitigation | Fallback |
|---|---|---|---|
| The change signal is weak on real CLIP | AUROC 0.59–0.75 | F3, F4; G1; val thresholds; sensitivity | **F-A:** "Directional Bayesian Fusion of Open-Vocabulary Features in Gaussian Splatting with View-Level Epistemic and Aleatoric Uncertainty"; Tab II moves to Limitations |
| The depth residual beats c | Geometry is fixed, so a moved object leaves a residual [A] | Keep that row; report semantic-only changes | Present c as a complement fused with the residual; drop "robust" |
| u_ale equals NormLift | estim_equiv.log | G2; view counting; the Kish option | **F-B:** "calibrated view-level confidence for training-free lifting" plus change; a later window if this is too thin |
| RIO10 access or size (each seqXX.zip holds every rescan); class map unchecked | README l.13–63 | Form on day 1; one scene at a time; unzip only _01/_02; no RIO10 mIoU until the class map is checked | Bonn by Oct 14 |
| GPU and the student are shared with USGLoc | 13,606/16,303 MiB used; `sang` holds 3.85 GB; USGLoc relaunches after 5.3 h | K1 gate and lock; night jobs; CPU work first | N = 4; fewer sweeps; Nov 27 |
| Disk: 48 GB free | df | fp16 caches; delete fits after evaluation | Ask before pruning usgloc/maps |
| Overlap with USGLoc | venue C | No localization, no 7-Scenes, none of its maps or numbers; disclose to the editor | none |

## 8 Decisions for the user

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Date | (a) Nov 15, slip to Nov 27 at the Oct 9 gate; (b) Nov 27 fixed; (c) Jan 29, RA-L only | (a) |
| D2 | Change dataset | (a) RIO10; (b) Bonn; (c) our own capture (3–10 pd) | (a), with (b) if there is no link by Oct 14 |
| D3 | GPU sharing | (a) nights plus a shared `flock` lock until USGLoc is submitted; (b) fixed time slots | (a) |
| D4 | Length | (a) 6 pages; (b) allow 7 ($175) | (b) |
| D5 | USGLoc coordination | Which paper goes first, the disclosure, the author list, and the "open-vocabulary" wording at usgloc_ral_draft.tex:72 | Agree this week; the supervisor decides |

## 9 Evidence map

| Fact | Report | Source |
|---|---|---|
| 6 pages including references, +2 at $175; ieeeconf for the first submission; double-anonymous review | venue A | https://www.ieee-ras.org/publications/ra-l/information-for-authors-ra-l ; main.tex:13 |
| IROS 2027 window Aug 1 2026–Apr 30 2027; final decision ≤ 5 months; ICRA transfer by Dec 31 2026; AIM and Humanoids windows | venue B | https://www.ieee-ras.org/publications/ra-l/ ; https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/ |
| Weekdays; 34 and 44 workdays | synth, judge | `date -d`; python count; WF/judge/dates.txt |
| u_ale (m0 = 0) equals NormLift: max difference 0.0, Spearman 1.0; Kish 0.978 | judge | WF/judge/estim_equiv.log (`python3 estim_equiv.py`) |
| NormLift reliability and intra/inter split; static only | sota §1–2 | https://arxiv.org/html/2609.18898 |
| r̃ has no effect at r̄ = 1 | math E7, code 1 | vmf_field.py:74-76; WF/math/c2_out.txt |
| About 50× pixel over-counting; stable fraction 1.00 | math E8 | WF/math/c3_out.txt [C3-ii] |
| Change AUROC 0.591/0.633 (0.725/0.745) | math E9, code | WF/math/c3b_out.txt; WF/code/clip_static_vs_moved.py |
| Joint prior 0.3330 vs 0.3333 | math E10 | WF/math/c6_out.txt |
| Decay leaves r̄ unchanged; thresholds 0.5 vs 0.55 | math E12, code 4 | consolidation.py:21,40-41; run_mapping.py:200 |
| Text energy 0.109/0.119; label flips 29%/13.2%; ds4 agreement 0.55–0.71 | code risks 2–3, math E15 | WF/code/; WF/math/c3_out.txt [C3-iv] |
| μ* is the weighted mean; κ has no effect | code 6 | WF/code/formula_checks.log |
| The gsplat test checks only linearity and mass | math E11 | tests/test_gsplat_gpu.py:53-75 |
| The gradient trick is scooped | sota C3 | WF/sota/gradbp_backproject.py:115-150 |
| 2.5–3 GB at d = 64; 1.81 s per frame on CPU | code | WF/code/scale_cpu.log |
| No CSS code; Feature 3DGS and LangSplat do not fit the GPU | data §2 | WF/data/logs/github_licences.txt |
| RIO10: form required, URL redacted, zip layout, aligned poses | judge, data | WF/judge/rio10_readme.md l.9, 13-63; WF/judge/setup_rio10.py:6,31 |
| RIO10 splits; 34,415 val frames; 5 cm stale rule | data §1, §4 | WF/data/logs/rio10_stats_summary.txt |
| Replica zip 6,762,780,327 B at 10.3 MB/s | judge (PM) | WF/judge/dbx_head.txt, dbx_range.txt |
| GPU 13,606/16,303 MiB; 48 GB free; tectonic missing | synth | WF/synth/checks.txt; `ls ~/.local/bin/tectonic` |
| `sang` uses 3,852 MiB; USGLoc gates relaunch after 5.3 h | venue D | venue report §D |
| USGLoc commit adea106 today; 15 uncommitted files | synth | `git -C usgloc log -1`; `git status --short \| wc -l` |
| uags at ab28fb4; 11 tracked .pyc files modified; docs/ untracked | synth | `git status --short` |
| 5/6 tests pass (plyfile missing) | code | WF/code/cpu_tests.log |
| Overlap and boundary with USGLoc | venue C | usgloc_ral_draft.tex:88, 150-156, 471; main.tex:67-68 |
| Bibliography mismatches | sota §4 | WF/sota/meta*.txt, cr*.txt |
