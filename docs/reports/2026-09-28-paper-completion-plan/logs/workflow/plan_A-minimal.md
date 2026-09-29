# UAGS plan A: minimal credible path to RA-L (2026-09-29 to 2026-11-15)

Tags point to the input reports: [code], [math Ex], [sota], [data], [venue]. **[A]** marks an assumption that the reports do not back. I ran no project code and wrote no files. The only commands I ran were `date` (to get the weekdays) and a grep of the section headings in `papers/main.tex`.

## 1 Scope decision

| Keep, narrowed | Cut or demote |
|---|---|
| C1: vMF conjugate BKI on splats, worded as "a directional generalization". Cite Mardia & El-Atoum, LatentBKI (continuous BKI, on voxels) and NormLift (concurrent) [sota]. | Localization: tab:loc (main.tex:456-480), the ATE column of tab:abl, and the keyword "localization". It collides with USGLoc, there is no change data for it, and refine_pose needs an initial pose [venue C]. |
| C2: epistemic/aleatoric read-out, with the fixed estimator from Section 2 and an explicit definition of calibration. | EFE section (main.tex:371-388). It is not an expected free energy, it has no experiment, and it costs page space [math E13]. |
| C4: consolidation, rebuilt as a change test and measured on real rearrangement data. | C3, the gradient trick. It is scooped by Gradient-Weighted Back-Projection [sota]. Keep one paragraph with that citation. Also drop Fig. 3 and the unused angular-variance formula [math E4]. |

**Honest core fact.** With κ0=0, μ* is exactly the normalized T·α-weighted mean feature [code mismatch 6; sota C1]. So our mIoU equals the averaging baseline. The paper claims **the same mIoU with better uncertainty**, not higher mIoU.

- **Title:** "Uncertainty-Aware Open-Vocabulary Gaussian Splatting for Semantic Mapping under Scene Change".
- **Fallback title (gate G3 fails):** "Directional Bayesian Fusion of Open-Vocabulary Features in Gaussian Splatting with Epistemic and Aleatoric Uncertainty".
- **Abstract, in order:**
  1. The problem: lifted CLIP features have no trust signal that separates "seen too little" from "inconsistent or changed".
  2. The method: a vMF posterior per Gaussian, evidence counted in views, closed-form epistemic and aleatoric channels, and a two-timescale change test.
  3. Replica: the same mIoU as training-free lifting, but lower AUSE and ECE, and the two channels separate error that more views remove from error that stays.
  4. RIO10: stale-content AP. Numbers go in only after the test runs.

## 2 Technical fixes before any experiment

| # | Fix | Why | Validation (pass test) |
|---|---|---|---|
| F1 | Count evidence in views. For each keyframe, take (R_v, m_v) from its backward pass. Then R = Σ w_v R_v/m_v and n = Σ w_v, with w_v = min(1, m_v/m0). Skip near-duplicate keyframes. Set τ_min in views. | n_eff counts pixels: one view gives n ≥ 13, and 100% of Gaussians are "stable" [math E8, C3-ii]. | Unit test: duplicating a frame or changing the downscale moves n by less than 1% [A]. Rerun C3-ii: the fraction "stable after one view" goes from 1.00 to about 0. |
| F2 | Replace r̃ with the joint-prior MAP: A_d(κ̂) = ‖R+m0‖/(n+c0), and set u_ale = 1 − A_d(κ̂). Tune c0 on val. | r̃ = 1 whenever r̄ = 1, so the "correction" does nothing [math E7]. The joint prior keeps (R, n) sufficient [math E10, C6]. | One view gives 1/(1+c0). In the C2-b setting, bias under pure conflict is lower than with r̃. With m0=0 this is nearly NormLift's reliability, so the paper must say so and cite it (my reading of [sota]). |
| F3 | Subtract the per-scene mean feature (from the mapping frames) before normalizing. Add a change score c = 1 − cos(μ_slow, μ_fast) from two exponentially weighted resultants. | On raw CLIP, changed content has r̄ of 0.81-0.95, so r_unstable=0.55 never fires. AUROC is only 0.59-0.73 [math E9, C3b; code risk 1]. | Smoke test: `c3b_change.py` AUROC above 0.633, the centred-r̄ number. Real test: RIO10 val (Tab II). |
| F4 | When the change test fires, reset (R, n) and set a `changed` flag. A Gaussian re-enters the robust set only after τ_min new views. Define λ per keyframe. Keep all thresholds in one config. | Decay leaves r̄ and μ* unchanged, and forgotten Gaussians come back [math E12; code mismatch 4]. r_unstable is 0.5 in one file and 0.55 in another. | Unit test on an A→B sequence: μ* follows B within k views, and the flag stays set. |
| F5 | Fit the 64-d PCA basis on image and text embeddings together. Score by decoding first, μ*ᵀBᵀt, without renormalizing Bᵀt. | Text keeps 11% of its energy in the image basis, and 29% of pixel labels change [code risk 2; math E15]. | Per-pixel argmax agrees with 512-d on at least 0.95 of Replica val pixels [A target]. Today it is 0.707. |
| F6 | Extract dense CLIP at full 640×480 and make laion2b the default weights. | At downscale 4, labels agree with full-res on only 55-71% of pixels, and the openai weights are not cached [code risk 3]. | Agreement table against full-res on val. |
| F7 | Per-pixel predictive: composite μ*, u_epi and u_ale with the same T·α weights. Define p(c) = softmax_c(β μ_pᵀt_c), with β fit on val. Use the same softmax for the deterministic rows, so their ECE is not "--". | Novel-view rendering is missing, and "calibrated" is never defined [math E16, §3]. | The mock-rasterizer result matches a brute-force reference, and ECE runs on val. |
| F8 | Say that κ only scales κ*. Report κ̂ from an MLE fit on val. | κ changes no decision in the method [math E5]. | One number in the Implementation paragraph. |
| F9 | Add a per-Gaussian gsplat identity test (N≈300 one-hot renders). Change "any rasterizer" to "any rasterizer affine in per-Gaussian features". | The GPU test checks only linearity and total mass [math E11]. | Max absolute difference below 1e-4 [A]. Under 1 min of GPU. |
| F10 | Make one `rasterization()` call with all channels and `channel_chunk`, and drop the `.clone()`. | Each of the 3 passes per frame redoes projection, and the clone costs memory [code, scalability]. | Measure peak memory once on office_0 while the GPU is idle. |

## 3 Experiments

**Rules for every row:**
- Same 3DGS geometry and same MaskCLIP ViT-B/16 laion2b features. Only the fusion rule differs.
- Tune on val. Run test once.
- Report mean ± std over N scenes.
- One AUSE definition, stated in the paper: per pixel, 0/1 label error, trapz over the removed fraction [A choice].

**Splits:**
- **Replica (Semantic-NeRF renders)** [data §4]. Val: office_0 and room_2. Test: room_0, room_1, office_1-4 (N=6). Map from Sequence_1 every 5th frame (180 frames). Evaluate on Sequence_2 every 10th frame (90 frames).
- **RIO10.** Map from seqXX_01, then update with the first half of seqXX_02 using GT poses. Val scenes: 01, 04, 06. Test scenes: at least 02, 03, 05, 07; add 08-10 if disk allows. The scene lists come from [data]; cutting test to 4 is my choice.

| Item | Question | Rows (fair) | Metrics | Success test | GPU-h [A] | Depends on |
|---|---|---|---|---|---|---|
| Tab I: seg + calibration (Replica test) | At equal mIoU, is our uncertainty ranked and calibrated better? | 2D MaskCLIP per frame; averaging + NormLift reliability; CSS categorical (one-hot of MaskCLIP argmax, same kernel) with variance and MI/entropy; Ours. Stretch: a gsplat distilled field | mIoU ↑, Acc ↑, AUSE_epi ↓, AUSE_ale ↓, ECE-15 ↓, NLL ↓ | Ours beats the best scalar row on AUSE and ECE in ≥5 of 6 scenes [A]. mIoU equals averaging (sanity check) | 3DGS fit 8×0.5 = 4; fusion 2 | F1-F7 |
| Fig: view sweep (Replica test, maps from 10/30/90/180 views) | Does u_epi track error that more views remove, and u_ale error that stays? | Ours vs NormLift reliability and CSS variance | 2×2 cross-AUSE. "View-reducible" = wrong at 10 views, right at 180. "Persistent" = still wrong at 180 | For Ours, diagonal AUSE < off-diagonal. No single scalar wins both | 1 | Tab I |
| Tab II: map under change (RIO10) | Does the change test find stale semantics, and does consolidation clean the map? | Photometric residual; u_ale alone; CSS variance; NormLift reliability drop; Ours change score; Ours with and without consolidation | Stale-Gaussian AP ↑ and AUROC ↑; stale mass left in the robust set ↓; static mass wrongly forgotten ↓ | Ours AP > photometric residual and > u_ale on ≥3 of 4 test scenes | Fit 7×1 = 7; runs 3; val sweep 4 | RIO10 access, F3, F4 |
| Tab III: ablations | What each fix buys | (i) categorical vs vMF; (ii) scalar vs two channels; (iii) pixel-count + r̃ vs F1+F2; (iv) PCA renorm vs F5; (v) raw vs F3 | Same metrics as Tab I / II | Each fix helps its target metric. A fix that does not help is reported and dropped | 2 | I, II |
| Fig: qualitative | Show the method working | One RIO10 val frame with high SemanticChange; one Replica test frame | RGB, query map, u_epi, u_ale, change score | none (illustration) | 0.2 | I, II |
| Timing | What it costs | Ours vs averaging, one scene, GPU idle | ms/frame, peak GB | Measured, not estimated | 0.2 | F10 |

- **Stale GT** [data §4]: a Gaussian is stale if it is more than 5 cm from the seq02 mesh and less than 5 cm from the seq01 mesh.
- **Hidden class** (my addition): Gaussians that sit behind the seq02 mesh depth in every update view form a third class. They are left out of AP and reported separately.
- **CPU extraction time.** Measured speed is 1.81 s/frame at 640×480 [code].
  - Replica: 8 scenes × 270 frames ≈ 1.1 CPU-h.
  - RIO10: about 3.5k frames [A] ≈ 3-6 CPU-h.
  - Cache PCA-64 fp16 features on the patch grid.
- **Total GPU:** about 23 GPU-h [A], run as gated night jobs.
- **Out of scope:** 7-Scenes, ScanNet, ACE/ACE-G/Marepo, Chu & Kim, active mapping.

**Gate before every GPU job:**
- `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` must show at least peak + 4 GB free (at least peak + 7 GB while a MASt3R batch is queued).
- `pgrep -f 'run_7scenes.py|exp_floor_matcher.py|train_map.py|train_encoders.py'` must be empty for long jobs [venue D].

## 4 Code tasks (in execution order)

| # | Task | Files | Person-days |
|---|---|---|---|
| 1 | `pip install --user --break-system-packages plyfile`; laion2b default; get tests to 6/6 | scripts/run_mapping.py:194 | 0.25 |
| 2 | F1 view counting, with tests | usplat/bki_accumulate.py, usplat/vmf_field.py, tests/test_view_count.py | 1 |
| 3 | F2 estimator and F8 κ̂, with tests | usplat/vmf_field.py, tests/test_estimators.py | 0.75 |
| 4 | F5 joint PCA; F6 extraction cache | usplat/feature_pca.py, scripts/extract_clip.py | 1.25 |
| 5 | F7 + F10 per-pixel renders; F9 test | usplat/gsplat_adapter.py, tests/test_gsplat_gpu.py | 1.75 |
| 6 | Replica Semantic-NeRF loader, label map, splits | usplat/datasets.py | 1 |
| 7 | 3DGS fitting adapted from usgloc train_map.py (disclosed in the paper) | scripts/fit_gs.py | 1.5 |
| 8 | Eval harness: mIoU/Acc, prompts, ignore labels, ECE + β, AUSE, cross-AUSE | usplat/evaluation.py, scripts/eval_seg.py | 2 |
| 9 | Baselines: 2D, averaging + NormLift, CSS Dirichlet | usplat/baselines.py | 1.5 |
| 10 | RIO10 loader, stale and hidden GT from meshes | scripts/rio10_stale_gt.py | 2.5 |
| 11 | F3 + F4 change score and consolidation; photometric residual baseline; tests | usplat/consolidation.py, tests/test_consolidation.py | 1.5 |
| 12 | Run scripts that save logs; figure scripts | scripts/run_tab1.sh, scripts/run_tab2.sh, scripts/make_figs.py | 1.5 |
| 13 | Timing and memory instrumentation | scripts/run_mapping.py | 0.5 |
| 14 | Stretch: gsplat distilled field | scripts/distill_field.py | 1.5 |
| | **Total without task 14** | | **≈18.5** |

## 5 Writing tasks (≈7 person-days)

| Task | main.tex lines | Person-days |
|---|---|---|
| Port to ieeeconf (`letterpaper, 10 pt, conference`) and check that `\IEEEPARstart`, `IEEEkeywords` and `\markboth` still work. Anonymise: remove lines 39-45 and the Acknowledgment. Run `pdfinfo` on the final PDF | 2, 13, 39-45, 81, 522 | 0.5 |
| Abstract, intro, contributions. Remove "principled", "without heuristics", "any rasterizer", "continuous generalization" and "localization" | 53-133 | 1 |
| Related work. Add LatentBKI, Grad-Weighted BP, NormLift, Mardia & El-Atoum, Occam's LGS, LEGaussians, SLIM-VDB, MIF, 3DGS-CD, CL-Splats, VG-Mapping, GaussianUpdate. Fix E2-BKI, PhysGS, LAMP, VISTA, gsplat, Marepo title, pages and DOIs. Fix l.150-152, 166, 175, 185. Cite USGLoc in the third person | 136-191, bib | 1.25 |
| Method. Add a 3DGS preliminaries equation (define T). Write F1-F4, F7, F8. Cut the gradient trick to one paragraph. Remove EFE. Add an algorithm box | 192-388 | 2 |
| Experiments, Implementation, Limitations (fixed geometry, CLIP common component, no geometry insertion) | 389-520 | 1.5 |
| Figures: merge Fig. 2 with Fig. 4, drop Fig. 3, add the view-sweep plot and the qualitative figure | figs/ | 0.75 |

**Page budget (6 pages including references) [A]:**

| Part | Pages |
|---|---|
| Intro | 0.9 |
| Related work | 0.6 |
| Method (1 figure + algorithm box) | 1.7 |
| Experiments (3 tables, 2 figures) | 2.0 |
| Conclusion and limitations | 0.2 |
| References (about 38 entries) | 0.7 |

**Compiling:** tectonic is no longer on the machine [venue], so use Overleaf or reinstall tectonic (size not checked).

**Admin:** author list, PaperCept PINs, and a note to the editor disclosing USGLoc.

## 6 Timeline

**Target: submit Sun 2026-11-15. Hard stop: Mon 2026-11-30.** The IROS 2027 RA-L transfer window closes 2027-04-30, and RA-L gives a final decision within 5 months [venue B]. ICRA 2027 is not possible.

| Week | Work | Gate |
|---|---|---|
| W1 Sep 29-Oct 4 | Day 1: submit the RIO10 form and start the Replica download with parallel range curl. Code tasks 1-4. ieeeconf port, anonymisation, bib fixes | G0: form sent; Replica on disk; tests pass |
| W2 Oct 5-11 | Code tasks 5-9. CPU extraction for Replica. Night jobs: F9 test and a peak-memory run on office_0 | G1: F1-F7 tests pass; label agreement ≥0.95; peak ≤3 GB [A]. If not, fix before fitting any maps |
| W3 Oct 12-18 | Fit 8 Replica maps (night, gated). Tab I and the view sweep on val. Tune c0, β, τ_min | G2: Ours beats NormLift and CSS variance on both val scenes (AUSE_epi on view-reducible error). If not, use fallback F-B. If there is no RIO10 link by Oct 14, switch to Bonn |
| W4 Oct 19-25 | Change dataset, val only: fit maps, build stale GT, code task 11, threshold sweep | G3: change AP > photometric residual on ≥2 of 3 val scenes. If not, use fallback F-A |
| W5 Oct 26-Nov 1 | Replica test and RIO10 test, once each. Ablations, figures, timing (GPU idle) | G4: all test logs saved. If late, cut the stretch row and the sensitivity plot |
| W6 Nov 2-8 | Write, compile, check pages | G5: full draft of at most 7 pages goes to the supervisor. If not ready, move to Nov 30 |
| W7 Nov 9-15 | Supervisor review, check every number against its log, anonymity check, submit | Submit |
| After W7 | Only if Nov 30 is missed: submit by about Dec 31 for the AIM 2027 window, which ends Jun 1, 2027 (the Dec 31 date is my own derivation) | |

## 7 Risks

| Risk | Evidence | Mitigation | Fallback |
|---|---|---|---|
| The change signal is too weak on real CLIP | AUROC 0.59-0.73 in simulations [math E9] | F3; thresholds set on val; sensitivity plot | **F-A:** static paper under the fallback title; Tab II moves to Limitations |
| The two channels do not separate | Untested | Gate G2 on val first | **F-B:** one calibrated uncertainty plus the change test. It is weak against NormLift, so consider the AIM window |
| RIO10 access or size | Form required; sizes not checked [data] | Download one scene at a time; delete zips after unpacking | Bonn RGB-D Dynamic (immediate access, 183 MB-5.8 GB per sequence, static GT cloud): ghost-Gaussian rate |
| Shared GPU | USGLoc used 13.7 of 16.3 GB; user `sang` holds 3.85 GB at all times [data, venue D] | CPU extraction; gated night jobs; `set_per_process_memory_fraction` | Shorter sweeps; N=4 test scenes |
| Novelty | C3 scooped; NormLift concurrent; LatentBKI [sota] | Narrow the claims; state the equivalences openly | none |
| USGLoc overlap and dual submission | [venue C] | No localization, no 7-Scenes; third-person citation; disclosure to the editor | none |
| Disk | 48 GB free, 95% full | fp16 PCA caches | none |
| Replica Semantic-NeRF link or size unknown | [data] | Start the download in W1 | Feature 3DGS Replica split (LSeg-derived labels) |

## 8 Decisions for the user

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Change dataset | (a) RIO10 (b) Bonn (c) own capture, 3-10 person-days (d) none | (a), with (b) as the fallback |
| D2 | Scope | (a) cut localization and EFE (b) keep one refinement row, with and without consolidation (c) keep everything | (a) |
| D3 | GPU priority (USGLoc also targets IROS 2027) | (a) UAGS runs at night only until USGLoc submits (b) a shared `flock` lock (c) fixed time slots | (a) + (b) |
| D4 | Disk space | (a) RIO10 subset of 7 scenes (b) external disk (c) prune usgloc/maps with its owner's consent | (a); (b) if the sizes turn out large |
| D5 | Length and date | (a) 6 pages, Nov 15 (b) allow 7 pages ($175), Nov 15 (c) Nov 30 | (b), with Nov 30 as the hard stop |