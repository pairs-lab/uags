# UAGS plan B (rigorous): placeholders to a reviewer-proof RA-L submission

Tags such as [code], [math E7] and [venue B] point to the five analysis reports. "Assumption" means the reports do not back it. All GPU-hours are estimates, not measured. A copy of this plan is at `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/wf/plan-B-rigorous/plan.md`.

## 1 Scope decision

| Claim (keep, narrowed) | Cut or demote | Why |
|---|---|---|
| C1: a directional (vMF) conjugate posterior per Gaussian, with evidence counted in views | "Continuous generalization of categorical mapping" becomes "shares the sufficient statistic and MAP label" | [math E14]. LatentBKI already did open-dictionary BKI on voxels [sota] |
| C2: two channels (not yet seen vs ambiguous or changed), with a calibrated per-pixel predictive; tested by cross-AUSE and ECE | The old shrinkage r̃, which does nothing at r̄=1 | [math E7]. It overlaps with CSS, NormLift and PhysGS [sota], so the gain must be measured |
| C3: consolidation driven by a temporal change test, tested on real rearrangements | The gradient trick becomes an implementation detail that cites Joseph et al. | Scooped [sota C3] |
| | tab:loc and the ATE column | Collides with USGLoc. refine_pose needs an initial pose, so the comparison is unfair [venue C] |
| | The EFE section | No experiment, not a true EFE, and costs pages [math E13, §4] |
| | "Without heuristics" | Becomes "class-agnostic thresholds, set once on a validation split" [math §3] |

**Title (suggestion):** "Uncertainty-Aware Open-Vocabulary Gaussian Splatting for Mapping in Changing Scenes". RIO10 shows objects being rearranged, not people moving.

**Abstract headline:**
- One number for calibration: ECE on the Replica test split.
- One number for finding stale content: AP on the RIO10 test split.
- No claim of an mIoU gain, because μ* equals the deterministic weighted mean [code mismatch 6].

## 2 Technical fixes before any experiment

Check each fix on the CPU first, with real MaskCLIP features on 7-Scenes as in [math C3/C3b]. Then check it on the val splits.

| # | Fix | Why | Validation (pass test) |
|---|---|---|---|
| F1 | Count evidence in views. For each keyframe keep (R_v, m_v). Set R=Σ w_v R_v/m_v and n=Σ w_v, with w_v=min(1, m_v/m0). Drop near-duplicate keyframes. Set τ_min in views. | n_eff over-counts correlated pixels about 50x. Every footprint is "stable" after one view [math E8, C3-ii]. | Unit test: per-Gaussian n changes by less than 5% between 160×120 and 640×480 (the 5% is an assumption). The fraction "stable after 1 view" is 0. |
| F2 | Replace r̃ with the joint conjugate prior p(μ,κ) ∝ C_d(κ)^{c0} exp(κ m0ᵀμ), so A_d(κ̂)=‖R+m0‖/(n+c0). u_ale comes from κ and u_epi from the spread of μ. | r̃ does nothing at r̄=1 and under-corrects [E7]. κ=10 is off by 1-2 orders of magnitude [E5]. The joint prior gives a real Bayesian split [E10]. | Re-run the C6 brute-force check (0.3330 vs 0.3333). Build a bias grid for ρ∈{0, 0.5, 0.9} and n∈{1, 2, 5, 20}, against Kish ρ̂. One observation must give 1/(1+c0). Choose c0 on val. |
| F3 | Centre the features: subtract the mean of the val scenes' mapping frames, then normalise. | Random CLIP pairs have cos ≈0.65. Changed content has r̄ 0.86-0.95 [code risk 1; E9]. | The random-pair floor drops to about 0. Change AUROC beats raw r̄ (0.591) and centred r̄ (0.633) [C3b]. |
| F4 | Split r̄ into r_within=Σ‖R_v‖/Σm_v and r_across. Add a two-timescale change score, cos(μ_slow, μ_fast). | r̄ mixes boundaries, occlusion, view noise and real change, and it ignores order [E9]. | AUROC on simulated A→B change, then boundary-vs-change separation on val. |
| F5 | Consolidation: λ per keyframe, a persistent "changed" flag, no NaN reset, and a depth-residual check before a Gaussian is re-admitted. | Decay never changes r̄, and forgotten Gaussians come back [E12]. | Re-run the C4 demo: the forgotten region stays out. Add a unit test. |
| F6 | Per-pixel predictive: η_p=Σ_n k_{p,n} η_n and p(c)=softmax_c(β μ_pᵀt_c), with β fit on val. | tab:seg cannot be filled without it [E16]. | ECE and NLL code, with tests. |
| F7 | Score μ*ᵀ(Bᵀt) without renormalising, or fit the PCA on image and text features together. | PCA-64 keeps only 0.109 of the text energy, and 13% of labels flip [code; E15]. | Argmax agrees with 512-d on at least 95% of val pixels (the 95% is an assumption). |
| F8 | Extract dense CLIP at 640×480, with laion2b as the default weights. | The downscale-4 default agrees with full resolution on only 55-71% of pixels [code risk 3]. | Same agreement test as F7. |
| F9 | Per-Gaussian gsplat identity check with one-hot renders (N≈300). Render all channels in one `rasterization` call with `channel_chunk`. | The current test checks only linearity and total mass [E11]. | Max per-Gaussian error below 1e-4 (assumption). One GPU test of under 5 min. |
| F10 | Text fixes: κ* is exact when κ0=0; use the exact variance 1−A_d(κ*); say "rasterizer affine in features"; add a note on the T·α kernel; use one AUSE definition. | [E2-E4, E11; code 10] | Supervisor read-through. |

## 3 Experiments

**Rules for every row:**
- Every method uses the same 3DGS geometry and the same MaskCLIP ViT-B/16 laion2b features, so only the fusion rule differs.
- κ/c0, m0, β, λ and all thresholds are set on val. Each test split runs once.
- Report mean ± std over scenes, with N.

| ID | Question | Data and split | Fair baselines | Metrics | Success test (val first) | GPU-h (est.) | Needs |
|---|---|---|---|---|---|---|---|
| T1 tab:seg | Is the uncertainty calibrated and well ranked, with no loss of accuracy? | Replica Semantic-NeRF renders. Val: office_0, room_2. Test: room_0, room_1, office_1-4. Map on Sequence_1, every 5th frame (180). Evaluate on Sequence_2, every 10th frame (90) [data §4]. | Per-frame 2D MaskCLIP. An optimised deterministic field in gsplat (same iterations). Weighted mean + NormLift reliability. CSS Dirichlet, reimplemented with the same kernel, with variance and MI/entropy. | mIoU ↑, Acc ↑, ECE (15 bins) ↓, NLL ↓, AUSE_epi ↓ vs per-pixel error, AUSE_ale ↓ vs boundary error | mIoU within 1 point of the deterministic mean (assumption). ECE below the deterministic field + softmax. AUSE_epi below the scalar baseline. | Fits for 8 scenes: 4-8. Fusion: 2. | F1-F8, K6-K9 |
| T2 view-sweep figure | Does u_epi fall with more views while u_ale does not? | Replica, 10/30/90/180 views | As in T1 | 2×2 cross-AUSE ↓ | Each channel wins on its own error type | 1 | T1 |
| T3 change table (replaces tab:loc) | Does consolidation remove stale content and keep static content? | RIO10. Map from seqXX_01, then update with the first half of seqXX_02 using GT poses. Val scenes: 01, 04, 06. Test scenes: 02, 03, 05, 07, 08, 09, 10 [data §4]. A Gaussian is stale if it is more than 5 cm from the seq02 mesh and less than 2 cm from the seq01 mesh (the 2 cm is an assumption). | Photometric render residual (3DGS-CD style). Depth residual. Scalar feature variance. CSS variance/MI. NormLift reliability. Raw r̄. | Stale AP ↑, AUROC ↑, stale share in the robust set ↓, static content wrongly forgotten ↓, mIoU on unchanged regions of held-out frames ↑; results split by SemanticChange | AP above the best baseline on val. Keep any row where a baseline wins. | Fits for 10 scenes: 5-10. Updates: 2. | F1-F5, RIO10, K10 |
| T4 tab:abl | Which parts matter? Remove one part at a time. | Replica test for (a, b, c, e, g). RIO10 test for (d, f). | Full model | As in T1/T3 | Drop every part whose removal does not hurt on val, and rename the method to what remains. | 3 | T1, T3 |
| T5 sensitivity figure | Are the thresholds robust? | Val only | none | AP and ECE vs τ_min, c0, λ | A stated range | 1 | T3 |
| T6 fig:qual | Show the result | A RIO10 val frame with high SemanticChange, and a Replica test frame | none | RGB, query map, u_epi, u_ale, stale mask | none | <0.5 | T1, T3 |
| T7 runtime | Is it practical? | 1 Replica scene and 1 RIO10 scene, only when the GPU is idle | Deterministic field | ms per keyframe ↓, peak GB ↓ at d=64 | Under 3 GB (the [code] memory model predicts 2.5-3 GB) | <0.5 | F9 |

**Ablations in T4:**
- (a) view vs pixel evidence
- (b) joint prior vs old r̃ vs none
- (c) no centring
- (d) change score vs raw r̄
- (e) vMF vs categorical
- (f) no consolidation
- (g) PCA decode vs joint PCA

**Compute:** about 25-40 GPU-h in total, as in [data §5] (not measured).

**Command pattern:** `python3 scripts/eval_seg.py --split val --scenes office_0 room_2 --out results/t1_val.json`

**GPU gate before any job** [venue D]:
- `nvidia-smi --query-gpu=memory.free --format=csv,noheader` must show at least the job's peak + 4 GB.
- `pgrep -f 'run_7scenes.py|exp_floor_matcher.py|train_map.py|train_encoders.py'` must return nothing.

## 4 Code tasks (in order)

| # | Task | Files | pd |
|---|---|---|---|
| K1 | Install plyfile, untrack `__pycache__`, add tests for consolidation and evaluation | tests/ | 1 |
| K2 | F1, F2, F7 | usplat/vmf_field.py, bki_accumulate.py, feature_pca.py | 3 |
| K3 | F3, F4, F5; take efe_planner out of the pipeline | consolidation.py, new usplat/change.py | 3 |
| K4 | CPU replay on 7-Scenes with real CLIP (needed for Gate G1) | scripts/cpu_check_change.py | 1 |
| K5 | F8, F9, memory cuts (drop `.clone()`), identity test | clip_dense.py, gsplat_adapter.py, tests/test_gsplat_gpu.py | 1.5 |
| K6 | Replica Semantic-NeRF loader, label map, prompt template | usplat/datasets.py | 1.5 |
| K7 | 3DGS fitting, ported from usgloc fit_gaussians and disclosed in the paper | scripts/fit_3dgs.py | 1.5 |
| K8 | Per-pixel render; mIoU/Acc/ECE/NLL/AUSE harness | usplat/evaluation.py, scripts/eval_seg.py | 3 |
| K9 | Baselines: CSS, deterministic field, NormLift, scalar variance, photometric and depth residual | usplat/baselines/ | 4 |
| K10 | RIO10 loader (540×960 portrait), stale ground truth from meshes, update protocol | datasets.py, scripts/eval_change.py | 3 |
| K11 | Val selection and sweeps; JSON results that record the command | scripts/select_on_val.py | 1 |
| K12 | Timing and memory instrumentation, figure scripts | scripts/ | 2 |
| | **Total** | | **25.5 (about 31 with a 20% buffer)** |

The [code] report estimated 30-45 pd. Cutting tab:loc (11-19 pd) and EFE (3-4 pd) pays for the math fixes.

## 5 Writing tasks

**Format and anonymity** [venue A, E]:
- Port to ieeeconf with `letterpaper, 10 pt, conference`.
- Check the IEEEtran-only macros at l.45, 73-76 and 81.
- Anonymise l.39-45 and l.522-523.
- There is no LaTeX on the machine. Use Overleaf, or install tectonic (download size not checked).

**Rewrite:**
- The abstract and the contributions (l.113-130).
- The Method, l.210-369: the F1-F6 equations, a 3DGS preliminaries equation that defines T, the per-pixel predictive, and a one-column algorithm box.
- Delete l.371-386 (EFE).
- The Experiments, l.389-505.
- Add a Limitations paragraph: fixed geometry, the shared CLIP component, Tango image quality.
- Add an Implementation paragraph with a hyperparameter table from val.

**Related work** [sota]:
- Add LatentBKI, Gradient-Weighted Back-Projection, NormLift (concurrent), Mardia & El-Atoum 1976, Occam's LGS, LEGaussians, SLIM-VDB, MIF, 3DGS-CD, CL-Splats, VG-Mapping and GaussianUpdate. Cite USGLoc in the third person.
- Fix l.150-152, 166, 175 and 185.
- Add references for MaskCLIP, AUSE, ECE and Kish (not checked).

**Bibliography** [sota §4]:
- Fix kim2025e2bki, liu2025physgs (key and authors), lee2026lamp, nagami2025vista, ye2024gsplat and the chen2024marepo title.
- Add pages and DOIs.
- Check every new reference by DOI or arXiv id.

**Figures:**
- Keep Fig. 1, merge Figs. 2 and 4, drop Fig. 3.
- Add fig:qual as a figure* and one 2-panel plot (view sweep + sensitivity).

**Page budget:** 6 pages including references (my estimate, not measured). Pay for one extra page only if needed.

| Part | Pages |
|---|---|
| Introduction | 0.8 |
| Related work | 0.6 |
| Method | 1.7 |
| Experiments (3 tables, 2 figures) | 2.2 |
| Limitations and conclusion | 0.2 |
| References | 0.5 |

## 6 Timeline

**Target: submit on Fri 2026-11-27.** That is 3 days before the Nov 30 limit for an IROS 2027 transfer. The 5-month final decision has to land before the transfer window closes on Apr 30, 2027 [venue B].

**Backup: Fri 2027-01-29.** This is RA-L only, with no IROS 2027 transfer. The Humanoids 2027 transfer window stays open until Jun 30 [venue B].

**Capacity:** about 45 pd, for about 31 pd of code plus 8 pd of writing (assumption: one student at 5 pd per week).

| Week | Work | Gate (Friday) |
|---|---|---|
| W1 Sep 29-Oct 4 | Submit the RIO10 form on Sep 29. Check the Replica download size with a HEAD request. K1, K2. ieeeconf port, anonymisation, bib fixes. Agree the paper boundary and a shared GPU lock with the USGLoc owners. | none |
| W2 Oct 5-11 | K3, K4, K5. Start the Replica download. | **G1, Oct 9 (math):** the change AUROC beats 0.591 and 0.633 [C3b], and n does not depend on resolution. If it fails: drop "u_ale detects change", rename u_ale "ambiguity", and consolidate on the change score only. |
| W3 Oct 12-18 | K6, K7, K8. Gated night-time fits of the Replica val scenes. | **G2, Oct 16 (data):** 3 RIO10 val scenes are on disk. If not: switch to the Bonn box sequences (ghost rate against the Leica cloud) or 3DGS-CD, and review the date. |
| W4 Oct 19-25 | K9. Replica val tuning and view sweep. | **G3, Oct 23 (seg val):** AUSE_epi is below the scalar baseline and ECE is at most the deterministic field's. If AUSE_epi fails: reframe as "closed-form directional lifting with calibrated confidence" and demote disentanglement to a diagnostic. |
| W5 Oct 26-Nov 1 | K10, K11. RIO10 val fits, T3 on val, ablations on val. | none |
| W6 Nov 2-8 | T5. Freeze all settings. Run each test split once (Replica 6 scenes, RIO10 7 scenes). | **G4, Nov 6:** on val, stale AP beats every baseline, including the photometric residual. If it fails: report it, present the change score as a complement fused with the photometric residual, and drop "robust" from the title. If the test runs have not started by then, move to Jan 29. |
| W7 Nov 9-15 | T4 on test, T7 on an idle GPU, figures, Experiments text | none |
| W8 Nov 16-22 | Full draft, supervisor review, check that every number links to a log | **G5, Nov 20:** the paper fits 6 pages, and one scene re-run from the saved commands gives the same numbers. |
| W9 Nov 23-27 | Check PDF metadata with pdfinfo, disclose USGLoc to the editor, submit | none |

## 7 Risks

| Risk | Evidence | Mitigation | Fallback framing |
|---|---|---|---|
| Real CLIP features do not separate change from static ambiguity | Measured AUROC 0.59-0.75 [C3b; code] | F3, F4; residual baselines; split results by boundary | "Two-channel semantic confidence for lifting", with change handling as a limitation |
| RIO10 access is delayed; its size is unknown; only 48 GB of disk is free | [data §5] | Submit the form now; download one scene at a time; delete fits after evaluation | Bonn or 3DGS-CD data, or the Jan 29 date |
| The GPU is shared with USGLoc and a resident 3.85 GB process | [venue D] | CPU work first; gating; memory cap; d=64; night runs | Cut the Replica test to the 4 Feature-3DGS scenes and say so |
| Novelty: C3 is scooped; NormLift and LatentBKI overlap | [sota] | Narrow the claims; run NormLift and CSS as measured baselines | none |
| Overlap with USGLoc | [venue C] | No localization table; no USGLoc maps or numbers; disclosure to the editor | none |
| mIoU equals the deterministic mean | [code 6] | Claim no mIoU gain | none |
| Tango image quality hurts CLIP | assumption | Check at G2 | none |
| Page overflow | Estimated 7-7.5 pages [math §4] | Cut EFE and Fig. 3 | Pay $175 for one extra page |

## 8 Decisions for the user

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Submission date | (a) Nov 27, 2026; (b) Jan 29, 2027 | (a), switching to (b) at G4 if the test runs are not done. Never submit with val-only results. |
| D2 | Localization | (a) cut it; (b) one ablation row starting from a perturbed pose, with no state-of-the-art baselines; (c) full RIO10 localization against ACE and ACE-G (11-19 pd [code]) | (a); add (b) only if a page is free |
| D3 | EFE | (a) cut it; (b) keep it with a vMF entropy gain and a pool-based test (3-4 pd) | (a), and mention it as future work |
| D4 | Change dataset | (a) RIO10; (b) Bonn or 3DGS-CD; (c) our own capture (3-10 pd) | (a); submit the form on Sep 29 |
| D5 | Coordination with USGLoc | Which paper goes first, editor disclosure, a shared GPU lock, disk space (usgloc/maps is 55 GB) | Agree this week: a shared GPU lock, disclosure in both papers, and UAGS uses its own 3DGS fits |