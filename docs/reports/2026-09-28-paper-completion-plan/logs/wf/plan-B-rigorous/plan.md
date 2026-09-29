# UAGS plan B (rigorous): placeholders to a reviewer-proof RA-L submission

Tags cite the five reports. "Assumption" = not backed by them. GPU-hours are estimates, not measured.

## 1 Scope decision

| Claim (keep, narrowed) | Cut or demote | Why |
|---|---|---|
| C1: a directional (vMF) conjugate posterior per Gaussian, with evidence counted in views | "continuous generalization of categorical mapping" becomes "shares the sufficient statistic and MAP label" | [math E14]. LatentBKI already did open-dictionary BKI on voxels [sota] |
| C2: two channels (not yet seen vs ambiguous/changed), with a calibrated per-pixel predictive; tested by cross-AUSE and ECE | Old shrinkage r̃ (a no-op at r̄=1) | [math E7]. Overlap with CSS, NormLift, PhysGS [sota], so the delta must be measured |
| C3: consolidation driven by a temporal change test, tested on real rearrangements | Gradient trick becomes an implementation detail that cites Joseph et al. | Scooped [sota C3] |
| | tab:loc and the ATE column | Collides with USGLoc; refine_pose needs an init, so the comparison is unfair [venue C] |
| | EFE section | No experiment, not a true EFE, page budget [math E13, §4] |
| | "without heuristics" | Becomes "class-agnostic thresholds, set once on a validation split" [math §3] |

Title direction (suggestion): "Uncertainty-Aware Open-Vocabulary Gaussian Splatting for Mapping in Changing Scenes". RIO10 is rearrangement, not moving people. The abstract headline gives one number for calibration (ECE, Replica test) and one for stale-content detection (AP, RIO10 test). It makes no mIoU-gain claim, because μ* equals the deterministic weighted mean [code mismatch 6].

## 2 Technical fixes before any experiment

Validate on CPU first (real MaskCLIP on 7-Scenes, as in [math C3/C3b]), then on the val splits.

| # | Fix | Why | Validation (pass test) |
|---|---|---|---|
| F1 | Count evidence in views. Per keyframe keep (R_v, m_v). Set R=Σ w_v R_v/m_v and n=Σ w_v, with w_v=min(1, m_v/m0). Drop near-duplicate keyframes. Set τ_min in views. | n_eff over-counts correlated pixels about 50x. Every footprint is "stable" after 1 view [math E8, C3-ii]. | Unit test: per-Gaussian n changes <5% between 160×120 and 640×480 (threshold is an assumption). "Stable after 1 view" fraction = 0. |
| F2 | Replace r̃ with the joint conjugate prior p(μ,κ) ∝ C_d(κ)^{c0} exp(κ m0ᵀμ), so A_d(κ̂)=‖R+m0‖/(n+c0). u_ale comes from κ, u_epi from the spread of μ. | r̃ does nothing at r̄=1 and under-corrects [E7]. κ=10 is off by 1-2 orders [E5]. This gives a real Bayesian split [E10]. | Re-run the C6 brute-force check (0.3330 vs 0.3333). Bias grid for ρ∈{0, 0.5, 0.9}, n∈{1, 2, 5, 20}, against Kish ρ̂. One observation must give 1/(1+c0). Pick c0 on val. |
| F3 | Centre the features: subtract the mean from the val scenes' mapping frames, then normalise. | Random CLIP pairs have cos ≈0.65. Changed content has r̄ 0.86-0.95 [code risk 1; E9]. | The random-pair floor drops to ≈0. Change AUROC beats raw 0.591 and centred-r̄ 0.633 [C3b]. |
| F4 | Split r_within=Σ‖R_v‖/Σm_v from r_across. Add a two-timescale change score, cos(μ_slow, μ_fast). | r̄ mixes boundary, occlusion, view noise and change, and it ignores order [E9]. | AUROC on simulated A→B change, then boundary-vs-change separation on val. |
| F5 | Consolidation: λ per keyframe, a persistent "changed" flag, no NaN reset, and a depth-residual check before re-admission. | Decay never changes r̄, and forgotten Gaussians re-enter [E12]. | Re-run the C4 demo: the forgotten region stays out. Add a unit test. |
| F6 | Per-pixel predictive: η_p=Σ_n k_{p,n} η_n and p(c)=softmax_c(β μ_pᵀt_c), with β fit on val. | tab:seg is undefined without it [E16]. | ECE/NLL code with tests. |
| F7 | Score μ*ᵀ(Bᵀt) without renormalising, or fit PCA on image and text together. | PCA-64 keeps 0.109 of text energy, and 13% of labels flip [code; E15]. | Argmax agreement with 512-d ≥0.95 on val frames (assumption). |
| F8 | Dense CLIP at 640×480, laion2b by default. | Downscale-4 agrees with full resolution on only 55-71% of pixels [code risk 3]. | Same agreement test. |
| F9 | Per-Gaussian gsplat identity check with one-hot renders (N≈300). One `rasterization` call with `channel_chunk`. | The current test checks only linearity and mass [E11]. | Max per-Gaussian error <1e-4 (assumption). One GPU test, under 5 min. |
| F10 | Text fixes: κ* is exact when κ0=0; exact variance 1−A_d(κ*); "rasterizer affine in features"; T·α kernel note; one AUSE definition. | [E2-E4, E11; code 10] | Supervisor read. |

## 3 Experiments

Rules: every method shares the same 3DGS geometry and MaskCLIP ViT-B/16 laion2b features, so only the fusion rule differs. κ/c0, m0, β, λ and thresholds are set on val; each test split runs once. Report mean ± std over scenes, with N.

| ID | Question | Data and split | Fair baselines | Metrics | Success test (val first) | GPU-h (est.) | Needs |
|---|---|---|---|---|---|---|---|
| T1 tab:seg | Calibrated, well-ranked uncertainty without an accuracy loss? | Replica Semantic-NeRF renders. Val: office_0, room_2. Test: room_0, room_1, office_1-4. Map on Sequence_1, every 5th frame (180). Evaluate on Sequence_2, every 10th frame (90) [data §4]. | Per-frame 2D MaskCLIP. Optimised deterministic field in gsplat (same iterations). Weighted mean + NormLift reliability. CSS Dirichlet, reimplemented with the same kernel, with variance and MI/entropy. | mIoU ↑, Acc ↑, ECE (15 bins) ↓, NLL ↓, AUSE_epi ↓ vs per-pixel error, AUSE_ale ↓ vs boundary error | mIoU within 1 point of the deterministic mean (assumption). ECE below the deterministic field + softmax. AUSE_epi below the scalar baseline. | Fits for 8 scenes: 4-8. Fusion: 2. | F1-F8, K6-K9 |
| T2 view-sweep figure | Does u_epi fall with more views while u_ale does not? | Replica, 10/30/90/180 views | As in T1 | 2×2 cross-AUSE ↓ | Each channel wins on its own error type | 1 | T1 |
| T3 change table (replaces tab:loc) | Does consolidation remove stale content and keep static content? | RIO10. Map from seqXX_01. Update with the first half of seqXX_02, GT poses. Val: scenes 01, 04, 06. Test: 02, 03, 05, 07, 08, 09, 10 [data §4]. Stale = >5 cm from the seq02 mesh and <2 cm from the seq01 mesh (2 cm is an assumption). | Photometric render residual (3DGS-CD style). Depth residual. Scalar feature variance. CSS variance/MI. NormLift reliability. Raw r̄. | Stale AP ↑, AUROC ↑, stale share in the robust set ↓, static wrongly forgotten ↓, mIoU on unchanged regions of held-out frames ↑; stratified by SemanticChange | AP above the best baseline on val. Keep any row where a baseline wins. | Fits for 10 scenes: 5-10. Updates: 2. | F1-F5, RIO10, K10 |
| T4 tab:abl | Which parts matter? Remove one at a time. | Replica test for (a, b, c, e, g). RIO10 test for (d, f). | Full model | As in T1/T3 | Drop every part whose removal does not hurt on val, and rename the method to what remains. | 3 | T1, T3 |
| T5 sensitivity figure | Are the thresholds robust? | Val only | none | AP, ECE vs τ_min, c0, λ | Stated range | 1 | T3 |
| T6 fig:qual | Show it | RIO10 val frame with high SemanticChange; Replica test frame | none | RGB, query, u_epi, u_ale, stale mask | none | <0.5 | T1, T3 |
| T7 runtime | Is it practical? | 1 Replica + 1 RIO10 scene, idle GPU only | Deterministic field | ms per keyframe ↓, peak GB ↓ at d=64 | Under 3 GB (model: 2.5-3 GB [code]) | <0.5 | F9 |

Ablations: (a) view vs pixel evidence; (b) joint prior vs old r̃ vs none; (c) no centring; (d) change score vs raw r̄; (e) vMF vs categorical; (f) no consolidation; (g) PCA decode vs joint PCA.

Total about 25-40 GPU-h, as in [data §5] (not measured). Command pattern: `python3 scripts/eval_seg.py --split val --scenes office_0 room_2 --out results/t1_val.json`. Before any GPU job: `nvidia-smi --query-gpu=memory.free --format=csv,noheader` ≥ peak + 4 GB, and `pgrep -f 'run_7scenes.py|exp_floor_matcher.py|train_map.py|train_encoders.py'` empty [venue D].

## 4 Code tasks (in order)

| # | Task | Files | pd |
|---|---|---|---|
| K1 | Install plyfile, untrack __pycache__, add tests for consolidation and evaluation | tests/ | 1 |
| K2 | F1, F2, F7 | usplat/vmf_field.py, bki_accumulate.py, feature_pca.py | 3 |
| K3 | F3, F4, F5; remove efe_planner from the pipeline | consolidation.py, new usplat/change.py | 3 |
| K4 | CPU replay on 7-Scenes with real CLIP (Gate G1) | scripts/cpu_check_change.py | 1 |
| K5 | F8, F9, memory cuts (drop `.clone()`), identity test | clip_dense.py, gsplat_adapter.py, tests/test_gsplat_gpu.py | 1.5 |
| K6 | Replica Semantic-NeRF loader, label map, prompt template | usplat/datasets.py | 1.5 |
| K7 | 3DGS fitting, ported from usgloc fit_gaussians with disclosure | scripts/fit_3dgs.py | 1.5 |
| K8 | Per-pixel render; mIoU/Acc/ECE/NLL/AUSE harness | usplat/evaluation.py, scripts/eval_seg.py | 3 |
| K9 | Baselines: CSS, deterministic field, NormLift, scalar variance, photometric/depth residual | usplat/baselines/ | 4 |
| K10 | RIO10 loader (540×960 portrait), stale GT from meshes, update protocol | datasets.py, scripts/eval_change.py | 3 |
| K11 | Val selection and sweeps; JSON results that include the command | scripts/select_on_val.py | 1 |
| K12 | Timing/memory instrumentation, figure scripts | scripts/ | 2 |
| | **Total** | | **25.5 (≈31 with a 20% buffer)** |

[code] estimated 30-45 pd; cutting tab:loc (11-19) and EFE (3-4) pays for the math fixes.

## 5 Writing tasks

- **Format:** port to ieeeconf `letterpaper, 10 pt, conference`; check IEEEtran-only macros (l.45, 73-76, 81); anonymise l.39-45, 522-523 [venue A, E]. No LaTeX on disk: use Overleaf or install tectonic (size not checked).
- **Rewrite:**
  - abstract and contributions (l.113-130)
  - Method l.210-369: F1-F6 equations, a 3DGS preliminaries equation that defines T, the per-pixel predictive, a one-column algorithm box
  - delete l.371-386 (EFE)
  - Experiments l.389-505
  - a Limitations paragraph: fixed geometry, CLIP common component, Tango image quality
  - an Implementation paragraph with a hyperparameter table from val
- **Related work:**
  - Add: LatentBKI, Gradient-Weighted Back-Projection, NormLift (concurrent), Mardia & El-Atoum 1976, Occam's LGS, LEGaussians, SLIM-VDB, MIF, 3DGS-CD, CL-Splats, VG-Mapping, GaussianUpdate, and USGLoc in the third person.
  - Fix l.150-152, 166, 175 and 185 [sota].
  - Add refs for MaskCLIP, AUSE, ECE and Kish (not checked).
- **Bibliography:** fix kim2025e2bki, liu2025physgs (key and authors), lee2026lamp, nagami2025vista, ye2024gsplat and the chen2024marepo title. Add pages and DOIs [sota §4]. Check every new reference by DOI or arXiv id.
- **Figures:** keep Fig. 1, merge Figs. 2 and 4, drop Fig. 3. Add fig:qual (figure*) and one 2-panel plot (view sweep + sensitivity).
- **Page budget** (6 pages including references; assumption, not measured):

| Part | Pages |
|---|---|
| Intro | 0.8 |
| Related | 0.6 |
| Method | 1.7 |
| Experiments (3 tables, 2 figures) | 2.2 |
| Limits and conclusion | 0.2 |
| References | 0.5 |

Pay for one extra page only if needed.

## 6 Timeline

Target submission is **Fri 2026-11-27**, 3 days before the Nov 30 limit for an IROS 2027 transfer: the 5-month final decision must land before the Apr 30, 2027 window closes [venue B]. The backup date is Fri 2027-01-29 (RA-L only; the Humanoids 2027 window stays open to Jun 30 [venue B]). Capacity is about 45 pd for about 31 pd of code plus 8 pd of writing (assumption: one student, 5 pd per week).

| Week | Work | Gate (Friday) |
|---|---|---|
| W1 Sep 29-Oct 4 | Submit the RIO10 form on Sep 29. HEAD-check the Replica size. K1, K2. ieeeconf port, anonymisation, bib fixes. Agree the boundary and a GPU flock with the USGLoc owners. | none |
| W2 Oct 5-11 | K3, K4, K5. Start the Replica download. | **G1 Oct 9 (math):** change AUROC beats 0.591/0.633 [C3b], and n is resolution-invariant. Fail: drop "u_ale detects change", rename u_ale "ambiguity", and consolidate on the change score only. |
| W3 Oct 12-18 | K6, K7, K8. Gated night fits of the Replica val scenes. | **G2 Oct 16 (data):** 3 RIO10 val scenes on disk. Fail: switch to Bonn box sequences (ghost rate vs the Leica cloud) or 3DGS-CD, and review the date. |
| W4 Oct 19-25 | K9. Replica val tuning and view sweep. | **G3 Oct 23 (seg val):** AUSE_epi below the scalar baseline and ECE ≤ the deterministic field. Fail: reframe as "closed-form directional lifting with calibrated confidence" and demote disentanglement to a diagnostic. |
| W5 Oct 26-Nov 1 | K10, K11. RIO10 val fits, T3 val, ablations on val. | none |
| W6 Nov 2-8 | T5. Freeze settings. Run each test split once (Replica 6 scenes, RIO10 7 scenes). | **G4 Nov 6:** on val, stale AP beats every baseline, including the photometric residual. Fail: report it, present the change score as a complement to the photometric residual (fused), and drop "robust" from the title. If the test runs have not started, move to Jan 29. |
| W7 Nov 9-15 | T4 test, T7 on an idle GPU, figures, Experiments text | none |
| W8 Nov 16-22 | Full draft, supervisor review, number-to-log check | **G5 Nov 20:** fits 6 pages; one scene re-run from the saved commands matches. |
| W9 Nov 23-27 | pdfinfo metadata check, disclosure to the editor, submit | none |

## 7 Risks

| Risk | Evidence | Mitigation | Fallback framing |
|---|---|---|---|
| Real CLIP does not separate change from static ambiguity | AUROC measured at 0.59-0.75 [C3b; code] | F3, F4; residual baselines; stratify by boundary | Two-channel semantic confidence for lifting; change handling becomes a limitation |
| RIO10 access delay; size unknown; 48 GB free | [data §5] | Form now; download per scene; delete fits after eval | Bonn/3DGS-CD, or Jan 29 |
| Shared GPU (USGLoc plus a resident 3.85 GB process) | [venue D] | CPU work first; gating; memory cap; d=64; night runs | Replica test cut to the 4 Feature-3DGS scenes, stated |
| Novelty (C3 scooped; NormLift; LatentBKI) | [sota] | Narrow the claims; NormLift and CSS as measured baselines | none |
| USGLoc overlap | [venue C] | No loc table; no USGLoc maps or numbers; disclosure | none |
| mIoU equals the deterministic mean | [code 6] | Claim no mIoU gain | none |
| Tango image quality hurts CLIP | assumption | Check at G2 | none |
| Page overflow | est. 7-7.5 pages [math §4] | Cut EFE and Fig. 3 | Pay $175 for 1 extra page |

## 8 Decisions for the user

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Date | (a) Nov 27, 2026; (b) Jan 29, 2027 | (a), switching to (b) at G4 if test runs are not done. Never submit with val-only results. |
| D2 | Localization | (a) cut; (b) one perturbed-start ablation row without SOTA baselines; (c) full RIO10 localization vs ACE/ACE-G (11-19 pd [code]) | (a); (b) only if a page is free |
| D3 | EFE | (a) cut; (b) vMF entropy gain + pool-based test (3-4 pd) | (a), mention as future work |
| D4 | Change dataset | (a) RIO10; (b) Bonn/3DGS-CD; (c) own capture (3-10 pd) | (a), submit the form Sep 29 |
| D5 | USGLoc coordination | Submission order, editor disclosure, GPU lock, disk (usgloc/maps is 55 GB) | Agree this week: shared flock, disclosure in both papers, UAGS uses its own fits |
