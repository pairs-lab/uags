# UAGS plan C: reuse and differentiate (2026-09-28)

**Target date.** Submit to RA-L on **Sun 2026-11-15**. The hard limit is **Mon 2026-11-30**. The IROS 2027 transfer window closes Apr 30, 2027, and RA-L promises a final decision within 5 months, so submission must be by about Nov 30 (venue B). ICRA 2027 cannot be reached: it needs acceptance by Dec 31, 2026.

**Evidence.** Everything comes from the five reports: code, math, sota, data and venue. **[A]** marks my own assumption, with no report behind it.

**GPU numbers.** Every GPU-hour figure is [A]. No UAGS run on a real scene has been timed.

## 1 Scope decision

**Boundary with USGLoc.** UAGS covers the map side: what the map believes about language semantics, and how that belief ages when the scene changes. USGLoc covers the query-side pose belief (venue C).

| Kept claim | Evidence |
|---|---|
| C1' A vMF (directional) conjugate posterior per Gaussian over CLIP features. It "shares the sufficient statistic and MAP label of categorical BKI". | Tab I, ablation (i) |
| C2' Evidence counted in views. Epistemic uncertainty comes from the view count. Aleatoric uncertainty comes from a bias-corrected resultant, split into within-view and across-view parts. | Tab I, Fig views, ablations (ii), (iv), (v) |
| C4' Consolidation. Class-agnostic thresholds, set once on val, flag stale content and forget it. | Tab II (RIO10 + 7-Scenes false alarms), ablation (iii) |

**Cut:**
- **C3 gradient trick.** It is scooped (sota: Joseph et al., SIGGRAPH Asia 2025, backproject.py:115-150). Keep it as a three-line implementation detail that cites them.
- **tab:loc.** It collides with USGLoc's main result (venue C, rows 1, 3, 4). Remove "localization" from the abstract, contributions and keywords.
- **EFE section.** It has no experiment and is not a real EFE (math E13).
- **Wording to change** (math §3): "continuous generalization", "without heuristics", "principled", "any differentiable rasterizer", "irreducible", and undefined "calibrated".

**Title direction:** "Uncertainty-Aware Open-Vocabulary Semantic Gaussian Splatting for Mapping in Changing Scenes".

**Abstract direction:**
1. Problem: semantic 3DGS maps give no trust signal and keep stale content.
2. Method: a vMF posterior with a view-level epistemic/aleatoric read-out, then consolidation.
3. Headline: ECE/AUSE on Replica and stale-content AP on RIO10.
4. Positioning: against LatentBKI, CSS and NormLift (concurrent). Cite USGLoc in the third person.

## 2 Technical fixes before any experiment

| # | Fix | Why | Validation |
|---|---|---|---|
| F1 | Count evidence in views. Per keyframe, take (R_v, m_v) from the backward pass. Set n = Σ_v w_v with w_v = min(1, m_v/m0), and express τ_min in views. | n_eff is pixel mass. One 160×120 view lets 287/300 Gaussians pass n_eff > 1 (code, test_gsplat_gpu.log). This overcounts by about 50× (math E8). | Unit test: n is the same at downscale 2 and 4. After 1 view, no Gaussian passes τ_min. |
| F2 | Replace r̃ with the view-level Kish form ρ̂² = (n_K r̄² − 1)/(n_K − 1). It is undefined when n_K ≤ 1. | At r̄ = 1, r̃ = 1 for every n (math E7). | Unit test: uniform directions give ρ̂ ≈ 0 (math C2-c: 0.062 / 0.021). n_K = 1 gives "no info". |
| F3 | Centre CLIP features with a mean taken from the val scenes. Add a two-timescale change test cos(μ_slow, μ_fast). Split r_within from r_across. | On changed content, r̄ stays at 0.86–0.95, so the 0.55 threshold never fires. Measured AUROC: 0.591 raw and 0.633 centred (math C3b-v); 0.725–0.745 in the code report's check. | Stale-Gaussian AUROC on RIO10 val beats raw r̄ (gate G3). |
| F4 | Add a "changed/forgotten" flag that keeps the gate closed. Apply λ per keyframe. Put all thresholds in one config. Add a depth-residual check before a Gaussian re-enters the robust set. | After forgetting, u_epi rises from 0.015 to 0.171 (math C4-c). Thresholds are 0.5, 0.55 and 0.8 in different files (code mismatch 4). | Turn math C4 into a test: a flagged Gaussian stays out of the robust set and its u_epi does not rise. |
| F5 | Fit PCA-64 on image and text embeddings together. | Text keeps only 0.109–0.119 of its energy in the image basis, and 13–29% of labels change (math E15, code risk 2). | Argmax agreement with 512-d is ≥ 0.95 on Replica val [A threshold]. |
| F6 | Extract CLIP features at full resolution. Make `laion2b_s34b_b88k` the default. | Downscale-4 labels agree with full-res on only 0.55–0.71 of pixels. The openai weights are not cached (code 12). | mIoU on val, full-res vs ds4. |
| F7 | Define the render η_p = Σ_n k_{p,n} η_n. Define calibration as p(c) = softmax(β μ*ᵀt_c), with β fit on val. | Tab I cannot be filled without these (math E16). | ECE, NLL and Brier can be computed on val. |
| F8 | State that κ only scales κ*. Estimate κ̂ by MLE on val. | κ changes no decision (code 6, math E5). | Text check. |
| F9 | Add a per-Gaussian one-hot identity check. Use a single rasterization call with `channel_chunk` and `covars`. | The GPU test checks only linearity and total mass (math E11). | Max abs diff < 1e-5 [A] on about 300 Gaussians. |

## 3 Experiments

**Rules for every row** (data §4):
- Every method uses the same geometry and the same MaskCLIP ViT-B/16 laion2b features.
- Hyperparameters and β are chosen on val only.
- The test split is run once.
- Report mean ± std with N.

| ID | Question | Data and split | Fair baselines | Metrics | Success test | GPU-h [A] | Depends on |
|---|---|---|---|---|---|---|---|
| E1 Tab I | At equal mIoU, is the uncertainty better ranked and better calibrated? | Replica Semantic-NeRF. Val: office_0, room_2. Test: room_0, room_1, office_1–4. Map on Seq_1 every 5th frame (180); evaluate on Seq_2 every 10th frame (90). | Per-frame 2D MaskCLIP; averaged field (Joseph / Occam mean, with feature variance as uncertainty); Feature-3DGS-style optimised field in gsplat; CSS Dirichlet, reimplemented | mIoU↑, Acc↑, ECE (15 bins)↓, NLL↓, AUSE_epi↓, AUSE_ale↓ | Ours has lower ECE and AUSE_epi than the averaged field and CSS on ≥ 4/6 test scenes [A]. mIoU equals the averaged field by construction, and the paper says so. | 4 fit + 2 eval + 4 optimised field | F1–F7, tasks 4–9 |
| E2 Fig views | Does u_epi fall as views are added, while u_ale does not? | Replica test, mapped with 10 / 30 / 90 / 180 views | Scalar variance | 2×2 cross-AUSE: each channel against coverage error and against boundary error | u_epi ranks coverage error best; u_ale ranks boundary error best | 1 | E1 maps |
| E3 Tab II | Does consolidation find and forget stale content? | RIO10. Map from seqXX_01. Update with the first half of seqXX_02 using GT poses, every 5th frame [A]. Val scenes: 01, 04, 06. Test scenes: 02, 03, 05, 07, 08, 09, 10. A Gaussian is stale if it is > 5 cm from the seq_02 mesh and near the seq_01 mesh. | Photometric residual; scalar variance; CSS variance; no consolidation | Stale AP↑, AUROC↑; stale content left after update↓; mIoU on changed vs unchanged regions↑ (only if the class map works; not checked) | Ours beats the photometric residual and scalar variance on val (G3) | 5 fit + 2 | RIO10 access, F3, F4, task 10 |
| E4 (Tab II row) | How many false alarms in static scenes? | 7-Scenes, all 7 scenes. **Reuse usgloc/maps geometry** (one variant) with the usgloc SfM-pGT loader, not usplat's DSLAM fx=585 loader. Map on train sequences; update on test sequences. | Same as E3 | % of Gaussians flagged↓ (every flag is false) | ≤ the photometric-residual rate [A] | 0 fit + 1 | Tasks 4, 5 |
| E5 Tab III | Which parts matter? | (i) categorical vs vMF; (ii) scalar vs 2-D uncertainty; (iii) consolidation on/off (RIO10 test); (iv) views vs pixel mass; (v) Kish correction on/off at ≤ 10 views; (vi) joint vs image-only PCA | Each part removed | From E1 and E3 | Drop any part whose removal does not hurt | 3 | E1, E3 |
| E6 Fig qual | Show the result | One RIO10 val frame with high SemanticChange, plus one Replica frame | none | RGB, query map, u_epi, u_ale | none | < 0.5 | E1, E3 |
| E7 Runtime | What does it cost? | One Replica and one RIO10 scene, measured only while the GPU is idle | none | ms/frame, peak GB, N, d | ≤ 3 GB at d=64 (code estimate: 2.5–3 GB) | 0.5 | Task 11 |

- **Total:** about 23 GPU-h [A]. The data report estimates 25–40 GPU-h for a larger scope; that estimate is not measured either.
- **Optional E8 (D1):** refine_pose from a perturbed start, with vs without the robust subset, on RIO10 test, with no state-of-the-art rows. About 1 GPU-h.

## 4 Code tasks (do them in this order)

| # | Task | What it reuses | pd |
|---|---|---|---|
| 1 | Install plyfile (`pip install --user --break-system-packages plyfile`). Untrack `__pycache__`. Write a gate wrapper `scripts/gpu_gate.sh` (details below). | venue D | 0.5 |
| 2 | F1, F2, F4, F8 plus tests | math C2 and C4 scripts, turned into tests | 3 |
| 3 | F3, F5, F6 | none | 1.5 |
| 4 | Loaders: 7-Scenes SfM pGT (port); Replica Semantic-NeRF; RIO10 (portrait about 540×960, following ACE-G `setup_rio10.py`) | usgloc/datasets.py:127 | 3 |
| 5 | Adapter for usgloc .pt maps: covs → gsplat `covars` | rendering.py:148-173 | 0.5 |
| 6 | Fit 3DGS on Replica and RIO10. Import usgloc/mapping.py (:74, :164) via PYTHONPATH and put a new dataset class in uags. Do not edit usgloc. | train_map.py | 1.5 |
| 7 | F7 renders through GsplatRenderer extra channels; F9 | rendering.py:135 | 1.5 |
| 8 | Eval harness: prompts, ignore labels, mIoU/Acc, ECE/NLL/Brier, one AUSE definition shared with USGLoc | usgloc/metrics.py:79, copied with a citation | 2.5 |
| 9 | Baselines: averaged field (0.5), CSS adapter (1.5), optimised field (1.5) | none | 3.5 |
| 10 | RIO10 change protocol: stale labels, GT-pose update, SemanticChange strata, photometric-residual baseline | none | 3.5 |
| 11 | Timing and memory instrumentation; dump of the hyperparameter table | none | 0.5 |
| 12 | Optional E8 | localization.py:60 | 1 |

The gate wrapper (task 1) starts a job only when all of these hold:
- Free memory is at least the UAGS peak + 4 GB, or + 7 GB while a MASt3R batch is queued.
- `pgrep -f 'run_7scenes|exp_floor_matcher|train_map|train_encoders'` finds nothing.
- `set_per_process_memory_fraction` is set.

**Total:** 21.5 pd (+1 optional). The full original scope was 30–45 pd (code report).

**Not reused, on purpose:**
- The ACE and Marepo code and runs. They live in another session's /tmp, and a localization table would collide with USGLoc.
- USGLoc's distilled field, its EDL head, its 25-class vocabulary and any of its numbers (venue C).

## 5 Writing tasks (about 8 pd)

| Task | pd |
|---|---|
| Port to `ieeeconf` (letterpaper, 10pt, conference). Check `\IEEEPARstart`, `IEEEkeywords` and `\markboth`. Build on Overleaf, because tectonic is missing [A]. | 0.5 |
| Anonymise: lines 39–45 and 522–523; no GitHub link; cite USGLoc in the third person; run `pdfinfo` on the final PDF. | 0.25 |
| Title, abstract, contributions (C1', C2', C4'), keywords without "localization". | 1 |
| Method: preliminaries equation that defines T; view-level evidence; Kish ρ̂; render equation; calibration definition; consolidation flag; algorithm box; the gradient trick as a cited detail; the wording list in math §3. | 2 |
| Related work: add LatentBKI, Joseph et al., NormLift, Mardia & El-Atoum, Occam's LGS, LEGaussians, SLIM-VDB, MIF, 3DGS-CD, CL-Splats and VG-Mapping. Fix lines 150-152, 166, 175 and 185. | 1 |
| Bib: fix the 7 mismatches listed in sota §4 and check each DOI. | 0.5 |
| Experiments, limitations (fixed κ, CLIP common component, static geometry, pixel correlation), and a note that the GPU was shared. | 2 |
| Figures: drop Fig 3, merge Fig 2 with Fig 4, add Fig qual and the views plot. | 1 |

**Page budget: 6 pages, at most 1 paid extra** [A split]:

| Part | Pages |
|---|---|
| Intro | 0.8 |
| Related work | 0.6 |
| Method | 1.9 |
| Experiments | 2.1 |
| Limitations and conclusion | 0.2 |
| References | 0.7 |

## 6 Timeline

| Week | Work | GPU | Gate |
|---|---|---|---|
| W1 Sep 29–Oct 4 | Send the RIO10 form on day 1. Check the Replica size and download it if ≤ 20 GB [A]. Tasks 1–3. Port and anonymise the paper. Draft the method rewrite. | None | none |
| W2 Oct 5–11 | Tasks 4, 5 and 7. E4 as gated night jobs. Measure the UAGS peak memory once. | Small, gated | **G1 (Oct 11):** if RIO10 data is not in hand, switch E3 to the Bonn box sequences (ghost Gaussians against the Leica cloud; no semantic GT). |
| W3 Oct 12–18 | Task 6 fits (8 Replica + 10 RIO10). Tasks 8 and 9. | About 10 | **G1b:** E4 false-alarm rate < 5% [A]. If not, recalibrate F3/F4 before using RIO10. |
| W4 Oct 19–25 | E1 and E2 on val; fit β, thresholds and m0. | About 3 | **G2:** AUSE_epi beats scalar variance and ECE ≤ the averaged field on val. If not, reframe C2' as a view-count epistemic map and drop the calibration claim. |
| W5 Oct 26–Nov 1 | Task 10. E3 on val. Related work and bib. | About 4 | **G3:** stale AP beats the photometric residual and scalar variance on val. If not, use the fallback in §7. |
| W6 Nov 2–8 | Test runs once. E5, E6, and E7 in an idle window. Code freeze Nov 8. | About 6 | **G4:** every table cell comes from a saved log. |
| W7 Nov 9–15 | Write-up. Supervisor review on Nov 12. Anonymity check. Note to the editor about USGLoc. **Submit Nov 15.** | None | none |
| Nov 16–30 | Buffer for a slipped gate only. | none | Hard stop Nov 30 |

## 7 Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| GPU contention. USGLoc runs until at least mid-October (venue estimate). `sang`'s 3.85 GB process is always resident. Free memory has dropped to 1761 MiB. | venue D | Gate wrapper, d=64, shared lock (D5), CPU work first |
| RIO10 access delay; its size is not checked. | data §5 | Send the form today; download per scene; Bonn as fallback |
| Disk: 48 GB free; usgloc/maps takes 55 GB. | data §3 | Download one scene at a time; delete fields (about 0.5 GB/scene at d=64); ask before pruning maps (D4) |
| The aleatoric signal does not separate change from static ambiguity (AUROC 0.59–0.75). | math E9, code risk 1 | F3 and F4; gate G3 |
| Novelty against NormLift, CSS and LatentBKI. | sota §2 | Narrowed claims; cite NormLift as concurrent |
| Overlap with USGLoc is noticed. | venue C | The §1 boundary; tell the editor; share no numbers or figures |
| Our mIoU equals the averaged field. | code 6 | Claim uncertainty quality at equal mIoU, and say so |

**Fallback framing if G3 fails.** "Bayesian directional fusion of open-vocabulary features in 3DGS with view-level epistemic/aleatoric uncertainty":
- Keep E1, E2, E4 and ablations (i), (ii), (iv), (v).
- Report the RIO10 change results as a limitation.
- Drop "changing scenes" from the title.

If G2 also fails, do not submit on Nov 15. The later options are Nov 30 (still in the IROS window) or CASE, RO-MAN or AIM (venue B).

## 8 Decisions for you

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Localization in UAGS | (a) none; (b) one downstream ablation row (E8) with no SOTA rows; (c) a RIO10 table vs ACE and ACE-G | **(b).** Option (c) needs an initializer that works under scene change: ACE gets 358 cm on RIO10 (ACE-G Tab. 3). |
| D2 | EFE section | (a) cut it; (b) derive the vMF entropy gain and run a pool-based experiment (3–4 pd) | **(a) cut**, and name it as future work |
| D3 | Change dataset | (a) RIO10; (b) Bonn; (c) our own capture (3–10 pd) | **RIO10 now, Bonn as fallback** |
| D4 | Reuse of USGLoc assets | (a) reuse the 7-Scenes map geometry and the train_map code, with disclosure; (b) keep UAGS fully independent | **(a), geometry only.** Ask the USGLoc owner which map variant is final, whether the others can be moved off the disk, and which paper goes to RA-L first. |
| D5 | GPU sharing | (a) a shared `flock` lock agreed with the USGLoc session; (b) wait until USGLoc submits | **(a) for small gated jobs now.** Run the main jobs after USGLoc submits. |