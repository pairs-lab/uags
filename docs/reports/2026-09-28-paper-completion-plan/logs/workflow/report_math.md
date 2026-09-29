# UAGS math and claims audit of `papers/main.tex`

**Scope.** I only read files in `uags`, `usgloc` and the 7-Scenes data. Scripts and outputs are in `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/wf/math/`. Tags like [C3-iii] point to one output file there; the commands are listed at the end. I ran no GPU test: `nvidia-smi` showed 14690/16303 MiB in use at 98 % load, held by the other session.

**Summary.** The conjugate update and the gradient identity are correct. Five things are seriously wrong:
- The small-sample correction does not do what the text says.
- n_eff counts correlated pixels as independent evidence.
- On real CLIP features the aleatoric channel barely separates change from static content. With the default thresholds, the "unstable" rule never fires.
- Forgetting interacts badly with the EFE gate.
- "Calibrated" is never defined.

## 1. Equation audit

| # | Item (main.tex line) | Verdict | Finding and evidence | Fix |
|---|---|---|---|---|
| E1 | vMF likelihood and vMF conjugate prior (221–230) | Correct | Standard result when κ is known. | Say "κ known". Cite Mardia & Jupp for the conjugacy. |
| E2 | Tempered posterior and closed form (232–252) | Correct, one minor issue | The numerical posterior matches vMF(η/‖η‖, ‖η‖) with max relative error 3.11e-09 in d=3 [C1-1]. CSS Eq. 7 requires k(x,x)=1 for a BKI kernel. T·α breaks this when α<1, even though CSS Eq. 3 uses the same T·α and claims it is valid [css_grep.txt]. | Write "a [0,1]-valued kernel, as in CSS". Also state the good property Σ_n k_{p,n} ≤ 1: one pixel carries at most unit mass. |
| E3 | κ* ≈ κ·n_eff·r̄ (316–318) | Correct, minor | It is exact when κ0=0 (40.667750 = 40.667750). Otherwise \|κ* − κ n r̄\| ≤ κ0 [C1-2]. "Two independent routes" (319) is wrong: r̄ depends on n, since E[r̄] ≈ 1/√n under pure conflict [C2-b]. | Write "=" when κ0=0 and give the bound otherwise. Drop "independent". |
| E4 | Posterior angular variance "closed form ≈ (d−1)/κ*" (255–256) | Minor, major if used | This is the mean squared angle, and only when κ* ≫ d. Error ratios: 21.21× at d=512, κ*=10; 4.70× at κ*=50; 1.10× at κ*=1000; 3.12× at d=64, κ*=10 [C1-3]. "Closed form ≈" contradicts itself. | Use the exact 1 − A_d(κ*) = E[1 − cos], or state the κ* ≫ d regime. It is unused anyway (`vmf_field.py:67-68` has no caller). |
| E5 | Fixed κ (219–220; code default κ=10, `run_mapping.py:198`) | Serious for any "posterior" or "calibrated" claim | κ=10 implies E[cos(z,μ)] = A_d(10) = 0.153 at d=64 and 0.0195 at d=512 [C1-4]. Real MaskCLIP static points give ρ̂ ≈ 0.85 across views [C3-iii]. Matching even A_d=0.9 needs κ ≈ 299 (d=64) or 2421 (d=512) [C1-4]. So the likelihood is off by 1–2 orders of magnitude. But κ changes none of μ*, r̄, u_epi, u_ale, consolidation or EFE: only κ* uses it, and κ* is never called. The text prior κ0 is also unused (grep of `kappa_post`, `angular_variance`, `prior_kappa`). | Either say that κ only scales κ*, or estimate κ by MLE on a validation split (A_d(κ̂) = pooled r̄). Do it in the space actually used: PCA-64 changes r̄ from 0.869 to 0.903 (medians) [C3-iv]. |
| E6 | Epistemic proxy 1/(1+n_eff) (322–325) | Minor for ranking, major for "calibrated" | AUSE is rank-based, so any monotone proxy gives the same AUSE. But n_eff is not scale-invariant: in [C2-c], Σw = 2.6 while Kish n = 7.1. It depends on resolution, on-screen area and opacity. The posterior's own spread depends on r̄ too (κ* = κ n r̄). | Take it from the posterior (1 − A_d(κ̂ n r̄)), or count views (E8). Show calibration against held-out error. |
| E7 | Shrinkage r̃ = r̄ − (1−r̄)/(n+1) (330–337; `vmf_field.py:74-76`) | **Serious flaw** | 1 − r̃ = (1 − r̄)(n+2)/(n+1), so r̄=1 gives r̃=1 for every n [C2-a]. The sentence "The correction is essential: with a single observation r̄=1…" is false: only the τ_min gate helps. It also under-corrects. Pure conflict (truth 0), d=64, n=5: r̄ 0.445, r̃ 0.353, unbiased ρ̂ 0.079. At ρ=0.9, n=2: r̃ 0.935 vs ρ̂ 0.900 [C2-b]. | Replace it with the standard bias correction in Kish form: ρ̂² = (n_K r̄² − 1)/(n_K − 1), with n_K = (Σk)²/Σk². It is undefined at n_K ≤ 1, which correctly says "no information". Its weighted form gives 0.062 / 0.021 for truth 0 [C2-c]. Or use prior shrinkage r = ‖R+m0‖/(n+c0) (E10). |
| E8 | n_eff = Σ over pixels of k (252) | **Serious flaw** | Real MaskCLIP on 7-Scenes: pixel cosine is 0.999 at 1 px offset and 0.920 at 8 px (ds=4, which has only 8×10 tokens) [C3-i]. A 197-px footprint covers 3.7 distinct 16-px tokens [C3-ii]: about 50× overcounting, and it grows with resolution. In one view, every footprint has r̄ ≥ 0.86 and n ≥ 13 > τ_min=5, so all are "stable" after one frame (fraction 1.00) [C3-ii]. Epistemic becomes a measure of screen area. Kish fixes unequal weights, not correlation. | Count views. Per keyframe, the backward pass already gives (R_v, m_v). Aggregate R = Σ_v w_v R_v/m_v and n = Σ_v w_v with w_v = min(1, m_v/m0). Remove near-duplicate keyframes. Set τ_min in views. View-level Kish from Σ w_v² is free. |
| E9 | Aleatoric = 1 − r̃ as "irreducible" (326–333) | **Serious** | r̄ mixes five causes: within-view spread (boundaries, Gaussians spanning two objects, 16-px token blur), occlusion (Gaussians behind a surface get the occluder's z with weight T·α), view-dependent CLIP noise, true change, and a large shared CLIP component. Random pixel pairs already have cos 0.654–0.679 [C3-i]. r̄ ignores order, so "A then B" and "A/B interleaved" look the same. Simulated A→B change on real features [C3b-v]: static r̄ 0.827/0.867/0.909 (10/50/90 %) vs changed 0.813/0.857/0.892. AUROC is 0.591 raw and 0.633 centred. 0 % of changed points reach r_unstable=0.55, and 61 % stay "stable". Caveat: "B" is another random static point from the same scene, not a real rearrangement. | (1) Split the dispersion: r_within = Σ‖R_v‖/Σm_v and r_across = ‖Σ w_v μ_v‖/Σ w_v. (2) Add a temporal change test, e.g. cos(μ_slow, μ_fast) from two exponentially weighted resultants. (3) Centre the features (subtract the dataset mean) or set thresholds relative to the random-pair floor. (4) Report AUSE_ale separately on boundary, occlusion and moved-object masks. |
| E10 | Banerjee κ̂ and "at the cost of closed-form conjugacy" (341–344) | Minor | The formula is correct. It is biased at small n: 5 uniform directions give κ̂ = 35.4 (d=64) and 285.9 (d=512) [C7]. The conjugacy claim is inaccurate. The joint prior p(μ,κ) ∝ C_d(κ)^{c0} exp(κ m0ᵀμ) keeps (R, n) sufficient, and its MAP gives A_d(κ̂) = ‖R+m0‖/(n+c0). Checked by brute force: 0.3330 vs 0.3333; 0.7500 vs 0.7500; 0.8823 vs 0.8824 [C6]. Literature refs for this prior: not checked. | Use the joint prior. It gives a real Bayesian split: the posterior over κ is the aleatoric part, and κ̂‖R‖ is the epistemic spread of μ. It also fixes E7: one observation gives 1/(1+c0). |
| E11 | Gradient identity (283–297) | Correct, with conditions | gsplat 1.5.3 with `sh_degree=None` uses a linear gather (`gsplat/rendering.py:471-487`). A background term is affine and does not change the gradient. Packed mode is a gather with scatter-add backward. Antialiased mode rescales opacity (`:452-453`), which is still linear. The kernel is truncated consistently (α<1/255 skipped, T ≤ 1e-4 stops; `RasterizeToPixels3DGSFwd.cu:149-154`). The identity breaks with SH colours (`clamp_min(colors+0.5,0)`, `rendering.py:525`), with alpha-normalised renders (kernel becomes k/A_p), and with the Feature-3DGS decoder. "Verified against an explicit weighted-sum reference" (296–297) holds only for the mock, which is literally `W @ f` (`mock_rasterizer.py:65`), so it is true by construction. The GPU test checks only linearity and total mass (`tests/test_gsplat_gpu.py:53-75`), not per-Gaussian values. "One render" is really ⌈(d+1)/32⌉ chunked renders (`gsplat_adapter.py:62-69`). | Add a per-Gaussian gsplat check with one-hot renders (N≈300 renders, cheap). Replace "any differentiable feature rasterizer" with "any rasterizer affine in per-Gaussian features". State the chunking. |
| E12 | Consolidation decay (360–369; `consolidation.py:30-41`) | **Serious** | (a) λ scales R and n alike, so r̄ does not change (0.6461 → 0.6461) [C4-a]. Forgetting acts only through later updates. (b) With no new data it decays to n=4.86 < τ_min and freezes. u_ale becomes undefined, the EFE discount resets to 1 (`efe_planner.py:40-42`), epistemic value rises 0.015 → 0.171, and the planner picks the dynamic region [C4-c; demo used r_unstable=0.8]. This contradicts lines 384–386. (c) Steady state is n* = λm/(1−λ) = 1800 in the example [C4-d]; λ is applied per call, every 10 frames (`run_mapping.py:152-153`), not per unit of time. (d) For an equal A/B mix, r̄ = ρ√((1+c)/2). Reaching 0.55 needs c ≤ −0.395 (ρ=1) or c ≤ −0.201 (ρ=0.87) [C7]. CLIP pairs sit near +0.65, so the rule never fires on raw features [C3b-v]. In the d=64 simulation r̄ bottoms at 0.616: no decay, n grows to 4000, μ drifts slowly to B [C4-b]. (e) By reasoning, not measured: after forgetting, moved-object Gaussians gather consistent "wall" evidence and can re-enter the robust set with stale geometry. | Define λ per second or per keyframe, and say what happens to r̄. Keep a "forgotten/changed" flag so the EFE gate stays closed. Use a two-timescale change test. Add a geometric check (depth or photometric residual) before re-admitting a Gaussian to the robust set. Set thresholds on a validation split and show sensitivity. |
| E13 | EFE (373–386) | Minor math, major wording | The sign is right. But it is not expected free energy. u_epi(1−u_ale) is not an information gain, s_n is a cosine and not a log-preference, and λ_p is hand-set, which is exactly what line 185 criticises in VISTA. The paper's own vMF gives a closed-form gain H(κ*_old) − H(κ*_new). For d=64, κ=10, m=5: n=5 gives 12.60 vs 8.89 nats (r̄ 0.95 vs 0.6); n=50 gives 2.82 vs 2.71 [C5-b]. The model's r̄ effect fades with n, while the paper's gate keeps a ratio of about 1.6–1.8. | Use the vMF entropy reduction and call the rest "EFE-inspired". Or cut the section: it is optional and has no experiment. |
| E14 | "Continuous generalization of categorical mapping" (61–62, 117–119, 257–259, 512–513) | Minor–major wording | With one-hot z, R equals the Dirichlet count increment: same statistic, same argmax. The model does not reduce to the Dirichlet, though. Counts (5,5,0) give Dirichlet (0.462, 0.462, 0.077) but vMF μ* = (0.707, 0.707, 0) with κ* = 70.7, rising to 707 at (50,50,0) [C5-a]. The vMF becomes confident in a direction that is neither label. | Write "shares the sufficient statistic and the MAP label of the categorical update". |
| E15 | Queries are "cosine similarity with a text embedding" (262–263); code queries in PCA space (`run_mapping.py:167`, `feature_pca.py:41-56`) | Major, and the paper does not describe it | The uncentred PCA-64 basis keeps 0.918 of image energy but only 0.119 (0.086–0.130) of text energy. Renormalising raises image–text cosine from 0.178 to 0.539 and changes the argmax label for 13.2 % of pixels over 12 prompts [C3-iv]. | Decode first: score μ*ᵀ(Bᵀt) without renormalising Bᵀt, i.e. cosine in the full space. Or fit PCA on image and text embeddings together. State d and ablate it. |
| E16 | Novel-view "per-pixel uncertainty follows CSS" (263–265) | Major gap | Not defined for the vMF: CSS renders Dirichlet α. mIoU, AUSE and ECE all depend on it. | Give the equation, e.g. η_p = Σ_n k_{p,n} η_n giving vMF(η_p), or composite μ*, u_epi and u_ale with the same weights. |

## 2. Claims that need experimental evidence

| Claim (line) | Needed evidence |
|---|---|
| Existing semantic 3DGS has no calibrated trust (53, 92–95) | Literature plus Tab. I with a fair ECE for deterministic features (softmax over cosines; the table's "--" is unfair). CSS already gives expectation and variance, so soften the claim. |
| A briefly seen region and an ambiguous or changing one get the same confidence (54–56, 95–99) | Show that a scalar uncertainty (CSS variance, feature variance) cannot rank both error types; this is ablation (ii). |
| A single posterior variance cannot split epistemic from aleatoric (105–108) | The Dirichlet already allows mutual information vs expected entropy. Add "CSS + MI/entropy split" as a baseline, or drop the claim. |
| Disentangled epistemic/aleatoric (58, 120–122, 513–514) | A 2×2 cross-AUSE (each channel against coverage error and against change/boundary error). Plus curves against number of views: u_epi falls on static regions, u_ale does not fall on dynamic ones. |
| The small-sample correction prevents false confidence (121–122, 334–337) | Currently false (E7). After the fix: ECE/AUSE at low n, with and without the correction. |
| Closed-form update "generalizes" categorical mapping (61–62, 117–119) | Math wording (E14), plus ablation (i) on a closed label set where both run (e.g. ScanNet-20 via LSeg, as CSS did). |
| The gradient trick plugs into any rasterizer without kernel changes (62–65, 123–125, 514–515) | A per-Gaussian gsplat vs explicit-weight check, a second rasterizer, and a time/memory table. |
| Consolidation finds stable structure and forgets transient content (65–67, 126–129) | Precision/recall of the unstable and robust masks against ground-truth moved objects. Needs a changing-scene dataset: RIO10, 3RScan or ScanNet rescans are candidates (size and licence not checked). |
| Robust to scene change without heuristics or a language model (67–68, 128–129, 516–517) | Tab. II on real change data vs Chu & Kim, ACE and Marepo; a sweep of τ_min, r_stable, r_unstable and λ; ablation (iii). |
| Arbitrary post-hoc queries with calibrated uncertainty (158–160) | Open-vocab mIoU with ECE/NLL on query sets not seen at mapping time. |
| EFE is principled and avoids wasted views on dynamic regions (186–189, 373–386) | An active-mapping run: views needed to reach a target mIoU/AUSE, and the share of views spent on dynamic regions, vs a VISTA-style score. The metrics in Tab. III cannot measure this. |
| Evaluation on 7-Scenes, ScanNet and "semi-dynamic" data (69–70, 132–133, 395–398) | 7-Scenes and ScanNet are static. A dynamic dataset is required, or the title claim has no support. |
| Headline result (70, 517) | TODO. |
| CSS names an open-dictionary continuous conjugate prior as future work (104–105) | Verified. CSS conclusion: "…may be extended to open-dictionary continuous splatting through a continuous conjugate prior" [css_grep.txt]. This also means the core idea is CSS's stated next step, which is a novelty risk. |
| 3DGS relocalization beats NeRF (142–144) | No citation given (e.g. GS-based relocalizers). |

## 3. Wording risks

| Phrase (lines) | Risk | Suggested change |
|---|---|---|
| "first" | Not used as a priority claim. Good. Keep it that way, since CSS proposed this extension. | — |
| "continuous generalization" (61–62, 117–119, 512–513) | Mathematically too strong (E14). | "shares the sufficient statistic of the categorical update" |
| "without hand-tuned dynamic-object heuristics / thresholds / semantic thresholds" (67–68, 128, 359, 368–369, 517) | τ_min, r_stable, r_unstable, λ, κ and λ_p are all hand-set. The conclusion (518–519) admits the thresholds come from a validation split. Defaults in `run_mapping.py:198-203` are 5, 0.85, 0.55, 0.9 and every 10 frames; this is contradicted by [C3b-v]. | "with class-agnostic thresholds on the map's own uncertainty, set once on a validation split" plus a sensitivity plot |
| "calibrated" (53, 94, 147, 160, 424) | Undefined for open vocabulary. AUSE measures ranking, not calibration. `evaluation.py:36` suggests r̄ as an ECE confidence, but r̄ ≥ ~0.8 on raw CLIP [C3-iii] and is not a probability of correctness. | Define p(label = c) = softmax_c(β μ*ᵀt_c) over a fixed evaluation vocabulary (β fit on validation), report ECE/NLL/Brier, and call AUSE "uncertainty ranking quality" |
| "principled" (187, 373) | Not an EFE (E13). | "EFE-inspired" or derive the information gain |
| "Crucially…drops out", "exact analogue" (243, 257) | Fine mathematically, but "exact analogue" overstates the model match (E14). | "analogue" |
| "independent routes" (319) | n and r̄ are coupled (E3). | "two routes" |
| "any differentiable feature rasterizer" (64, 125, 294, 514) | False for SH, normalised or decoder rasterizers (E11). | "any rasterizer affine in per-Gaussian features" |
| "ATE/ARE" (411, 464–469) | ATE is a trajectory metric, not per-image relocalization error. | "median translation / rotation error" |
| "irreducible / more sensing cannot resolve" (327–329) | Boundary and view-dependent spread can be reduced with more resolution or views (E9). | Qualify it |

## 4. Presentation

| Issue | Evidence | Fix |
|---|---|---|
| Page budget | The committed `papers/main.pdf` (same commit ab28fb4) has 6 pages: 703/931/665/705/863/34 words, so page 6 holds one reference (`pdfinfo`, `pdftotext -f p -l p`). RA-L allows 6 pages including figures and references, plus up to 2 extra pages at $175 each (WebFetch of https://www.ieee-ras.org/publications/ra-l/information-for-authors-ra-l, 2026-09-28). My estimate, not measured: after adding implementation, dataset, results text, a real qualitative figure, an algorithm box and limitations, it reaches about 7–7.5 pages. | Cut the EFE section (optional, no experiment). Drop Fig. 3 (the identity is one line). Merge Fig. 2 with Fig. 4. Shorten Related Work. |
| Figure balance | 5 figures: 4 method schematics plus a placeholder qualitative box. No results figure. 3 tables, all placeholders. | Add a qualitative figure* and sparsification or change-curve plots. |
| Missing pieces | `algorithmic` is loaded but no algorithm is used. There is no 3DGS preliminaries equation (T_{p,n} is never defined). Limitations are one sentence (518–520). No runtime/memory numbers. The dataset is a TODO (398), the implementation is a TODO (415–417), and the render equation is missing (E16). | Add a one-column algorithm (per-keyframe: render/backward, then consolidate), a preliminaries equation, a limitations paragraph (fixed κ, pixel correlation, CLIP common component, static geometry) and a timing table. |
| Anonymity | RA-L review is double-anonymous: "leave space for an author byline, but do not include the author information" (same URL). The draft has named placeholders, funding and affiliation (39–43) and an Acknowledgment section (522–523). | Anonymise the byline, remove thanks and acknowledgment, cite own work in the third person. |
| Sibling paper overlap | The same group's USGLoc (Dirichlet/EDL semantic 3DGS, calibrated localization on 7-Scenes) overlaps in theme. | Position the two papers explicitly (cite anonymously if it is public). Avoid overlapping claims and the same 7-Scenes tables. |
| Bib entries marked "% TODO verify" | From arXiv abs pages [arxiv_out.txt]. **kim2025e2bki**: arXiv 2509.11964 exists; title and authors match; arXiv says "Accepted to IEEE RA-L", so cite RA-L (volume/pages not checked). **liu2025physgs**: arXiv 2511.18570 exists; authors are Chopra, Liang, Seneviratne, Manocha, so the key is wrong and the `\todo{authors}` needs filling ("Submitted to CVPR 2026"). **ye2024gsplat**: arXiv 2409.06765 exists; authors match "Ye et al."; comment says JMLR MLOSS, so cite JMLR (volume not checked). **zhou2024feature**: not checked. | Fix as listed. |
| Other bib entries | arXiv IDs 2411.02547 (CSS), 2503.05949 (Bayesian Fields), 2507.01125 (VISTA) and 2602.11862 (LAMP) exist with matching titles and authors. LAMP's arXiv journal-ref says "IEEE RA-L, 2025", so update its venue. All venues and page numbers: not checked. | Also add missing refs: GS relocalizers, MaskCLIP/LSeg, AUSE, ECE, Kish, joint vMF prior. |

## 5. Prioritised fix list

**Critical**
1. **E7:** replace r̃ with the Kish-form unbiased ρ̂ or the joint-prior shrinkage. Rewrite lines 121–122 and 334–337.
2. **E8:** count evidence in views, not pixels, and set τ_min in views. Otherwise every single-view Gaussian is "stable" [C3-ii].
3. **E9 / E12:** centre the features or set thresholds relative to the floor, add a temporal change test, and fix the forgetting/EFE interaction. Show all of this on real rearrangement data before writing any localization claim.
4. Define calibration and the per-pixel predictive (E16, §3 "calibrated"). Tab. I cannot be filled without them.
5. Get a changing-scene dataset. Without one, the title and Tab. II have no support.
6. Anonymise the paper for double-anonymous review.

**Major**
7. Change the "continuous generalization" wording (E14).
8. EFE: derive the information gain or cut the section, and drop "principled" (E13).
9. Replace the "without heuristics" wording and add a threshold sensitivity study.
10. Handle κ (E5): MLE estimate or joint prior, and say it does not affect decisions today.
11. Use the exact angular variance (E4).
12. Add a per-Gaussian gsplat identity check and replace "any" (E11).
13. Fix the PCA query distortion (E15).
14. Add the algorithm box, limitations, runtime table and a page plan.
15. Split within-view and across-view dispersion, and report AUSE per error type (E9).

**Minor**
16. κ* is exact when κ0=0 (E3).
17. The T·α kernel breaks k(x,x)=1, same as CSS (E2).
18. Rename ATE/ARE.
19. Fix the bib entries.
20. Code uses r̄ for consolidation (`consolidation.py:33`) but r̃ for u_ale; make them consistent.
21. The text prior κ0 is never exercised in code; test it or drop it.

## Commands and outputs (all in the scratch dir above)

| Tag | Command | Output file |
|---|---|---|
| C1 | `python3 c1_vmf_posterior.py` | c1_out.txt |
| C2 | `python3 c2_rbar_shrinkage.py` | c2_out.txt |
| C3 | `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" python3 c3_real_clip.py` (CPU, 25.7 s; the project's `OpenCLIPDense`, ViT-B-16 laion2b_s34b_b88k; 7-Scenes chess seq-01 frames 0–190 stride 10; depth intrinsics 585 used for colour too, 5 cm depth test) | c3_out.txt |
| C3b | same env, `python3 c3b_change.py` (simulated change by swapping observation halves between static points) | c3b_out.txt |
| C4 | `PYTHONDONTWRITEBYTECODE=1 python3 c4_consolidation_efe.py` (imports usplat read-only) | c4_out.txt |
| C5 | `python3 c5_dirichlet_efe.py` | c5_out.txt |
| C6 | `python3 c6_joint_prior.py` | c6_out.txt |
| C7 | inline python (Banerjee κ̂ on uniform data; mixture threshold) | c7_out.txt |
| Refs | `curl -sSL https://arxiv.org/abs/<id>` for 7 IDs; `curl -sSL https://arxiv.org/html/2411.02547` then grep | arxiv_out.txt, css_grep.txt |