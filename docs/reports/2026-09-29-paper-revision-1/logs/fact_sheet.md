# Fact sheet for the paper revision (2026-09-29)

Every drafter writes from this sheet. Code is the ground truth: /home/thanhnc/gauss_ws/src/uags/usplat/vmf_field.py
(FieldConfig defaults), predictive.py, evaluation.py, vmf_math.py, clip_dense.py, feature_pca.py, bki_accumulate.py.
Plan: docs/reports/2026-09-28-paper-completion-plan/1-plan.md. Latest measured findings: docs/reports/2026-09-29-data-and-baselines/README.md.

## A. Method as implemented (defaults in brackets)

1. Features. Per keyframe v, a dense extractor gives unit per-pixel vision-language features z_p that are cosine-comparable
   with text embeddings t_c. The extractor was chosen on the Replica validation scenes (2026-09-29): LSeg [li2022lseg] (ViT-L/16 image encoder,
   CLIP ViT-B/32 text encoder, 512-d, 60x80 token grid), PCA to 64 dims, 5-template prompt ensemble. Candidates it beat:
   ClearCLIP (ViT-B/16 laion2b), Talk2DINO, dino.txt, ProxyCLIP. Write "LSeg" where the name is needed.
   Features are compressed to d = 64 by an uncentred PCA fit on the mapping frames only (no text, no labels). A text query is
   projected WITHOUT renormalisation: mu*^T (B^T t) equals the cosine between the decoded direction B mu* and t in the full space
   (renormalising the projected text changes the argmax label; unit test tests/test_estimators.py).
2. Accumulation. The splat weight w_{p,n} = T_{p,n} alpha_{p,n} is the BKI kernel (as in CSS). One render and one backward pass
   give, for every Gaussian n, R_{v,n} = sum_p w_{p,n} z_p and m_{v,n} = sum_p w_{p,n} (the gradient identity of
   Gradient-Weighted Back-Projection, joseph2025gradbp; we stack several target channels -- features, a mass channel, optional
   observed depth -- into the same backward pass). Needs a rasterizer affine in per-Gaussian features (gsplat colours, no SH).
3. View-level evidence (key change). One view is ONE observation: u_{v,n} = R_{v,n}/m_{v,n} (||u|| <= 1; its length is the
   within-view coherence), view weight w_{v,n} = min(1, m_{v,n}/(a0 H W)) [a0 = 1e-4]. Reason: pixels of one view are strongly
   correlated (one CLIP token covers ~16x16 px), so a pixel count over-states evidence and grows with resolution; with pixel
   counts every Gaussian passed a 5-sample threshold after ONE view (unit test).
4. Sufficient statistics: R = sum_v w u, W1 = sum w, W2 = sum w^2, Q = sum w^2 ||u||^2.
   Direction posterior (vMF likelihood with concentration kappa, tempered by w): eta = kappa R + kappa0 mu0 ; mu* = eta/||eta||.
   Concentration: joint conjugate prior p(mu,kappa) ∝ C_d(kappa)^{c0} exp(kappa m0^T mu) with m0 = 0 [c0 = 1] gives the MAP
   A_d(kappa_hat) = ||R||/(W1 + c0), A_d(k) = I_{d/2}(k)/I_{d/2-1}(k); kappa_post = kappa_hat ||R||.
   With one-hot z the statistic R equals the categorical (Dirichlet) count increment: same sufficient statistic and same MAP
   label, but NOT the same distribution (do not write "continuous generalization"; write "directional counterpart").
5. Epistemic: u_epi = 1/(1+s), s = min(n_K, W1), n_K = W1^2/W2 (Kish effective number of views).
6. Aleatoric: u_ale = 1 - rho_hat, rho_hat^2 = (||R||^2 - Q)/(W1^2 - W2), clamped to [0,1]; the argument is unbiased for
   rho^2 = ||E u||^2 under independent views; defined once s >= tau_min [3 views]. r_bar = ||R||/W1 is biased upward (~1/sqrt(n))
   and a prior-shrunk agreement ||R||/(W1 + c0) mixes the AMOUNT of evidence into the dispersion (NormLift's R(j) =
   ||f|| N/(N+1) has that form). rho_hat does neither, so "little evidence" and "conflicting evidence" stay separate.
   The old draft's shrinkage r_tilde = r_bar - (1 - r_bar)/(n+1) returns 1 at r_bar = 1: it did not do what the text claimed.
7. Predictive for pixel p of a novel view: m_p = sum_n w_{p,n} f_n (un-normalised render, so low opacity lowers confidence),
   p(c|p) = softmax_c(beta m_p^T t_c) with f_n = mu*_n (P0) or E[mu_n] = A_d(kappa_post,n) mu*_n (P1, mean-field over the vMF
   posteriors: weakly supported Gaussians pull m_p toward 0). P2: beta_p = beta0 exp(-a u_epi,p - b u_ale,p - g 1[u_ale undefined]),
   uncertainties composited with the same weights. beta and (beta0, a, b, g) are fit on the validation split by NLL.
   Every baseline row gets its own temperature on the same split.
8. Change test, per Gaussian and per view, on the raw channel. Family: a recent bucket F <- beta F + w u and an older bucket
   S <- S + (1 - beta) F_old (disjoint, S + F = all evidence); c = 1 - cos(S, F); a change must persist k views
   (score = min of the last k values). Chosen on a tuning scene: beta = 0, k = 2 [defaults], i.e. c_t = min over the last two
   views of 1 - cos(all previous evidence, current view). STATE PLAINLY that this member equals the classic feature-residual
   test (MIF / CL-Splats style); the contribution is not the statistic but its calibration and the Bayesian bookkeeping.
   Centring the features (subtracting the scene mean) was tested and hurt; it is off.
9. Calibration (no labels, no scene-specific tuning). The mapping sequence is assumed static. While mapping, every defined
   per-view score is stored in a histogram per history-size bucket of the older bucket (views: <4, 4-6, 6-10, 10-20, 20-40, >=40).
   The threshold for a Gaussian is the q-quantile of the null of ITS history size [q = 0.999: per-view false-alarm rate 1e-3];
   nearest bucket when a bucket has < 200 samples. Reasons (measured): a fixed threshold did not transfer between scenes, and a
   per-Gaussian maximum over the mapping pass is inflated by the short early histories.
10. Reset and re-admission. When c > theta(h): reset the Gaussian's statistics (forget) and flag it "changed". It re-enters the
   robust set after 3 new views AND a mean |observed depth - Gaussian-centre depth| <= 5 cm (observed depth from a sensor or a
   depth estimator, never the rendered mesh depth); with no depth stream at all, after the views alone. A removed object's
   Gaussians thus cannot return with stale geometry. Robust set = s >= tau_min, rho_hat >= 0.6, not changed.
11. Geometry is fixed after mapping: content that moved is forgotten at its old place but not inserted at its new place (limitation).
12. FieldConfig.legacy() reproduces the original draft's method exactly (tested against git): the ablation baseline.

## B. What is measured so far (internal; NO number goes into the paper, paper numbers stay \ph placeholders)
- Replica validation, 2D features only: worst-scene mIoU 23.2 with LSeg (ClearCLIP 13.4); LSeg chosen. 3D Table I running.
- 7-Scenes proxy (simulated swaps): our score = feature residual; per-scene calibrated operating point transferred between two scenes.
- No RIO10 data yet (access form pending).

## C. Scope, venue, honesty
- IEEE RA-L, DOUBLE-ANONYMOUS; first submission in the RAS conference class: \documentclass[letterpaper, 10 pt, conference]{ieeeconf}.
  No author names, affiliations, funding, acknowledgments, lab links. Leave space for a byline ("Anonymous Authors").
- 6 pages + 1 paid page planned (7 pages including references).
- Do NOT claim localization (the same group's concurrent localization paper covers it); remove the localization table,
  the ATE columns and the keyword. Remove the active-view-selection (EFE) section (no experiment); park removed text in
  papers/parked_sections.tex.
- Novelty wording (from the audits): the gradient identity is prior work (joseph2025gradbp); open-dictionary continuous BKI
  exists on voxels with a Gaussian model (wilson2025latentbki); CSS names a continuous conjugate prior for open dictionaries as
  future work (wilson2025css); NormLift (zang2026normlift, concurrent, arXiv Sept 2026) counts effective views with Kish and uses
  a shrunk norm as reliability for lifted CLIP features on static scenes; LEGaussians learns a per-Gaussian uncertainty
  (shi2024legaussians) -> never write "existing semantic 3DGS provides no uncertainty". PhysGS (CVPR 2026) and LAMP (RA-L 2025)
  are published, not concurrent. Change detection / long-term 3DGS: lu2025gscd, ackermann2025clsplats, zeng2025gaussianupdate,
  he2026vgmapping, jiang2026mif; semantic 3DGS localization under change: chu2026semantic (related work only).
- No "first", no "principled", no "without heuristics" (thresholds exist; say "class-agnostic thresholds set on a validation
  split or calibrated from the map's own mapping pass"), no "any rasterizer" (say "any rasterizer affine in per-Gaussian
  features"). "Calibrated" only for the predictive, defined by ECE/NLL, and only as something the experiments test.
- Placeholders: numbers are \ph (macro exists) and open items are \todo{...}. Never type a number that is not in this sheet.

## D. Style (the user's rules)
- Plain simple English, short sentences, active voice. No em-dash chains ("---"), no semicolon chains.
- All numbers in a table column carry the same number of decimals. Cited numbers only from the primary paper.
- Cite only keys that exist in papers/ref.bib (list in the prompt). If a new reference is essential, give its DOI/arXiv id in
  your answer instead of adding it.

## E. Experiments to describe (placeholders; plan section 3)
- Data: Replica (Semantic-NeRF renders; zhi2021semanticnerf, straub2019replica), 8 scenes, 101 classes, 0 = unlabelled.
  Validation office_0, room_2; test room_0, room_1, office_1-4. Map on Sequence_1 every 5th frame (180 frames); evaluate on
  Sequence_2 every 10th frame (90). Vocabulary = the classes present in each scene; prompt template frozen on validation.
  RIO10 (wald2020rio10; 3RScan annotations wald2019rio) for real rearrangements: validation scenes 01, 04, 06, test the other 7;
  map on seqXX_01, update with seqXX_02; stale ground truth from object annotations (moved or removed objects); pending access (\todo).
- Metrics: mIoU and pixel accuracy; ECE (15 bins), NLL, Brier, AURC of the predictive; AUSE (tie-aware, mean-gap) of u_epi
  against reducible error e_red (wrong with k views, right with the full map) and of u_ale against irreducible error e_irr
  (wrong at both); stale-content AP/AUROC at Gaussian and object level; false-alarm rate on static content; ms per keyframe, GB.
  Coverage rule: a pixel with rendered opacity < tau_cov counts as wrong.
- Tab I (Replica test): rows 2D features per frame, averaging (joseph2025gradbp), NormLift (zang2026normlift, re-implemented
  as defined), CSS Dirichlet (closed vocabulary given at mapping time; wilson2025css), Ours P0 / P1 / P2.
- Fig views: cross-AUSE of (u_epi, u_ale) x (e_red, e_irr) against the number of mapping views (10/30/90/180).
- Tab II (RIO10 test): NCC photometric residual, non-oracle depth residual, oracle depth residual (upper bound, labelled),
  feature residual (k = 2), CSS variance, Yager conflict (kim2025e2bki), NormLift reliability drop, dispersion 1 - rho_hat,
  Ours (calibrated test + reset + re-admission), Ours fused with non-oracle depth (rank average).
- Tab III ablations: views vs pixels; Kish vs prior-shrunk vs old r_tilde; change family (beta, k); calibration
  (history-conditioned vs pooled vs per-Gaussian max); re-admission depth check on/off; predictive P0/P1/P2; centring on/off;
  full legacy method.
- Timing and memory; one qualitative figure (placeholder box).
- Limitations: fixed geometry; feature extractor bounds absolute accuracy; the change statistic is a known residual;
  re-implemented baselines; synthetic calibration data (Replica) if the RIO10 class map fails.
