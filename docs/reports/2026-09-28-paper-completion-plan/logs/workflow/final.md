# UAGS final plan (revised after critique, 2026-09-28)

This is the synth plan (WF/synth/plan.md), revised against two critiques: completeness (21 gaps) and hostile reviewer (8 gaps). Section 10 maps every gap to its answer.

- **[A]** marks an assumption that no report or command backs.
- WF = `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/wf`.
- I changed no project file. My new checks are in WF/finalize/: `workdays.py/.log`, `budget.py/.log`, `load2.py/.log`, `load3.py/.log`, `strata.py/.log`. A copy of this plan is at WF/finalize/plan.md.

**The six largest changes:**
1. Change handling (C4) is now the main claim, and G3 is the go/no-go for submission. The static Replica block becomes a parity and calibration check (G2).
2. Stale ground truth now comes from object annotations, not from depth. The rendered-depth residual becomes an "oracle" upper-bound row. A non-oracle depth residual is added.
3. The predictive now uses the posterior (F6). "Calibrated" stays in the paper only if this beats plug-in + temperature on val.
4. NormLift is implemented as its paper defines it: view-level Kish over per-view weights. Fallback F-B is deleted.
5. The old F3 (subtract n·z̄) is replaced by a separate centred-and-renormalised channel.
6. The planned date moves to **Fri Nov 27**. The must-have work is 31.75 pd against 41 workdays, a 22.6% buffer. The buffer reaches 25% only if the supervisor takes the figure redraw (D6).

**Where the critiques or earlier plans disagreed, and what I chose:**
- **Date:** Nov 27 is the plan. Nov 15 is dropped, because it needs 32.25 pd in 32 workdays, which leaves 0% buffer (`python3 WF/finalize/workdays.py` → workdays.log; `python3 WF/finalize/load3.py` → load3.log).
- **Holidays:** I use Japanese holidays (Oct 12, Nov 3, Nov 23) because the draft affiliation is JAIST (WF/critic_complete/evidence.txt, the commented `\thanks`). But the machine clock is UTC+7 (`date +%z` gives +0700). Where the student works is not checked. If no holidays apply, the buffer gains 3 days.
- **Estimator:** build the joint prior and Kish, and pick one on val. If they tie, use the joint prior.
- **Optimised deterministic field:** moved from "required" to defer item 1. The completeness critic named it as a deferral candidate. It matters less now that change handling is the main claim.
- **Bonn:** no switch to Bonn inside the November plan. The rule is fixed in advance (§6).
- **G1:** kept only as a cheap early check (defer item 4). G3 decides.
- Unchanged from the synth plan: no localization; the static false-alarm row runs on Replica; USGLoc code is copied at pinned commit adea106; all 7 RIO10 test scenes are used (the 4 smallest if disk runs short, chosen before any result is seen).

## 1 Scope decision

| Keep (narrowed) | Cut or demote |
|---|---|
| **C4 (main claim):** a temporal change test plus consolidation on fixed geometry, which detects and forgets stale semantics. Tested on real RIO10 rearrangements with object-level labels. | tab:loc (main.tex:456-480), the ATE column, and the keyword "localization" |
| **C2:** two channels. The epistemic channel is a view count (at parity with NormLift's N_eff). The aleatoric channel is dispersion on a centred-and-renormalised channel. The posterior feeds the predictive. | The EFE section (main.tex:371-388) |
| **C1:** a vMF conjugate posterior per Gaussian. Word it as "a directional generalization; shares the sufficient statistic and MAP label of categorical BKI". Cite LatentBKI and Mardia & El-Atoum. | The gradient trick (main.tex:278-307) shrinks to one paragraph that cites Joseph et al. Drop Fig. 3. |
| | Inserting new geometry where an object moved to. Geometry is loaded once (scripts/run_mapping.py:114). This goes in Limitations. |
| | A ScanNet block. The form is sent anyway (D7), so the block stays possible for the revision round. |

**State these facts openly in the paper:**
- μ* is the T·α-weighted mean feature. So mIoU equals averaging, by construction.
- With raw features, m0 = 0 and c0 = β, 1 − u_ale equals NormLift's R(j) exactly (WF/judge/estim_equiv.log: max difference 0.0).
- NormLift already counts views. It applies Kish "to the per-view rendering weights" W_j^(v) = Σ_{i∈R_v} A_ij (NormLift Sec. 3.3.2, Eq. 6 and 8; WF/reviewer/normlift.txt, fetched from https://arxiv.org/html/2609.18898). So on static scenes our read-out reduces to NormLift. Cite it as concurrent (arXiv, 2026-09-16).
- NormLift's own limitation (3) reads: "generalization to building-scale, outdoor, or dynamic scenes remains to be verified" (normlift.txt).
- Geometry is fixed after seq01. A moved object is forgotten at its old place, but it is not mapped at its new place.

**What the paper must show** (this replaces the old "left to prove" line):
1. **Go/no-go (G3).** On RIO10 real rearrangements, with stale labels from object annotations that no method sees, one of two things holds. Either c finds stale semantics better than every non-oracle residual (NCC photometric, non-oracle depth, feature residual). Or c adds to non-oracle depth in a fused score.
2. **Claimed only if the G2 cell allows it.** The posterior predictive beats plug-in + temperature on NLL and ECE.
3. **Parity, not a claim.** u_epi ranks reducible error at least as well as NormLift's N_eff, and better than max-softmax (MSP).

**Problem statement:** "no view-level epistemic signal for open-vocabulary features, and no handling of stale semantics". Do not write "no trust signal": CSS, LEGaussians, NormLift and LAMP all have one.

**Title and claims:** these are chosen from the G2 × G3 grid in §6.

**Abstract, in order:**
1. The problem, reworded as above.
2. The method.
3. RIO10: stale-content AP against the residuals, and mIoU on changed regions after consolidation.
4. Replica: equal mIoU, ranking at parity, and calibration only if G2 passes.

Numbers go into the abstract only after the test runs.

## 2 Technical fixes before any experiment

**Symbols:**
- a0 = view-mass floor.
- z̄ = the scene mean feature, taken from the mapping frames.
- (m0, c0) = the prior.
- τ_cov = the coverage floor.
- ε = the minimum ‖z − z̄‖ for the centred channel.

| # | Fix | Why | Validation |
|---|---|---|---|
| F1 | Store (R_v, m_v) per keyframe. Set R = Σ w_v R_v/m_v and n = Σ w_v, with w_v = min(1, m_v/(a0·H·W)). Keep Σ w_v² for view-level Kish. Drop near-duplicate keyframes. Set τ_min in views. | n_eff counts pixels, about 50× too many (math E8). | n changes by less than 1% between ds4 and full resolution [A]. The share of Gaussians "stable" after 1 view drops from 1.00 to 0. |
| F2 | Joint prior: A_d(κ̂) = ‖R+m0‖/(n+c0). Kish: ρ̂² = (n_K r̄²−1)/(n_K−1). **u_ale is computed on the F3 channel.** κ̂ for the predictive (F6) is computed on the raw channel. | r̃ has no effect when r̄ = 1 (vmf_field.py:74-76). | One view gives 1/(1+c0) on both channels. Uniform directions give ρ̂ ≈ 0. The C6 check gives 0.3330 vs 0.3333. |
| **F3 (replaced)** | Accumulate a second channel from ẑ = (z − z̄)/‖z − z̄‖. z̄ is fixed per scene from the mapping frames, before accumulation. Pixels with ‖z − z̄‖ < ε get zero weight. **Queries use the raw R.** Dispersion, u_ale and c use the ẑ channel. | The old F3 left the unit sphere. Content with zero spread got r̄_c p5/p50/p95 = 0.451/0.586/0.737, so u_ale = 0.33/0.467/0.59 where it should be near 0. Static-vs-changed AUROC: raw 0.807, old F3 0.739, renormalised 0.843. Source: `python3 WF/reviewer/centring_check.py` → centring_check.log. Caveat: the "changed" pairs are synthetic. | Unit test: identical observations give r̄ = 1 for every class. Re-run G1 with this channel. Memory: +N(d+1)·4 B = 0.26 GB at N = 1M, d = 64 (budget.log). |
| F4 | Change score c = 1 − cos(μ_slow, μ_fast) on the ẑ channel. **slow** = the ẑ channel with decay λ. **fast** = an EWMA with its rate set on RIO10 val. That makes 3 channels in total (0.78 GB at 1M, budget.log). Split r̄ into r_within and r_across. | Changed content still has raw r̄ of 0.86–0.95, and r̄ ignores order. | G1, G3 |
| F5 | When c fires: reset (R, n) on every channel, and set a persistent `changed` flag. Re-admit a Gaussian only after τ_min new views **and** a small **non-oracle** depth residual. Replica uses its sensor depth. RIO10 uses Tango depth (via 3RScan) or MASt3R depth, **never the rendered mesh depth**. Apply λ per keyframe. | Decay leaves r̄ unchanged, and forgotten Gaussians come back. Ours must read no oracle input. | Unit test on an A→B sequence. Ablation (e). |
| **F6 (replaced)** | Three predictives. **P0 plug-in:** p(c) = softmax(β m_pᵀt_c), with m_p = Σ_g w_pg μ*_g. **P1 posterior (mean-field):** m_p = Σ_g w_pg A_d(κ_post,g) μ*_g, with κ_post = κ̂·n·r̄ (raw channel); this uses E_vMF[μ] = A_d(κ)μ*. **P2:** β(u_epi, u_ale) = β0·exp(−a·u_epi − b·u_ale) [A: the form], fit on Replica val. Every baseline row gets P0 with its own β fit on val, so "averaging + temperature scaling" is the matched baseline. Composite u_epi and u_ale with T·α. | Under the old F6, Ours' predictive ignored u_epi, u_ale, κ̂ and n. P1 matters where views are few. At d = 64, r̄ = 0.85: A_d = 0.826 (n = 1), 0.909 (2), 0.962 (5), 0.981 (10), 0.996 (50). Source: `python3 WF/reviewer/ad_check.py` → ad_check.log. | Mock rasterizer vs brute force. A Monte Carlo check of P1 on one val scene (defer item 6). Tab III (i). |
| F7 | Decode-first query: μ*ᵀ(Bᵀt). **The PCA basis and z̄ are fit on the mapping frames only**: no text, no val labels. | Text keeps 11% of its energy, and 13–29% of labels flip. | Argmax agrees with 512-d on at least 0.95 of pixels [A]. It is 0.707 now. |
| **F8 (changed)** | Extract at full resolution, with laion2b as the default. **Cache the ViT token grid** (fp16, 512-d, before upsampling and normalisation), then upsample and normalise at load time. Size: 1.23 MB per Replica frame, 2.65 GB for all 2,160 frames; 2.09 MB per RIO10 frame (budget.log). Per-pixel caches are not allowed. | A per-pixel fp16 PCA-64 cache is 39.3 MB per frame: 56.6 GB for the Replica mapping frames and 228 GB for the RIO10 updates (WF/critic_complete/sizes.log). Only 48 GB is free. | The cached path matches direct extraction within 1e-3 cosine [A]. |
| F9 | Per-Gaussian gsplat identity test (one-hot, N ≈ 300). One `rasterization()` call with `channel_chunk`. Drop the `.clone()`. | The current test checks only linearity and mass. | Error below 1e-4 [A]. One GPU test under 5 min. |
| F10 | Text fixes: κ only scales κ*; exact variance 1 − A_d; "rasterizer affine in features"; one AUSE equation (tie-aware, §3). | math E3–E5, E11 | Supervisor read-through |
| **F11 (new)** | NormLift as its paper defines it. Per-view W_j^(v) = Σ_{i∈view v} A_ij and f_j^(v) (its Eq. 6). N_eff = (Σ_v W)²/Σ_v W². R(j) = ‖f_j‖·N_eff/(N_eff+β) with β = 1 (its Eq. 8). C_intra·C_inter. Its epistemic score is 1/(1+N_eff). Also add a trivial "visible-view count" row. | The old row ("pixel Kish") was a weakened NormLift. | A unit test on a toy scene checks R against Eq. 6 and 8 computed by hand. |

## 3 Experiments

**Rules for every row:**
- All methods use the same 3DGS geometry and the same MaskCLIP ViT-B/16 laion2b features.
- All parameters are set on val. The test runs once, from a frozen commit.
- Report mean ± std with N. Keep every row where a baseline wins.
- **Re-implementations.** Averaging, NormLift and CSS are our own re-implementations. The paper says so, and it says that CSS receives the evaluation vocabulary at mapping time. The averaging row is checked against the official Gradient-Weighted Back-Projection code (WF/sota/gradbp_backproject.py) on one val scene: max feature difference below 1e-4 [A].
- **Coverage rule** (fixed before any val run):
  - Every GT-labelled pixel counts.
  - For 3D rows, a pixel with Σα < τ_cov counts as wrong. Rows without an uncertainty channel get u = 1 there.
  - A "covered pixels only" variant is also reported.
  - 2D MaskCLIP sees the evaluation frame itself, so the rule does not apply to it.
- **MSP everywhere.** Every row, Ours included, also gets MSP and predictive-entropy AUSE from its P0 predictive.
- **Tie-aware AUSE.** Use the mean-gap definition, with the error averaged over each tied block. This equals the expectation over random tie order.
  - Unit tests: permutation invariance; constant u gives the random curve; the oracle gives 0.
  - It replaces the behaviour at usgloc/usgloc/metrics.py:100 (`np.argsort(-u)`), which breaks ties by sort order. View counts will have many ties.
- **Error targets (Replica).** At a k-view map:
  - e_red = 1[wrong at k views and right at 180 views]
  - e_irr = 1[wrong at k views and wrong at 180 views]
  - A secondary target for e_irr is the GT boundary band (3 px [A]).
  - All k-view fields use the same 180-frame geometry.
- **Vocabulary.** Freeze one file on val before any test run. It holds: Semantic-NeRF class id → name → prompt, the ignore list (void and undefined), the merges, and the prompt template(s). The code default is "a photo of a {}." (usplat/clip_dense.py:120). The queries are the classes present in each scene.

**Splits:**
- **Replica (Semantic-NeRF):**
  - Val: office_0 and room_2. Test: room_0, room_1 and office_1–4 (N = 6).
  - Map on Seq_1, every 5th frame (180). Evaluate on Seq_2, every 10th frame (90).
  - If the zip has no Seq_2: evaluate on Seq_1 frames offset by 2 from the mapping frames [A]. The paper then says this is weaker.
- **RIO10:**
  - Val: scenes 01, 04 and 06 (4,929 seq02 frames). Test: scenes 02, 03, 05, 07, 08, 09 and 10.
  - Map on seqXX_01, every 10th frame [A].
  - Update with the first half of seqXX_02, every 5th frame [A], using GT poses.
  - The semantic block uses the second half, every 10th frame.

**Stale ground truth (RIO10), built from annotations that no method reads:**
1. **Object GT (primary).**
   - Instances come from semanticsXX.zip (`instances.txt`, `frame-*.instances.png`) and modelsXX.zip (`labels.ply`) (rio10_readme.md l.41-55).
   - An object is **moved** if its centroid shifts by more than δ_obj or it rotates by more than θ_obj [A: 10 cm, 10°]. It is **removed** if it is absent from seq02.
   - A Gaussian is **stale** if its nearest labelled seq01 point, within 5 cm, belongs to a moved or removed object.
   - AP is reported at the Gaussian level and at the object level (object score = mean over its Gaussians).
   - Whether instance IDs name the same object in seq01 and seq02 is **not checked**. Check it on val 01 on the day the data arrives: compare label names, and the centroids of static IDs within the alignment p95.
2. **If the IDs do not match:** use the 3RScan annotations ("rigid", "removed" and "nonrigid" per rescan; WF/data/3rscan_README.md:108-110). These need the 3RScan form and a RIO10→3RScan scan-ID map. That map is not in RIO10 metadata.json and is otherwise not checked.
3. **If neither works by G3:** use the geometric two-surface rule, and write plainly that "change is evaluated only as geometric change".
- **GT agreement:** report the IoU of the object-GT and geometric-GT stale sets on val 01.
- **Alignment check:** on val 01, measure the seq01-vs-seq02 surface distance on static structure (wall, floor and ceiling in labels.ply). Report the median and p95. **Set δ above p95.** The check has its own verification row.
- **Hidden:** behind the seq02 surface in every update view that covers the Gaussian's location. **Unobserved:** outside all update frusta, or Σ T·α < τ_cov. Both are left out of AP and reported separately.
- **SemanticChange bins:** each Gaussian gets the mean SemanticChange of the update frames that observe it (stats.txt).

**Depth sources:**
- **Oracle:** the seq02 rendered depth from the mesh. RIO10 has no sensor depth (rio10_readme.md l.13-63). It is used only for an upper-bound row labelled "oracle", and never inside Ours or G3.
- **Non-oracle:** Tango depth from 3RScan, if the form and the scan map work. Otherwise MASt3R depth on update-frame pairs, with the scale fixed by the GT poses.
  - The MASt3R checkpoint is on disk: 2.6 GB in `third_party/mast3r_ckpt` (`du -sh`).
  - USGLoc's MASt3R process used 6,866 MiB (venue §D), so run this at night only.

| Item | Question | Rows (fair) | Metrics | Success | GPU-h [A] | Needs |
|---|---|---|---|---|---|---|
| **Tab II (main; RIO10 test)** | Does c find stale semantics beyond non-oracle residuals, and does consolidation clean them out? | NCC photometric residual (exposure-normalised); **non-oracle depth residual**; **oracle depth residual (upper bound)**; **feature residual** 1−cos(rendered μ*, new z), lifted with T·α, flagged after k consecutive keyframes (MIF / CL-Splats style); scalar variance; CSS variance/MI; **Yager conflict on CSS one-hot counts** (cite E2-BKI, not USGLoc); NormLift R drop; raw r̄; Ours u_ale; Ours c; **fusion c + non-oracle depth** (pre-registered rank average, no fitting); 3DGS-CD on val 01 only (defer item 2); consolidation on/off | Object- and Gaussian-level stale AP↑, AUROC↑; stale mass left↓; static mass wrongly forgotten↓; hidden and unobserved reported apart; SemanticChange bins | G3 criteria (§6) | ≤ 13 | K12, K13 |
| **Tab II semantic block** (RIO10 test, real data) | Do semantics improve where the scene changed? | Averaging, CSS, NormLift, Ours before and after consolidation | mIoU, ECE, AUSE on the second half of seq02, split into changed and unchanged regions (object GT); a class map to the frozen vocabulary | Reported. If the class map fails, write "calibration is shown on synthetic data only" in Limitations. | in Tab II | K12 |
| Tab II static row | How many false alarms when nothing changed? | Same rows. Replica: map on Seq_1, update on Seq_2, thresholds from RIO10 val with no re-tuning. | % flagged↓ | At most the NCC residual's rate | 0.5 | Tab I maps |
| **Tab I (Replica test; parity + calibration)** | At equal mIoU, is our uncertainty ranked at least as well, and does the posterior calibrate better? | 2D MaskCLIP; averaging + temperature; NormLift (F11); visible-view count; CSS variance/MI; Ours P0, P1, P2; optimised gsplat field (defer item 1) | mIoU↑, Acc↑, ECE-15↓, NLL↓, Brier↓, AURC↓, AUSE↓ (own u and MSP), and the covered-only variant | mIoU within 1 point of averaging [A]. G2 grid (§6). | ≤ 10 | F1–F11 |
| **Fig. views** | Does u_epi track reducible error and u_ale the error that stays? | Ours u_epi and u_ale; **Ours-scalar**; NormLift R and 1/(1+N_eff); CSS variance; averaging MSP. Fields from 10/30/90/180 views. | 2×2 cross-AUSE: (u_epi, u_ale) × (e_red, e_irr) | Ours' diagonal is below its off-diagonal. u_epi on e_red is below MSP. | 1 | Tab I |
| **Tab III** (test, one run each from the frozen list; parts chosen on val) | Which parts matter? | (a) views vs pixels; (b) joint prior vs Kish; (c) ẑ channel vs raw vs old subtraction; (d) c vs raw r̄ vs one-shot feature residual; (e) no re-admission check (defer item 3); (f) vMF vs categorical; (g) query fix (defer item 3); **(h) Ours-scalar: κ* or 1 − A_d(κ*), all else fixed**; (i) predictive P0/P1/P2 | As in Tab I / II | Drop every part that does not help on val, then rename the method to what remains | 3 | I, II |
| Fig. sensitivity (val) | Are the thresholds and poses robust? | none | Stale AP, static false-alarm rate and ECE against τ_min, c0, λ, a0, δ ∈ {3, 5, 10} cm, the **c threshold**, the **EWMA fast/slow rates**, and **pose noise** of 1/2/5 cm and 0.5/1/2° on the update poses (reuses the maps) | A stated working range | 1 | II |
| Selective query | Can a user trust abstentions? | Same rows as Tab I | AURC in Tab I (must-have). Risk-coverage on RIO10 before/after consolidation (defer item 5). | Reported | 0 | K10 |
| Fig. qualitative | Show the result | One RIO10 val frame (high SemanticChange), one Replica frame | RGB, query, u_epi, u_ale, c | none | 0.2 | I, II |
| **Timing** | What does it cost? | One Replica val scene and one RIO10 val scene. Rows: CLIP extraction on CPU and GPU, 3DGS fit, Ours per keyframe, averaging, CSS, NormLift, optimised-field training (if run). | ms per keyframe, peak GB | At most 3.5 GB at d = 64 [A] | 0.5 | K15 |
| G5 re-run | Can it be reproduced? | Replica room_0 and one RIO10 test scene, re-run from the saved commands | Metrics from the saved maps | \|ΔmIoU\| ≤ 0.2 points, \|ΔAP\| ≤ 0.005, \|ΔECE\| ≤ 0.002 [A]. The gsplat backward pass uses atomics, so the results are not bit-exact. | 0.5 | all |

**Timing condition:** no other compute job runs, and `sang`'s resident 3.85 GB process is present (venue §D). Book one slot for this (D3).

**Hardware line** (`lscpu`, `free -g`, `nvidia-smi`, 2026-09-28 23:14): Intel Core i5-14600KF (20 threads), 31 GiB RAM, RTX 5070 Ti 16,303 MiB, driver 595.84.

**Threshold config** (one file; each row names the val set that fixes it):

| Threshold | Fixed on |
|---|---|
| a0, c0, m0, τ_min, τ_cov, near-duplicate keyframe threshold, β, P2 parameters | Replica val (office_0, room_2) |
| PCA basis, z̄, ε | The mapping frames of each scene. No labels, no text. |
| EWMA fast rate, decay λ, c threshold, re-admission depth threshold, feature-residual threshold and k | RIO10 val (01, 04, 06) |
| δ | Alignment p95 on val 01. Not tuned on AP. |
| δ_obj, θ_obj | Fixed before any method output is seen [A: 10 cm, 10°] |
| Replica static row | The RIO10 val thresholds, not re-tuned |

**Compute** [A, not measured]:
- **GPU:** 18 3DGS fits (8 Replica, 10 RIO10 seq01), MASt3R on about 3,442 update frames, and the field runs. At most 30 GPU-h. The real limit is free memory.
- **Peak memory:** measured in W2 on office_0 (K9). Cap maps at 1M Gaussians [A] and fit at ≤ 640×480 pixels per frame [A], so the fit peak stays within the free memory seen most of the time. A 1M-Gaussian map is 0.236 GB (budget.log).
- **CPU extraction:**
  - Replica: 2,160 frames × 1.81 s ≈ 1.1 h.
  - RIO10: re-estimate from one measured frame when the data arrives. Scaling by pixel count gives 2.9 h for the update frames alone (sizes.log). The seq01 and evaluation frames are not counted.

**Disk budget** (48 GB free, `df -h /`; `python3 WF/finalize/budget.py` → budget.log):

| Item | Size | Note |
|---|---|---|
| Replica zip | 6.76 GB | WF/judge/dbx_head.txt. Kept until `unzip -t` passes, because the server ignored Range and a failed transfer restarts (dbx_range.txt). |
| Replica extracted (needed members only) | ≈ 8.8 GB [A: 1.3× the zip] | Peak during unzip ≈ 15.6 GB. Delete the zip after the check. |
| Replica token cache | 2.65 GB | |
| 18 maps at the 1M cap | 4.25 GB | Keep the maps of the two G5 scenes. Delete the others only after G5. |
| RIO10 zips (each seqXX.zip holds every rescan) | **not checked** | Run a HEAD on each zip as soon as the link arrives. If the host honours Range (`curl -r 0-99`), read only the seqXX_01 and _02 members with Python `zipfile` over a range-reading file object. Otherwise handle one scene at a time and delete each zip after extraction. |
| RIO10 token cache | 1.08 GB per scene, plus the seq01 frames | |
| Non-oracle depth | ≤ 1.04 MB per frame; ≤ 3.57 GB for all update frames (uncompressed) | |
| Saved fields (G5 and qualitative scenes, 3 channels) | 0.78 GB each | Ablation fields are not saved: only JSON and a few PNGs. |
| Headroom for USGLoc | 10 GB [A] | Ask before pruning usgloc/maps (55 GB). |

**Out of scope:**
- 7-Scenes and 12-Scenes. The only exception is K5, an internal CPU replay on chess that is not in the paper.
- ScanNet (the form is sent).
- ACE/ACE-G/Marepo, Chu & Kim, active mapping, and inserting new geometry.

## 4 Code tasks (in this order; * = defer list)

| # | Task | Files | pd |
|---|---|---|---|
| K1 | `gpu_gate.sh`: start a job only if free memory ≥ peak + 4 GB (+ 7 GB while a MASt3R batch is queued), `pgrep -f 'run_7scenes\|exp_floor_matcher\|train_map\|train_encoders'` finds nothing, and the `flock` lock is free. Then call `set_per_process_memory_fraction`. Install plyfile. Make laion2b the default. | scripts/, run_mapping.py:194 | 0.5 |
| K2 | F1 with view-level Kish and tests | bki_accumulate.py, vmf_field.py, tests/test_view_count.py | 1.0 |
| K3 | F2 and F3 (ẑ channel), with unit tests: identical observations give r̄ = 1; one view gives 1/(1+c0) | vmf_field.py, tests/test_estimators.py | 1.25 |
| K4 | F4, F5 and the config file with its val-set column | usplat/change.py, consolidation.py, tests/ | 1.5 |
| K5* | G1 replay of c on the ẑ channel (C3b protocol, 7-Scenes chess, CPU). Reuse WF/math/c3b_change.py and WF/reviewer/centring_check.py. | scripts/cpu_check_change.py | 0.5 |
| K6 | F7, plus the F8 token cache with PCA and z̄ fit on the mapping frames | feature_pca.py, scripts/extract_clip.py | 1.0 |
| K7 | F6 (P0/P1/P2) and F9. *MC check of P1: 0.25 (defer item 6). | gsplat_adapter.py, tests/test_gsplat_gpu.py | 1.5 |
| K8 | Replica loader and splits; `unzip -l` check; frozen vocabulary file; **2D MaskCLIP check on CPU** (20 office_0 frames; laion2b per-pixel vs FastSAM mask pooling via usgloc MaskClipEncoder, usgloc/usgloc/mapping.py:560; pass at ≥ 15 mIoU [A]) | datasets.py, configs/vocab_replica.yaml | 1.5 |
| K9 | fit_gs.py copied from `git -C ../usgloc show adea106:usgloc/mapping.py`, credited "adapted from [anonymous]"; peak-memory log; Gaussian and resolution caps | scripts/fit_gs.py | 1.25 |
| K10a | AP/AUROC; tie-aware AUSE with unit tests | evaluation.py, tests/test_ause.py | 0.5 |
| K10b | mIoU/Acc, ECE/NLL/Brier, reliability diagrams, AURC, e_red/e_irr targets, the coverage rule, MSP/entropy for every row | evaluation.py, scripts/eval_seg.py | 1.5 |
| K11 | Averaging + GradBP check 0.5; NormLift (F11) + visible-view count + unit test 0.75; CSS variance/MI + Yager 1.0; Ours-scalar 0.25. *Optimised field: 1.5 (defer item 1). | usplat/baselines/ | 2.5 |
| K12 | RIO10 loader (540×960 portrait); object GT, geometric GT and their agreement; alignment check; hidden/unobserved; class map for the semantic block | datasets.py, scripts/rio10_stale_gt.py | 3.0 |
| K13 | NCC residual; oracle and non-oracle depth (Tango or MASt3R); feature residual with temporal filter; fusion row; SemanticChange bins. *3DGS-CD on val 01: 1.0 (defer item 2). | scripts/eval_change.py | 1.75 |
| K14 | Run scripts that write JSON with the command, git hash, seed and config hash (**in W2, before the first real run**) | scripts/ | 0.75 |
| K15 | Timing and memory instrumentation | run_mapping.py | 0.5 |
| K16 | Sweeps (thresholds and pose noise) 0.5; figures 0.25. *RIO10 risk-coverage: 0.75 (defer item 5). | scripts/ | 0.75 |
| | **Code total (must-have)** | | **20.75** |

**Other must-have work:**
- Admin and downloads: 0.5 pd.
- Running and analysing experiments: 4.0 pd (val Tab II 1.25, val Tab I 1.25, test 1.25, G5 0.25).
- Student writing: 6.5 pd.
- **Student total: 31.75 pd.**

**Defer list** (4.25 pd), cut in this order:
1. Optimised field (1.5)
2. 3DGS-CD (1.0)
3. Ablations (e) and (g) (0.25)
4. K5 (0.5)
5. RIO10 risk-coverage (0.75)
6. MC check of P1 (0.25)

Source: `python3 WF/finalize/load3.py` → load3.log.

## 5 Writing tasks

| Task | main.tex lines | Who | pd | Week |
|---|---|---|---|---|
| Port to ieeeconf. Check `\IEEEPARstart`, `IEEEkeywords` and `\markboth`. Anonymise. Build on Overleaf (tectonic is missing). | 2, 13, 39–45, 73–81, 522–523 | student | 0.5 | W2 |
| Related work and bibliography. Add the 12 must-cite papers (sota §3) plus RIO10 and 3RScan. Fix the sota §4 mismatches and lines 150–152, 166, 175 and 185. Cite NormLift as concurrent and USGLoc per D5. Check the MaskCLIP, AUSE, ECE and Kish references by DOI. | 136–191, 526–624 | supervisor [A] | 1.5 | W3–W7 |
| Method: a preliminaries equation that defines T; F1–F6 and F11; the reduction to NormLift on static scenes; an algorithm box; the AUSE equation; gradient trick cut to one paragraph; EFE deleted | 192–388 | student | 1.75 | W4 (0.5), W8 (1.25) |
| Figures: merge Fig 2 and Fig 4, drop Fig 3, add the views/sensitivity panel and fig:qual. **Redraw** pipeline.tex:92-100 (it still draws EFE and "robust subset for localization") and uncertainty_plane.tex:57 (the decay trajectory). | figs/ | student, or supervisor for a 25% buffer (D6) | 1.0 | W8 |
| Experiments, an implementation table with the hardware line, and limitations. The limitations cover: fixed geometry (no insertion), the shared CLIP component, Tango image quality, re-implemented baselines, and "synthetic-only calibration" if the RIO10 class map fails. Also restore the published-numbers footnote: Feature 3DGS 0.787 on its own split and CSS 28.4 on room_0, with a different-split caveat (copied from the data report, not re-checked). | 389–520 | student | 1.75 | W9 |
| Abstract, intro and contributions. Use the reworded problem statement. Remove "principled", "without heuristics", "any rasterizer", "continuous generalization" and "localization". Use "calibrated" only if the G2 cell allows it. | 53–133 | student | 1.0 | W9 |
| Revise after the supervisor review | all | student | 0.5 | W9 |
| Admin: 2–5 replacement keywords; PaperCept PINs; co-author sign-off; an anonymous code mirror (anonymous.4open.science, never github.com/pairs-lab); `pdfinfo` metadata and an anonymity grep; the editor note on USGLoc | — | student + supervisor | in W9 | W9 |

**Page budget:** 7 pages from the start (6 + 1 paid page at $175, D4).

| Part | Pages |
|---|---|
| Intro | 0.8 |
| Related work | 0.6 |
| Method | 1.7 |
| Experiments | 2.7 |
| Limitations | 0.2 |
| References | 0.8 (24 references already fill most of the page-5 right column in the current main.pdf, and 13 or more are being added) |
| **Total** | **6.8** |

## 6 Timeline (planned Fri 2026-11-27; hard stop Mon 11-30)

**Why these dates:**
- The IROS 2027 transfer window closes on Apr 30, 2027, and RA-L gives a final decision within 5 months. So Nov 30 is the last safe date.
- Workdays with JP holidays (`python3 WF/finalize/workdays.py` → workdays.log): 32 from Sep 29 to Nov 13, and 41 from Sep 29 to Nov 27. Without holidays: 34 and 44.
- Load: must-have 31.75 pd in 41 workdays, a **22.6% buffer**. With every defer item kept it is 36.0 pd, a 12.2% buffer (`python3 WF/finalize/load3.py` → load3.log). The buffer is 25% only if the supervisor takes the 1.0 pd figure task.
- The pd figures include running and analysing experiments, but not USGLoc revision work. The date of USGLoc's revision round is not checked, and the defer list absorbs it.

| Week (workdays) | Work (student pd) | Gate |
|---|---|---|
| W1 Sep 29–Oct 4 (4) | **Day 1:** submit the RIO10 and 3RScan forms (plus ScanNet, D7). Start the Replica download (6.76 GB, about 11 min at 10.3 MB/s), then run `unzip -l`. Ask for the USGLoc order (D5). Propose the `flock` lock and weekend slots to the USGLoc owner. Then K1, K2, K3. (3.0) | **G0 Fri Oct 2:** forms sent; `unzip -l` shows Seq_1, Seq_2 and the class names for all 8 scenes; 6/6 tests pass; D5 settled |
| W2 Oct 5–11 (5) | K14, K8 (with the 2D MaskCLIP check), K9, port. Night (gated): office_0 fit with the peak-memory log. CPU: Replica token extraction (≈ 1.1 h). (4.0) | **Fri Oct 9, capacity check:** MaskCLIP ≥ 15 mIoU [A], or the fallback chosen (mask pooling first, then openai weights, ≈ 600 MB, size not checked). Peak memory measured and caps set. If W1–W2 are not done: cut defer items 1–3 now. If more than 2 pd behind: re-decide D1. |
| W3 Oct 12–18 (4; Oct 12 holiday) | K4, K6, K5*. RIO10 val download after a HEAD size check. Night: the 7 other Replica fits. (3.25) | **Wed Oct 14:** RIO10 rule (below). **G1 Fri Oct 16** (if K5 is kept): c AUROC > 0.633 on the C3b replay, and the identical-observation test passes. A G1 failure changes nothing until G3. |
| W4 Oct 19–25 (5) | K12 (val 01 first: ID check, alignment p95, δ, GT agreement), K10a, method text part 1. Night: seq01 fits for the 3 val scenes, and non-oracle depth. (4.0) | none |
| W5 Oct 26–Nov 1 (5) | K13, Tab II and the static row on val, K16 sweeps (including pose noise), K15. (4.0) | **G3 Fri Oct 30 (go/no-go)**, below |
| W6 Nov 2–8 (4; Nov 3 holiday) | K7 (P0/P1/P2, F9 GPU test), K10b. Background: download, extract and fit the RIO10 test scenes one at a time. These are inputs only; no labels are read. (3.0) | none |
| W7 Nov 9–15 (5) | K11. Tab I and Fig. views on val. (3.75) | **G2 Fri Nov 13:** grid below |
| W8 Nov 16–22 (5) | **G4 Mon Nov 16:** freeze the list of test runs and the commit, then run each test once. The list is Tab I, Tab II and its semantic block, the static row, Tab III and Fig. views; sensitivity and timing run on val. Figures, method text part 2, the timing slot. (3.75) | G4: every table cell comes from a JSON log |
| W9 Nov 23–27 (4; Nov 23 holiday) | Experiments text, abstract/intro, G5 re-run, supervisor review Nov 24–25, revision, admin. **Submit Fri Nov 27.** (3.5) | **G5 Thu Nov 26:** the re-run is within tolerance; at most 7 pages; anonymity check passes |

**RIO10 rules (fixed now):**
- **No link by Wed Oct 14:** swap the tracks. W4 does K7 and K10, and W5 does K11 and Tab I on val, so G2 moves to Oct 30. The RIO10 track moves to W6–W7, so G3 moves to Nov 13.
- **No RIO10 data by Wed Oct 28:** no November submission. Bonn is **not** a November switch: it has no semantic labels and needs a new protocol. Take the decision to D2.
- **Bonn protocol**, written now and run only after a D2 decision (1.5 pd, for a later window):
  - Use the box sequences. Map from the frames before the box event, and update with the frames after it.
  - A Gaussian is stale if it lies more than δ from the static Leica BLK360 cloud and inside the post-event frusta.
  - Report AP/AUROC and the ghost rate. This tests geometric change only.

**G3 (go/no-go; RIO10 val 01/04/06; object GT; non-oracle rows only).** The static false-alarm rate on Replica val must be at most the NCC rate in every case.
- **Pass (i):** Ours c has higher AP than every non-oracle residual (NCC, non-oracle depth, feature residual) on at least 2 of 3 val scenes. Claim: "detects stale semantics better than residuals".
- **Pass (ii):** fusion (c + non-oracle depth) has higher AP than non-oracle depth alone, by at least 0.03 [A], on at least 2 of 3 val scenes. Claim: "c adds to depth".
- **Fail:** no November submission. If G2 later lands in cell A, the supervisor may pick a static paper (F-A) for a later window, but only with the reduction to NormLift stated. Otherwise defer.

**G2 grid (Replica val, both scenes).**
- **Epistemic pass:** u_epi AUSE on e_red is below averaging-MSP **and** at most NormLift's 1/(1+N_eff) + 0.005 [A].
- **Calibration pass:** P1 or P2 beats P0 and averaging + temperature on both NLL and ECE.

| | Calibration pass | Calibration fail |
|---|---|---|
| **Epistemic pass** | **A.** "Uncertainty-Aware Open-Vocabulary Gaussian Splatting for Changing Scenes". Claims: change (G3), a calibrated posterior predictive, and two channels (epistemic at parity with NormLift). | **B.** "Detecting and Forgetting Stale Open-Vocabulary Semantics in Gaussian Splatting with View-Level Evidence". No "calibrated" or "lower ECE". C1 is called an interpretation. |
| **Epistemic fail** | **C.** "Calibrated Open-Vocabulary Gaussian Splatting that Forgets Stale Semantics". The epistemic channel is reported, not claimed. | **D.** "Detecting and Forgetting Stale Open-Vocabulary Semantics in Gaussian Splatting". A change paper only. The Replica block shrinks to one parity table. C1 is an interpretation. |

The titles are drafts, and the supervisor picks. **F-B is deleted.** Never submit val-only results. My estimates for later windows (window close minus 5 months) are about Dec 31 for AIM 2027 and about Jan 29 for Humanoids 2027.

## 7 Risks

| Risk | Evidence | Mitigation | Fallback |
|---|---|---|---|
| The change signal is weak on real CLIP | Raw r̄ AUROC 0.59–0.75 (math E9, code). The ẑ channel gives 0.843 on synthetic pairs (centring_check.log). | F3 (ẑ), F4, thresholds from val, sensitivity plot | G3 fail rule (§6) |
| Depth beats c, and the depth-weak stratum is tiny | Frames with top-quartile SemanticChange and bottom-quartile GeometricChange are 0.0–3.4% of seq02 frames per scene. Spearman(SemanticChange, GeometricChange) is 0.46–0.91. Source: `python3 WF/finalize/strata.py WF/data/rio10_stats.txt` → strata.log; the frame counts sum to 34,415. | Headline "c adds to non-oracle depth" through the pre-registered fusion row. Report the stratum, but do not headline it. Non-oracle depth is noisier than the oracle [A], which leaves room for fusion. | Cell B or D wording; G3 pass (ii) |
| No independent stale labels | Instance ID consistency is not checked; the 3RScan map is not checked | Check on val 01 on arrival; 3RScan route | Geometric GT only, stated plainly, with oracle depth excluded from G3 |
| Alignment error near δ | Not measured | δ above the measured p95 | If p95 > 10 cm [A]: object-level AP only |
| u_epi and u_ale tie with NormLift | estim_equiv.log; NormLift Sec. 3.3.2 | Faithful NormLift; the G2 grid treats a tie as parity | Cells B, C or D |
| MaskCLIP is too weak on Replica | Not measured; CSS reports LSeg alone at 25.5 mIoU on room_0 (data report, copied, not re-checked) | W2 check with a pre-set threshold | Mask pooling, then openai weights |
| Replica zip contents unchecked; no Range support | dbx_range.txt | Day-1 `unzip -l`; keep the zip until `unzip -t` passes | Seq_1 offset frames, stated as weaker |
| RIO10 access or size | README l.9, 13-63 | Forms on day 1; HEAD sizes; Range extraction; one scene at a time | Oct 14 / Oct 28 rules |
| GPU and student shared with USGLoc | 13,178/16,303 MiB used at 94% (`nvidia-smi`, 23:14); `sang` holds 3.85 GB; USGLoc's gates relaunch after 5.3 h | K1 gate and lock; nights; weekend slots (D3); peak memory measured in W2; caps | Defer list; N = 4 RIO10 test scenes |
| Disk: 48 GB free | `df -h /` | Budget table (§3) | Ask before pruning usgloc/maps |
| Overlap with USGLoc, including its "temporal conflict model" future-work line (usgloc_ral_draft.tex:471) | venue C | No localization, no 7-Scenes, none of its maps or numbers. A short paragraph states the difference. The Yager row cites E2-BKI. Disclose to the editor. Suggest a rewording to the USGLoc owner (we are read-only there). | none |
| Holidays assumed to be Japanese | Clock UTC+7; JAIST affiliation | Conservative count | +3 days of buffer if they do not apply |

## 8 Decisions for the user

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Date | (a) Nov 27 with the defer list; (b) Nov 15 (0% buffer); (c) a later window (about Dec 31 for AIM, estimate) | (a). Re-decide at the Oct 9 check if more than 2 pd behind. |
| D2 | Change dataset | (a) RIO10, plus the 3RScan form for annotations and Tango depth; (b) Bonn, for a later window only; (c) our own capture (3–10 pd) | (a). No Bonn switch in November. |
| D3 | GPU sharing | (a) night jobs, a shared `flock` lock, weekend slots booked with the USGLoc owner, and one timing slot in W8; (b) fixed daily slots | (a) |
| D4 | Length | (a) 6 pages; (b) 7 pages from the start ($175) | (b) |
| D5 | USGLoc coordination | Settle by G0 (Oct 2): which paper goes first, disclosure, the author list, and the "open-vocabulary" wording at usgloc_ral_draft.tex:72. If USGLoc goes first and is not public: cite it as "anonymous, under review", upload the anonymised PDF as supplementary material (the RA-L rule for this is not checked), and credit copied code "adapted from [anonymous]". | Settle this week; the supervisor decides |
| D6 | Supervisor load | Related work and bibliography (1.5 pd); optionally the figure task (1.0 pd) | Both. This gives the 25% buffer. |
| D7 | Extra forms | 3RScan (fallback annotations and Tango depth); ScanNet (a possible revision-round block) | Send both on day 1 |

## 9 Evidence map

| Fact | Source |
|---|---|
| 6 pages including references, +2 at $175; ieeeconf; double-anonymous review | venue A; https://www.ieee-ras.org/publications/ra-l/information-for-authors-ra-l |
| IROS 2027 window to Apr 30, 2027; final decision within 5 months | venue B; https://www.ieee-ras.org/publications/ra-l/ |
| Workdays: 32 / 41 (JP holidays), 34 / 44 (none) | `python3 WF/finalize/workdays.py` → workdays.log |
| Load: 31.75 pd must-have, 22.6% buffer; 36.0 pd with all defer items, 12.2% | WF/finalize/load3.log |
| Clock +0700; JAIST affiliation | `date +%z`; WF/critic_complete/evidence.txt |
| u_ale (m0 = 0, c0 = 1) equals NormLift: max difference 0.0 | WF/judge/estim_equiv.log |
| NormLift view-level Kish on W_j^(v) = Σ A_ij (Sec. 3.3.2, Eq. 6, 8); "Calibrated by effective multi-view support ... training-free"; limitation (3) on dynamic scenes | WF/reviewer/normlift.txt (curl of https://arxiv.org/html/2609.18898) |
| A_d(κ_post) = 0.826 / 0.909 / 0.962 / 0.981 / 0.996 at n = 1/2/5/10/50 | `python3 WF/reviewer/ad_check.py` → ad_check.log |
| Old F3 breaks zero spread (r̄_c 0.451/0.586/0.737); AUROC raw 0.807, old F3 0.739, renorm 0.843 | WF/reviewer/centring_check.log |
| μ* is the weighted mean; κ has no effect | WF/code/formula_checks.log |
| Per-pixel cache 39.3 MB per frame, 56.6 GB for mapping, 228 GB for RIO10; token grid 2.65 GB; RIO10 CPU 2.9 h | `python3 WF/critic_complete/sizes.py` → sizes.log |
| Map 0.236 GB at 1M; field 0.26 GB per channel; RIO10 token 2.09 MB per frame; unzip peak 15.55 GB [A] | `python3 WF/finalize/budget.py` → budget.log |
| RIO10 has only rendered depth; zip layout; labels.ply, instances | WF/judge/rio10_readme.md l.13-63 |
| 3RScan rigid/removed/nonrigid annotations | WF/data/3rscan_README.md:108-110 |
| SemanticChange vs GeometricChange strata; 34,415 seq02 frames | `python3 WF/finalize/strata.py WF/data/rio10_stats.txt` → strata.log |
| AUSE breaks ties by sort order | usgloc/usgloc/metrics.py:100-101 |
| Prompt template "a photo of a {}." | usplat/clip_dense.py:120 |
| Figures still draw EFE, localization and decay | papers/figs/pipeline.tex:92-100; uncertainty_plane.tex:57 (grep in WF/critic_complete/evidence.txt) |
| MASt3R checkpoint 2.6 GB; USGLoc MASt3R used 6,866 MiB | `du -sh third_party/mast3r_ckpt`; venue §D |
| Hardware: i5-14600KF, 20 threads, 31 GiB, RTX 5070 Ti, driver 595.84 | `lscpu`, `free -g`, `nvidia-smi` (2026-09-28 23:14) |
| GPU 13,178/16,303 MiB at 94%; 48 GB disk free | `nvidia-smi` 23:14:37; `df -h /` |
| MaskClipEncoder, FastSAM-s.pt | usgloc/usgloc/mapping.py:560; `ls usgloc/FastSAM-s.pt` |
| Geometry loaded once | scripts/run_mapping.py:114 |
| USGLoc "temporal conflict model" | usgloc/paper/usgloc_ral_draft.tex:471 |
| Replica zip 6,762,780,327 B; Range ignored; 10.3 MB/s | WF/judge/dbx_head.txt, dbx_range.txt |
| Earlier rows (math E7–E12, code risks, sota, data, venue) | as in the synth plan's evidence map (WF/synth/plan.md §9) |

## 10 Changes after critique

**Completeness critic**
1. **Circular stale GT (critical)** → The stale GT now comes from object annotations (RIO10 instances and labels.ply, then 3RScan, then geometric only, stated plainly). The rendered-depth residual is labelled oracle. A non-oracle depth residual is added (Tango or MASt3R). G3 uses non-oracle rows only. GT agreement is reported on val 01. F5 re-admission uses non-oracle depth only.
2. **No mechanism for lower ECE/NLL (critical)** → F6 now has P1 (a posterior mean-field predictive using A_d(κ̂·n·r̄)) and P2 (β(u_epi, u_ale)), plus the rows "Ours P0, no uncertainty" and "averaging + temperature". "Lower ECE" leaves the abstract unless the G2 calibration test passes (cells B and D).
3. **Feature cache does not fit on disk** → The cache holds the ViT token grid (2.65 GB for Replica). A disk budget table is in §3. RIO10 zips get a HEAD check and Range extraction.
4. **AUSE targets and per-row uncertainty undefined** → e_red and e_irr are defined. MSP/entropy AUSE is computed for every row. G2 requires u_epi to beat averaging-MSP on e_red. AUSE is tie-aware and unit-tested (K10a).
5. **No rule for unobserved content** → The coverage rule is fixed before any val run: Σα < τ_cov counts as wrong, u = 1 for rows without uncertainty, plus a covered-only variant. RIO10 gets an "unobserved" class next to "hidden", both left out of AP.
6. **Missing change baselines** → Added: the feature residual with a matched temporal filter, the pre-registered c + non-oracle depth fusion row, and a Yager conflict row. 3DGS-CD on val 01 is defer item 2. If it does not run, the paper says why.
7. **Weekly load too high; pd incomplete** → The plan is rebuilt week by week with JP holidays and 80% capacity per week. It includes running and analysis pd, K14 moves to W2, and the date is Nov 27. The buffer is 22.6%, not 25–30%. It reaches 25% only if the supervisor takes the figure task (D6). The deferrals are named and ordered: optimised field, 3DGS-CD, ablations (e)/(g), K5, RIO10 risk-coverage, MC check.
8. **G2 fallback contradicts itself** → F-B is deleted. G2 is a 2×2 grid (epistemic × calibration), each cell with a title and claim. When both fail, the paper stands on G3 as a change paper (cell D). If G3 fails, it is deferred.
9. **No fair ablation of the main C2 claim** → "Ours-scalar" (κ* or 1 − A_d(κ*)) is added to Tab III (h) and to the Fig. views cross-AUSE.
10. **No early feature-quality check** → The 2D MaskCLIP CPU check is in W2 (K8): 20 office_0 frames, per-pixel vs FastSAM pooling, pass at ≥ 15 mIoU [A], with fallbacks fixed. Openai vs laion2b is compared only if the openai weights are downloaded (≈ 600 MB, not checked).
11. **Vocabulary and zip contents undefined** → A frozen vocabulary file (class → prompt, ignore list, merges, templates) is made in K8 on val. `unzip -l` runs on day 1, and G0 requires Seq_1, Seq_2 and class names for all 8 scenes.
12. **Cross-scan alignment unchecked** → The seq01-vs-seq02 static-surface distance (median, p95) is measured on val 01, δ is set above p95, and the check has its own verification row.
13. **Semantic evidence is synthetic only** → A RIO10 semantic block is added: mIoU, ECE and AUSE on the second half of seq02, changed vs unchanged regions, before and after consolidation, with a class map in K12. If the class map fails: "synthetic-only calibration" goes in Limitations.
14. **Bonn fallback has no protocol** → The protocol is written in §6 (1.5 pd), but Bonn is **not** a November switch. The pre-decided rules are: swap tracks on Oct 14; no November submission if there is no RIO10 by Oct 28.
15. **GPU peak memory unmeasured** → Peak memory is measured on office_0 in the first W2 night slot (K9). Maps are capped at 1M Gaussians and ≤ 640×480 pixels [A]. Weekend slots are booked with the USGLoc owner (D3).
16. **Timing and hardware incomplete** → The timing condition is "no other job, `sang` resident", in a booked slot. Rows are added for extraction (CPU and GPU), fit, field training, CSS and NormLift. The hardware line is measured.
17. **Thresholds without a val split** → A threshold table names the val set for each threshold. The c threshold and EWMA rates are swept. PCA and z̄ are fit on the mapping frames only. The Replica static row uses RIO10-val thresholds.
18. **Page and figure plan too light** → The plan is 7 pages from the start, with 0.8 page of references. The pipeline and plane figures are redrawn (inside the 1.0 pd figure task).
19. **Test-once and reproducibility gaps** → At G4 the list of test runs is frozen (Tab I, Tab II and its semantic block, static row, Tab III and Fig. views on test; sensitivity and timing on val). The maps of the two G5 scenes are kept, seeds are fixed, and G5 tolerances are set [A].
20. **Smaller evaluation gaps** → Gaussians are binned by the mean SemanticChange of the frames that observe them. RIO10 CPU time is re-estimated from one measured frame. The photometric residual is NCC.
21. **Claims and admin** → The problem is reworded to "no view-level epistemic signal ... and no handling of stale semantics". NormLift's R gets a unit test against Eq. 6/8. The paper states that both rows are re-implementations and that CSS gets the vocabulary at mapping time. Keywords, PINs, sign-off and the anonymous mirror are in W9.

**Hostile reviewer**
22. **R1: weakened NormLift (critical)** → NormLift is implemented as defined (F11), with its N_eff as the epistemic score and a visible-view-count row. The paper states the reduction to R(j) on static scenes and cites NormLift as concurrent. F-B is deleted. C4 is the main claim, G3 is the go/no-go, G2 is a parity check. The "left to prove" line is rewritten (§1).
23. **R2: circular change test with no semantic outcome (critical)** → Object-level GT (see line 1). The headline is either "c beats non-oracle residuals" or "c + depth beats depth". Strata are reported (hidden, and high SemanticChange with low GeometricChange), but this stratum covers only 0–3.4% of frames (strata.log), so it is not headlined. A semantic outcome is added (the RIO10 semantic block). If no independent labels exist by G3, the paper says "geometric change only".
24. **R3: posterior unused in calibration (critical)** → P1/P2 predictives, Tab III (i) with P0/P1/P2 on ECE, NLL and Brier, reliability diagrams and AURC. If P1/P2 do not beat plug-in + temperature on val, "calibrated" is removed and C1 becomes an interpretation (cells B and D).
25. **R4: F3 math flaw** → F3 is replaced by the ẑ channel (+0.26 GB at 1M, d = 64). The unit test requires r̄ = 1 for identical observations. G1 is re-run with this variant (K5, defer item 4). The F3 text is rewritten.
26. **R5: missing change baselines** → The feature-residual row with a temporal filter, and Tab III (d) "c vs one-shot feature residual". The 3DGS-CD try is defer item 2, and the paper explains if it is missing.
27. **R6: synthetic-only; all baselines re-implemented** → Option (b): the RIO10 semantic block. Option (a), ScanNet, is **not done** in November; the form is sent for a revision round. The averaging row is checked against the official GradBP code. The published-numbers footnote is restored with its caveat. **Left open:** NormLift and CSS have no public code, so they stay re-implementations with a unit test only.
28. **R7: idealised protocol, no robot use** → A pose-noise sweep (1/2/5 cm, 0.5/1/2°) is added to the sensitivity figure. **Insertion is not done.** The scope is retitled to "detect and forget stale semantics" (titles B and D), and this goes in Limitations. A selective query with abstention is added (AURC in Tab I; RIO10 risk-coverage is defer item 5). **Left open:** there is no robot experiment.
29. **R8: USGLoc overlap details** → The order is settled at G0 (D5). An anonymous citation and an anonymised supplementary PDF are planned; the RA-L rule for the PDF is **not checked**. A paragraph states the difference, including USGLoc's "temporal conflict model" line. The Yager row is cited to E2-BKI. Copied code is credited "adapted from [anonymous]".