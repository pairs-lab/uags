# UAGS related-work threat and bibliography audit (label: sota, 2026-09-28)

**Bottom line:** 1 of 4 contributions is scooped: the gradient-based accumulation (C3). The other three are partly new but must be narrowed. The paper also misses 3 must-cite papers (LatentBKI, Gradient-Weighted Feature Back-Projection, NormLift). 7 bib entries are wrong or out of date.

I edited no project files. Scratch evidence is in `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/wf/sota/`:
- `meta1..12.txt`: arXiv metadata, made with `python3 absmeta.py <ids>` / `python3 absmeta2.py <ids>`, which scrape `https://arxiv.org/abs/<id>`.
- `cr1..5.txt`: venue data, made with `python3 cr.py <title|DOI>`, which calls `api.crossref.org`.
- `s2cit1-2.txt`: citing papers, made with `python3 s2cit.py`, which calls the Semantic Scholar API.
- `gradbp_backproject.py`: prior-art code, fetched with `curl https://raw.githubusercontent.com/JojiJoseph/3dgs-gradient-backprojection/main/backproject.py`.

Contributions as numbered in `papers/main.tex:113-130`:
- **C1** vMF posterior per Gaussian with a BKI update.
- **C2** epistemic/aleatoric split with a small-sample correction.
- **C3** gradient-based accumulation.
- **C4** uncertainty-driven consolidation.
- **C5** the optional EFE view selection.

## 1. Threat table

Numbers in this table come from WebFetch summaries of the linked html pages. I did not check them against the PDF tables.

| Paper | Venue, arXiv | What it does | C1 | C2 | C3 | C4 | Code | Datasets / metrics |
|---|---|---|---|---|---|---|---|---|
| **Gradient-Weighted Feature Back-Projection** (Joseph, Amrutur, Bhatnagar) | SIGGRAPH Asia 2025 Conf. Papers pp.1-12, DOI 10.1145/3757377.3763926; arXiv 2411.15193 | Training-free lifting of CLIP/DINO features to 3DGS. Uses ∂C/∂c_k = α_k T_k (their Eq. 3). Feature = Σ F αT / Σ αT. Paper: "after Euclidean normalization on f_k, both equations become equivalent". | partial: same mean direction | none | **overlap (scoops C3)** | none | github.com/JojiJoseph/3dgs-gradient-backprojection | 3D-OVS, LERF-Mask, affordance; mIoU, recall, time |
| **LatentBKI** (Wilson, Xu, Sun, Ewen, Zhu, Barton, Ghaffari) | IEEE RA-L 10(4):3102-3109, 2025, DOI 10.1109/LRA.2025.3539091; arXiv 2410.11783 | Open-dictionary BKI over VLM latent features. Gaussian likelihood with Normal-Inverse-Wishart conjugate, on voxels. Uncertainty from covariance eigenvalues. | **overlap** (open-vocab continuous BKI already done; the model is Gaussian, not vMF) | partial | none | none | github.com/UMich-CURLY/LatentBKI | SemanticKITTI (mIoU 61.54), Matterport3D (mIoU 14.18) [html](https://arxiv.org/html/2410.11783) |
| **NormLift** (Zang, Li, Engel, Park, Viola) | arXiv 2609.18898, 2026-09-16 (concurrent) | Casts per-Gaussian lifting as cosine alignment on the CLIP unit sphere. The closed form is the L2-normalized weighted mean. Reliability R = ‖f‖·N_eff/(N_eff+β), with Kish N_eff and β=1. Splits the norm into intra-view × inter-view consistency. Deterministic, no Bayes, no epistemic/aleatoric wording. | partial | **overlap** (r̄ with small-sample shrinkage ≈ our r̃) | partial (αT weights) | none: static only, per their limitations | not given | ScanNet 10 scenes, OpenGaussian protocol; mIoU 35.77 vs LUDVIG 33.90 (their Tab. 1, 19 classes) [html](https://arxiv.org/html/2609.18898) |
| **CSS** (Wilson et al.) | ICRA 2025 pp.3284-3290 | Dirichlet posterior per Gaussian. BKI kernel = αΠ(1-α), the same kernel we use. Two uncertainty measures: variance for "few observations" and expectation for "conflicting categories". | partial (closed set) | **partial**: the same split in closed-set form, not named epistemic/aleatoric | none | none | not given | KITTI seq. 15, Replica room_0; mIoU, sparsification |
| **LAMP** (Lee, Yu, Kim, Choi) | IEEE RA-L 10(12):12365-12372, Dec 2025 | Implicit language field whose network outputs vMF (μ, κ). κ has a Gamma prior. Trained by negative log-posterior. Models "aleatoric uncertainty" only. | partial: vMF on CLIP, but implicit, learned κ, no conjugate update | partial | none | none | project page only | Isaac Sim, real 28-floor building; SR, SPL |
| **SLIM-VDB** (Sheppard, Ewen, Wilson, ...) | IEEE RA-L 11(3):2386-2393, 2026; arXiv 2512.12945 | Open-set Bayesian fusion: Normal-Inverse-Gamma per CLIP dimension, per voxel. | partial | partial | none | none | github.com/umfieldrobotics/slim-vdb | SceneNet, SemanticKITTI; mIoU, FPS, memory |
| **Occam's LGS** (Cheng, Zaech, Van Gool, Paudel) | BMVC 2025 (per repo tag and bmva-archive); arXiv 2412.01807 | Maximum-likelihood mean feature under Gaussian noise = αT-weighted average. No uncertainty. | partial (same mean) | none | partial | none | github.com/insait-institute/OccamLGS | LERF, 3D-OVS; mIoU |
| **LUDVIG** (Marrie et al.) | ICCV 2025 pp.7440-7450; arXiv 2410.14462 | Learning-free weighted uplifting of features plus graph diffusion | partial (same mean) | none | partial | none | github.com/naver/ludvig | not checked |
| **LEGaussians** (Shi, Wang, Duan, Guan) | CVPR 2024 pp.5333-5343 | Learnable per-Gaussian semantic uncertainty u∈[0,1] (their Sec. 3.4.2, Eq. 10-11) | none | partial: contradicts our "no uncertainty" claim at `main.tex:150-152` | none | none | not checked | not checked |
| **Bayesian Fields** (Maggio, Carlone) | arXiv 2503.05949 (no venue found) | Recursive Bayes on a per-task relevance probability; tasks fixed at mapping time; static scene | partial | none | none | none | github.com/MIT-SPARK/Bayesian-Fields | Clio (Cubicle, Apartment), LeRF; strict/relaxed-osR, IoU |
| **E2-BKI** (Kim, Jeon, Min, Kwak, Seo) | IEEE RA-L 11(3):2378-2385, 2026 | Evidential BKI with ellipsoidal kernels on Gaussian primitives. Closed set, outdoor. | partial | partial (EDL / Dempster-Shafer) | none | none | e2-bki.github.io | RELLIS-3D (47.3 mIoU), OffRoad, KITTI-360 (Brier) |
| **PhysGS** (Chopra, Liang, Seneviratne, Manocha) | CVPR 2026 pp.18980-18990 | Bayesian 3DGS for physical properties. Dirichlet plus NIG. Var = E[σ²] (aleatoric) + Var[μ] (epistemic). VLM (GPT-4V) priors, no CLIP. | none | **partial**: explicit split, but not semantic embeddings | none | none | project page only | ABO-500, indoor, outdoor |
| **MIF** (Jiang, Liu, Jin, Wang, Li) | RSS 2026; arXiv 2605.21935 | Confidence-aware semantic 3DGS with distilled CLIP/DINO features. A discrepancy score (includes 1−cos of features) triggers local memory updates in dynamic scenes. | none | none | none | **partial**: semantic discrepancy drives the update; relocation success in non-static scenes 12% to 94% | ziya-jiang.github.io/MIF-homepage | real office, Unitree G1; success rates |
| **Chu & Kim** | IEEE RA-L 11(3):3852-3859, Mar 2026, DOI 10.1109/LRA.2025.3643336 | Semantic 3DGS map (OpenGaussian). An LLM picks "robust" objects. Coarse-to-fine render-and-match localization under scene change. | none | none | none | partial (discrete LLM gating) | none found | Abstract (Semantic Scholar API) gives no numbers. Full text is paywalled (IEEE Xplore returned HTTP 420; Semantic Scholar says CLOSED). **Numbers not checked.** Crossref references include 7-Scenes (10.1109/ISMAR.2013.6671777) and ScanNet (10.1109/CVPR.2017.261). USGLoc notes say 7-Scenes Tab. 1-2, ScanNet Tab. 3-4, custom rearranged set Tab. 5-6, baselines Marepo/ACE/PNeRFLoc/CrossFire/GS-CPR/STDLoc/iComMa/SplatLoc (copied from `usgloc/docs/sota_gapfill_2026-06.md:65`, not re-checked). |
| 3DGS-CD (Lu, Ye, Leonard) | IEEE RA-L 10(3):2662-2669, 2025 | Render-and-compare plus EfficientSAM to find rearranged objects | none | none | none | partial | github.com/520xyxyzq/3DGS-CD | own 5 real scenes + NeRF-CD; P/R/F1/IoU |
| CL-Splats (Ackermann et al.) | ICCV 2025 pp.7808-7817 | DINOv2 render-vs-real change masks, then local re-optimization | none | none | none | partial | cl-splats.github.io | CL-Splats set, CL-NeRF set; PSNR/SSIM/LPIPS |
| GaussianUpdate (Zeng et al.) | ICCV 2025 pp.25800-25809 | Continual 3DGS update with generative replay | none | none | none | partial | zju3dv.github.io/GaussianUpdate | not checked |
| GaME (Yugay, ..., Carlone, ..., Schmid) | arXiv 2506.06909 v3 | Removes stale Gaussians by photometric/geometric conflict plus SAM masks | none | none | none | partial | vladimiryugay.github.io/game | Flat, Aria, TUM; PSNR, depth L1 |
| VG-Mapping (He, Yu, Chen, Zhang) | IEEE RA-L 11(10):11561-11568, 2026; arXiv 2510.09962 | TSDF-based change regions drive Gaussian insert/delete in semi-static scenes | none | none | none | partial | github.com/heyicheng-never/VG-Mapping | VG-Scene (6 synthetic + 3 real pre/post-change sequences) |
| DualMap (Jiang et al.) | IEEE RA-L 10(12):12612-12619, 2025 | Open-vocabulary object map with an object-level status check in changing scenes | none | none | none | partial | eku127.github.io/DualMap | not checked |
| Also seen, weaker overlap | — | GS-LTS (2503.17733); Gaussian Difference (ICASSP 2025); GD-DIFF (2605.07203); online change detection (CVPR 2026, 2511.12370); LTGS; ChronoGS; CrossMaps; VLMM (2607.16173); VALA (3DV 2026 pp.1791-1800); NRGS (2604.22439) | — | — | — | partial | — | not checked |
| **VISTA** (Nagami et al.) | IEEE RA-L 11(3):3150-3157, 2026 | Receding-horizon view selection, G = Σγ(c·G_I + G_S), with a hand-set, decaying weight c. G_S is similarity to the query, not an information gain. No epistemic/aleatoric split. | C5: partial | | | | "released upon acceptance" | 6 static scenes; PSNR/SSIM/LPIPS; hardware success rate |
| **ActiveSGM** (Chen, Zhan, Yin, Xu, Mordohai) | NeurIPS 38 pp.37293-37320, 2025; arXiv 2506.00225 | 3DGS active semantic mapping. Closed-set semantic entropy × geometry. No epistemic/aleatoric split. | C5: partial | | | | github.com/lly00412/ActiveSGM | Replica, MP3D; mIoU 84.89 (Replica) |
| HELIOS, GauSS-MI, OUGS, NBV for semantic/dynamic 3DGS (2512.22771), GAVIS (CVPR 2026) | various | Information gain from Dirichlet consistency, Shannon MI, or Fisher information | C5: partial | | | | | not checked |

## 2. Verdict per contribution

| # | Verdict | Evidence and required narrowing |
|---|---|---|
| C1 vMF posterior + BKI | **Partially novel** | Searches found no paper with a conjugate vMF (directional) BKI update on 3DGS primitives (queries in `axs1.txt`, `axs2.txt`); "not found" is not proof. However, "open-dictionary continuous BKI" is already LatentBKI (voxels, NIW) and SLIM-VDB (NIG). With κ0=0 our posterior mean direction μ* equals the normalized Gradient-Weighted Back-Projection / Occam's LGS / NormLift feature. The vMF conjugacy itself is textbook (Mardia & El-Atoum 1976). The new part is only the directional conjugate model with a text prior (κ0 μ0) on splat primitives. Change "the continuous generalization" (abstract, `main.tex:120`) to "a directional generalization". `main.tex:104-105` must add that LatentBKI already realized the continuous conjugate prior, on voxels. |
| C2 epistemic/aleatoric split | **Partially novel (close concurrent overlap)** | CSS already separates "few observations" (variance) from "conflicting categories" (expectation) in the closed-set case. NormLift (2026-09-16) uses ‖f‖ (= r̄) with shrinkage N_eff/(N_eff+β) as reliability for CLIP on 3DGS, which is nearly our r̃. PhysGS names the aleatoric/epistemic split for Bayesian 3DGS. LAMP uses vMF κ as aleatoric. Still new: an explicit two-axis (n_eff, r̄) epistemic/aleatoric read-out from one vMF posterior for open-vocabulary 3DGS. NormLift must be cited as concurrent. |
| C3 gradient trick | **Scooped** | The Gradient-Weighted Back-Projection code does exactly this with gsplat: render a zero buffer, backward of Σ render·feats gives Σ w z (`backproject.py:115-130`); a second render of a zero buffer, backward of Σ gives Σ w, the denominator (`:133-150`). Public since at least Jan 2025 (GitHub commit 2025-01-21) and in the arXiv v1 of Nov 2024. Our only delta is fusing both into one pass with a ones-channel (`usplat/bki_accumulate.py:13-42`), which is trivial. Demote this to an implementation detail that cites that paper. |
| C4 consolidation | **Partially novel** | I found no work that uses high-n_eff / low-r̄ CLIP disagreement to decay evidence in 3DGS. Neighbours: MIF (semantic cosine discrepancy triggers updates, RSS 2026), 3DGS-CD, CL-Splats, GaME, VG-Mapping (photometric or geometric change), Chu & Kim (LLM gating). Evidence decay itself is standard. Main risk: r̄ is also low for *static* ambiguity (boundaries, mixed pixels, view-dependent CLIP; NormLift's intra-view term, VALA's "view-specific noise"). A reviewer will ask how you tell moved objects from ambiguous static ones. Also, `usplat/consolidation.py:40-41` scales R and n_eff by the same λ, so r̄ never changes under decay. Only n_eff falls, until it drops below `min_obs`. |
| C5 EFE (optional) | **Minor / partial** | VISTA, ActiveSGM, HELIOS, GauSS-MI and FisherRF already do semantic or geometric information-gain view selection. The EFE framing plus the aleatoric gate is new wording rather than a new mechanism. Cite ActiveSGM. |

Text fixes in `main.tex`:
- **Line 166:** PhysGS is CVPR 2026, not "concurrent".
- **Line 175:** LAMP was published in RA-L Dec 2025, not "concurrent".
- **Lines 150-152:** "provide no uncertainty" is contradicted by LEGaussians.
- **Line 185:** VISTA's semantic term is query similarity, not an information gain.

## 3. Must-cite additions (bib-ready)

- **LatentBKI:** J. Wilson, R. Xu, Y. Sun, P. Ewen, M. Zhu, K. Barton, M. Ghaffari, "LatentBKI: Open-dictionary continuous mapping in visual-language latent spaces with quantifiable uncertainty," IEEE RA-L, vol. 10, no. 4, pp. 3102-3109, 2025, doi:10.1109/LRA.2025.3539091.
- **Gradient-Weighted Feature Back-Projection:** J. Joseph, B. Amrutur, S. Bhatnagar, "Gradient-weighted feature back-projection: A fast alternative to feature distillation in 3D Gaussian splatting," in Proc. SIGGRAPH Asia 2025 Conf. Papers, pp. 1-12, doi:10.1145/3757377.3763926.
- **NormLift:** Y. Zang, D. Li, D. Engel, S. Park, I. Viola, "NormLift: From lifted features to semantic reliability in 3D Gaussian splatting," arXiv:2609.18898, 2026.
- **Mardia & El-Atoum:** K. V. Mardia and S. A. M. El-Atoum, "Bayesian inference for the von Mises-Fisher distribution," Biometrika, vol. 63, no. 1, pp. 203-206, 1976, doi:10.1093/biomet/63.1.203. This is the correct source for the conjugacy statement at `main.tex:229`.
- **Occam's LGS:** J. Cheng, J.-N. Zaech, L. Van Gool, D. P. Paudel, "Occam's LGS: An efficient approach for language Gaussian splatting," BMVC 2025 (pages not checked), arXiv:2412.01807.
- **LEGaussians:** J.-C. Shi, M. Wang, H.-B. Duan, S.-H. Guan, "Language embedded 3D Gaussians for open-vocabulary scene understanding," CVPR 2024, pp. 5333-5343.
- **SLIM-VDB:** A. Sheppard et al., "SLIM-VDB: A real-time 3D probabilistic semantic mapping framework," IEEE RA-L, vol. 11, no. 3, pp. 2386-2393, 2026, doi:10.1109/LRA.2026.3652875.
- **MIF:** P. Jiang, H. Liu, J. Jin, W. Wang, X. Li, "Learning to evolve: Multi-modal interactive fields for robust humanoid navigation in dynamic environments," RSS 2026, arXiv:2605.21935.
- **ActiveSGM:** L. Chen, H. Zhan, H. Yin, Y. Xu, P. Mordohai, "Understanding while exploring: Semantics-driven active mapping," NeurIPS 38, pp. 37293-37320, 2025.
- **3DGS-CD:** Z. Lu, J. Ye, J. Leonard, IEEE RA-L, vol. 10, no. 3, pp. 2662-2669, 2025, doi:10.1109/LRA.2025.3533457.
- **CL-Splats:** J. Ackermann et al., ICCV 2025, pp. 7808-7817.
- **GaussianUpdate:** L. Zeng et al., ICCV 2025, pp. 25800-25809.
- **VG-Mapping:** Y. He, J. Yu, G. Chen, H. Zhang, IEEE RA-L, vol. 11, no. 10, pp. 11561-11568, 2026, doi:10.1109/LRA.2026.3723741.
- **Optional:** LUDVIG (ICCV 2025, pp. 7440-7450); DualMap (IEEE RA-L, vol. 10, no. 12, pp. 12612-12619, 2025); GaME (arXiv:2506.06909).

For the "custom dynamic dataset" TODO, these public pre/post-change sets exist: VG-Scene, the 3DGS-CD set, the CL-Splats set, and GaME Flat/Aria. I did not check their licenses or download sizes.

## 4. Bibliography verification table

| Key | Status | Exact finding | Source |
|---|---|---|---|
| kerbl2023 | OK | TOG 42(4), pp. 1-14. Article no. 139 not re-checked; Crossref lists no article number. | api.crossref.org/works/10.1145/3592433 |
| mildenhall2020nerf | OK, incomplete | Add LNCS pp. 405-421 | Crossref 10.1007/978-3-030-58452-8_24 |
| radford2021 | OK | PMLR 139, pp. 8748-8763 | proceedings.mlr.press/v139/radford21a.html |
| wilson2025css | OK, incomplete | All 9 authors match. Add pp. 3284-3290, doi:10.1109/ICRA55743.2025.11128341 | Crossref; arxiv.org/abs/2411.02547 |
| vegabrown2014 | OK | NeurIPS vol. 27, 2014. The official bibtex has an empty pages field. | proceedings.neurips.cc ...a99cb6ad...-Bibtex.bib |
| gan2020bayesian | OK | RA-L 5(2):790-797 | Crossref 10.1109/LRA.2020.2965390 |
| kim2025e2bki | **MISMATCH** | Not arXiv 2025: published IEEE RA-L vol. 11, no. 3, pp. 2378-2385, Mar. 2026, doi:10.1109/LRA.2026.3653367. Authors correct. Title uses "Uncertainty-Aware". | Crossref; arxiv.org/abs/2509.11964 |
| liu2025physgs | **MISMATCH** | Authors missing and key wrong. Should be S. Chopra, J. Liang, G. Seneviratne, D. Manocha, CVPR 2026, pp. 18980-18990. | CVF CVPR2026 html bibtex; arxiv.org/abs/2511.18570 |
| maggio2025bayesian | OK | arXiv:2503.05949 (v1 2025-03-07). No venue found on arXiv, Crossref or web search. | arxiv.org/abs/2503.05949 |
| lee2026lamp | **MISMATCH** | Published IEEE RA-L vol. 10, no. 12, pp. 12365-12372, Dec. 2025, doi:10.1109/LRA.2025.3619820. The arXiv v1 (Feb 2026) came after publication. Authors correct. | Crossref; arXiv jref |
| nagami2025vista | **MISMATCH (outdated)** | Now IEEE RA-L vol. 11, no. 3, pp. 3150-3157, Mar. 2026, doi:10.1109/LRA.2026.3653276. All 8 authors match. | Crossref |
| mardia2000directional | OK | Wiley 2000; Crossref online date 1999-01-03 | Crossref 10.1002/9780470316979 |
| banerjee2005vmf | OK | JMLR 6:1345-1382 | jmlr.org/papers/v6/banerjee05a.html |
| friston2010 | OK | 11(2):127-138 | Crossref 10.1038/nrn2787 |
| parr2022active | OK | MIT Press 2022 | Crossref 10.7551/mitpress/12441.001.0001 |
| ye2024gsplat | **MISMATCH (outdated)** | Published JMLR vol. 26, no. 34, pp. 1-17, 2025. 11 authors, first is Vickie Ye. | jmlr.org/papers/v26/24-1476.html |
| qin2024langsplat | OK, incomplete | Add pp. 20051-20060 | CVF CVPR2024 bibtex |
| zhou2024feature | OK, incomplete | First author Shijie Zhou is correct (10 authors, last A. Kadambi). Add pp. 21676-21685. | CVF CVPR2024 bibtex |
| wu2024opengaussian | OK | NeurIPS 37, pp. 19114-19138 | NeurIPS 2024 bibtex |
| chen2024marepo | **Minor mismatch** | Official title is "...for Visual Re-Localization". Pages 20665-20674 are correct. | CVF CVPR2024 bibtex |
| brachmann2023ace | OK | pp. 5044-5053 | CVF CVPR2023 bibtex |
| chu2026semantic | OK | vol. 11, no. 3, pp. 3852-3859, Mar. 2026. Add doi:10.1109/LRA.2025.3643336. | Crossref |
| glocker2013 | OK | ISMAR 2013, pp. 173-179 | Crossref 10.1109/ISMAR.2013.6671777 |
| dai2017scannet | OK (CVF pages) | CVF gives pp. 5828-5839. IEEE Xplore pagination is pp. 2432-2443 (doi:10.1109/CVPR.2017.261); pick one. | CVF html; Crossref |

## 5. Not checked / limits
- The number of rearranged objects in Chu & Kim's dataset is not checked, and neither are their result numbers (paywalled).
- Venues not found for Bayesian Fields, GaME and GS-LTS. The 3DV venue for 2512.22771 comes only from the Semantic Scholar venue field.
- The arXiv API returned HTTP 406 and DBLP reset the connection, so I used arXiv abs pages, Crossref and Semantic Scholar instead.
- No GPU use and no downloads larger than 50 MB.