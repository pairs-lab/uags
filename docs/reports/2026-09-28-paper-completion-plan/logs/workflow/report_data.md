# UAGS data and baseline design report (label: data)

Date: 2026-09-28. All work was read-only. Scratch files and evidence are in `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/wf/data/` (logs are in `logs/`). I ran no GPU test. At the time, `nvidia-smi` showed 13,668 of 16,303 MiB in use at 89% utilisation (`logs/nvidia_smi.txt`).

## Key findings

1. **No single dataset covers all three needs** (open-vocabulary semantic ground truth, scene-change pairs, ground-truth poses) except RIO10/3RScan and ReplicaCAD. RIO10 and 3RScan are real data from a Tango phone camera. ReplicaCAD is synthetic and needs habitat-sim. The minimal credible set is:
   - Replica (Semantic-NeRF renders) for Table seg and ablations (i), (ii) and (iv).
   - RIO10 for Table loc, scene-change robustness (2) and ablation (iii).
   - 7-Scenes, already on disk, as a static sanity block.
2. **RIO10 gives a strong, ready-made story.** ACE collapses under scene change on RIO10 validation: average median error 358.4 cm / 58.7°. ACE-G gets 37.0 cm / 12.1° and Reloc3r 47 cm / 9.4° (ACE-G, ICCV 2025, arXiv 2510.11605, Table 3).
3. **The three closest baselines have no runnable code.**
   - CSS: no URL in the paper, and a GitHub search returned 0 repos.
   - Chu & Kim: the paper is paywalled and GitHub search returned 0 repos.
   - Bayesian Fields: the repo holds only a LICENSE file of 1,298 bytes (`logs/bayesian_fields_repo_contents.txt`).
   - So CSS must be reimplemented, and that reimplementation is the same thing as ablation (i). The code has no categorical path; the only mention is a docstring at `usplat/vmf_field.py:4`.
4. **Chu & Kim datasets.** All 25 references are deposited in Crossref. The only dataset papers among them are 7-Scenes (ISMAR 2013) and ScanNet (CVPR 2017). There is no 3RScan, RIO10, TUM or Bonn reference (`logs/chu_kim_crossref_refs.txt`, `logs/chu_kim_refs_resolved.txt`). Their "dynamic" data is probably their own and unreleased (not checked).
5. **The planned Table loc columns do not fit RIO10.** They are "ATE", 5cm/5° and 2cm/2°, but median errors on RIO10 are tens of cm. Use the RIO10 metrics instead: DCRE(0.05), DCRE(0.15), Pose(5 cm, 5°), Outlier(0.5), plus median t/r. Also rename "ATE" to "median translation error" (ATE is a trajectory metric).
6. **Two open-vocabulary baselines do not fit a shared 16 GB GPU.** LangSplat recommends 24 GB of VRAM (README). OpenGaussian used 16–20 GB (paper). The fair runnable baseline is a deterministic feature field built in our own gsplat pipeline with the same MaskCLIP features. Published ScanNet numbers can be cited as context only.

## 1. Dataset table

Fit ratings: G = good, P = partial, N = no.

| Dataset | Semantic GT | Change pairs | Poses / depth | Size (source) | Access / licence | Standard protocol and metrics in prior work | Fit (1) OV-seg + calib | Fit (2) map vs change | Fit (3) loc under change |
|---|---|---|---|---|---|---|---|---|---|
| **Replica, Semantic-NeRF renders** | Yes. 88 classes from mesh annotations via Habitat; remapped to NYUv2-13 for fusion (arXiv 2103.15875) | No (static) | Exact poses and depth; 640×480, 90° HFOV; 2 sequences × 900 frames per scene | Not checked (Dropbox) | Immediate. Replica Research Terms, research only (`logs/github_licence_text.txt`) | SplatLoc: 8 scenes, localize seq2 against seq1 (arXiv 2409.14067) | **G** | N | P (static sanity only) |
| Replica, Feature 3DGS split | LSeg-derived labels, "manually re-label some pixels" following NeRF-DFF (arXiv 2312.03203) | No | Yes | Not checked (Google Drive) | Immediate | room_0, room_1, office_3, office_4; 80 images each, every 8th image from the 3rd is test (10 test images); mIoU and accuracy (Table 2) | P (labels are not human GT) | N | N |
| Replica, CSS split | Yes, 28 classes | No | Yes | Not released | None | room_0, 405 images, manual trajectory; mIoU, accuracy, sparsification curves (arXiv 2411.02547, Table IV) | Not reproducible | N | N |
| Replica, NICE-SLAM + raw mesh | Raw mesh has class and instance labels; the NICE-SLAM zip has no 2D labels | No | 2000 frames, 1200×680 | NICE-SLAM zip 12.44 GB (HEAD request); raw Replica v1 33.86 GB (GitHub release API, `logs/replica_release_sizes.txt`); OpenScene replica_3d 103 MB (`logs/remote_sizes.txt`, labels not checked) | Replica Terms | OVO / HOV-SG: 8 scenes, 51 classes, 3D mIoU/mAcc (arXiv 2411.15043) | G, but too large for 48 GB of free disk | N | N |
| LangSplat / OpenGaussian | They do not use Replica. LangSplat uses LERF + 3D-OVS (README); OpenGaussian uses LERF + ScanNet (paper) | – | – | – | – | – | – | – | – |
| **ScanNet v2** | 3D instance semantics | Rescans exist but are not aligned and have no change labels (not checked) | BundleFusion poses, depth | Full set not checked; OpenScene processed 2D 6.27 GB + 3D 3.75 GB (`logs/remote_sizes.txt`) | ToS form emailed; reply "within a week" (ScanNet README); non-commercial | OpenGaussian: 10 scenes, 19/15/10 classes, 3D mIoU/mAcc (Table 2). OVO: HOV-SG 5 scenes or 312 val scenes, ScanNet20/200 | G (optional large-scale) | N | P (static; Chu & Kim likely) |
| ScanNet++ | Long-tail, label-ambiguous semantics | None mentioned | DSLR 33 MP + iPhone RGB-D + laser | Not checked | Account, application, approval, token; ScanNet++ ToU | NVS and 3D semantic/instance benchmarks | G but heavy | N | N |
| **3RScan** | 3D instance labels with IDs kept across rescans | **Yes.** Rescans have a global alignment T plus per-instance rigid, non-rigid and removed changes (README) | Tango RGB-D, poses, meshes | Not checked | Data behind a ToS form; code MIT | Scene-graph and instance relocalization; split files have 384/46/45 lines (train/val/test) | P (Tango image quality) | **G** | P (no standard relocalization protocol besides RIO10) |
| **RIO10** (subset of 3RScan) | Per-frame instance PNGs for train and val sequences (`semanticsXX.zip`); class mapping not checked | **Yes.** 10 scenes: seqXX_01 is mapping, seqXX_02 is validation with GT, 54 hidden test sequences. Per-frame change statistics (SemanticChange, GeometricChange, ...) | Poses aligned across scans, rendered depth, meshes per sequence. About 540×960 portrait (intrinsics cx≈270, cy≈493, `logs/rio10_intrinsics.txt`). Validation = 34,415 frames (`logs/rio10_stats_summary.txt`) | Not checked. Download is per scene: seq, models and semantics zips | Google Form; the URL comes after the form. CC BY-NC-SA 4.0 (comment in ACE-G `datasets/setup_rio10.py` line 11) | DCRE(0.05/0.15), Pose(5 cm, 5°), Outlier(0.5), NaN; Score = 1 + DCRE(0.05) − Outlier (leaderboard). ACE-G uses median t/r on validation | P | **G** | **G (best available)** |
| ReplicaCAD / Habitat rearrangement | Yes, via the simulator | **Yes, controlled.** 5 macro × 20 micro variations; 84 of 105 public | Exact | 0.84 GB baked + 0.16 GB interactive (HF API, `logs/replicacad_sizes.txt`) | CC BY 4.0. Needs habitat-sim: PyPI has only a macOS-arm64 wheel, and conda is not on PATH | Embodied rearrangement, not a mapping benchmark | G (synthetic) | G | G (synthetic), but **setup risk is high** |
| TUM RGB-D dynamic | None | No; people move during capture | Mocap GT at 100 Hz; 640×480 at 30 Hz | 0.44–2.55 GB per sequence, 9 dynamic sequences (`logs/tum_sizes.txt`) | Immediate. CC BY 4.0 (TUM page) | Dynamic SLAM, ATE/RPE | N | P (no change GT) | N |
| Bonn RGB-D Dynamic | None | Within-sequence: placing, removing and moving boxes | Mocap poses; static-scene GT cloud from a Leica BLK360 | 16.4 GB all; 183 MB–5.8 GB per sequence; GT cloud 676 MB subsampled or 4.0 GB full (Bonn page) | Licence not stated on the page | ReFusion: ATE and reconstruction versus GT cloud | N | P/G ("ghost" Gaussians measured against the static GT cloud) | N |
| Chu & Kim RA-L 2026 | Not checked | Claimed "dynamic scene change" | Not checked | Not checked | Paywalled; no code | Datasets cited: only 7-Scenes and ScanNet (Crossref refs) | – | – | Only if their PDF numbers use the same split |
| 7-Scenes / 12-Scenes | None | No (static) | SfM and DSLAM pseudo-GT on disk | 21 GB / 13 GB on disk | Already here | Median t/r, 5cm5°, 2cm2° with SfM pseudo-GT (GS-CPR Tables 1, 2, 4) | N | N | Static sanity only |
| 3DGS-CD (optional) | None | 5 real scenes, pre- and post-change images | Not checked | Not checked | Apache-2.0 code | Change masks | N | P (small) | N |

## 2. Baseline table

"Runs here" means it was already run on this machine by USGLoc (`usgloc/results/`). All published numbers are cited with their source and were not re-run by me.

| Method | Code / licence | Runs on our stack (sm_120, torch 2.11, CUDA 12.8) | Published numbers (source) | GPU-h (est.) | Use |
|---|---|---|---|---|---|
| **CSS** (Wilson ICRA'25) | None found | Must be reimplemented in our BKI pipeline (categorical Dirichlet on one-hot labels) | Replica room_0 mIoU/Acc: LSeg alone 25.5/70.7; CSS with LSeg 28.4/76.7; CSS with GT segmentation 73.7/97.0 (arXiv 2411.02547, Table IV). Uncertainty shown only as sparsification curves; no AUSE number | Same as ours | Table seg row = ablation (i) |
| Feature 3DGS | github.com/ShijieZhou-UCLA/feature-3dgs; Inria non-commercial | NOT RUN. Custom N-dim rasterizer, tested on CUDA 11.8. Our GCC is 13.3 and the Inria-lineage `rasterizer_impl.h` uses `uint32_t` without `<cstdint>` (known GCC-13 break, likely a one-line patch). Also needs the LSeg checkpoint (size not checked) | Replica, 4 scenes: mIoU 0.787 / Acc 0.943 (full); 0.782 / 0.943 (speed-up); NeRF-DFF 0.636 / 0.864 (Table 2, labels derived from LSeg) | 2–4 | Official run on its own split (context row) plus a gsplat re-implementation as the fair row |
| LangSplat | github.com/minghanqin/LangSplat; Inria | 24 GB of VRAM recommended (README), so it does not fit | ScanNet 19/15/10 classes: 3.78 / 5.35 / 8.40 mIoU; LERF mean 9.66 (OpenGaussian Tables 2 and 1) | – | Cite only |
| OpenGaussian | github.com/yanmin-wu/OpenGaussian; Inria | Needs pytorch3d and the ashawkey rasterizer; 16–20 GB, 15 min per ScanNet scene on a V100 (paper). Does not fit | ScanNet 10 scenes: 24.73 / 30.13 / 38.29 mIoU; LERF 38.36 (Tables 2 and 1) | 2.5 on a free ≥24 GB GPU | Cite only, unless a ScanNet block is added |
| LEGaussians | github.com/buaavrcg/LEGaussians; MIT | Not checked | ScanNet 3.84 / 9.01 / 12.82; LERF 16.21 (OpenGaussian Tables 2 and 1) | – | Cite |
| Gaussian Grouping | github.com/lkeab/gaussian-grouping; Apache-2.0 | Not checked | Replica 71.15 mIoU / 66.52 PQ; ScanNet 68.70 / 61.83 (Table 4, closed-set panoptic) | – | Related work only |
| Bayesian Fields | github.com/MIT-SPARK/Bayesian-Fields; BSD-2 | **Code not released** (repo = LICENSE only) | Clio Cubicle / Apartment strict open-set recall 0.89 / 0.59 (Table 1); 8–12 min per scene on an RTX 3090 | – | Discuss as the closest probabilistic open-vocabulary method |
| **ACE** | github.com/nianticlabs/ace; Niantic non-commercial | **Runs here** | 7-Scenes (SfM pseudo-GT): 1.1 cm / 0.34°, 97.1% / 83.3%. 12-Scenes: 0.7 / 0.26, 100 / 97.2 (GS-CPR Tables 1, 2, 4). **RIO10 validation: 358.4 cm / 58.7°** (ACE-G Table 3). Reproduced here, full 7-Scenes: 1.09 / 0.34, 97.0 / 83.1. 12-Scenes 100-frame subset: 0.98 / 0.26, 99.9 / 91.6, a translation gap to publication (`logs/usgloc_baseline_avgs.txt`) | 5 min mapping per scene (ACE paper); about 1–2 GPU-h for RIO10 | Table loc, both blocks |
| **ACE-G** | github.com/nianticspatial/ace-g; Niantic Spatial non-commercial. Weights 186 MB; ships `setup_rio10.py` | Not tried. The default setting needs 21 GB of VRAM, so `--max_buffer_size` and `--batch_size` must be lowered ("results might be slightly worse", README) | RIO10 validation: 37.0 / 12.1 (25 min) and 41.1 / 13.8 (5 min); Reloc3r 47 / 9.4; DINO-ACE 83.8 / 15.9; GLACE 114 / 33.4 (Table 3) | 1–4 | **Strongest fair baseline under scene change** |
| Marepo | github.com/nianticlabs/marepo; non-commercial | **Runs here** | 7-Scenes 2.9 / 1.04, 84.0 / 33.7; 12-Scenes 2.1 / 1.04, 95 / 50.4 (GS-CPR). Reproduced here (100 frames per scene): 2.87 / 1.03, 83.4 / 34.3. No RIO10 number | 1–2 | Table loc (APR row) |
| DSAC* | github.com/vislearn/dsacstar; BSD-3 | Only the pose-solver egg is built (`~/.local/.../dsacstar-0.0.0-py3.12...egg`). Training takes about 2.5 days per scene (DSAC* paper, K80) | 7-Scenes 1.1 / 0.34, 97.8 / 80.7; 12-Scenes 0.5 / 0.25, 99.8 / 96.7 (GS-CPR) | Too costly | Cite in the static block |
| GS-CPR | github.com/XRIM-Lab/GS-CPR; CC BY-NC-SA 4.0 parts + Inria | NOT RUN. Tested on CUDA 12.1 / torch 2.5.1. Needs tiny-cuda-nn, a depth rasterizer and MASt3R (checkpoint on disk). The curope build **failed** on torch 2.11 because `kernels.cu:101` uses `tokens.type()` (`logs/curope_build_head.log`); a slow PyTorch fallback may exist (not checked) | ACE+GS-CPR: 7-Scenes 0.8 / 0.25, 100 / 93.1; 12-Scenes 0.5 / 0.21, 100 / 98.7. Scaffold-GS trained 30k iterations on an A6000; 0.19 s per query on an RTX 4090 (Tables 1, 2, 4, 6; `logs/gscpr_tables_v4.txt`) | – | Static block, published numbers |
| Chu & Kim | None; paywalled | – | Not checked | – | Row only with PDF numbers on the same data and pseudo-GT |
| RIO10 leaderboard (hidden test) | Server at vmnavab26.in.tum.de/RIO10; the page loaded on 2026-09-28; submission not tested | Via the server | DCRE(0.05) / Score: MegLoc 0.889 / 1.836; NeuralRouting 0.538 / 1.441; KAPTURE-R2D2-FUSION 0.447 / 1.338; Grove v2 0.416 / 1.162; D2-Net 0.392 / 1.247; Active Search 0.185 / 1.166; HF-Net 0.064 / 0.373 (leaderboard page) | – | Only if we submit the test once |

## 3. On-disk inventory (commands: `ls`, `du -sh`, `find / -maxdepth 5`)

| Item | Path | Size |
|---|---|---|
| 7-Scenes, all 7 scenes | `usgloc/data/7scenes_source` | 21 GB |
| 12-Scenes (apt1, apt2, office1, office2) | `usgloc/data/12scenes_source` | 13 GB |
| SfM + DSLAM pseudo-GT (Brachmann 2021) | `usgloc/data/visloc_pseudo_gt_limitations/pgt/{sfm,dslam}` | 65 MB |
| USGLoc gsplat maps (94 files) | `usgloc/maps` | **55 GB** |
| ACE / Marepo per-frame results | `usgloc/results/{ace_dsacstar_full,marepo_100,ace_dsacstar_12scenes_100}` | – |
| ACE, Marepo, GS-CPR clones and ACE models (**in /tmp, can be deleted at any time**) | `/tmp/claude-1005/.../887d461d-.../scratchpad/{ace,marepo,GS-CPR,ace_models}` | 74 MB / 955 MB / 88 MB / 105 MB |
| MASt3R checkpoint; dust3r and mast3r code | `/home/thanhnc/gauss_ws/third_party/` | 2.75 GB |
| DINOv2 ViT-L/14 (usable for retrieval init), LoFTR indoor, RoMa indoor/outdoor | `~/.cache/torch/hub/checkpoints` | 1.22 GB, 46 MB, 446 MB × 2 |
| CLIP ViT-B-16 laion2B | `~/.cache/huggingface` | 577 MB |
| FastSAM-s | `usgloc/FastSAM-s.pt` | – |
| Replica, ScanNet, 3RScan, RIO10, TUM, Bonn | **None found** (shallow `find`) | – |
| Free disk | `/` | 48 GB (95% used, `logs/df.txt`) |

**Code gaps in uags:**
- `ReplicaDataset` hard-codes the NICE-SLAM layout: 1200×680, fx=600 (`usplat/datasets.py:51-53`). The Semantic-NeRF Replica layout needs a new loader.
- `SevenScenesDataset` hard-codes fx=585, the DSLAM intrinsics (`usplat/datasets.py:105`). SfM pseudo-GT needs a per-scene focal length.
- There is no RIO10 loader.
- There is no categorical (CSS) baseline.
- AUSE and ECE already exist (`usplat/evaluation.py:15,35`).
- Dense features are MaskCLIP only (`usplat/clip_dense.py:28`).

## 4. Recommended protocol per table

**Rules for all tables:**
- Every method uses the same 3DGS geometry and the same dense features (MaskCLIP ViT-B/16), so only the 3D fusion rule differs.
- Hyperparameters (κ, τ_min, r_stable, r_unstable, λ, and the ECE temperature) are chosen only on the val split.
- The test split is run once.
- Report mean ± std over scenes, with N stated.

**Table seg (1): open-vocabulary segmentation and calibration, Replica (Semantic-NeRF renders)**
- Val scenes: office_0 and room_2.
- Test scenes: room_0, room_1, office_1, office_2, office_3, office_4. This includes all four Feature 3DGS scenes.
- Per scene, the map uses Sequence_1, every 5th frame (180 frames, as in Semantic-NeRF). Evaluation uses Sequence_2, every 10th frame (90 frames on a new trajectory).
- Text queries are the classes present in each scene, with a fixed prompt template.
- Metrics:
  - mIoU and pixel accuracy.
  - AUSE_epi against per-pixel error. Also sweep the number of mapping views (10 / 30 / 90 / 180): epistemic uncertainty must predict the error that more views remove.
  - AUSE_ale against error at boundaries (dilated GT edges) and on ambiguous classes.
  - ECE with 15 bins, temperature fitted on the val scenes.
- Rows:
  - Per-frame 2D MaskCLIP (no 3D fusion).
  - Deterministic feature field (Feature-3DGS style, in gsplat).
  - CSS categorical (reimplemented).
  - Ours.
- Context only, in a footnote: published Feature 3DGS 0.787 and CSS 28.4. They use a different split and different labels.
- Optional large-scale block: ScanNet, the 10 OpenGaussian scenes, 3D mIoU at 19/15/10 classes, with published rows for LangSplat, LEGaussians and OpenGaussian.

**Scene-change robustness (2): RIO10 train + val, with GT poses to isolate the mapping part**
- Build the map from seqXX_01. Update it with the first half of seqXX_02 (in time), using GT poses.
- Label-free GT for "stale" Gaussians: a Gaussian is stale if it lies more than δ (for example 5 cm) from the seqXX_02 mesh while lying near the seqXX_01 mesh. Both meshes are in `modelsXX.zip`.
- Metrics:
  - AP and AUROC of the instability score (ours aleatoric, ours epistemic, scalar variance, CSS Dirichlet variance, photometric residual).
  - mIoU on held-out frames from the second half of seqXX_02, using the instance semantics.
  - Stratify results by the per-frame SemanticChange value from `stats.txt`.
- Optional extra check: Bonn box sequences, measuring the ghost-Gaussian rate against the static Leica point cloud.

**Table loc (3): localization under change, RIO10**
- Val: scenes 01, 04, 06. Their mean semantic change is 0.205, 0.367 and 0.093 (low to highest), with 4,929 validation frames in total. Use them to tune the method and refinement settings.
- Test: seqXX_02 of scenes 02, 03, 05, 07, 08, 09, 10, run once. Queries are the second-half frames not used in (2).
- Optional final test: submit the 54 hidden test sequences to the server once.
- Metrics: the official RIO10 evaluation code, reporting DCRE(0.05), DCRE(0.15), Pose(5 cm, 5°), Outlier(0.5), NaN rate and median t/r.
- Rows:
  - ACE, re-run here, 5 min per scene.
  - ACE-G (with the VRAM-reduced setting stated).
  - Marepo.
  - Retrieval (DINOv2) + render refinement on the full map.
  - Ours: the same with consolidation and the robust subset.
- Static block: 7-Scenes with SfM pseudo-GT. Tune on held-out train frames (as in USGLoc plan 2.6). Show published ACE, DSAC*, Marepo and GS-CPR next to our reproduced ACE and Marepo.
- Design risk: our refinement basin is about 10–20 cm (USGLoc 2.5 diagnostics, not re-checked). An initializer under change therefore has to be shown to work.

**Table abl (4):**
- (i) Categorical vs vMF: Replica test.
- (ii) Scalar vs 2-D uncertainty: Replica AUSE and RIO10 stale-Gaussian AP.
- (iii) Consolidation on/off: RIO10 test, DCRE and median t.
- (iv) EFE vs hand-weighted score, pool-based: pick 30 of the 180 Sequence_1 frames; measure mIoU and AUSE on Sequence_2. This needs no simulator.
- Replace the "ATE" column with RIO10 DCRE(0.05) or median t.

**Fig qual:** a RIO10 validation frame with a moved object (high SemanticChange) plus a Replica frame.

## 5. Access and compute risks

| Risk | Evidence | Mitigation |
|---|---|---|
| GPU effectively unavailable: USGLoc uses 13.7 of 16.3 GB at 89% | `logs/nvidia_smi.txt` | Schedule after the USGLoc runs finish. Rough total estimate 25–40 GPU-h (not measured) |
| Disk: 48 GB free; usgloc/maps is 55 GB; RIO10 and Semantic-NeRF Replica sizes not checked | `du`, `df` | Download RIO10 one scene at a time (only 3 val + 7 test scenes' train/val sequences). The user decides whether to prune usgloc/maps |
| Network: about 200 KB/s per connection | Task brief | Parallel range curl (worked for 12-Scenes) |
| Access delays: RIO10 form (turnaround not checked); 3RScan form; ScanNet "within a week"; ScanNet++ approval | READMEs | **Submit the RIO10 form today.** ScanNet only if the optional block is kept |
| Licences are all non-commercial (fine for an academic paper): Niantic (ACE, Marepo, ACE-G), Inria (Feature 3DGS, LangSplat, OpenGaussian), CC BY-NC-SA 4.0 (RIO10, DUSt3R) | `logs/github_licence*.txt` | State the licences in the paper |
| Old CUDA extensions: curope fails on torch 2.11 (`.type()`); Inria rasterizer lacks `<cstdint>` under GCC 13.3; habitat-sim has no Linux py3.12 wheel | `logs/curope_build_head.log`, `rasterizer_impl.h`, PyPI JSON | Patch the extensions; skip ReplicaCAD; skip running GS-CPR (use published numbers) |
| VRAM: LangSplat 24 GB, OpenGaussian 16–20 GB, ACE-G 21 GB by default | READMEs / paper | Cite published numbers; run ACE-G with a reduced buffer and say so |
| Unfair comparisons: LSeg vs MaskCLIP features; Feature 3DGS labels are LSeg-derived; SfM vs DSLAM pseudo-GT (0.95 cm gap per USGLoc plan 1.5) | Papers / plan | Same features for every row; SfM pseudo-GT only; published rows go in a separate context block |
| Our 12-Scenes ACE subset reproduction is 0.98 cm vs 0.7 cm published (40% worse) | `logs/usgloc_baseline_avgs.txt` | Check which pseudo-GT that run used before citing it |
| The RIO10 test server may be slow or down | Page loaded; submission not tested | Test-once on the 7 validation test scenes is the primary test; the server is optional |
| No code for CSS, Chu & Kim or Bayesian Fields | GitHub API | Reimplement CSS; Chu & Kim only with PDF numbers on the same split, obtained through institutional access |

**Evidence files** (all in `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/wf/data/logs/`):
- `usgloc_baseline_medians.txt`, `usgloc_baseline_avgs.txt`
- `gscpr_tables_v4.txt`
- `rio10_stats_summary.txt`, `rio10_intrinsics.txt`
- `chu_kim_crossref_refs.txt`, `chu_kim_refs_resolved.txt`
- `github_licences.txt`, `github_licence_text.txt`, `bayesian_fields_repo_contents.txt`
- `remote_sizes.txt`, `replica_release_sizes.txt`, `tum_sizes.txt`, `replicacad_sizes.txt`
- `curope_build_head.log`, `batch_ace12_tail.log`
- `nvidia_smi.txt`, `df.txt`

Numbers that came from web pages were read with WebFetch and cite a URL plus table number. They were not re-computed.