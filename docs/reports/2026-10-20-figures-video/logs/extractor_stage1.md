# Stronger open-vocabulary extractor for every Table I row: stage 1

2026-10-24 | root: /home/thanhnc/gauss_ws/src/uags | git: master@741572d, +14 untracked new files (nothing committed, no existing file edited) | part of [Figures, video and stronger results](../README.md)

**Verdict: stage 1 DONE** (2026-10-24 02:38-03:55). Three candidates chosen and implemented; each matches its official code (tokens equal to float precision); 17 CPU tests pass; the prereg is drafted; both drivers refuse to run until the lead commits it. Only validation smoke frames were read (3 per extractor); no test scene was touched.

**Next step:** the lead decides Q1-Q6 (section 9), sets `registered`, commits the prereg and the other 12 new code, config and test files (the 14th untracked file is this log). Then `scripts/extractor_select.sh` runs stages A and B (~20 min GPU).

## 1 Candidates (survey)

LSeg already beat ClearCLIP, dino.txt, Talk2DINO ViT-B and ProxyCLIP ViT-B/16 on Replica validation ([attempt 05](../../2026-09-29-data-and-baselines/attempts/05-extractor-study.md)). So the new candidates are stronger *published configurations* that were not tried. A candidate must give class-agnostic, text-aligned pixel features (methods that refine a class prediction with the class list cannot be lifted).

| Name | Method (venue) | Setting used (the paper's ADE20K / Context setting) | Published mIoU | Compute | Code / licence |
|---|---|---|---|---|---|
| `radseg` | RADSeg (CVPR'26 Findings, arXiv 2511.19704), no SAM refinement | C-RADIOv3-B + SCRA, 336/224 windows, SCGA, SigLIP2 adaptor head; short side 576 | 48.00 mean of 5 benchmarks (mid res) vs Trident-H 47.02, ProxyCLIP-H 43.58; 3D Replica 32.3, ScanNet 36.3 | 106M; 0.115 s/img (V100 fp32) | MIT; RADIO code NVIDIA NC (research only) |
| `proxyclip_h` | ProxyCLIP (ECCV'24, arXiv 2408.04883), best config | CLIP ViT-H/14 laion2b + DINO ViT-B/8 proxy; short side 448, 336/112 | 44.4 mean of 8 (vs 42.3 for ViT-B/16, the config tried in attempt 05) | ~0.7B | NTU S-Lab (non-commercial) |
| `trident` | Trident (ICCV'25, arXiv 2411.09219), no SAM refinement | CLIP ViT-H/14 + DINO ViT-B/16 per window, SAM-H global affinity; short side 576, 336/224 | 47.1 mean of 8 unrefined (48.6 refined) | 1.75B; 1.45 s/img (V100 fp32) | Apache-2.0 |
| `lseg` (incumbent) | LSeg (ICLR'22) | Table I as is (5 templates) | — | 0.07-0.09 s/frame here | MIT |

Not chosen: NACLIP/SCLIP/ClearCLIP with larger CLIP (below ProxyCLIP; SCLIP ViT-H 29.1), RayFronts/NARADIO (superseded by RADSeg, 42.1 vs 48.0), dino.txt on DINOv3 (gated weights), TextRegion (SAM2 masks, ~10x RADSeg cost), RADSeg+/Trident-refined/CorrCLIP/LPOSS (class-dependent refinement), CAT-Seg/SAN/FC-CLIP (score a class list, no pixel feature). Numbers above are copied from the papers (arXiv HTML, read 2026-10-24), not re-checked.

**Templates:** all three candidates publish the same 80 OpenAI ImageNet templates (checked: identical lists in the three repos, read with `ast`). Rule: each method's published set (candidates 80, LSeg its 5).

## 2 Files (all new; no existing file edited)

| File | Role |
|---|---|
| [usplat/extractors/radseg.py](../../../../usplat/extractors/radseg.py) | RADSeg re-implementation (SCRA, windows, SCGA, adaptor head, SigLIP2 text). RADIO model built by NVlabs' own `hubconf.radio_model` from the local checkpoint (read with `weights_only=True`); the SigLIP2 head built without RADIO's auto-download of the full model |
| [usplat/extractors/proxyclip_h.py](../../../../usplat/extractors/proxyclip_h.py) | subclass of the existing `ProxyCLIPDense` (unchanged): CLIP ViT-H/14 + DINO ViT-B/8 proxy |
| [usplat/extractors/trident.py](../../../../usplat/extractors/trident.py) | Trident re-implementation (proxy attention per window, splice, SAM-H affinity); SAM from the official repo, its last attention captured as Trident does |
| [usplat/extractors/_vfm.py](../../../../usplat/extractors/_vfm.py) | shared loaders: official DINO v1 ViT (private import names), CLIP ViT-H/14 from the local file, text averaging. Not an extractor (no `build`) |
| [configs/vocab_replica_extractors.json](../../../../configs/vocab_replica_extractors.json), [configs/vocab_scannet_extractors.json](../../../../configs/vocab_scannet_extractors.json) | the frozen vocabularies plus `templates_by_extractor` for the 3 candidates; every other key checked equal to the source (sha256 in `derived_from`) |
| [configs/extractor_prereg.json](../../../../configs/extractor_prereg.json) | pre-registration draft (section 4) |
| [scripts/extractor_2d.py](../../../../scripts/extractor_2d.py) | stage A: Table I's 2D-row path without a map, validation scenes only |
| [scripts/extractor_select.py](../../../../scripts/extractor_select.py) | `select` (R0 + stage B rule + gate), `winner`, `select-c`, `check-test`, `look`, `claim` |
| [scripts/extractor_select.sh](../../../../scripts/extractor_select.sh) | stage A/B driver |
| [scripts/extractor_tab1.sh](../../../../scripts/extractor_tab1.sh) | stage C driver (`DATASET`, `STAGE=val/test/table/claim`) |
| [tests/test_extractor_candidates.py](../../../../tests/test_extractor_candidates.py), [tests/test_extractor_select.py](../../../../tests/test_extractor_select.py) | CPU tests (section 8) |

Registry: unchanged. `get_extractor(name)` imports `usplat.extractors.<name>` and calls its `build()`, so new files register themselves. LSeg's code path is untouched.

Weights and code (outside the repo; sha256 pinned in the prereg `weights`, checked by both drivers):

| File (under /home/thanhnc/gauss_ws/data/weights) | Size | Source (revision) | Licence | sha256 (first 12) |
|---|---|---|---|---|
| radseg/c-radio_v3-b/c-radio_v3-b_half.pth.tar | 547 MB | HF nvidia/C-RADIOv3-B @ 44653a0 | NVIDIA Open Model License | 530b0e827b1d (= HF LFS) |
| radseg/siglip2-so400m-patch16-naflex/model.safetensors (+ tokenizer, configs) | 4.5 GB | HF google/siglip2-so400m-patch16-naflex @ cc24074 | Apache-2.0 | 11a61a206880 (= HF LFS) |
| clip/ViT-H-14-laion2b_s32b_b79k/open_clip_model.safetensors | 3.9 GB | HF laion/CLIP-ViT-H-14-laion2B-s32B-b79K @ 1c2b849 | MIT | 0084e75319a5 (= HF LFS) |
| dino/dino_vitbase8_pretrain.pth | 343 MB | dl.fbaipublicfiles.com/dino (torch.hub dino_vitb8) | Apache-2.0 | 575f0efc4838 |
| dino/dino_vitbase16_pretrain.pth | 343 MB | dl.fbaipublicfiles.com/dino (dino_vitb16) | Apache-2.0 | bf34ad0f424b |
| sam/sam_vit_h_4b8939.pth | 2.6 GB | dl.fbaipublicfiles.com/segment_anything | Apache-2.0 | a7bf3b02f3eb (md5 4b8939a8… = file name) |

Third-party code (cloned to /home/thanhnc/gauss_ws/third_party): RADIO @ 4ffaf15 (imported), DINO @ 7c446df (imported), segment-anything @ dca509f (imported); RADSeg @ 3fe8789, Trident @ 7feea10 and ProxyCLIP @ a0e38d1 are read only (re-implemented). A SigLIP2-giant snapshot fetched by mistake (RADIO's `siglip2` teacher is so400m, read from the checkpoint) was deleted.

## 3 Fidelity: re-implementation vs the official code (CPU fp32, no dataset frame)

Input: the RADSeg repo's demo image `assets/example1.jpg` (a street scene, not a dataset frame) resized to 640x480; 11 class names; the 80 templates. The official side runs the authors' code (RADSeg's `RADSegEncoder` with RADIO's real SigLIP2 adaptor; Trident's and ProxyCLIP's bundled `open_clip` and SAM, their window / splice / slide code copied verbatim, only the mmseg glue written out). Harnesses and logs: /home/thanhnc/gauss_ws/data/uags_results/extractor/fidelity/ (`*_vs_official.py`, `*_official.py`, `*.log`).

| Extractor | Token cos (min) | Text cos (min) | Pixel argmax agreement | Why not 100% |
|---|---|---|---|---|
| radseg | 0.9999994 (42x56 grid) | 0.9999998 | 97.2% | the official upsamples per-class softmax(100 cos) probabilities, the pipeline upsamples unit tokens (Table I's convention for every extractor) |
| trident | 0.9999997 (48x64 grid, 9 windows, input 588x770 as official) | 0.9999998 | 99.997% | — |
| proxyclip_h | 0.9999996 (per 336 crop, 2 crops) | 0.9999998 | 95.9% at 640x480; **99.67%** on a 672x448 frame | inherited from `proxyclip.py`: both sides rounded to the 8-px proxy grid (597 -> 600 px wide) and window tokens averaged instead of per-pixel window logits; on a frame that needs no rounding the gap is 0.3% |

Verdict: all three compute the published features (tokens equal to float32 precision). The pixel-level gaps are pipeline conventions shared by every row, plus ProxyCLIP-H's resize rounding (open decision Q6).

## 4 Pre-registration draft: [configs/extractor_prereg.json](../../../../configs/extractor_prereg.json) (uncommitted, `registered` empty)

Written 03:08, before any candidate touched a dataset frame. The smoke check (section 6) ran after it and filled only the GPU caps. Current draft sha256 06ce7a689544.

| Rule | Content |
|---|---|
| Templates | Each method's published set: candidates the 80 OpenAI ImageNet templates (same list in all three repos), LSeg its 5 (Table I). Nothing chosen on our data (Q1). |
| Stage A (validation only) | Per extractor and dataset: per-frame 2D segmentation of every validation evaluation frame through Table I's 2D-row path: tokens → upsample to the label grid → PCA-64 fit like `run_mapping.py` on the first 20 mapping frames (seed 0) → projected text → argmax. Replica office_0, room_2 (90 frames each); ScanNet scene0000_00, scene0062_00 (279, 37). Metric: per-scene mIoU (variant all), A_D = mean over D's 2 scenes. |
| R0 (independent check) | LSeg's stage-A mIoU must equal Table I's validation `2d_features` per scene within 0.1 points (4 scenes). Else stop: stage A would measure something else. |
| Stage B | Δ_D(x) = A_D(x) − A_D(LSeg). Winner = max of min(Δ_Replica, Δ_ScanNet); ties within 0.1 points → order radseg, proxyclip_h, trident (cheapest first). **Gate: winner's min Δ ≥ 1.0 point (both datasets)**, else the attempt ends on validation, LSeg stays, no test look (Q2). |
| Why per dataset | Table I and its claims are per dataset. A pixel pool is dominated by ScanNet (scene0000_00 = 18.6M of 33.5M validation pixels) and could hide a Replica loss. Scene means weigh scenes like Table I. |
| Stage C | Winner only. Re-map all 18 scenes on the existing 3DGS maps, exactly Table I's `run_mapping.py` call except `--extractor`, `--vocab`; distill (val: f3dgs + λ_u ∈ {1,2,4}; test: f3dgs + chosen λ_u); `eval_seg.py` as Table I (`--with-2d`, baselines + distill rows, no v2 rows). New dirs: uags_fields/extractor/W, uags_results/extractor/W. Refit betas, CSS α0, P3, λ_u on validation (make_tables `choose`). |
| Consistency (independent check) | Before test: the winner's 2D row in the stage C validation record must equal its stage-A numbers within 0.1 points per scene. |
| Test once | Clean committed tree equal to the validation commit; `look_test.json` appended before the run; a finished test record is never rerun. |
| C1 | "W improves our map on D" only if ours P2 test mean mIoU (with W) > Table I's ours P2 (LSeg) AND better in ≥ 4/6 Replica / ≥ 5/8 ScanNet scenes. Same rule computed for Acc, ECE, NLL, Brier, AURC (descriptive unless it holds). |
| vs averaging | Reported as is (means, x/N); no claim registered, none added after the test. |
| Disclosure | Later attempt; further Replica test look (after Table I v1 and 6 other looks: views, Table IV, Table I v2, qualitative, smoothing, fusion); 4th ScanNet look. Sentence drafted in the prereg (also for the gate-fails case). |
| Open decisions | Q1 templates, Q2 margin, Q3 drop Trident?, Q4 what C1 changes in the paper, Q5 licences (RADIO and ProxyCLIP code non-commercial), Q6 ProxyCLIP-H resize rounding |

## 5 Drivers and gates

| Gate | extractor_select.sh (A/B) | extractor_tab1.sh (C) |
|---|---|---|
| prereg committed, unmodified, `registered` set, names this pipeline | yes | yes |
| new code + vocab files committed; no uncommitted tracked change | yes | val: yes; test: scripts/ usplat/ configs/ clean and equal to the validation commit (`make_tables.check_code`) |
| `UAGS_LSEG_*` / `UAGS_PROXYCLIP_*` overrides refused | yes | yes |
| weight sha256 = prereg; third-party commits = prereg | all extractors | the winner's files |
| output dirs | not a Table I / smooth / fusion dir, not under uags_fields | only under uags_results/extractor and uags_fields/extractor; Table I dirs refused |
| runs once | refused once `selection_a.json` exists or stage C started | runs only if `winner` passes (gate + inputs unchanged); `check-test` before test; `look_test.json` appended before the test evaluation; a finished test record is never rerun |
| text cache | a COPY of Table I's cache (LSeg); new caches for candidates | a COPY of stage A's cache |
| GPU | `gpu_gate.sh` at measured caps, headroom 1024 MiB, never beside USGLoc; no `empty_cache` anywhere in the new code | same; maps never refitted (missing map refused); `disk_guard.py` before each field |

How to run (after the lead's decisions and the commit): `scripts/extractor_select.sh` → `DATASET=replica STAGE=val scripts/extractor_tab1.sh`, same for scannet → `STAGE=test` per dataset → `STAGE=table` per dataset → `STAGE=claim`.

## 6 Smoke check (GPU, 3 validation frames per extractor)

Frames: Replica office_0 Sequence_1 frame 0 (a 1-frame PCA) and Sequence_2 evaluation frames 0 and 10; `scripts/extractor_2d.py` smoke mode (`--scene office_0 --max-pca-frames 1 --max-views 2`, records say `smoke: true`; the selection refuses them). RTX 4070 Ti, 2026-10-24 03:32:50-03:33:49 (one minute in all). Runner, records and logs: /home/thanhnc/gauss_ws/data/uags_results/extractor/smoke/ (`run_smoke.sh`, `smoke_replica_<x>.json`).

| Extractor | Build (s) | ms / frame (2 frames, after warm-up) | Peak GPU alloc (MiB) | Peak RSS (GiB) | Pixel acc, 2 frames (sanity) |
|---|---|---|---|---|---|
| lseg | 9.0 | 70 | 1876 | 3.8 | 74.8% |
| radseg | 9.3 | 70 | 3838 (incl. 2.8 GB fp32 text tower) | 4.5 | 74.8% |
| proxyclip_h | 9.9 | 281 | 3985 | 8.2 | 67.4% |
| trident | 15.5 | 353 | 4896 | 8.2 | 72.2% |

All four load, run and give plausible labels (7 classes in these frames). The records also hold a 2-frame mIoU (LSeg 37.6, RADSeg 47.7, ProxyCLIP-H 41.0, Trident 42.2, PCA fit on 1 frame): NOT used by any rule and not evidence (2 frames, 1 scene). The rules were written before this run.

## 7 Cost estimate (stages A-C)

Basis: smoke ms/frame above; Table I's measured LSeg costs (ScanNet walls on this 4070 Ti from `uags_results/scannet_tab1/*.log`; Replica distillation training times from `uags_results/tab1/distill_*.log` on the earlier 5070 Ti; evaluation `wall_s` of the Table I records). Estimates, not measurements.

| Stage | Frames extracted | GPU wall (gated, one job at a time) | Disk | Host RAM peak |
|---|---|---|---|---|
| A (4 extractors x 2 datasets, validation) | 576 per extractor (2,304) | extraction 0.7 / 0.7 / 2.7 / 3.4 min (LSeg / RADSeg / ProxyCLIP-H / Trident); with loading, labels, text: ~15-20 min total | < 50 MB | ~8 GB |
| B | 0 | seconds (CPU) | — | — |
| C, LSeg-equivalent part | — | ScanNet 10.3 h (map 0.2, distill val 3.1, test 4.3, eval 2.6) + Replica 3.9 h (distill ~3.1, map 0.2, eval 0.6) = **14.1 h** | fields ~24 GB (Table I: Replica 9.8 GB, ScanNet 14 GB; the field is 64-d whatever the extractor) | tokens of ScanNet scene0000_00 (541 frames): LSeg ~5.5, RADSeg 5.6, Trident 6.6, ProxyCLIP-H 9.0 GB |
| C, extra extraction vs LSeg (14,732 frames: mapping 3,898, distillation 8,958, 2D row 1,876) | 14,732 | RADSeg ≈ 0, ProxyCLIP-H +48 min, Trident +66 min | — | — |
| **C total** | | **RADSeg ~14 h, ProxyCLIP-H ~15 h, Trident ~15.2 h** | ~24 GB (473 GB free) | |

Weights already on disk: 12.3 GB.

## 8 Verification

Test command: `cd uags && PATH=$HOME/venvs/uags/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. HF_HUB_OFFLINE=1 UAGS_SLOW_TESTS=1 python -m pytest -q -s -p no:cacheprovider tests/test_extractor_candidates.py tests/test_extractor_select.py` → **17 passed in 424.7 s** (03:44-03:51). Without `UAGS_SLOW_TESTS`: 8 passed, 4 skipped (183 s). After a last driver edit, `tests/test_extractor_select.py` again: 5 passed (03:52).

| # | Check | Test / command | Result |
|---|---|---|---|
| V1 | RADSeg sizes (576x768 → 896x672), window cover, SCRA = RADSeg's own attention (transcribed), SCGA convex / fixed point | test_radseg_sizes_and_windows, test_scra_equals_official, test_scga_properties | PASS |
| V2 | Trident: SAM similarity = transcription of `get_sam_sim` (both `refine_neg_cos`), row-stochastic; attention capture = SAM's `Attention.forward` (rel-pos on), shapes | test_trident_sam_similarity_equals_official, test_trident_capture_attention_is_sams | PASS |
| V3 | With weights, CPU fp32, each of the 3: token grid, unit norms, **two calls bit-identical**, unit text, distinct classes | test_shapes_norms_determinism[radseg, proxyclip_h, trident] | PASS (42x56x1152, 56x75x1024, 48x64x1024) |
| V4 | Text/pixel alignment on a toy (non-dataset demo image): asphalt block labelled 'road' among {road, sky, tree, car} | test_text_pixel_alignment_toy[...] | PASS (100% for all 3) |
| V5 | Stage A on a toy (CPU, toy extractors): mIoU 1.0 sharp / 0.951 blurred / 0.250 biased; deterministic; refuses a non-prereg extractor, a protocol change, a test scene; smoke flag | test_extractor_2d_toy_and_refusals | PASS |
| V6 | Stage B rule (per-dataset minimum, ties to the earlier name); R0 failure blocks the gate; margin gate; a record edited after the selection, smoke / foreign-prereg / mixed-commit / unfinished / wrong-scene records refused | test_stage_b_rule_ties_margin, test_select_gates | PASS |
| V7 | Claim C1: mean and min wins per dataset, ties no win, lower-is-better metrics, vs averaging as is | test_claim_c1 | PASS |
| V8 | Drivers end to end in a scratch git repo: refused `UAGS_LSEG_FLIP`, `UAGS_PROXYCLIP_SHORT`, OUT in a Table I dir / in uags_fields, uncommitted prereg, unregistered prereg, tampered weight, dirty tree, stage C before stage B, stage C test before val, stage C roots outside `extractor/`; stage A+B → winner toya, R0 PASS; a second stage A run refused; stage C dry-run plan = 7 steps with the new paths and `--extractor toya` | test_drivers_end_to_end | PASS |
| V9 | Real repo: both drivers refuse (prereg not committed) | `scripts/extractor_select.sh`; `DATASET=replica STAGE=val scripts/extractor_tab1.sh` | PASS (rc 1) |
| V10 | Independent check: existing tests of the code the new files build on | `pytest tests/test_seg_eval.py tests/test_distill.py tests/test_feature_pca.py tests/test_clip_dense.py tests/test_extractor_proxyclip.py::test_proxyclip_shapes_and_norms` | PASS (33 passed, 1 skipped) |
| V11 | Fidelity vs official code | section 3 | PASS (tokens equal) |
| V12 | No existing file edited | `git diff` empty; `git status`: 14 untracked new files | PASS |
| V13 | New vocabularies equal the frozen ones except the added templates | key-by-key comparison | PASS |
| I3 | References | arXiv 2408.04883, 2411.09219, 2511.19704 opened (abstract + HTML tables); RayFronts 2504.06994 via search only | PASS / RayFronts NOT RUN |
| — | Stages A, B, C | need the lead's decisions and a commit | NOT RUN (by design) |

## 9 Open decisions and issues

| # | Item |
|---|---|
| Q1 | Templates: published sets (drafted: 80 ImageNet templates for the candidates, LSeg keeps 5) or the 5 LSeg templates for all. |
| Q2 | Margin 1.0 point on both datasets (drafted), 0.5, or 0. |
| Q3 | Keep all 3 candidates (drafted) or drop Trident (1.75B parameters, 5x RADSeg's time here, +66 min in stage C). |
| Q4 | If C1 holds on both datasets: replace Table I and move the LSeg table to the appendix (drafted); else the new table goes to the appendix. |
| Q5 | Licences: RADIO code (NVIDIA, research only) and ProxyCLIP code (NTU S-Lab, non-commercial). Weights: C-RADIOv3 NVIDIA Open Model License, the rest MIT / Apache-2.0. |
| Q6 | ProxyCLIP-H resize rounding (section 3): keep the `proxyclip.py` convention (drafted) or drop the candidate. |
| Issue | `run_mapping.py`'s `MAPPING_CODE` hash lists `lseg.py` but no new extractor file (it may not be edited), so `eval_seg.py --betas` alone would not notice a changed candidate module. Covered by the drivers (`check_code`: scripts/ usplat/ configs/ must equal the validation commit). |
| Issue | `eval_seg.py` computes uncached text on `--clip-device cpu`: for RADSeg ~11 s per class with 80 templates (CPU), for Trident / ProxyCLIP-H it builds a second full extractor on the CPU (~4-7 GB RAM). Stage A pre-fills the validation classes; test-only classes cost minutes. |
| Issue | The smoke ms/frame come from 2 frames; stage C times are estimates from Table I's logs (Replica distillation measured on the earlier 5070 Ti). |
| Risk | Two validation scenes per dataset: the 1-point gate can still pass on noise, and a pass is not a test result. C1 needs the test scenes (≥ 4/6, ≥ 5/8). |
