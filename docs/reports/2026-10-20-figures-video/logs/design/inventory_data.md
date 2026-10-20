# Data inventory for figures and video (read-only, 2026-10-20)

Nothing was written to the repo or the data folders, and no GPU job ran. I made these scratch files to look at the NOR scenes (one sheet per four scenes, `nor_0XX_0YY.jpg`):
- `/tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/nor_sheets/` holds the before/after sheets with the object region outlined in red.
- `/tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/nor_sheets/selected_frames.json` lists the frame shown per scene. The rule was fixed before looking: the map frame and the update frame with the largest `masks_real` area.
- `/tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/nor_sheet.py` builds the sheets.

## 1. NeRF-Object-Removal (`/home/thanhnc/gauss_ws/data/nerf_object_removal/nerf-object-removal/0XX`, 5.0 GB including a 1.4 GB zip)
- **Per-frame files:**
  - `images/NNN.jpg`: 720x540 colour.
  - `depth/depth_i.bin`: float32 192x256, with `confidence/`.
  - `masks_real/NNN.npy`: 192x256 region of the object. It is present before AND after the removal, and non-empty in almost every frame.
  - `masks_synthetic/`: a virtual object, frames ≥ split_end only.
  - `sparse/` has one COLMAP camera per frame.
- **Per-scene files:** `boxes/bbox.txt` gives 8 box corners in the COLMAP frame, and `meta.json` gives the splits. Map frames are [0, split_start), update frames are [split_end, N), and the frames in between show the removal.
- **No meshes or point clouds:** every `.ply` and the lidar tar are 160-byte iCloud stubs.
- **Orientation:** P means the phone was held in portrait. The stored frame is sideways (image-right points along world +y, which is gravity), so rotate it 90° clockwise to show it upright. L means the frame is already upright. I worked this out from the mean camera axes.

| Scene | Split | Removed object (what the frames show) | Or. | Box m (L) | N: map/between/upd | Used map→KF / upd→KF (stride) | Map N Gs, PSNR | Map + relmax field |
|---|---|---|---|---|---|---|---|---|
| 001 | val | white fan heater on office carpet, beside a plant | P | 0.57×0.33×0.29 (54) | 510: 257/65/188 | 129→65 / 94→55 (2) | 1.00M, 20.4 | yes / yes |
| 002 | val | black mesh office chair at a desk | P | 1.05×0.74×0.67 (517) | 405: 169/80/156 | 85→52 / 78→48 (2) | 1.00M, 22.8 | yes / yes |
| 003 | val | pastel plush teddy bear on a grey sofa | P | 0.34×0.32×0.27 (30) | 406: 153/96/157 | 153→41 / 157→42 (1) | 1.00M, 29.9 | yes / yes |
| 004 | val | grey swivel armchair with orange cushion | P | 0.80×0.71×0.65 (367) | 363: 156/68/139 | 156→49 / 139→52 (1) | 1.00M, 25.8 | yes / yes |
| 005 | test | paperback book on a wooden desk by a monitor | P | 0.30×0.25×0.07 (5) | 240: 100/46/94 | 100→27 / 94→22 (1) | 1.00M, 31.6 | yes / yes |
| 006 | test | torn black-yellow cardboard box on stone paving by a wall (outdoor) | P | 0.50×0.19×0.17 (16) | 259: 108/49/102 | 108→32 / 102→35 (1) | 1.00M, 29.6 | yes / yes |
| 007 | test | black jacket on a granite slab (outdoor) | L | 0.48×0.42×0.13 (27) | 332: 152/49/131 | 152→33 / 131→39 (1) | 1.00M, 24.9 | yes / yes |
| 008 | test | grey-brown backpack against a stone block (outdoor) | L | 0.38×0.38×0.29 (41) | 488: 209/61/218 | 105→50 / 109→61 (2) | 1.00M, 21.3 | yes / yes |
| 009 | test | brown fabric laundry hamper on carpet by a white door | P | 0.66×0.65×0.62 (266) | 623: 298/48/277 | 149→33 / 139→33 (2) | 392k, 31.9 | yes / yes |
| 010 | test | white fan heater on parquet by a radiator | P | 0.50×0.44×0.44 (95) | 592: 235/184/173 | 118→33 / 87→29 (2) | 402k, 33.9 | yes / yes |
| 011 | test | black backpack on a low white box/panel on parquet | P | 0.63×0.53×0.53 (175) | 816: 251/101/464 | 126→34 / 232→41 (2) | 256k, 32.0 | yes / yes |
| 012 | excluded | black backpack on a staircase | P | 0.50×0.50×0.50 (125) | 537: 201/127/209 | — | — | no / no |
| 013 | test | brown paper shopping bag on the floor by a wall | P | 0.54×0.45×0.45 (109) | 633: 287/113/233 | 144→38 / 117→32 (2) | 153k, 38.2 | yes / yes |
| 014 | test | blue bicycle (pink saddle) at a yellow-brick wall corner (outdoor) | L | 1.68×0.99×0.40 (660) | 793: 396/100/297 | 132→85 / 99→79 (3) | 1.00M, 25.4 | yes / yes |
| 015 | test | grey bicycle with orange lock at a dark-brick wall (outdoor) | L | 1.53×0.92×0.36 (506) | 702: 286/147/269 | 143→66 / 135→69 (2) | 1.00M, 23.7 | yes / yes |
| 016 | excluded | rental e-bike between steel bollards; brand logo visible (outdoor) | L | 1.91×1.21×0.55 (1269) | 484: 149/132/203 | — | — | no / no |

- **Exclusions:** 012 and 016 are out because their before→after median depth error is 3.54 cm and 2.50 cm, over the 1 cm limit. Every other scene is between 0.24 and 0.68 cm (`/home/thanhnc/gauss_ws/data/uags_results/nor/check_nor.log`).
- **Box sizes, by volume only (no results used):** the largest val object is 002 (office chair). The largest included test object is 014 (bike).
- **File names:** maps are `/home/thanhnc/gauss_ws/data/uags_maps/nor_0XX_map.pt` (8.6–56 MB). Fields are `/home/thanhnc/gauss_ws/data/uags_fields/nor_0XX_lseg_relmax.pt` (0.17–1.13 GB), and they hold the mapping-pass state only.
  - **Path trap:** the repo's `configs/nor_{val,test}.json` point to `nor_0XX_lseg.pt`, which does not exist. The real paths are in `.../nor/tab2_relmax/config_{val,test}.json`.
- **Update pass is not saved:** `eval_change_nor.py` replays it, which takes 44–179 s per scene and about 5.8 GB of reserved GPU memory. It has no option to dump per-Gaussian scores or flags, so any change figure or video needs a new dump/render step.
- **Per-scene records** are in `/home/thanhnc/gauss_ws/data/uags_results/nor/tab2_relmax/test/0XX.json` and hold:
  - mapping and update keyframe indices;
  - per update frame: `n_covered`, `n_pos` (stale pixels), `n_mask`;
  - counts (stale, in-box, unobserved).

**Test results only** (G = Gaussian level, P = pixel level, AP in %). Source is `metrics.scenes[s].rows[row]` in `/home/thanhnc/gauss_ws/data/uags_results/nor/tab2_relmax/change_test.json`. Row names in that file: "ours" is "Ours: test, reset, re-admission", "fused" is ours + depth, "normlift" is the NormLift drop, "depth" is the depth residual.

| Scene | G ours | G ours+depth | G NormLift | G depth | P ours | P ours+depth | P NormLift | P depth | stale / all Gs | unobserved |
|---|---|---|---|---|---|---|---|---|---|---|
| 005 | 80.8 | 53.5 | 80.2 | 23.2 | 73.1 | 50.1 | 73.5 | 15.2 | 124.6k / 999k | 0.26 |
| 006 | 20.7 | 75.2 | 46.1 | 71.2 | 13.6 | 55.7 | 36.8 | 48.4 | 52.9k / 999k | 0.24 |
| 007 | 38.2 | 61.4 | 29.6 | 13.9 | 47.1 | 63.3 | 65.8 | 29.2 | 19.6k / 1.00M | 0.25 |
| 008 | 89.4 | 93.8 | 81.6 | 46.8 | 63.8 | 59.8 | 72.3 | 24.5 | 21.2k / 1.00M | 0.12 |
| 009 | 18.0 | 55.6 | 69.2 | 55.0 | 50.5 | 57.6 | 68.6 | 31.3 | 50.6k / 392k | 0.28 |
| 010 | 59.2 | 81.1 | 66.1 | 85.9 | 36.6 | 36.0 | 43.2 | 26.4 | 51.1k / 402k | 0.31 |
| 011 | 73.5 | 87.1 | 77.0 | 53.4 | 65.7 | 63.4 | 72.7 | 41.8 | 35.5k / 256k | 0.32 |
| 013 | 49.9 | 78.9 | 62.8 | 61.8 | 56.0 | 63.2 | 64.4 | 23.4 | 23.4k / 153k | 0.50 |
| 014 | 53.2 | 83.0 | 59.1 | 87.3 | 81.5 | 82.9 | 72.7 | 75.2 | 64.6k / 1.00M | 0.24 |
| 015 | 51.4 | 80.3 | 65.9 | 75.6 | 83.6 | 86.3 | 87.9 | 81.3 | 104.3k / 1.00M | 0.19 |
| **mean** | 53.4 | 75.0 | 63.8 | 57.4 | 57.2 | 61.8 | 65.8 | 39.7 | | |

The means match the paper's `tab:change`.

## 2. Replica (`/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset`, 7.6 GB)
- **Files:** `<scene>/Sequence_{1,2}/` hold `rgb/rgb_i.png` (640x480), `depth/depth_i.png` (uint16, mm), `semantic_class_i.png` and `vis_sem_class_i.png`, and `traj_w_c.txt` (c2w, OpenCV). There are 900 frames per sequence, plus `semantic_info/<scene>/info_semantic.json`. No meshes.
- **Protocol:** mapping uses every 5th frame of Sequence_1 (180 frames, 176–180 kept as keyframes). Evaluation uses every 10th frame of Sequence_2 (90 frames).

| Scene | Split | Map `uags_maps/<s>_seq1.pt` N, PSNR | KF | Full field `<s>_lseg.pt` | Other fields in `/home/thanhnc/gauss_ws/data/uags_fields` | Eval classes |
|---|---|---|---|---|---|---|
| office_0 | val | 620k, 41.5 | 179 | 782 MB | f3dgs, legs_l1/l2/l4, centre, legacy, pixels | 24 |
| room_2 | val | 954k, 37.1 | 179 | 1.2 GB | f3dgs, legs_l1/l2/l4, centre, legacy, pixels | 21 |
| room_0 | test | 1.00M, 34.2 | 179 | 1.3 GB | f3dgs, legs_l4, centre, legacy, pixels, **+ M=30 k-field** | 28 |
| room_1 | test | 1.00M, 36.2 | 178 | 1.3 GB | f3dgs, legs_l4, centre, legacy, pixels | 24 |
| office_1 | test | 459k, 40.5 | 176 | 572 MB | same as room_1 | 22 |
| office_2 | test | 1.00M, 37.3 | 178 | 1.3 GB | same | 21 |
| office_3 | test | 916k, 36.8 | 179 | 1.2 GB | same | 26 |
| office_4 | test | 516k, 29.6 | 180 | 619 MB | same | 16 |

- **What the variant fields are:**
  - f3dgs and legs: the distilled baselines.
  - centre, legacy, pixels: the ablation re-maps (`replica_tab3.sh`).
  - text caches: `uags_fields/text_cache_lseg.pt` and `uags_fields/scannet/text_cache_lseg.pt`.
- **k-view fields (M = 10/30/90) are deleted** (the views script ran with KEEP=0). The only one left is `/home/thanhnc/gauss_ws/data/uags_results/qual/kfields/room_0_lseg_k30.pt` (572 MB, 30 keyframes).
- **Fig. qual:** room_0 (test), evaluation frame 29, which is Sequence_2 `rgb_290.png`. It was picked by the rule "most e_red pixels" (`render_qual.py --list-frames`).
  - Counts: 76800 labelled pixels, 4170 e_red, 18007 e_irr.
  - Outputs: `/home/thanhnc/gauss_ws/data/uags_results/qual/qual_room_0_f29.{pdf,png,json}`, copied to `papers/figs/qual.pdf`.
- **Per-frame counts:** `/home/thanhnc/gauss_ws/data/uags_results/qual/frames_room_0.json` covers all 90 room_0 evaluation frames. Fields per frame: labelled, right_full, wrong_full, ale_defined, right_k, e_red, e_irr.
  - Top e_red frames: 29, 30, 36, 37, 32. Top e_irr frames: 89, 68, 67, 32, 69.
  - No other scene has per-frame counts, and the Table I eval JSONs hold per-scene values only.
- **Table I results:** `/home/thanhnc/gauss_ws/data/uags_results/tab1/eval_{val,test}.json`, plus `tab1_v2/` (negative v2) and `tab3/` (ablations).

## 3. ScanNet v2 (`/home/thanhnc/gauss_ws/data/scannet`, 23 GB)
- **Raw data** in `scans/<s>/` (21 GB): `.sens`, `color/` (1296x968 jpg), `depth/` (640x480 uint16 png), `label-filt/` (1296x968 uint16 raw ids, every frame), `pose/`, `intrinsic/`. **No 3D mesh or 3D labels.**
- **Prepared data** in `uags_f2/<s>/` (1.8 GB): `color/` (648x484 jpg, q95, all frames), `prepare.json`, and `depth/`, `label-filt/` and `pose/` symlinked to `scans/`.
  - `eval_s20o5/` holds the evaluation frames (5, 25, 45, …) as a colour copy, with `view.json` listing the frames and the "lost" ones (invalid pose).
- **Labels:** raw id → NYU40 via `scannetv2-labels.combined.tsv`, scored on the 19 OpenGaussian classes (`configs/vocab_scannet.json`).
- **Maps:** `/home/thanhnc/gauss_ws/data/uags_maps/scannet/<s>.pt`. Fields: `/home/thanhnc/gauss_ws/data/uags_fields/scannet/<s>_lseg.pt` (~1.2 GB) plus `_f3dgs` and `_legs_l1` for all 10 scenes; `_legs_l2/l4` exist for the val scenes only.

| Scene | Split | Frames | Map frames (stride 10) → KF | Eval frames | Classes | Map N, PSNR |
|---|---|---|---|---|---|---|
| 0000_00 | val | 5578 | 558 → 521 | 279 | 14 | 999k, 19.4 |
| 0062_00 | val | 730 | 73 → 62 | 37 | 6 | 999k, 20.5 |
| 0070_00 | test | 1322 | 133 → 124 | 66 | 9 | 999k, 26.4 |
| 0097_00 | test | 756 | 76 → 64 | 38 | 7 | 841k, 30.1 |
| 0140_00 | test | 4294 | 430 → 397 | 215 | 8 | 999k, 24.5 |
| 0200_00 | test | 1110 | 90 (21 bad poses) → 78 | 46 (10 lost) | 8 | 999k, 17.5 |
| 0347_00 | test | 1080 | 108 → 80 | 54 | 7 | 999k, 19.1 |
| 0400_00 | test | 1261 | 123 (4 bad poses) → 117 | 61 (2 lost) | 7 | 999k, 18.8 |
| 0590_00 | test | 2689 | 269 → 243 | 135 | 14 | 999k, 30.4 |
| 0645_00 | test | 4503 | 450 (1 bad pose) → 424 | 225 | 16 | 999k, 21.7 |

Results are in `/home/thanhnc/gauss_ws/data/uags_results/scannet_tab1/eval_{val,test}.json`, per scene only, with no per-frame data.

## 4. Views experiment (`/home/thanhnc/gauss_ws/data/uags_results/views/`, 1.5 MB)
- **Per-scene files** `views_{val,test}_<scene>.json`: val is office_0 and room_2, test is the 6 scenes.
  - For each M in {10, 30, 90}: `n_mapping_views`, the k-field path (deleted), and rows ours_P0, ours_P2, averaging, normlift, viewcount.
  - Each row has:
    - AUSE of msp, entropy, u_epi and u_ale against e_red and against e_irr;
    - n_pixels, n_hole, n_e_red, n_e_irr, n_undefined, miou_k;
    - `on_ale_defined`, the same values on the subset where u_ale is defined.
  - `full` holds the full-map mIoU per row only. The M = 180 point comes from the Table I file (`plot_views.py`).
- **Merged files:** `views_val.json` (N=2) and `views_test.json` (N=6), with mean/std per M, row, score and target.
- **Logs:** `map_<s>_k{10,30,90}.*.log`, `eval_*.log`, `scenes_*.json`.
- **No per-pixel or per-frame data.** A new views figure or video needs the k-fields mapped again.
- `/home/thanhnc/gauss_ws/data/uags_results/views_repro/` holds the room_0 cross-machine repeat, same structure.

## 5. Disk and tools
- **Disk:** `/` (which also holds `/tmp`) has 428 GB free of 1.9 TB (77% used).
- **`/home/thanhnc/gauss_ws/data` is 88 GB:**

| Folder | Size |
|---|---|
| uags_fields | 43 GB |
| scannet | 23 GB |
| replica_semnerf | 7.6 GB |
| weights | 5.4 GB |
| nerf_object_removal | 5.0 GB |
| uags_cache | 2.5 GB |
| uags_maps | 1.5 GB |
| uags_results | 645 MB |

- **Tools in `~/venvs/uags`:** imageio-ffmpeg 0.6.0, whose bundled ffmpeg 7.0.2 binary exists in the venv; PIL 12.3; OpenCV 4.11; matplotlib 3.11.2; imageio 2.38.
- **GPU at the time of the check (read only):** 5.0 of 12.3 GB in use by other users.
- **Third-party marks to avoid in figures:** the 016 e-bike logo, product branding on the 006 box, and readable book titles in some 004 frames. These are not lab marks, but better left out.