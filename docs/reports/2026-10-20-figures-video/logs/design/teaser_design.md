# Teaser (Fig. 1) design, layout A: one NeRF-Object-Removal validation scene (002, office chair removed). It shows one update camera pose before and after the update in a 2x5 grid, plus a curve of forgetting over the update. Supporting scratch files, all CPU-only with no results rendered: /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/teaser/teaser_layout_mockup.png (layout at print size, placeholders only, 7.16 x 2.57 in), /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/teaser/premise_val.json (labels of the stale Gaussians before any update, val 001-004), /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/teaser/check_premise.py, /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/teaser/check_support.py, /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/teaser/val_frames_rotated.jpg, /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/teaser/002_060_335_full.jpg. Nothing was written in the repo or the data folders (checked with find -newer), and no GPU job ran.

I chose layout A: a single NOR validation scene (002, office chair removed; picked by the rule "largest removed-object box volume", 517 L), shown at one update camera pose (frame 335, the update keyframe with the largest masks_real area). Row 1 is the map after mapping: render, p("chair") for averaging and for ours, u_ale and u_epi. Row 2 is after the full Table III update: the real after-removal photo, averaging's p("chair"), ours' p("chair"), the Gaussians our test reset, and u_epi. Panel (k) plots, over the 48 update keyframes, the share of the chair's Gaussians whose top class is still "chair" and the share of static Gaussians that were relabelled. The comparison is truthful by construction, which a CPU check confirmed: for every Gaussian with no full view, ours' mu* has exactly the direction of the averaging feature (median cosine 1.000 on the stale Gaussians of all 4 val scenes). So the two rows are the same map before the update and differ afterwards only where our test reset Gaussians. The ground truth is the rendered D2 stale Gaussians, not the coarse masks_real, and every caption word depends on a measured number. One new script (scripts/render_teaser.py) replays the exact Table III update path by importing eval_change_nor.py functions without editing them, checks itself against the saved val record, and dumps arrays for a CPU compose stage. Expected cost is about 12 GPU-minutes. The main risk is that averaging also dilutes the chair away; the figure is rendered by the fixed rules whatever the outcome.

# Teaser (Fig. 1) design for UAGS

Design phase only (2026-10-20). Nothing in the repo or data folders was written (checked with `find -newer` on both trees), and no GPU job ran. Scratch folder: `/tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/teaser/` (S below). R = `/home/thanhnc/gauss_ws/src/uags`, D = `/home/thanhnc/gauss_ws/data`.

## 0. Facts checked in this phase (CPU only, validation scenes only)
| Fact | Value | Source |
|---|---|---|
| Box volume of the removed object (val) | 001 54 L, **002 517 L** (1.05x0.74x0.67 m), 003 30 L, 004 367 L | `usplat/box3d.py:77` read_box_corners + box_extents on `<scene>/boxes/bbox.txt` |
| 002 update keyframe with the largest masks_real area (of the 48 Table III keyframes) | **frame 335** (38th of 48), 36,552 of 49,152 px; next 333, 337 | `D/uags_results/nor/tab2_relmax/val/002.json` views.update + `masks_real/335.npy` |
| Top class of the D2-stale Gaussians after mapping (18 cached ScanNet prompts; bookshelf not cached) | 002: chair 80.5 % (floor 13.1 %); 004: chair 74.8 %; 001: wall 94.8 %; 003: bed 99.8 % | `CUDA_VISIBLE_DEVICES="" python3 S/check_premise.py S/premise_val.json` (1.9 min) |
| cos(mu*, averaging feature) on stale Gaussians after mapping | median 1.000 in all 4 val scenes | same run |
| Why the two are equal | `vmf_field.py:428` w = min(1, m/(a0HW)). When every view has m < a0HW, R = sum_v R_v/(a0HW), which is parallel to averaging's F = sum_v R_v (`baselines.py:201-203`). Also mu* = R/norm(R) (`vmf_field.py:719`, prior off). So the two rows are the same map before the update. Afterwards they differ only at Gaussians our test reset (`vmf_field.py:532`), or that got a full view. | code |
| Mapping keyframes that see a stale Gaussian, 002 / 004 | median 42 of 52 / 42 of 49 (mass-weighted median support 3.08 / 5.80 views) | `python3 S/check_support.py 002 004` (ViewCount and NormLift state in the field file) |
| Update keyframes covering a stale Gaussian, 002 | median 15 of 48 | val/002.json counts.update_views_stale |
| u_ale defined (support >= 3) | 002: 3.0 % of Gaussians, 39.6 % of the mapping pixel mass (001 52.5 %, 003 60.1 %, 004 51.1 %) | inline CPU command on raw_W1, raw_W2 and the NormLift S1 |
| Ours on 002 (val record, Table III code) | flagged: Rec 53.6 %, FA 14.0 %; changed at the end: Rec 50.6 %, FA 2.7 %; 117,170 resets; threshold pooled 0.1845; 84,205 stale and 493,103 static observed Gaussians | val/002.json metrics |
| Stale share in the robust set | 002: 10.5 % after mapping -> 9.0 % after the update; test mean 8.2 % -> 4.8 % | val/002.json and test/0XX.json metrics.ours.kept_after_mapping / kept |
| 002 frames are portrait | rotating 90 deg clockwise makes them upright (checked visually for 002, 003, 004) | S/val_frames_rotated.jpg |
| Third-party marks in frame 335 | monitor logo (top edge), mouse-pad logo (upper middle) | S/002_060_335_full.jpg |
| masks_real is a coarse dilated region | in 002 it also covers the desk and the monitor | S/val_frames_rotated.jpg |

## 1. Three layouts considered
| | Layout | For | Against |
|---|---|---|---|
| **A (chosen)** | One NOR val scene, one update camera pose, 2 rows (after mapping / after the update) x 5 square panels, plus a curve panel (k) | One story. Both uncertainties shown. Both methods before and after, so the after-update difference can only come from the reset. (k) gives whole-scene numbers, not one frame. | Dense (10 panels). The u_ale panel is partly grey on NOR. |
| B | Three 2.3 in hero panels: map + query before; averaging after; ours after; u_epi and u_ale as small insets; reset contour | Readable at a glance | Uncertainties only as tiny insets. Does not show that averaging equals ours before the update. No scene-level number, so one frame carries the claim. |
| C | Timeline: 6 time steps (3 mapping and 3 update keyframes, each at its own pose) x rows (camera, ours p, averaging p, u_epi) | Shows the dynamics | Panels at different poses are hard to compare. Needs a mapping replay. 18-24 tiny panels. Panel (k) of A shows the same dynamics in one panel. Better suited to the video. |
| (D, dropped) | Replica uncertainty (left) + NOR change (right) | Each claim shown on its best data | Incoherent, and it repeats Fig. qual |

## 2. Layout A in detail
Size: 7.16 x ~2.6 in, full-width title block (checked with the mock-up S/teaser_layout_mockup.png, which has no results, at 7.16 x 2.57 in). Geometry:
- 0.17 in left strip for the row labels "After mapping" / "After the update".
- 5 square panels of 1.0 in with 0.04 in gaps.
- 0.38 in gap, then plot (k), about 1.45 in wide, spanning both rows.
- 0.15 in title line above each row and a 0.27 in colour-bar strip.
- Fonts: RQ.RC (`scripts/render_qual.py:69`, serif TrueType), 7 pt titles, 6 pt labels.
- Panels (b) and (g) have a thin vermillion #D55E00 frame (averaging); (c), (h), (i), (j) have a blue #0072B2 frame (ours). These are the `plot_views.py:69-79` colours.

All image panels are rendered at full resolution (720x540, the frame's own K), rotated 90 deg clockwise, and cropped by rule R7.

| Panel | Content | Map state | Code path | Colours / marks |
|---|---|---|---|---|
| (a) Map render | RGB of the 3DGS map at pose t* | after mapping | `render_qual.py:126` render_rgb | RGB |
| (b) Averaging: p("chair") | softmax over the 19 ScanNet classes of beta * F.t, with averaging's own beta 38.66 | after mapping (saved NormLift state) | TensorRow(buf=nl.feature()) `rows.py:67`, `baselines.py:206` -> render_row `seg_eval.py:109` -> class_scores `seg_eval.py:161` | #CC79A7 overlay with alpha 0.85p on the dimmed grey of (a); no tint where alpha < 0.5 |
| (c) Ours: p("chair") | same with ours P0 (mu*), beta 39.21 | after mapping | freeze(field_rows(ours, modes=("P0",))[0]) `rows.py:133,338`; P0 buffer = mu_post `predictive.py:45` | same |
| (d) Ours: u_ale | rendered 1 - rho_hat; grey where undefined (< 3 views, or > 50 % of alpha undefined) | after mapping | rr.unc["u_ale"] (`vmf_field.py:748`), undef_frac 0.5 | viridis from 0 to its q99, `render_qual.py:211` u_image, `:248` u_max |
| (e) Ours: u_epi | rendered 1/(1+s) | after mapping | rr.unc["u_epi"] (`vmf_field.py:734`) | viridis, one scale shared with (j) (q99 of both) |
| (f) Camera, chair removed | the real photo of frame t* | — | dataset frame `datasets.py:613` | RGB; dashed GT outline |
| (g) Averaging: p("chair") | as (b) | after all 48 update keyframes (saved NormLift on ours' hooks) | as (b) | overlay on the grey of (f); GT outline; text "mean p = x.xx" over the GT region |
| (h) Ours: p("chair") | as (c) | after the update (the Table III end state) | as (c) | same |
| (i) Ours: reset by test | alpha-weighted share of the Gaussians reset at least once in the update (Table III "flagged", Rec./FA) | after the update | TensorRow(buf=zeros(N,1), unc={"reset": (n_resets > r0)}) -> render_row (pattern of `eval_change_nor.py:659` pixel_level) | #0072B2 overlay with alpha 0.85 x share on grey (f); GT outline; text "reset: A % of the chair's, B % of static Gaussians" |
| (j) Ours: u_epi | as (e) | after the update | as (e) | viridis, scale of (e); GT outline |
| (k) Top class during the update | x = update keyframes 0..48 (0 = after mapping = row 1; 48 = row 2). Solid: % of the stale observed Gaussians whose top class (argmax over the 19 classes) is "chair". Dashed: % of the static observed Gaussians whose top class differs from their class after mapping. | per step | per-Gaussian argmax(f . t_c), no temperature | ours #0072B2, averaging #D55E00; end values labelled |

GT outline: contour at 0.5 of the rendered D2-stale share, the Gaussian ground truth of Table III (`eval_change_nor.py:572`). It is drawn white-dashed (0.6 pt) over a black halo (1.0 pt) on every row-2 panel. masks_real is not used for the outline because it is too coarse.

## 3. How "averaging keeps / ours forgets" is shown truthfully
1. **Same map, same pose, same predictive form.** Each row uses its own validated temperature. Row 1 shows that the two maps are identical before the update (cosine 1.000, section 0). Row 2 is after the same 48 update views. The only mechanism that differs is our test's reset of the four sums (`vmf_field.py:529-543`). Averaging keeps adding every view (`baselines.py:201`), and NormLift has the same feature, so averaging stands for both. The caption says so.
2. **The base image in row 2 is the real photo after the removal.** The overlay is the map's render. A tinted region over empty floor is therefore stale semantics. Geometry is fixed and the caption says so.
3. **The ground truth is the Table III D2 stale set**, rendered (section 2).
4. **Numbers, not adjectives.** Panels (g), (h) and (i) print measured means over the GT region and the whole-scene reset shares. Panel (k) covers all 84,205 stale observed Gaussians of the scene, not one frame.
5. **Caption wording depends on the measured values**, as rules C1-C3 below. Nothing is re-selected after rendering. If averaging also lost "chair", the figure is published anyway with neutral wording ("averaging forgets only by dilution").
6. **The robust set is not shown.** It already excludes about 90 % of the stale Gaussians before any update (002: 10.5 % kept; test mean 8.2 %), so it would wrongly credit the change test.

## 4. Data and settings
- Scene: NOR 002 (val).
  - Map: `D/uags_maps/nor_002_map.pt` (999,658 Gaussians, PSNR 22.8).
  - Field after the mapping pass: `D/uags_fields/nor_002_lseg_relmax.pt`. It has the NormLift state with with_averaging=True (F, S1, S2, Sn) and FieldConfig relmax, fast_min = slow_min = 0.5, k = 2, q = 0.999.
  - Val record: `D/uags_results/nor/tab2_relmax/val/002.json`.
- Update: align "none", update stride 2, the 48 keyframes of the record. Grid 360x270 (Table III). LiDAR high-confidence depth.
- Query: "chair". Vocabulary: the 19 classes of `configs/vocab_scannet.json`, with its 5 LSeg templates (`templates_by_extractor.lseg`).
  - Text embeddings come from `ES.text_embeddings` (`eval_seg.py:141`) with a COPY of `D/uags_fields/scannet/text_cache_lseg.pt`. The copy holds 18 prompts; "bookshelf" is computed by the LSeg text encoder on the CPU.
  - Texts are projected with the scene PCA without renormalising (`feature_pca.py:59`).
- Temperatures: `D/uags_results/scannet_tab1/eval_val.json` (ES.load_betas `eval_seg.py:555`): ours_P0 39.21, averaging 38.66. These are for display only; (k) uses no temperature.

## 5. Selection rules
Fixed 2026-10-20, before any render. Validation results of every row were already known when these rules were written; the protocol allows that for validation scenes.
- R1 Scene: the NOR validation scene whose removed object has the largest 3D box volume. This uses bbox.txt only, no method output, and gives 002.
- R2 Technical fallback only: 004 (second-largest box). Use it only if 002 fails a technical check:
  - the replay is off the val record by more than 1 % in resets;
  - fewer than 90 % of the crop's pixels are covered (alpha >= 0.5) at t*;
  - brand text can be neither cropped nor blurred.

  Never switch because of how averaging or ours look.
- R3 Pose t*: the update keyframe (of the 48 used by Table III) with the largest masks_real area, ties going to the earliest. This gives frame 335. Both rows use this pose.
- R4 States: row 1 = the saved mapping-pass field (the Table III input). Row 2 = after all 48 update keyframes through the Table III code path: calibrate q = 0.999, then update_view with LiDAR depth, with reset and re-admission.
- R5 Query and vocabulary:
  - The query is the removed object's ScanNet-19 class ("chair"). The vocabulary is the 19 ScanNet classes (frozen 2026-10-10).
  - No prompt tuning. No other query is tried; any extra query must be disclosed.
- R6 Predictive: P0 for ours and the averaging row, each with its own ScanNet-validation temperature. Panel (k) uses the argmax, so it has no temperature.
- R7 Display:
  - Rotate 90 deg clockwise (portrait capture).
  - Crop a 540x540 square from the 540x720 upright frame. Centre it vertically on the centroid of the pixels whose rendered D2-stale share is >= 0.5, clamped to the frame. Use the same crop for every panel.
  - If brand text is still readable in the crop, blur it and say so in the caption.
- R8 Scales, fixed in advance:
  - p in [0,1];
  - u_epi from 0 to the q99 of the shown covered pixels of (e) and (j) together;
  - u_ale from 0 to the q99 of (d);
  - reset share in [0,1];
  - coverage alpha >= 0.5 (tau_cov, as Table I).
- R9 Populations in (k) are those of Table III: stale = D2 stale and covered by at least 1 update keyframe; static = covered and not stale.
- R10 Text on the panels = means over the pixels with D2 share >= 0.5 and alpha >= 0.5 at t*. Reset shares are taken over the R9 populations.
- C1-C3 Caption words:
  - C1: "still answers 'chair'" only if the mean p of averaging over the GT region is >= 0.5 after the update.
  - C2: "does not" only if ours' mean p there is < 0.5.
  - C3: "u_epi rises where reset" only if the mean u_epi over the GT region is higher in (j) than in (e).
  - Otherwise the caption is neutral and gives the numbers.
- R11 Supplement (recommended): the same layout and rules for all four val scenes. Queries are fixed now: 001 "heater", 002 "chair", 003 "teddy bear", 004 "chair". "heater" and "teddy bear" are added as a 20th class.

## 6. Code plan: `scripts/render_teaser.py` (new; never edit any file hashed by `eval_change_nor.py:191` CODE)
The script has two sub-commands: `render` (GPU, writes an NPZ with every map and curve) and `compose` (CPU, NPZ -> PDF/PNG + meta JSON). Splitting them lets the layout be changed without the GPU.

**render**
1. Reserve GPU memory: `usplat/gpu_reserve.py:23` apply.
2. Load the scene with `ctx = EC.load_context(Namespace(field=F, align="none", ply="", data="", update_stride=0, max_update_keyframes=K, device, backend, channel_chunk=0))` (`eval_change_nor.py:697`). This:
   - checks that the field is mapping-only (`:197`) and was built on this map (sha256);
   - selects the keyframes (`time_field.py:136`);
   - extracts LSeg tokens on the full frames (`:251`);
   - builds the frozen colours and mu*.
3. Check that `[v["index"] for v in ctx["upd_views"]]` equals the record's views.update (skipped in smoke mode). Set t* by R3.
4. Text: ES.load_vocab (`eval_seg.py:87`) -> ES.text_embeddings (`:141`, writes only the cache copy) -> pca.project_text. Betas: ES.load_betas (`:555`), with ES.check_fit_settings as in render_qual.
5. D2 truth:
   - `corners = read_box_corners(ctx["map_ds"].box_path)`
   - `P, C, _ = EC.update_depth_points(ctx["data"], None, corners, EC.INFLATE + EC.NEAR)` (`:551`)
   - `gt = EC.stale_ground_truth(means, corners, P[C >= EC.HIGH_CONFIDENCE])` (`:572`), the same D2 path as EC.ground_truth (`:745`).
6. Fields: `ours = DirectionalSemanticField.from_state(ckpt["field"], dev)` (`vmf_field.py:811`) and `nl = EC.make_normlift(ckpt, ours, saved=True)` (`eval_change_nor.py:349`).
   - EC hangs the saved NormLift on `disp`, a copy with no change test (`:481-483`).
   - Hanging it on ours gives identical sums: update_view builds ViewStats R_v, m_v from geometry and z only (`vmf_field.py:374-391`). _ingest does not change them in place, and the hook runs after _ingest (`:391`).
   - The CPU test below checks this equality.
7. Full-resolution rasterizer at t*: frame from `make_dataset("nor", data, stride=1, part="update")`, `viewmat = inv(c2w)`, `ras = ctx["rasterizer"](splats, viewmat, K_full, 720, 540)` (`time_field.py:189`).
   - Row-1 renders: rgb, rr_ours (P0 + u_epi, u_ale), rr_avg, and the GT share (TensorRow unc={"stale"}).
   - Also store lab0 = the per-Gaussian argmax of mu* . t and of avg . t (-1 where the feature is 0).
8. `ours.calibrate_change_threshold(0.999)` (`vmf_field.py:581`, as `eval_change_nor.py:479`). Assert that thr_by_hist equals the record's metrics.ours.threshold.by_history. Set `r0 = ours.n_resets.clone()`.
9. For each v in ctx["upd_views"] (as `eval_change_nor.py:499-509`):
   - `z, ras_v, gz, kw = EC.view_inputs(ctx, v)` (`:278`)
   - `ours.update_view(ras_v, z, num_pixels=v["h"]*v["w"], **kw)` (`vmf_field.py:344`)
   - `observed |= EC.covered(ours, stats)` (`:302`)
   - store int8 labels argmax(mu* . t) and argmax(nl.feature() . t) for all N, about 1 MB per step.
10. Row-2 renders at t* (end state): rr_ours (P0 + u_epi), rr_avg, reset share, changed share (meta only), and the camera image.
11. Checks against the record (stop if they fail):
    - resets: |sum(n_resets - r0) - 117,170| <= 1 %;
    - Rec/FA of (n_resets > r0) on the R9 populations within 0.01 of 0.536 / 0.140;
    - n_stale_observed within 0.5 % of 84,205.
12. Curves (k) from the stored labels. Save an NPZ with float16 maps in the stored orientation, plus the curves and a meta block (rules, t*, betas, checks, timing, `torch.cuda.max_memory_allocated`, sha256 of the field, map and script, git commit).

Optional for the video: `--snapshots each` dumps the same row-2 maps at every update view's own pose.

**compose**: read the NPZ; rotate with np.rot90(k=-1); crop by R7. Use RQ.RC, RQ.u_image and RQ.u_max for colours, matplotlib contour for the GT outline, the overlays of section 2 and plot (k). Write with RQ.save (`render_qual.py:363`: PDF at 600 dpi plus PNG). The meta JSON holds every printed number (callouts, shares, vmax, checks) so the caption is filled from it, never typed by hand.

## 7. CLI and run order (implementation phase)
```bash
cd /home/thanhnc/gauss_ws/src/uags
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
D=/home/thanhnc/gauss_ws/data; O=$D/uags_results/teaser; mkdir -p $O/002
cp $D/uags_fields/scannet/text_cache_lseg.pt $O/text_cache_lseg.pt          # copy: ES.text_embeddings writes it
python3 -m pytest -q tests/test_render_teaser.py                            # CPU toy, mock backend
A="--field $D/uags_fields/nor_002_lseg_relmax.pt --record $D/uags_results/nor/tab2_relmax/val/002.json --vocab configs/vocab_scannet.json --extractor lseg --query chair --betas $D/uags_results/scannet_tab1/eval_val.json --text-cache $O/text_cache_lseg.pt --frame-rule max-mask --rotate cw --res full"
scripts/gpu_gate.sh 6000 python3 scripts/render_teaser.py render $A --max-update-keyframes 6 --out $O/002/smoke.npz   # smoke, no record check
scripts/gpu_gate.sh 6000 python3 scripts/render_teaser.py render $A --out $O/002/teaser_data.npz 2>&1 | tee $O/002/render.log
python3 scripts/render_teaser.py compose --data $O/002/teaser_data.npz --out $O/teaser_002.pdf --meta $O/002/teaser_meta.json
# optional supplement: same with 001 (--query heater), 003 (--query "teddy bear"), 004 (--query chair)
# after review: cp $O/teaser_002.pdf papers/figs/teaser.pdf and add a row to papers/figs/README.md
```

## 8. Tests: `tests/test_render_teaser.py` (CPU, mock backend)
Reuse the toy NOR scene of `tests/test_eval_change.py` (`build`, `Fake`): a red object in front of a textured wall, removed in the update. Monkeypatch `EC.load_extractor` to Fake (as `test_eval_change.py` run does) and the text step to one-hot texts (identity PCA: Fake channel 0 = wall, 1 = object). Checks:
1. The teaser replay's resets equal ours' resets from `EC.main` on the same toy.
2. NormLift on ours' hooks equals NormLift on a copy with no change test (EC's `disp`): F and S1 within 1e-6.
3. Before the update, p(object) > 0.5 on object pixels for both rows, and mu* is parallel to the averaging feature for Gaussians with only sub-full views.
4. After the update, on the object region, ours' p(object) is below averaging's. Curve point 0 equals the mapping-state share and point V equals the end-state share.
5. compose writes a PDF/PNG 7.16 in wide with TrueType fonts (as `test_render_qual.py:139`).

## 9. LaTeX placement (implementation phase; one source line per paragraph)
IEEEtran journal cannot put a figure* on page 1. Use the title block instead: `\IEEEaftertitletext` is placed inside `\twocolumn[...]` with the title (IEEEtran.cls line 4772; it is locked out only in conference mode).
- Preamble:
  ```latex
  \makeatletter\newcommand{\teasercaption}[1]{\def\@captype{figure}\caption{#1}}\makeatother
  ```
- Before `\maketitle`:
  ```latex
  \IEEEaftertitletext{\vspace{-0.5\baselineskip}\noindent\begin{minipage}{\textwidth}\centering\includegraphics[width=\textwidth]{figs/teaser.pdf}\teasercaption{...}\label{fig:teaser}\end{minipage}\vspace{0.5\baselineskip}}
  ```
- Cite Fig.~\ref{fig:teaser} in the first paragraph of the Introduction. The pipeline figure becomes Fig. 2.
- Compile with TinyTeX pdflatex + bibtex (tectonic is not installed) and check page 1.

## 10. GPU time and memory
- Table III val timings for 002: load + LSeg 66 s, update 32 s, ground truth 15 s; peak alloc 4.56 GB, which included a second field copy and the residual baselines.
- The teaser run skips the replay and the pixel metrics, so expect about 2 min wall and a peak of about 3.5 GB.
- Gate: `scripts/gpu_gate.sh 6000`.
- Total: smoke about 1.5 min + 002 about 2 min + supplement (001, 003, 004) about 5 min, so about 9 min. **Budget 12 GPU-minutes.**
- Host RAM < 6 GB.

## 11. Caption template (numbers from teaser_meta.json; words gated by C1-C3)
"A semantic 3D-GS map that knows how much evidence it has and forgets what changed. NeRF-Object-Removal [weder2023removing], validation scene 002: a chair is removed between the mapping and the update recordings. Both rows show the same update camera pose; the map geometry is fixed. Top, after mapping: (a) map render; (b, c) probability of the text query 'chair' for averaging [joseph2025gradbp] (also NormLift's feature) and for our field, identical before any change; (d) u^ale, whether the views agree (grey: fewer than 3 views); (e) u^epi, how much view evidence there is. Bottom, after the update: (f) camera image (dashed: the removed chair's Gaussians, ground truth); (g) averaging still answers 'chair' there (mean p = X); (h) ours does not (Y), because (i) our test reset A % of the chair's Gaussians (B % of the static ones), and (j) their u^epi rises. (k) Over the 48 update keyframes: share of the chair's Gaussians whose top class is 'chair' (solid) and of static Gaussians whose top class changed (dashed)."

RULES:['R1 Scene: the NeRF-Object-Removal validation scene whose removed object has the largest 3D box volume (boxes/bbox.txt, no method output). This gives 002, office chair, 1.05x0.74x0.67 m = 517 L (001 54 L, 003 30 L, 004 367 L). Written 2026-10-20 before any teaser render; validation results of all rows were already known, which the protocol allows for validation scenes.', "R2 Technical fallback only: scene 004 (second-largest box). Use it only if 002 fails a technical check: the replay is off the val record by more than 1 % in resets, fewer than 90 % of the crop's pixels are covered at t*, or brand text cannot be removed. Never switch because of how averaging or ours look.", 'R3 Pose t*: the update keyframe (of the 48 used by Table III) with the largest masks_real area, ties to the earliest. This gives frame 335 (36,552 of 49,152 px; 38th of 48). Both rows are rendered at this pose.', 'R4 States: row 1 is the saved mapping-pass field (Table III input). Row 2 is the state after all 48 update keyframes through the Table III code path: calibrate q = 0.999, update_view with LiDAR depth, reset, re-admission.', "R5 Query and vocabulary: the query is the removed object's ScanNet-19 class, 'chair'. The vocabulary is the 19 ScanNet classes of configs/vocab_scannet.json (frozen 2026-10-10) with its 5 LSeg templates. No prompt tuning; no other query is tried, and any extra query is disclosed.", 'R6 Predictive: P0 for ours and the averaging row, each with its own ScanNet-validation temperature (39.21, 38.66; scannet_tab1/eval_val.json). The curves in (k) use the argmax, so no temperature.', 'R7 Display: rotate 90 deg clockwise (portrait capture). Crop a 540x540 square from the 540x720 upright full-resolution frame, centred vertically on the centroid of the pixels whose rendered D2-stale share is >= 0.5, clamped to the frame; the same crop for every panel. Blur brand text only if it is still readable, and say so.', 'R8 Colour scales fixed in advance: p in [0,1]; u_epi from 0 to the q99 of the shown covered pixels of the before and after panels together; u_ale from 0 to its q99; reset share in [0,1]; coverage alpha >= 0.5.', 'R9 Populations in (k) are those of Table III: stale = D2 stale and covered by at least one update keyframe; static = covered and not stale.', 'R10 Text on the panels = means over the pixels with rendered D2 share >= 0.5 and alpha >= 0.5 at t*. Reset shares are taken over the R9 populations.', "C1-C3 Caption words: 'still answers chair' only if averaging's mean p over the GT region is >= 0.5 after the update; 'does not' only if ours' mean p there is < 0.5; 'u_epi rises' only if the mean u_epi over the GT region rises. Otherwise use neutral wording with the numbers. No scene, frame or query is re-selected after rendering.", "R11 Supplement: the same layout and rules for all four validation scenes. Queries fixed now: 001 'heater', 002 'chair', 003 'teddy bear', 004 'chair'. 'heater' and 'teddy bear' are added as a 20th class."]
SCRIPTS:['scripts/render_teaser.py (render: GPU replay of the Table III update by importing eval_change_nor.py functions, dumps an NPZ with every map and curve; compose: CPU, writes the PDF/PNG and the meta JSON). CLI: python3 scripts/render_teaser.py render --field F --record VAL_JSON --vocab configs/vocab_scannet.json --extractor lseg --query chair --betas scannet_tab1/eval_val.json --text-cache COPY --frame-rule max-mask --rotate cw --res full [--max-update-keyframes N] [--snapshots each] --out NPZ ; python3 scripts/render_teaser.py compose --data NPZ --out PDF --meta JSON', "tests/test_render_teaser.py (CPU, mock backend, reuses the toy NOR scene of tests/test_eval_change.py: resets equal EC.main's, NormLift on ours' hooks equals NormLift on EC's no-reset copy, mu* parallel to averaging before the update, ours' p(object) below averaging's after it, curve end points consistent, 7.16 in TrueType output)"]
GPU:12
RISKS:["Averaging may also forget (look clean) by dilution. On 002 a stale Gaussian is seen in a median 42 of 52 mapping keyframes and covered in a median 15 of 48 update keyframes. The background seen through it shifts with parallax, so it should average to an incoherent direction and 'chair' should mostly survive, but this is unverified. The figure is rendered by the fixed rules anyway, and the caption uses numbers with neutral wording if C1 fails.", 'Ours resets only about half of the chair on 002 (val record: 53.6 % of the stale Gaussians flagged, 50.6 % still changed at the end) and has many false resets (FA 14.0 %, 2.7 % at the end). So (h) may keep a partial ghost and (i) may show speckles on static content such as the desk the chair hid. The numbers go into the panel text and the caption.', 'The u_ale panel (d) may be mostly grey. On 002, u_ale is defined for only 3.0 % of the Gaussians and 39.6 % of the mapping pixel mass (support >= 3 views). The caption explains grey; undef_frac stays 0.5 as in Table I.', "The rise of u_epi after the reset may be weak at the end of the update, because reset Gaussians keep gaining support from later update views. 'u_epi rises' is claimed only if it is measured (C3).", "Do not use the robust set to show forgetting. It already holds only 10.5 % of 002's stale Gaussians after mapping (test mean 8.2 % -> 4.8 % after the update). This also affects the paper's sentence 'only 4.8 % of the stale Gaussians stay in our robust set', which should give the 8.2 % reference.", 'masks_real is a coarse dilated region (in 002 it covers the desk and the monitor). The outline therefore uses the rendered D2 stale Gaussians. The masks_real-based pose rule R3 still only picks a pose.', 'Render quality at an update pose: the 002 map has PSNR 22.8, and t* is a novel pose for the map, so floaters or blur may appear in (a). The real photo in (f) offsets this.', 'Third-party marks: the monitor logo and the mouse-pad logo in frame 335. Rule R7 should crop the monitor; blur anything still readable and say so. Double-anonymous: no names or lab marks anywhere.', 'The display temperatures are ScanNet-validation fits applied to NOR (no NOR labels exist). This affects only the p shading; (k) and the label decisions use the argmax.', "The 'bookshelf' text embedding is not in the ScanNet cache. It must be computed by the LSeg text encoder on the CPU into a COPY of the cache, because ES.text_embeddings rewrites --text-cache.", 'IEEEtran journal cannot place a figure* on page 1. The teaser must go into \\IEEEaftertitletext with a caption helper, and this must pass a compile check (TinyTeX pdflatex + bibtex; tectonic is not installed).', 'Do not edit any file hashed by eval_change_nor.py code_state (eval_change_nor.py, time_field.py, eval_seg.py, change_baselines.py, evaluation.py, seg_eval.py, rows.py, nor_align.py, box3d.py), or the Table III provenance changes. render_teaser.py imports them only.', 'The GPU replay is not bit-exact. Checks against the val record (thresholds equal; resets within 1 %; Rec/FA within 0.01; n_stale_observed within 0.5 %) must pass before the figure is used. If they fail, stop and investigate instead of publishing a figure from a different state.', 'The forgetting curve (k) is a new, validation-only quantity. Putting it on the 10 test scenes as a paper metric would be an extra test look: pre-register the definition, run it once, and disclose it.', 'Validation results of all rows were seen before the rules were written (allowed for validation). The recommended supplement with all four validation scenes in the same layout is the guard against cherry-picking.']