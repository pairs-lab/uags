# A design for the qualitative results, returned in `spec`. I changed nothing in the repo or the data folders and ran no GPU job. The design has four figures (N1 stale content, N2 text query, S baseline comparison, F failure cases) and two supplementary grids. For each one it gives the selection rule, which is fixed now and already applied on the CPU from published JSONs and GT labels only, plus the panels, colormaps, the scripts to write with their CLIs, the verification checks and a GPU budget. Evidence from this phase is in /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/qual_design/: selection.json, frames_gt_replica.json, frames_gt_scannet.json, crop_check.jpg, static_frames_rgb.jpg and the three scripts that produced them (select_rules.py, pick_frames_gt.py, crop_check.py).

I rank the NOR stale-content figure first. Table III is the main claim and has no picture yet. N1 shows the map before the update, the frame after the removal with the GT mask, and our test, the NormLift drop, the depth residual and the fused score rendered into the frame, plus our flags at the label-free threshold against the D2 truth. Its three test scenes are the best, median and worst by our pixel-level AP (015, 013, 006), so one failure case is built in. Second is the text-query figure N2, which shows the stale-semantics problem from the Introduction. It compares query relevance before the update, after it without the change test, and after it with ours. It comes from the same dump and adds a disclosed N=10 contrast number. Third is a merged Replica + ScanNet baseline figure S showing labels, errors and uncertainty for 2D features, averaging/NormLift, Feature 3DGS, LEGaussians and ours P2. It uses the median-mIoU test scene and the frame with the most GT classes (room_1 frame 66, scene0140_00 frame 57), and it shows Table I's AUSE_u (LEGaussians' u close to random order). I do not propose label-only figures, because the label maps differ by 0.1-2 mIoU. All three need two new GPU scripts that import eval_seg.py, eval_change_nor.py and render_qual.py without editing them, about 30 min of gated GPU wall time in total.

# Qualitative results: design (UAGS, 2026-10-20, read-only phase)

Nothing in the repo or the data folders was changed, and no GPU job ran. Every selection outcome below was computed on the CPU from published result JSONs or GT labels, before any new render.

Paths: R = /home/thanhnc/gauss_ws/src/uags, D = /home/thanhnc/gauss_ws/data, S = /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/qual_design

## 1. Ranking for an RA-L reviewer (do first: N1, N2, S)
| Rank | Candidate | Figure | Why a reviewer cares | Gated GPU wall time |
|---|---|---|---|---|
| 1 | (3) NOR stale content | **N1** `fig:qual-change` | It pictures the main claim (Table III), which has no figure yet. Real removals, every compared row, the label-free operating point, and the worst scene by rule. | 3 scenes ~5 min (all 10 test scenes ~19 min) |
| 2 | (4) Query before/after removal | **N2** `fig:qual-query` | Shows the robotics consequence named in the Introduction (stale answers to queries) and what the reset changes. Teaser candidate. | 0 extra (same dump as N1) |
| 3 | (1)+(2) merged: Replica + ScanNet vs baselines | **S** `fig:qual-compare` | The standard baseline comparison, including real data with labels. Makes Table I's AUSE_u visible: LEGaussians' u is close to random order (19.7 vs 20.6), and u_ale is on par with NormLift 1-C_inter. | ~2 min |
| 4 | (5) Failure cases | **F** `fig:qual-fail` | The NOR scenes with the highest false-alarm rate (011) and the lowest recall (009), plus static "confident errors". Supports Limitations. | ~10 min (static search); the NOR part reuses the dumps |
| 5 | (1)/(2) as label-only figures | supplementary grids only | Averaging, F3DGS and ours differ by 0.1-2 mIoU, which is invisible at figure size. | ~10 min for all 14 static test scenes |

## 2. Facts verified in the code (file:line)
| # | Fact | Where |
|---|---|---|
| F1 | `render_qual.load` forces a single row (`args.rows = args.row`). A multi-row figure must call `ES.load_scene` itself. | scripts/render_qual.py:101-108 |
| F2 | `load_scene` builds every factory row, keeps `--rows` and freezes them on the device. If the 2D row is requested, it runs LSeg on EVERY frame. | scripts/eval_seg.py:256-305, 271-277, 378-408 |
| F3 | `load_frames` keeps the image only with `with_2d`. The 2D row is `PixelSource` plus `frame_features_2d` (tokens, bilinear, map PCA). | scripts/eval_seg.py:216, 430-452 |
| F4 | `evaluate_view` returns pred (-1 = no decision), conf, covered and u{} per pixel. Pass `gt = zeros(P)` to get every pixel, as render_qual does. | usplat/seg_eval.py:223-276; scripts/render_qual.py:143 |
| F5 | `render_row` is one call on [buf, 1, u...]. A pixel's u is NaN when less than (1 - undef_frac) of its alpha is defined. The LEGaussians decoder is applied after compositing. | usplat/seg_eval.py:109-158 |
| F6 | averaging and normlift (with channel normlift_1-Cinter) come from ckpt baselines. f3dgs and legs_l<λ> come from files next to `config.out`. The fields store local paths (checked for room_1, scene0140_00 and nor_013). | usplat/baselines.py:217-278, 733-750; usplat/distill.py:212-246 |
| F7 | ScanNet P2 was fit with a = -0.80 and b = -2.06, so β grows with u. Replica has a = 0.36, b = 11.83. | scannet_tab1/eval_test.json betas_used; usplat/predictive.py:212-216 |
| F8 | `run_update` builds `ours` (calibrated) and `disp` (change test off, saved NormLift on its hooks) internally and returns only tensors. | scripts/eval_change_nor.py:472-547 |
| F9 | After a fire, the four sums restart and the firing view is added. So μ* of a reset stale Gaussian becomes the surface seen through it. | usplat/vmf_field.py:531-548 |
| F10 | Table III's pixel level renders the per-Gaussian scores with TensorRow + render_row into the update keyframes, with masks_real as positives, on covered pixels only. | scripts/eval_change_nor.py:659-684 |
| F11 | eval_change_nor.py hashes its own source into Table III's provenance, so never edit it; import it instead. | scripts/eval_change_nor.py:191-193, 241-248 |
| F12 | `fuse_rank` and `normalized_rank` use mid-ranks; NaN gets the lowest rank. | usplat/change_baselines.py:496-538 |
| F13 | `ES.text_embeddings` writes new prompts into `--text-cache`, so always pass a copy. | scripts/eval_seg.py:149-163 |
| F14 | `gpu_gate.sh` waits until free memory ≥ need + 2048 MiB. | scripts/gpu_gate.sh:20 |
| F15 | NOR world +y is gravity. Portrait scenes need one clockwise quarter turn (mean camera axes, all 14 scenes). | S/crop_check.py |

## 3. Selection rules (fixed 2026-10-20, before any new render) and outcomes
| Fig. | Rule | Outcome | Source / command |
|---|---|---|---|
| N1, N2 | NOR test scenes with the highest, lower-median and lowest pixel-level AP of "ours" (Table III per-scene JSON) | **015** (83.6), **013** (56.0), **006** (13.6) | `python3 S/select_rules.py` → S/selection.json. Reads D/uags_results/nor/tab2_relmax/test/0XX.json `metrics.pixel.rows.ours.ap` |
| N1, N2, F, supp | Frame = the update keyframe with the most covered stale pixels (`pixel.frames[].n_pos`); ties go to the lower index | 015→535 (51526 px), 013→552 (15820), 006→178 (8357), 011→466, 009→504. All test scenes: 005→201, 007→212, 008→424, 010→541, 014→595 | same |
| N1, N2 | Display: the quarter turn that brings image-down closest to world +y; square crop of side min(H, W), centred on the masks_real bounding box and clamped to the image | 015: no turn; 013, 006, 011, 009, 002: one clockwise turn. Crop side 540 px at 720×540 | `python3 S/crop_check.py` → S/crop_check.jpg |
| N2 | Query = the most frequent argmax over the 101 classes of configs/vocab_replica.json (LSeg 5-template ensemble) of the mapping-pass μ*, taken over the D2-stale Gaussians with W1 > 0. Use the class as computed; never re-prompt. | computed by the dump, written to meta.json | dump_nor_qual.py |
| S | Scene per dataset = the test scene with the lower-median ours_P2 mIoU | Replica **room_1** (26.79), ScanNet **scene0140_00** (68.52) | S/selection.json, from tab1/eval_test.json and scannet_tab1/eval_test.json |
| S, supp | Frame = the evaluation frame with the most GT classes, each ≥ 0.5% of the frame's labelled pixels; ties go to the earliest. Uses labels only. | room_1 frame **66** (Sequence_2 rgb_660, 13 classes); scene0140_00 frame **57** (scan frame 1145, 7 classes); all 14 scenes listed in S/frames_gt_*.json | `CUDA_VISIBLE_DEVICES="" python3 S/pick_frames_gt.py --vocab configs/vocab_{replica,scannet}.json --scenes configs/{replica,scannet}_test.json --out S/frames_gt_{replica,scannet}.json` |
| F | NOR test scene with the highest FA of ours, and with the lowest Rec of ours | **011** (FA 18.0%), **009** (Rec 8.5%) | S/selection.json (`operating_point.ours`) |
| F | Static: over all Replica + ScanNet test evaluation frames, the frame with the most covered wrong ours-P2 pixels whose u_epi and u_ale are both ≤ their frame medians | needs a GPU listing | qual_compare.py `--list-confident-errors` |
| teaser (val) | Val scene with the largest box volume (002, office chair), at its max-n_pos keyframe | 002→335 | S/selection.json |

Display parameters (percentile ranks, colour ranges, crops) are fixed here. Any later change after seeing a figure must be written into the report.

## 4. Figures
### N1 `fig:qual-change` (figure*, 7.16 in; 3 rows × 7 square panels of ~0.97 in)
- **Row labels:** each scene with its Table III pixel-level AP (ours / NormLift / depth / fused):
  - 015: 83.6 / 87.9 / 81.3 / 86.3
  - 013: 56.0 / 64.4 / 23.4 / 63.2
  - 006: 13.6 / 36.8 / 48.4 / 55.7
- **Columns:**
  - (a) Map before the update: 3DGS colour render at the update pose (frozen map, black background).
  - (b) Frame after the removal, with the masks_real outline (red, 1 px).
  - (c) Ours (test, reset, re-admission), Table III score `ours` = c_max.
  - (d) NormLift reliability drop.
  - (e) Depth residual.
  - (f) Ours fused with depth.
  - (g) Our flags at the label-free threshold (`flags["ours"]` = reset at least once, the Table III FA/Rec point) against the D2 truth, tinted on the map render. Flagged and stale = yellow; stale but not flagged = cyan; flagged but static = magenta; other observed Gaussians = map colour × 0.5; unobserved = dark grey.
- **How (c)-(f) are drawn:** the per-Gaussian scores are rendered exactly as in EC.pixel_level, then each pixel is shown as its percentile rank among the frame's covered pixels with a defined score.
  - Colours: viridis from 0 to 1; grey (RQ.UNDEF) = no score; black = not covered (α < 0.5); a thin white mask outline on every score panel.
  - One colorbar, labelled "rank in frame".
- **Resolution:** render at 720×540 with the frame's own K (from `ds.K[index]`; the principal point moves per frame). Upsample the masks from 192×256 with nearest neighbour.
- **Caption must state:** NormLift's drop scores higher than ours at pixel level in all three rows, as in Table III; the masks are dilated silhouettes.

### N2 `fig:qual-query` (figure, 3.5 in, 3 rows × 4 columns; or figure* with a 5th column)
- **Rows:** the same scenes, frames and crops as N1.
- **Columns:**
  - (a) Frame after the removal + outline; the query class goes in the row label.
  - (b) Map before the update (mapping-pass field).
  - (c) After the update with the change test off (EC's `disp` copy: the same views, no reset).
  - (d) After the update, ours.
  - (e) Optional: averaging updated with the same views (NormLift's `feature()`).
- **Value shown:** cos_p = F_p·t / ‖F_p‖, with t = Bᵀt̃ not renormalised. This equals the full-space cosine between the decoded rendered direction and the unit text (usplat/feature_pca.py:59-66).
- **Colours:** viridis on [p5, p99] of panel (b)'s covered pixels, shared across the row; black = not covered.
- **Number (all 10 test scenes):** query contrast Δ = mean cos inside masks_real − mean cos outside, over covered pixels of every update keyframe on the 360×270 grid.
  - Reported for before / no change test / averaging / ours, as mean ± std (N=10), plus the number of scenes where ours < no change test.
  - This is a NEW test-set measurement: disclose it, and let no method choice depend on it.

### S `fig:qual-compare` (figure*; two blocks, one RQ.compose call each, 3 rows × 6 columns, eval grid 320×240 / 324×242)
- **Blocks:** Replica room_1 frame 66; ScanNet scene0140_00 frame 57.
- **Columns:** reference | 2D features (per frame) | Averaging / NormLift | Feature 3DGS | LEGaussians (λu = 4 on Replica, 1 on ScanNet) | Ours P2.
- **Row 1:** 3DGS render | label maps (RQ.scene_palette; black = no decision or not covered).
- **Row 2:** GT | errors (RQ.error_image: white right, red wrong, black not scored).
- **Row 3:** empty | uncertainty as percentile rank among covered pixels with a defined value (RQ.u_image with vmax=1; grey = undefined; black = not covered):
  - 2D features and F3DGS: 1 − max p
  - Averaging: NormLift 1 − C_inter
  - LEGaussians: legs_u
  - Ours: u_ale
- **Panel annotation:** that frame's AUSE, computed with SegAccumulator on the real GT.
- **Do not draw P2 confidence on ScanNet** (see F7).
- **Caption must state:** this is the median scene by rule, and on scene0140_00 ours (68.5 mIoU) is below averaging (69.1), Feature 3DGS (69.6) and the 2D features (69.4).

### F `fig:qual-fail` (figure*, 3 rows × 6 columns)
- **Rows 1-2:** 011 frame 466 (highest FA) and 009 frame 504 (lowest Rec), with N1 columns (a), (b), (c), (d), (e) and (g).
- **Row 3:** the static confident-error frame: RGB, GT, ours labels, ours errors, u_ale rank, and the confident-error mask in magenta.

### Supplementary
- **supp_change_all:** N1 layout for all 10 NOR test scenes.
- **supp_compare_all:** the S block for all 14 static test scenes, using the frames in S/frames_gt_*.json.
- An optional CSS column (reference only) is allowed in the supplementary only.

## 5. Code plan (new files only; eval_seg.py, eval_change_nor.py and render_qual.py stay unchanged)
Environment, from R:

`export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.`

Before any run, copy D/uags_fields/text_cache_lseg.pt and D/uags_fields/scannet/text_cache_lseg.pt into D/uags_results/qual/ (F13).

1. **scripts/pick_frames.py** (CPU). Promotes S/pick_frames_gt.py and S/select_rules.py.
   - `python3 scripts/pick_frames.py static --vocab configs/vocab_replica.json --scenes configs/replica_test.json --min-share 0.005 --out D/uags_results/qual/select/frames_gt_replica.json`
   - `python3 scripts/pick_frames.py results --tab1 D/uags_results/tab1/eval_test.json --scannet D/uags_results/scannet_tab1/eval_test.json --nor-dir D/uags_results/nor/tab2_relmax --out D/uags_results/qual/select/selection.json`
2. **scripts/qual_compare.py** (GPU, ~300 lines). Parser = ES.build_parser() + extra arguments, the same pattern as render_qual.py:477.
   - Command: `scripts/gpu_gate.sh 4500 python3 scripts/qual_compare.py --vocab configs/vocab_replica.json --scenes configs/replica_test.json --scene room_1 --frame 66 --betas D/uags_results/tab1/eval_val.json --text-cache D/uags_results/qual/text_cache_lseg.pt --row-factory usplat.baselines:row_factory,usplat.distill:row_factory --cols "2d_features:msp,averaging+normlift:normlift_1-Cinter,f3dgs:msp,legs_l4:legs_u,ours_P2:u_ale" --out D/uags_results/qual/compare/replica_room_1_f66.pdf --meta ...json --dump ...npz`
   - For ScanNet, use vocab_scannet, scannet_test, scene0140_00, `--frame 57`, the scannet_tab1/eval_val.json betas and legs_l1. Add `--list-confident-errors` for Fig. F.
   - Steps:
     - (a) Check the betas: ES.check_fit_settings and ES.load_betas.
     - (b) Set `args.with_2d=True` so frames keep their images (F3), and keep `2d_features` out of `--rows` so LSeg does not run on every frame.
     - (c) Load LSeg (ES._extractor), take the chosen frame's tokens to the CPU, then clear `ES._EXTRACTOR` and empty the CUDA cache.
     - (d) `S = ES.load_scene(...)` with rows ours_P2, averaging, normlift, f3dgs, legs_l*.
     - (e) Render with `ras = ES.make_rasterizer`, then one call `evaluate_view(rows, ras, S["text"], zeros(P), betas, 0.5, undef_frac=0.5)`.
     - (f) 2D row: `ES.frame_features_2d(S, dict(tok2d=tok, h, w), args)` → `ES.PixelSource(z)` → `evaluate_view([TensorRow("2d_features","feature",buf=zeros(1,d))], ...)`.
     - (g) Percentile ranks with a mid-rank, as in normalized_rank. Per-frame AUSE from SegAccumulator.add(evaluate_view with the real GT).metrics(15, 20).
     - (h) Draw with RQ.compose, label_image, error_image, u_image, legend_classes and save. RQ.compose has six fixed columns, which matches this layout.
3. **scripts/dump_nor_qual.py** (GPU, ~350 lines). Parser = EC.build_parser() + `--table --vocab --text-cache --frames auto|i,j --full-res --per-view --save-features`; `--out` is a directory.
   - Command: `scripts/gpu_gate.sh 5000 python3 scripts/dump_nor_qual.py --field D/uags_fields/nor_015_lseg_relmax.pt --align none --table D/uags_results/nor/tab2_relmax/test/015.json --vocab configs/vocab_replica.json --text-cache D/uags_results/qual/text_cache_lseg.pt --frames auto --full-res --per-view --save-features --out D/uags_results/qual/nor/015`
   - Steps:
     - (a) `ctx = EC.load_context(args)` (:697).
     - (b) Run `EC.replay_mapping` and `replay_check`, then `thr = {r: n.threshold(q)}`, as in EC:777-784.
     - (c) Capture the fields: around `upd = EC.run_update(...)`, temporarily set `EC.DirectionalSemanticField` to a subclass whose `from_state` records each instance. Then ours = made[0], disp = made[1], and NormLift = `disp.view_hooks[0].__self__` (baselines.py:165).
       - Assert exact equality: `upd["scores"]["ours"]` equals `where(isfinite(ours.c_last), ours.c_max, nan)`, and `upd["robust_end"]` equals `ours.robust_mask()`.
       - Fallback if the capture is not accepted: copy EC:472-547 and test it for equality on the toy scene.
     - (d) `--per-view`: during run_update only, wrap `EC.residuals_view` (called at :516 with `ras` and `v`). For each update keyframe, render ours' c_last where tested, last_fired, changed and robust, plus the map RGB (`ren["rgb"]`), saved as fp16.
     - (e) `gt = EC.ground_truth(ctx, args)`, then fused = fuse_rank([ours, depth], mask=observed) (:798).
     - (f) Verify: `EC.ranking_table` (G level) and `EC.pixel_level` (P level) per row against `--table`; |ΔAP| ≤ 0.005, otherwise exit 2.
     - (g) Query rule (section 3). Text embeddings via ES.text_embeddings on the CPU with the cache copy, then `ctx["pca"].project_text`.
     - (h) Save gauss.pt with:
       - the scores (7 rows) and flags
       - observed, seen, stale, in_box, changed_end, robust_end, robust_map, n_views
       - query {name, top-3 shares, t}
       - per-Gaussian μ·t for the map before, no change test, averaging and ours
       - u_epi and u_ale before and after
       - optionally fp16 features: about 0.5 GB per 1M Gaussians
     - (i) For each rule frame, at 720×540 and 360×270: photo, mask, map RGB, alpha, the 7 score renders, the decision tint (composited colours, as RQ.render_rgb) and the 4 cosine maps. Also save the rotation and the crop.
     - (j) Query contrast over all update keyframes.
     - (k) meta.json: argv, wall time, peak MiB, sha256 of this script plus EC.code_state(), field/map sha256, verification, selection.
4. **scripts/plot_nor_qual.py** (CPU). Subcommands `change | query | failures | grid | contrast`, each taking `--dumps D/uags_results/qual/nor/<s> ... --out papers/figs/<name>.pdf`.
   - Uses an N-column grid generalised from RQ.compose, with square panels, outlines and tint legends.
   - Reuses RQ.RC (TrueType fonts), RQ._legend, RQ._colorbar and RQ.save.
   - `contrast` writes JSON and LaTeX rows.
5. **scripts/qual_figs.sh** runs the whole chain under the gate, using the cache copies.
6. **Tests:**
   - tests/test_qual_compare.py: the mock toy from tests/test_seg_eval.py `_cli_toy`. Columns must equal evaluate_view with the real GT.
   - tests/test_dump_nor_qual.py: the toy NOR scene from tests/test_eval_change.py `build`, mock backend. Scores must equal EC.evaluate's, the captured state must match, and the query and rotation rules are checked.
7. **Rules for all new files:** no names, lab marks or Claude attribution in code headers or figures.

## 6. GPU budget (wall time under the gate; estimated from measured per-scene EC timings)
- **NOR dumps:**
  - All 10 test scenes: 800 s measured EC time × ~1.4 for the extra renders ≈ 19 min.
  - Val 002: ≈ 3.5 min.
  - All 4 val scenes: ≈ 14 min.
- **Static:** 2 main scenes ≈ 2 min; all 14 scenes ≈ 10 min; the confident-error search ≈ 10 min.
- **First three figures** (N1, N2 with the N=10 number, S): ≈ 25-30 min.
- **Everything:** ≈ 55 min.

## 7. Verification before a figure enters the paper
| ID | Check |
|---|---|
| V1 | Every dumped scene reproduces Table III G- and P-level AP per row within 0.005. |
| V2 | The captured ours score exactly equals EC's score. |
| V3 | qual_compare's labels and coverage equal evaluate_view's on the labelled pixels. |
| V4 | The new tests pass, and tests/test_render_qual.py and tests/test_eval_change.py are unchanged. |
| V5 | PDFs use TrueType fonts; no names or marks; the rule outcome is recorded in a meta JSON next to each PDF. |
| V6 | `git diff` shows eval_seg.py, eval_change_nor.py and render_qual.py unchanged. |

## 8. Hand-offs
- **Teaser:** an N2 row (015 bike) or val 002 (chair), using the features in gauss.pt.
- **Video:** the `--per-view` stacks along the update trajectory. Any pose can be rendered from the per-Gaussian arrays: render F once and compute cos per pixel; for scores, render the score as a buffer channel.

RULES:["N1/N2 scenes: the NOR test scenes with the highest, lower-median and lowest pixel-level AP of 'ours' in the Table III per-scene JSONs. Outcome: 015 (83.6), 013 (56.0), 006 (13.6).", 'NOR frame: the update keyframe with the most covered stale pixels (pixel.frames[].n_pos); ties go to the lower index. Outcome: 015→535, 013→552, 006→178, 011→466, 009→504, 005→201, 007→212, 008→424, 010→541, 014→595, val 002→335.', 'NOR display: rotate by the quarter turn that brings the image-down axis closest to world +y (gravity); square crop of side min(H, W), centred on the masks_real bounding box and clamped. Outcome: 015 no turn; 013, 006, 011, 009 and 002 one clockwise turn.', 'N2 query: the most frequent argmax over the 101 classes of configs/vocab_replica.json (LSeg 5-template ensemble) of the mapping-pass μ*, taken over the D2-stale Gaussians with W1 > 0. Report the class as computed and never re-prompt.', "N1 and S score/uncertainty panels: show the percentile rank among the frame's covered pixels with a defined value (viridis 0-1; grey = undefined; black = α < 0.5). N2: cosine on [p5, p99] of the before panel's covered pixels, shared across the row.", 'S scene: the test scene with the lower-median ours_P2 mIoU in the Table I JSONs. Outcome: Replica room_1 (26.79), ScanNet scene0140_00 (68.52).', "S frame (labels only): the evaluation frame with the most GT classes, each ≥ 0.5% of the frame's labelled pixels; ties go to the earliest. Outcome: room_1 frame 66 (rgb_660, 13 classes); scene0140_00 frame 57 (scan frame 1145, 7 classes). All 14 test scenes are listed in frames_gt_*.json for the supplementary grid.", 'F (NOR): the test scene with the highest FA of ours (011, 18.0%) and the one with the lowest Rec of ours (009, 8.5%), each at its max-n_pos frame.', 'F (static): over all Replica and ScanNet test evaluation frames, the frame with the most covered wrong ours-P2 pixels whose u_epi and u_ale are both ≤ their frame medians. Needs a GPU listing pass.', 'Teaser/val candidate: the val scene with the largest object box (002, office chair) at frame 335. Label it as a validation scene.']
SCRIPTS:['scripts/pick_frames.py (CPU): applies the selection rules from GT labels and published JSONs and writes the selection JSON (promotes the scratch select_rules.py and pick_frames_gt.py)', 'scripts/qual_compare.py (GPU, gate 4500 MiB): Fig. S blocks (labels, errors and rank-normalised uncertainty for 2D features, averaging/NormLift, Feature 3DGS, LEGaussians and ours P2) plus --list-confident-errors for Fig. F; reuses ES.load_scene, evaluate_view and RQ.compose and its helpers', 'scripts/dump_nor_qual.py (GPU, gate 5000 MiB): one NOR scene. Runs EC.load_context, replay_mapping and run_update (fields captured, not copied) and ground_truth; checks the result against Table III; applies the query rule; renders the rule frames at 720x540 and 360x270; computes the query contrast; optional per-view stacks for the video', 'scripts/plot_nor_qual.py (CPU): composes Figs. N1, N2, F (NOR rows) and the supplementary NOR grid, and writes the N=10 query-contrast JSON and LaTeX rows', 'scripts/qual_figs.sh: driver for the chain under scripts/gpu_gate.sh, using copies of the text caches', 'tests/test_qual_compare.py and tests/test_dump_nor_qual.py: CPU mock-backend smoke and equality tests on the existing toy scenes']
GPU:30
RISKS:["NormLift's drop has a higher pixel-level AP than ours in all three N1 rows (87.9 vs 83.6, 64.4 vs 56.0, 36.8 vs 13.6). This matches Table III, and the caption must say so. Do not change the rule to a nicer scene.", "N2 may show only a small effect. Ours differs from 'no change test' only on Gaussians that were reset (recall 29-54% in the three chosen scenes). Keep the rule and report the N=10 contrast whatever it shows; it is a new test-set measurement, so disclose it and base no choice on it.", 'Re-running the update on the GPU is not bit-exact, so resets on borderline Gaussians may differ from Table III. V1 requires |ΔAP| ≤ 0.005 per row before a dump is used; if it fails, stop and report.', "Capturing the fields by temporarily replacing EC.DirectionalSemanticField is a monkeypatch. Guard it with exact-equality asserts against EC's score tensor, or copy the update loop and prove equality on the toy scene.", 'On ScanNet the P2 fit has a=-0.80, b=-2.06, so β grows with u. Any P2-confidence panel on ScanNet would look inverted, so Fig. S shows u_ale, not P2 confidence. A reviewer may still ask; one sentence of disclosure is needed.', 'The median ScanNet scene (scene0140_00) has ours below averaging, F3DGS and the 2D features in mIoU (68.5 vs 69.1, 69.6, 69.4). The label row will not favour ours, and the rule must stay.', 'gpu_gate needs need + 2048 MiB free. At gate 5000 that is about 7 GB, and other users were holding 5-6 GB, so the jobs may wait. Lowering need below the measured peak (EC 4.56 GB at the eval grid) risks an OOM; 720x540 renders add memory (keep channel_chunk 32).', 'masks_real are dilated silhouettes (8-74% of the chosen frames). The pixel-level truth includes background around the object, and captions must say this.', "The query rule may return a generic or odd class (for example 'cloth' for a jacket). It is fixed and must be reported as computed.", "Third-party content: the 006 map render may show the box's product branding, and the ScanNet frame has a partly visible poster. Neither is an anonymity mark; blur only if the user asks, and disclose it. No lab marks or names may appear in figures, PDF metadata or script headers, and no Claude attribution anywhere.", 'ES.text_embeddings writes into --text-cache, so always pass a copy. Feature dumps take about 0.5 GB per 1M Gaussians; save them only for the 3 main scenes and 002.', "Percentile-rank colouring hides absolute scale. Captions must say 'rank within the frame'. The per-panel AUSE helps, but it is noisy on a single frame."]