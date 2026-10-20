VERDICT revise
ISSUES:
- GPU gates are too low and will likely hit out-of-memory under the hard cap. gpu_gate.sh:54 exports need/total, and usplat/gpu_reserve.py turns that into torch.cuda.set_per_process_memory_fraction. The Table III run alone (eval_change_nor.py) peaked at 4558 MiB allocated and reserved 5768 MiB (test/015.json provenance; the run started at >=7012 MiB free per the eval_015 log). dump_nor_qual adds 720x540 renders, captured fields and feature dumps on top of that, yet is gated at 5000. qual_compare is gated at 4500 with LSeg on the GPU, which needs >3.3 GB by itself (eval_seg.py:49). Every eval_seg gate that included the 2D row was 7000 (replica_tab1.sh:36, scannet_tab1.sh:429). Use 6500 for NOR (nor_tab2.sh:65, headroom 1024 as in nor_tab2.sh:44). Split qual_compare into a token step and a render step at 4500 each.
- The text-cache copies collide. The plan copies D/uags_fields/text_cache_lseg.pt (51 prompts) and D/uags_fields/scannet/text_cache_lseg.pt (18 prompts; only 11 keys are shared) into D/uags_results/qual/ under the same basename. It then passes that one file to the Replica, ScanNet and NOR runs. The second copy overwrites the first, so the prompts used for Table I get recomputed. The NOR query step also writes 50 extra prompts into the cache the static figures read (ES:149-163). Use three copies with distinct names.
- Per-frame percentile ranks can flatter weak scores (N1 (c)-(f), S row 3, F). A rank taken within one frame always spreads that frame's values over [0,1]. In 006 (ours pixel AP 13.6) the object can therefore look bright even when its scores are low compared with false alarms in other keyframes. Table III's pixel AP ranks the pooled covered pixels of all update keyframes (EC:659-684, e.g. n_pix 6,568,454 for 015), and AUSE pools every evaluation frame of a scene. Show ranks against those pooled populations.
- The square crop centred on the mask hides a quarter of the frame, including false alarms outside the object region. The max-n_pos frame rule also picks frames where the object region is cut by the image border: 015->535, 013->552, 011->466, 009->504, 014->595, 010->541, 007->212 and val 002->335 (mask covers 74% of the frame) all touch the border (review/frame_rule_check.json). Use full upright frames and a rule based only on the ground-truth masks: the largest masks_real region that lies fully inside the frame (015->675, 013->556, 006->178, 011->496, 009->354, 002->373), falling back to max n_pos.
- N1 leaves out the feature residual (CL-Splats/MIF). It is the closest prior art to our score and beats ours on the Table III means (Gaussian AP 56.6 vs 53.4, pixel AP 57.8 vs 57.2) and on 015 (pixel AP 85.7 vs 83.6). Leaving it out reads as hiding a stronger baseline. Add it as a column.
- S row 3 mixes score types. It shows 1 - max p (a total uncertainty) for 2D features and Feature 3DGS next to the own uncertainties of NormLift (1-C_inter), LEGaussians (u) and ours (u_ale). Table I's AUSE_u column has '--' for those two rows. Show own uncertainties only and leave those cells empty, or add a separate 1 - max p row covering every method.
- Per-frame AUSE annotations are new, noisy numbers on the test set, which the design itself admits. The per-scene AUSE of each method's own uncertainty is already in the Table I JSONs: room_1 has u_ale 10.3, NormLift 1-C_inter 11.0, LEGaussians u 21.8; scene0140_00 has 3.6, 3.5 and 8.7. Print those instead. On scene0140_00 NormLift ranks errors slightly better than ours, and the caption must say so.
- The teaser hand-off offers 015, which is the best test scene by ours' pixel AP, so the teaser would be cherry-picked. Use validation scene 002 at its rule frame 373, or state the selection rule in the teaser caption.
- Fig. F's false-alarm row (011) uses the max-n_pos frame, which shows the stale region, while false alarms sit on static content elsewhere. Pick the frame with a worst-case rule on the dumped flags instead. The static confident-error rule uses medians within each frame, so even a frame that is uncertain everywhere has 50% 'confident' pixels. It also does not say how undefined u_ale is handled. Use scene-level medians and count undefined u_ale as not confident.
- There is no validation-first step. Figure code, display parameters and the new N2 number would first be run on test scenes. Develop on NOR 001-004, Replica office_0/room_2 and ScanNet scene0000_00/0062_00, freeze everything, then render each test scene once and log every look at test data. The N2 contrast is a new test metric, so define and debug it on validation first. A stale-answer rate (the share of the removed object's former pixels still labelled as the query class) maps more directly to the Introduction's claim than delta-cosine does.
- Internal names would leak into captions. 'D2 truth', 'c_max', 'flags["ours"]', 'u_ale', 'mu*' and 'W1' must become the paper's terms. Write 'stale Gaussians' (box enlarged by 2 cm and more than 2 cm from high-confidence after-removal depth). Write 'Ours: test, reset, re-admission', scored by the largest persistent score $\bar c_n$ over the update. Write FA/Rec at the label-free threshold, $u^{\mathrm{ale}}$, $\boldsymbol\mu^\star_n$ and $s_n$, and use the Table III row names. Panel (a) and the N2 'after' panels must state that geometry is fixed: the object's Gaussians are still rendered and only their semantics are reset. Otherwise readers will take 'after' to mean the geometry was removed.
- qual_compare step (c) says to clear ES._EXTRACTOR and empty the CUDA cache. That contradicts usplat/gpu_reserve.py (never call empty_cache after apply) and the pattern at eval_seg.py:277. The step also needs the frame image before load_scene, but load_frames runs inside load_scene (ES:268). Read the frame with ES.load_frames first, or extract tokens in a separate gated step. The new GPU scripts must call gpu_reserve.apply, as EC:768-770 and RQ:454-456 do. They must never call update_view under inference_mode (vmf_field.py:354-356).
- The reproduction check is weaker than it can be. The dump command omits --update-stride, which every Table III run passed explicitly (argv in test/0XX.json). The value happens to equal the field stride in all 14 scenes, but the dump should take every setting from the record and assert identical mapping and update keyframe lists. It should also assert identical per-frame n_covered/n_pos (forward render only), not just |delta AP| <= 0.005. Row labels must print the published values, never the re-run ones.
- V6 protects only 3 files. EC.code_state hashes 9 files (eval_change_nor.py:191-193) and the mapping provenance hashes 12 more (run_mapping.py:378-381). Editing any of them silently changes the provenance of Tables I-IV. Freeze all of them, plus render_qual.py, plot_views.py and make_tables.py.
- Paper consistency: the design flags this risk but no deliverable acts on it. The ScanNet P2 fit has a=-0.80 and b=-2.06 (scannet_tab1/eval_test.json betas_used; predictive.py:212-216), so beta grows with uncertainty, against the stated P2 rationale. The paper needs one disclosure sentence whatever the figures show. On room_1, P2's ECE and NLL are also worse than P0's (15.5 vs 14.0, 1.188 vs 1.155). Fig. S must therefore show no P2 confidence panel on either dataset.
- Colour and legibility: the tint colours yellow, cyan and magenta are not safe for colour-blind readers (magenta and cyan get confused). Use the Okabe-Ito colours, as plot_views.py does. A plain white outline disappears on the yellow end of viridis, so give it a dark halo.
- The anonymity check is incomplete. Add pdfinfo/pdffonts checks (metadata, no Type 3 fonts). Ban usernames and absolute paths in any shipped file: configs/*.json and the meta JSONs contain /home/<user>/... paths. Look at every NOR photo for faces and number plates. Third-party branding (the 006 box, the 008 backpack pattern) is not an anonymity mark, but the report should list it.
- The GPU budget is understated. The ~30 min estimate leaves out the validation runs (NOR validation ~14 min), the pooled-rank pass and gate waits. A realistic figure is ~45 min for N1+N2+S and ~70 min with F and the supplementary grids. At gate 6500 + headroom 1024 the NOR jobs need ~7.5 GB free, so they will wait while other users hold more than ~4.8 GB.

REVISED SPEC:
I've finished the review. The design needs revision before anything runs on the GPU, so the corrected spec is below. Most of its foundation holds:
- Every code path it cites exists and checks out. That is facts F1-F15, plus the claimed line numbers in `eval_change_nor.py` (EC), `eval_seg.py` (ES), `render_qual.py` (RQ), `vmf_field.py` and `baselines.py`.
- All the data files it relies on are present.
- Its numbers match the published JSONs.

The problems are concrete, and each comes with the change that fixes it:

1. **GPU gates are too low.** The NOR dump at 5000 MiB and qual_compare at 4500 MiB will likely run out of memory. `gpu_gate.sh` turns the requested amount into a hard cap. The Table III run alone reserved 5768 MiB. LSeg needs more than 3.3 GB on its own. *Fix:* gate the NOR dump at 6500 MiB, and split qual_compare into two steps (tokens, then render) at 4500 MiB each.
2. **Text caches collide.** The Replica and ScanNet caches share a file name, so copying both into one folder overwrites one of them. *Fix:* use three copies with distinct names (Replica, ScanNet, NOR).
3. **Per-frame ranks can flatter weak methods.** A rank within one frame always spreads that frame's values over [0,1], so scene 006 (our pixel AP 13.6) could look good. *Fix:* rank against the same pooled pixels that Table III's AP and the AUSE use.
4. **Crops and frame choice hide context.** The square crop hides a quarter of the frame, including false alarms. The "most stale pixels" rule picks frames where the object region is cut by the image border: 015, 013, 011, 009, 014, 010, 007 and validation 002 (mask covers 74% of the frame). *Fix:* show full upright frames, and pick by a rule that uses only the ground-truth masks: the largest object region fully inside the frame. The new frames are 015→675, 013→556, 006→178, 011→496, 009→354 and 002→373.
5. **N1 leaves out the closest baseline.** The feature residual (CL-Splats/MIF) beats our score on the Table III means: Gaussian AP 56.6 vs 53.4, pixel AP 57.8 vs 57.2. *Fix:* add it as a column.
6. **Fig. S mixes kinds of uncertainty.** It puts "1 − max probability" for two methods next to the own uncertainty of the others. Table I has "--" for those two methods. *Fix:* show own uncertainties only, with empty cells where Table I has "--".
7. **Per-frame AUSE numbers are new noisy test numbers.** *Fix:* print the per-scene AUSE already stored in the Table I JSONs. On scene0140_00, NormLift is slightly better than ours (3.5 vs 3.6), and the caption must say so.
8. **The teaser would be cherry-picked.** The hand-off suggests scene 015, which is the best test scene. *Fix:* use validation scene 002 at frame 373.
9. **Fig. F does not show its failures.** The false-alarm row's frame shows the removed object, not the false alarms. The static rule uses medians within each frame. *Fix:* pick the false-alarm frame by a worst-case rule on the dumped flags, and use scene-level medians.
10. **No validation-first step.** *Fix:* develop all figure code and layouts on validation scenes, freeze them, then render each test scene once and log every test look. The new query number in N2 must be defined on validation first.
11. **ScanNet's adaptive predictive (P2) is fit backwards.** Its temperature grows with uncertainty (a = −0.80, b = −2.06), against the paper's rationale for P2. *Fix:* the paper needs one disclosure sentence whatever the figures show, and Fig. S shows no P2 confidence panel at all.

Smaller fixes are also in the spec: paper terms in captions instead of internal names, stricter reproduction asserts, freezing all 21 hashed source files (not just 3), colour-blind-safe colours, and a fuller anonymity check.

Nothing in the repo or data folders was changed and no GPU job ran. The two check scripts I wrote, and their outputs, are in `/tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/review/`:
- `frame_rule_check.py`
- `frame_rule_check.json`
- `full_frames.py`
- `full_frames.jpg`

---

# Qualitative results: corrected design (UAGS)

**Status.** This is a design only.
- Nothing in the repo or the data folders was changed, and no GPU job ran.
- Every outcome below comes from the published result JSONs, ground-truth masks and labels, or photos, all read on the CPU.

**Evidence folders.**
- S = `/tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/qual_design/` holds the first pass: `selection.json`, `frames_gt_replica.json`, `frames_gt_scannet.json`, `select_rules.py`, `pick_frames_gt.py`, `crop_check.py`.
- S2 = `/tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/review/` holds this review: `frame_rule_check.py`, `frame_rule_check.json`, `full_frames.py`, `full_frames.jpg`.

**Paths.**
- R = `/home/thanhnc/gauss_ws/src/uags`
- D = `/home/thanhnc/gauss_ws/data`
- EC = `scripts/eval_change_nor.py`, ES = `scripts/eval_seg.py`, RQ = `scripts/render_qual.py`

## 0. Global rules

**Environment.** Run from R:

```bash
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
```

**GPU.**
- Run every GPU job as `scripts/gpu_gate.sh <MiB> ...` with `UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5`, as `nor_tab2.sh:43-45` does.
- Each new GPU script calls `usplat.gpu_reserve.apply(device)` first, as EC:768-770 and RQ:454-456 do.
- After that call, never call `torch.cuda.empty_cache()` (see `gpu_reserve.py`).
- Never wrap `update_view` in `torch.inference_mode` (`vmf_field.py:354-356`). `torch.no_grad` is fine.

**Frozen files.** Edit none of these. Their hashes are part of the provenance of the published tables.
- EC.CODE (EC:191-193): `eval_change_nor.py`, `time_field.py`, `eval_seg.py`, `usplat/change_baselines.py`, `evaluation.py`, `seg_eval.py`, `rows.py`, `nor_align.py`, `box3d.py`.
- `run_mapping.MAPPING_CODE` (`run_mapping.py:378-381`).
- `render_qual.py`, `plot_views.py`, `make_tables.py`.
- New code goes in new files only, which import these modules.

**Text caches.** ES:149-163 writes new prompts into `--text-cache`, so never pass the originals. Make three copies with distinct names:
- `D/uags_fields/text_cache_lseg.pt` → `D/uags_results/qual/text_cache_lseg_replica.pt`
- `D/uags_fields/scannet/text_cache_lseg.pt` → `D/uags_results/qual/text_cache_lseg_scannet.pt`
- `D/uags_fields/text_cache_lseg.pt` → `D/uags_results/qual/text_cache_lseg_nor.pt`. The NOR queries add 50 prompts to this copy only.

**Validation first.**
- Build and debug every script, layout and display parameter on validation scenes only:
  - NOR 001-004
  - Replica office_0 and room_2
  - ScanNet scene0000_00 and scene0062_00
- Then freeze them and render each test scene once.
- Log every test look in the run report: scene, script, git hash and purpose.
- Log any display change made after a test render, with its reason.
- No method change is allowed.

**Numbers in figures and captions.**
- Scripts read them from the published JSONs. Never type them by hand, and never take them from a re-run.
- Table I comes from `D/uags_results/tab1/eval_test.json`. It has the same sha256 as `docs/reports/2026-09-29-replica-tab1/logs/eval_test_224fe1e.json`. ScanNet comes from `D/uags_results/scannet_tab1/eval_test.json`.
- Table III comes from `D/uags_results/nor/tab2_relmax/test/0XX.json`. Their per-scene means reproduce Table III exactly: Gaussian AP 53.4/75.0/63.8 and pixel AP 57.2/61.8/65.8 for ours/fused/NormLift.

**Anonymity.**
- No names, affiliations, lab marks, usernames or absolute paths in figures, captions, PDF metadata, the video or any supplementary file. `configs/*.json` and the meta JSONs contain `/home/<user>` paths, so never ship them.
- No Claude attribution in code headers, commits or the paper.
- `main.tex` keeps one source line per paragraph and per caption.

## 1. Facts verified in the code

| # | Fact | Where |
|---|---|---|
| F1 | `RQ.load` forces a single row (`args.rows = args.row`). | RQ:101-108 |
| F2 | `load_scene` builds every factory row, keeps `--rows` and freezes those rows on the device. If the 2D row is requested, it runs LSeg on every frame before the rows go to the GPU. | ES:256-305 (271-277), 378-408 |
| F3 | Frames keep their image only with `with_2d`. The 2D row is `PixelSource` plus `frame_features_2d`. | ES:186-217, 430-452 |
| F4 | `evaluate_view` returns pred (−1 = no decision), conf, covered and the u values. Passing gt = zeros(P) labels every pixel. | `seg_eval.py:223-276`; RQ:143 |
| F5 | `render_row` is one call. It gives NaN where less than (1 − undef_frac) of α is defined, and decodes LEGaussians after compositing. | `seg_eval.py:109-158` |
| F6 | The averaging and NormLift rows come from the checkpoint's baselines. The f3dgs and legs_l* rows come from `<field stem>_*.pt`. All needed files exist for room_1 (`_f3dgs`, `_legs_l4`) and scene0140_00 (`_f3dgs`, `_legs_l1`). | `baselines.py:184-278, 733-750`; `distill.py:212-246` |
| F7 | P2 uses β = β0·exp(−a·u_epi − b·u_ale − g·1[undefined]). On ScanNet the fit has a = −0.80, b = −2.06, so β grows with uncertainty. On Replica a = 0.36, b = 11.83. | `predictive.py:212-216`; `betas_used` in the eval JSONs |
| F8 | `run_update` builds `ours` (from_state, then calibrate) and `disp` (from_state with the change test off, plus the saved NormLift). It returns only tensors: scores, flags, observed, seen, n_views, changed_end, robust_end, robust_map. | EC:472-547 |
| F9 | After a fire, the sums restart and the firing view is added. So μ* of a reset Gaussian becomes the features seen through it. | `vmf_field.py:531-548` |
| F10 | Table III's pixel level renders each row's per-Gaussian scores with TensorRow + render_row into every update keyframe. masks_real is resized by nearest neighbour, only covered pixels count, and pixels are pooled over keyframes. | EC:659-684 |
| F11 | Fused = `fuse_rank([ours, depth], mask=observed)`, using mid-ranks with NaN lowest. | EC:797-798; `change_baselines.py:496-538` |
| G1 | The gate needs need + headroom MiB free and exports need/total, which `gpu_reserve` turns into a hard per-process cap. | `gpu_gate.sh:20, 54`; `gpu_reserve.py` |
| G2 | Table III's EC run peaked at 4558 MiB allocated and reserved 5768 MiB. It started at ≥ 7012 MiB free. The NOR default gate is 6500. | `test/015.json` provenance; `eval_015` log; `nor_tab2.sh:65` |
| G3 | LSeg in fp32 needs more than 3.3 GB. The ES gates that included the 2D row were 7000. | ES:49; `replica_tab1.sh:36`; `scannet_tab1.sh:429` |
| G4 | `from_state` is a classmethod that calls `cls(...)`. `NormLift.attach` appends its bound `on_view` to `field.view_hooks`. | `vmf_field.py:810-829`; `baselines.py:165-167` |
| G5 | `run_update` calls the module-level `residuals_view` at :516, after `ours.update_view` (:502) and `disp.update_view` (:510). | EC:499-521 |
| G6 | Table III passed `--update-stride` explicitly. It equals the field's stride in all 14 scenes, and `views.update` lists the keyframes. | per-scene JSON argv and settings |
| G7 | NOR `c2w` and `K` are indexed by frame number for every part, and K changes per frame. | `datasets.py:475-560` |
| G8 | The per-scene AUSE of each row's own uncertainty is stored in the Table I JSONs. | `rows.<row>.all.ranking.<channel>.ause` |

## 2. Selection rules and outcomes

These rules were fixed on 2026-10-21, before any new render. They read only published JSONs, ground-truth masks and labels, and photos.

| Use | Rule | Outcome |
|---|---|---|
| N1, N2 scenes | NOR test scenes with the highest, the lower-median (5th of 10, ascending) and the lowest pixel-level AP of "Ours: test, reset, re-admission". Source: `test/0XX.json` `metrics.pixel.rows.ours.ap`. | **015** (83.6), **013** (56.0), **006** (13.6) |
| NOR frame (every NOR panel) | Among the scene's Table III update keyframes (`views.update`), take the one with the largest masks_real area whose bounding box stays at least 2 px inside the raw 192×256 grid. Ties go to the lower index. If no keyframe qualifies, take the keyframe with the most covered mask pixels (`pixel.frames[].n_pos`). Uses ground truth only. | Test: 005→201, 006→178, 007→270, 008→424, 009→354, 010→519, 011→496, 013→556, 014→748, 015→675. Validation: 001→450, 002→373, 003→307, 004→266. Source: S2/`frame_rule_check.json`. Mask pixels over covered pixels: 015/675 41878/94646, 013/556 15061/97198, 006/178 8357/97144. |
| NOR display | Full frame, no crop. Rotate clockwise by the quarter turns that bring image-down closest to world +y, and apply the same turn to every panel of the scene. Portrait scenes take one turn; 007, 008, 014 and 015 take none. | Checked on 6 frames (S2/`full_frames.jpg`) |
| N2 query | The most frequent argmax over the 101 classes of `configs/vocab_replica.json`, using the LSeg 5-template ensemble. Text is projected with the scene's PCA, without renormalising. The argmax is taken over the mapping-pass μ* of the stale Gaussians with W1 > 0. Report the class as computed and never re-prompt. | Written to the dump's meta |
| S scenes | Test scene with the lower-median ours_P2 mIoU. | **room_1** (26.79), **scene0140_00** (68.52) |
| S frame | Evaluation frame with the most ground-truth classes, each covering at least 0.5% of the labelled pixels. Ties go to the earliest frame. Uses labels only. | room_1 → 66 (rgb_660, 13 classes); scene0140_00 → 57 (index 1145, 7 classes). All 14 scenes are listed in S/`frames_gt_*.json`. |
| F, NOR false alarms | Scene with the highest FA of ours (011, FA 18.0%). Frame: the update keyframe with the most covered pixels outside masks_real where the rendered share of ours-flagged Gaussians is ≥ 0.5, on the 360×270 grid. Ties go to the lower index. | 011; frame comes from the dump |
| F, NOR misses | Scene with the lowest Rec of ours (009, Rec 8.5%), at its NOR-rule frame. | 009 → 354 |
| F, static confident errors | Over the 14 static test scenes, the evaluation frame with the most covered labelled pixels that ours P2 gets wrong while u_ale is defined and ≤ the scene's median u_ale, and u_epi ≤ the scene's median u_epi. Medians are over covered pixels with a defined value. Undefined u_ale counts as not confident. Ties go to config order, then the earliest frame. | From `qual_compare --list-confident-errors` |
| Teaser (if it uses NOR) | Validation scene with the largest box volume (002, 1.05×0.74×0.67 m), at its NOR-rule frame. A test scene may appear in a teaser only with its rule stated in the caption. | 002 → 373 |

## 3. Figures

### N1 `fig:qual-change`

A figure*, placed in Sec. exp-change after Table III.

**Rows.** Three rows: 015/675, 013/556 and 006/178. All frames are full and upright: 015 is landscape, 013 and 006 are portrait. A grid composer allows rows of mixed orientation; the panel width is about 0.85 in.

**Row labels.** The scene ID plus the published pixel-level AP (%) of columns (c)/(d)/(e)/(f)/(g):

| Scene | (c) | (d) | (e) | (f) | (g) |
|---|---|---|---|---|---|
| 015 | 85.7 | 87.9 | 81.3 | 83.6 | 86.3 |
| 013 | 44.1 | 64.4 | 23.4 | 56.0 | 63.2 |
| 006 | 13.4 | 36.8 | 48.4 | 13.6 | 55.7 |

**Columns.**
- (a) The 3D-GS map rendered at the update pose. Geometry is fixed, so the removed object's Gaussians are still rendered.
- (b) The update frame after the removal, with the masks_real outline.
- (c) Feature residual, k=2.
- (d) NormLift reliability drop.
- (e) Depth residual.
- (f) Ours: test, reset, re-admission, scored by the largest persistent score $\bar c_n$ over the update.
- (g) Ours fused with depth residual.
- (h) Our flags at the label-free threshold against the stale Gaussians, composited per Gaussian like RGB, using Okabe-Ito colours:
  - flagged and stale: #009E73
  - stale, not flagged: #E69F00
  - flagged and static: #CC79A7
  - other observed Gaussians: map colour × 0.5
  - not covered by any update keyframe: #404040

**Values in (c)-(g).**
- Render exactly as EC.pixel_level does: `TensorRow` with zero buffer and unc = scores, then `render_row` with alpha_eps = 1e-6 and undef_frac = 0.5.
- Colour each pixel by its mid-rank among the pooled covered pixels with a defined score from ALL update keyframes of the scene at 360×270. This is the Table III population; look it up with searchsorted in the sorted pooled array.
- Use viridis from 0 to 1, with one colorbar labelled "rank among the scene's update pixels".
- Grey means no score (ranked last in AP). Black means not covered (α < 0.5).
- Draw the mask outline with a dark halo so it stays visible on viridis.
- Render at 720×540 with K = `ds.K[index]`. Upsample the masks from 192×256 by nearest neighbour.

**Caption must say:**
- The scenes and frames come from the rules above.
- Geometry is fixed.
- Ranks are pooled over the scene's update keyframes.
- NormLift's drop has a higher pixel-level AP than our test in all three scenes, as in Table III (9 of 10 scenes).
- The masks are dilated silhouettes.
- FA/Rec of ours are counted over all update views: 015 5.9/54.2, 013 12.6/51.7, 006 3.3/29.1.

### N2 `fig:qual-query`

**Rows.** The same scenes, frames and rotations as N1.

**Columns.**
- (a) The update frame with the mask outline. The query class goes in the row label.
- (b) Before the update: the mapping-pass μ*.
- (c) After the update with the change test off: EC's `disp` copy, which saw the same views with no reset.
- (d) After the update, ours.
- (e) Optional: averaging after the update (`NormLift.feature()`).

**Value.**
- Show cos_p = F_p·t/‖F_p‖. F_p is the P0 render (μ*, or the averaging feature).
- t = Bᵀt̃ is not renormalised, so cos_p is the cosine between the decoded rendered direction and the unit text in the full feature space (`feature_pca.py:57-65`).
- Use one viridis scale per row, from p5 to p99 of panel (b)'s covered pixels, shared by (b) through (e). Black means not covered.

**Numbers.** These are new. Define and debug them on validation scenes 001-004, then run the test scenes once and disclose that this is a new test-set measurement. No choice may depend on them. Per scene, over the covered pixels of all update keyframes at 360×270:
- **Stale-answer rate:** the share of covered masks_real pixels whose argmax over the 101 classes (P0 render) is the query class.
- **Δcos:** mean cos_p inside masks_real minus mean cos_p outside.
- Report before, no change test, ours and averaging, each as mean ± std over N=10. Also report how many scenes have ours below the no-change-test field.
- Write them as JSON plus LaTeX rows.

**Caption must say:**
- The query rule.
- Geometry is fixed.
- Only Gaussians our test flags are reset (Rec. 54.2/51.7/29.1% in these scenes), so the missed ones keep the old class.

### S `fig:qual-compare`

**Layout.**
- Two blocks: room_1 frame 66 and scene0140_00 frame 57.
- Each block is one `RQ.compose` call with 6 columns × 3 rows, at the evaluation grid: 320×240 for Replica, 324×242 for ScanNet.

**Columns.**
1. Reference
2. 2D features, per frame
3. Feature 3DGS
4. Averaging / NormLift
5. LEGaussians (λu = 4 on Replica, 1 on ScanNet)
6. Ours P2

**Rows.**
- Row 1: the 3D-GS render | label maps from `RQ.scene_palette` (black = no decision).
- Row 2: ground truth | errors from `RQ.error_image`.
- Row 3:
  - Slots 0-1 hold the class legend (`compose` puts it there automatically).
  - Slot 2 (Feature 3DGS) stays empty with "no own uncertainty", like the "--" in Table I.
  - The remaining slots show NormLift 1−C_inter, LEGaussians' learned u and our u_ale.
  - Each value is a mid-rank among the covered pixels with a defined value over ALL evaluation frames of the scene, the AUSE population.
  - Grey means undefined; black means not covered.

**Annotations.** Read these from the Table I JSONs; never recompute them.
- The per-scene mIoU of each method.
- The per-scene AUSE (%) of each own uncertainty:

| Scene | u_ale (ours) | 1−C_inter (NormLift) | LEGaussians u |
|---|---|---|---|
| room_1 | 10.3 | 11.0 | 21.8 |
| scene0140_00 | 3.6 | 3.5 | 8.7 |

**Rules.**
- No P2 confidence panel on either dataset (see F7). On room_1, P2's ECE is 15.5 vs P0's 14.0.

**Caption must say:**
- The scenes and frames come from the rules above.
- On scene0140_00, ours (68.5 mIoU) is below averaging (69.1), Feature 3DGS (69.6) and the 2D features (69.4).
- On scene0140_00, NormLift's 1−C_inter ranks errors slightly better than u_ale (3.5 vs 3.6).

### F `fig:qual-fail`

Optional; otherwise it goes to the supplementary.
- **Row 1:** 011 at its false-alarm frame, with N1 columns (a), (b), (d), (e), (f), (h).
- **Row 2:** 009 at frame 354, with the same columns.
- **Row 3:** the static confident-error frame: 3D-GS render, ground truth, ours labels, errors, u_ale rank (pooled), and the confident-error mask in #CC79A7 over the dimmed render.
- **Caption:** states the rules and names the Limitations each row shows.

### Supplementary

- **supp_change_all:** the N1 layout for all 10 test scenes at their rule frames, plus the 4 validation scenes, labelled as validation.
- **supp_compare_all:** the S block for all 14 static test scenes at these frames:

| Scene | Frame |
|---|---|
| room_0 | 18 |
| room_1 | 66 |
| office_1 | 25 |
| office_2 | 75 |
| office_3 | 15 |
| office_4 | 29 |
| scene0070_00 | 46 |
| scene0097_00 | 2 |
| scene0140_00 | 57 |
| scene0200_00 | 36 |
| scene0347_00 | 0 |
| scene0400_00 | 0 |
| scene0590_00 | 104 |
| scene0645_00 | 156 |

- A CSS column (closed set, not ranked) is allowed here only.

### Constraints passed on to the teaser and video

- **Teaser:** validation scene 002 at frame 373.
- **Video:**
  - Per-view stacks at 360×270 by default, using the same scenes and rule frames.
  - No UI, usernames or paths anywhere in the frames.

## 4. Code plan (new files only)

### 1. `scripts/pick_frames.py` (CPU)

- **Subcommands:**
  - `static`: the ground-truth class rule for the S frames.
  - `nor`: the masks plus the Table III JSON, which reproduces S2/`frame_rule_check.py`.
  - `results`: the scene rules.
- **Output:** `D/uags_results/qual/select/selection.json`.
- **Check:** assert that the outcomes match section 2 exactly.

### 2. `scripts/dump_nor_qual.py` (GPU, gate 6500)

**Interface.**
- Parser: `EC.build_parser()` plus `--table --vocab --text-cache --frames rule|i,j --full-res --per-view --save-features`.
- `--out` is a directory.

**Example call:**

```bash
UAGS_GPU_HEADROOM_MIB=1024 scripts/gpu_gate.sh 6500 python3 scripts/dump_nor_qual.py \
  --field D/uags_fields/nor_015_lseg_relmax.pt --align none \
  --table D/uags_results/nor/tab2_relmax/test/015.json \
  --vocab configs/vocab_replica.json \
  --text-cache D/uags_results/qual/text_cache_lseg_nor.pt \
  --frames rule --full-res --out D/uags_results/qual/nor/015
```

**Steps.**

a. Call `gpu_reserve.apply`. Set update_stride, tau_cov, undef_frac, k and change_quantile from the record's settings, and refuse any conflicting value.

b. Run `ctx = EC.load_context(args)`. Assert that the mapping and update keyframe index lists equal the record's `views.mapping` and `views.update` exactly.

c. Run `EC.replay_mapping` and `EC.replay_check`, then compute `thr` as EC:784 does.

d. Capture the fields, only around `upd = EC.run_update(...)`:
   - Temporarily set `EC.DirectionalSemanticField` to a subclass whose `from_state` records each instance.
   - With `--per-view`, also wrap `EC.residuals_view` so it renders per view after the original call.
   - Restore both in a `finally` block.
   - Then `ours, disp = made[0], made[1]` and `nl = disp.view_hooks[0].__self__`.
   - Assert all of these:
     - `ours.cfg.change_test` is true and `disp.cfg.change_test` is false.
     - `upd["scores"]["ours"]` equals `where(isfinite(ours.c_last), ours.c_max, nan)`, NaN-aware.
     - `upd["robust_end"]` equals `ours.robust_mask()`.
     - `upd["scores"]["dispersion"]` equals `where(observed, disp.aleatoric()[0], nan)`.
   - Fallback: if any assert fails, copy EC:472-547 into the new script and prove equality on the toy scene.

e. Run `gt = EC.ground_truth(ctx, args)` and add `scores["fused"]`.

f. Reproduction check:
   - `EC.ranking_table` (Gaussian level) and `EC.pixel_level` (pixel level): |ΔAP| ≤ 0.005 for every row.
   - Per-frame n_covered and n_pos equal to the record (the forward render is deterministic).
   - Ours FA/Rec within 0.5 points.
   - Otherwise exit 2.
   - Keep each row's pooled covered scores, sorted, for the rank display.

g. Query:
   - Text via `ES.text_embeddings(vocab, all 101 classes, args)` with `extractor="lseg"` and `clip_device="cpu"`, using the NOR cache copy, then `ctx["pca"].project_text`.
   - Take the argmax over the stale set only, in chunks.

h. Save `gauss.pt` with:
   - the scores and flags
   - observed, seen, stale, in_box, changed_end, robust_end, robust_map, n_views
   - the query (name, top-3 shares, t)
   - μ·t before, with no change test, for ours and for averaging
   - u_epi and u_ale before and after
   - With `--save-features`, only for 015, 013, 006 and 002: fp16 features, about 0.5 GB per 1M Gaussians for the four sets.

i. For each rule frame, at 720×540 (K = `ds.K[index]`) and at 360×270, save:
   - photo, mask, map RGB, α
   - the 7 score renders and the flag tint
   - the cos maps and argmax maps
   - the rotation

j. Compute the N2 numbers over all update keyframes and write `contrast.json`.

k. Write the false-alarm frame listing, which Fig. F needs.

l. Write `meta.json`:
   - argv, wall time and peak MiB
   - the sha256 of this script and `EC.code_state()`
   - the field and map sha256
   - the reproduction deltas and `replay_check`
   - the selection outcomes

### 3. `scripts/qual_compare.py` (GPU)

**Interface.**
- Parser: `ES.build_parser()` plus `--stage tokens|render --scene --frame --cols --tokens --meta --dump --list-confident-errors`.
- Row names: `ours_P2,normlift,f3dgs,legs_l4`. Use `legs_l1` on ScanNet.

**Example calls (Replica):**

```bash
scripts/gpu_gate.sh 4500 python3 scripts/qual_compare.py --stage tokens \
  --vocab configs/vocab_replica.json --scenes configs/replica_test.json \
  --scene room_1 --frame 66 --out D/uags_results/qual/compare/room_1_f66_tok.pt

scripts/gpu_gate.sh 4500 python3 scripts/qual_compare.py --stage render \
  --vocab configs/vocab_replica.json --scenes configs/replica_test.json \
  --scene room_1 --frame 66 \
  --tokens D/uags_results/qual/compare/room_1_f66_tok.pt \
  --betas D/uags_results/tab1/eval_val.json \
  --text-cache D/uags_results/qual/text_cache_lseg_replica.pt \
  --row-factory usplat.baselines:row_factory,usplat.distill:row_factory \
  --out D/uags_results/qual/compare/replica_room_1_f66.pdf
```

For ScanNet, use `vocab_scannet`, `scannet_test`, `scene0140_00`, `--frame 57`, the betas in `scannet_tab1/eval_val.json` and `text_cache_lseg_scannet.pt`.

**Tokens stage.**
- Read the frame image with `ES.load_frames` (with `with_2d=True`) on the CPU.
- Extract the tokens with `ES._extractor(args).extract_tokens(img)` and save them on the CPU.

**Render stage.**

a. Call `gpu_reserve.apply`, then `ES.check_fit_settings(ES.load_fit_settings(betas), args, betas)` and `ES.load_betas`.

b. Set `args.rows` as above, `no_p3=True`, `with_2d=False`. Run `S = ES.load_scene(...)` and `ES.betas_used`.

c. For the chosen frame:
   - `ras = ES.make_rasterizer(...)`.
   - One call to `evaluate_view(S["rows"], ras, S["text"], zeros(P), betas, tau_cov, undef_frac=...)`.
   - `RQ.render_rgb`.

d. For the 2D row:
   - `z = ES.frame_features_2d(S, dict(tok2d=tok, h=h, w=w), args)`.
   - Then `evaluate_view([TensorRow("2d_features", "feature", buf=zeros(1, d))], ES.PixelSource(z), ...)`.

e. For pooled ranks: run `render_row` on every evaluation frame for the three own-uncertainty channels, and keep the sorted covered defined values.

f. Draw with `RQ.compose` and its helpers. Read the per-scene mIoU and AUSE from the published JSON, then save the figure and `meta`.

g. `--list-confident-errors`: ours_P2 only, over all frames of the 14 scenes. Compute scene medians first, then counts, and write JSON. Gate 4500.

### 4. `scripts/plot_nor_qual.py` (CPU)

- Subcommands `change | query | failures | grid | numbers`.
- A grid composer for rows of mixed orientation.
- Reuses `RQ.RC`, `RQ._legend`, `RQ._colorbar` and `RQ.save`.

### 5. `scripts/qual_figs.sh`

- Runs the whole chain: validation first, then test, with the environment from section 0.
- Writes one log per step.

### 6. Tests (CPU, mock backend)

**`tests/test_dump_nor_qual.py`.**
- Uses the toy scene from `tests/test_eval_change.py:165` `build()`.
- Patch `ec.load_extractor` to `Fake`, as that test does, and patch the text embeddings.
- Check:
  - The dumped scores and flags equal `EC.evaluate`'s.
  - The capture asserts hold.
  - The pooled-rank function is correct.
  - The frame and rotation rules give the expected outcomes.

**`tests/test_qual_compare.py`.**
- Uses the toy from `tests/test_seg_eval.py:535` `_cli_toy`.
- Check:
  - Labels and coverage equal `evaluate_view`'s with the real ground truth.
  - Pooled ranks are correct.
  - The 2D-row path works.

### 7. `papers/figs/README.md`

Add each new PDF with its generating command.

## 5. Verification before a figure enters the paper

| ID | Check |
|---|---|
| V1 | Every NOR dump: keyframe lists identical; per-frame n_covered/n_pos identical; Gaussian- and pixel-level AP per row within 0.005; ours FA/Rec within 0.5 points. |
| V2 | The capture identity asserts pass. |
| V3 | `qual_compare` labels and coverage equal `evaluate_view`'s. Every printed number equals its published JSON value; captions are generated from the JSON. |
| V4 | The new tests pass, and `tests/test_render_qual.py`, `test_eval_change.py` and `test_seg_eval.py` still pass. |
| V5 | `pdffonts` shows no Type 3 fonts. `pdfinfo` metadata holds no names. No paths or usernames appear in any figure or video. Every photo is checked visually for faces and number plates. |
| V6 | `git diff --stat` touches no frozen file (section 0); only new files, `papers/` and `docs/` change. |
| V7 | N2 uses one vmin/vmax per row (asserted in code). The N1 rank population size equals the record's pixel n_pix minus that row's NaN pixels. |
| V8 | The run report lists every test look and every display change made after a test render. |

## 6. GPU budget

Wall time is under the gate, estimated from the measured per-scene EC timings × 1.4. Waits for free memory come on top.

| Job | Estimate |
|---|---|
| NOR validation dumps 001-004 | ≈ 14 min (EC 609 s) |
| NOR test dumps, all 10 scenes | ≈ 19 min (EC 802 s), plus ≈ 3 min for the pooled-rank pass |
| Static S, validation development plus 2 test scenes | ≈ 5 min |
| supp_compare_all, 14 scenes | ≈ 10-15 min |
| Confident-error listing | ≈ 10 min |

- N1 + N2 + S: about 45 min. With F and the supplementary grids: about 70 min.
- The NOR jobs need about 7.5 GB free (6500 + 1024 headroom), so they will wait while other users hold more than about 4.8 GB.

## 7. Required disclosures and remaining risks

- **NormLift wins at pixel level.** NormLift's drop beats our test at pixel level in all three N1 scenes (9 of 10 in Table III). Keep the rule and say so in the caption.
- **N2 may show a small effect.** Only flagged Gaussians are reset (Rec. 29-54% in the N2 scenes). Report the numbers whatever they show, and disclose them as a new test-set measurement.
- **ScanNet P2 coefficients.** The P2 fit has a, b < 0, so β grows with uncertainty. Add one sentence to the paper (Sec. exp-replica). No figure shows P2 confidence.
- **Re-run noise.** Re-running the update on the GPU is not bit-exact. If V1 fails, stop and report; never retune anything.
- **The capture monkeypatch** is guarded by the identity asserts. The fallback is to copy the update loop and test it for equality on the toy scene.
- **The N2 query rule** may return a generic class. Report it as computed.
- **Masks are dilated silhouettes** covering 8-52% of the chosen frames. Say this in the captions.
- **Third-party product branding** appears on the 006 box and the 008 backpack. List it in the report; blur it only if the user asks.
