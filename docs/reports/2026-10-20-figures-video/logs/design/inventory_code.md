# UAGS inventory: code to reuse for rendering images and video frames (read-only, 2026-10-20)

Path shorthands:
- R = /home/thanhnc/gauss_ws/src/uags
- D = /home/thanhnc/gauss_ws/data
- RQ = R/scripts/render_qual.py, ES = R/scripts/eval_seg.py, EC = R/scripts/eval_change_nor.py, TF = R/scripts/time_field.py

Two facts apply to everything below:
- **Real renders need the GPU (gsplat 1.5.3).** The CPU `--backend mock` builds a dense (P, N) weight matrix with a Python loop over N, so it only works on toy scenes. It runs RQ and EC on CPU toys in R/tests/test_render_qual.py and R/tests/test_eval_change.py; use those to smoke-test new scripts.
- **Nothing was changed.** `git status` is clean and no GPU job was run. Scratch tests are in /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/

## 1. Load a 3DGS map and a saved field
| file:function | inputs → outputs | cost | example |
|---|---|---|---|
| usplat/gs_io.py:73 `load_gaussians(path, device)` | map .pt → dict: means (N,3), quats (N,4, wxyz, unit), scales, opacities (N,), colors (N,3). Colors are the SH DC term only, because every map stores `sh0` only. | about 56 MB GPU per 1M Gaussians | `splats = load_gaussians(f"{D}/uags_maps/room_0_seq1.pt", device="cuda")` |
| `torch.load(F, map_location="cpu", weights_only=False)` | field ckpt keys: field, pca, config{ply, data, dataset, stride, downscale, extractor, out}, baselines, timing{keyframes}, extractor_options | files: Replica 0.6–1.35 GB, ScanNet about 1.2 GB, NOR 0.17–1.13 GB | `ck = torch.load(f"{D}/uags_fields/room_0_lseg.pt", ...)` |
| usplat/vmf_field.py:811 `DirectionalSemanticField.from_state(ck["field"], device)`; usplat/feature_pca.py:73 `StreamingPCA.from_state(ck["pca"], device)` | field object (read-outs :719 mu_post, :734 epistemic, :748 aleatoric → (u, valid), :765 robust_mask); PCA basis 512×64 | RAM ≥ 3 × N×64×4 B (R, Rf, Rs) | `field = DSF.from_state(ck["field"], "cpu")` |
| ES:256 `load_scene(sc, vocab, args, device)` | scenes entry → dict: field (CPU), splats (device), frames[{idx, viewmat (w2c), K scaled by --downscale, w, h, gt}], text (K,64) projected, pca, rows (frozen TensorRows on device), class_names. Needs labelled frames: Replica and ScanNet only, not NOR. | see section 3 | `S = ES.load_scene(sc, ES.load_vocab(v), args, "cuda")` |
| RQ:101 `load(sc, vocab, betas, args, field=None)` | (S, row) with only `args.row` built. `field=` swaps in a k-view field. Refuses a row without u_epi/u_ale (RQ:90 figure_row). | — | `S, row = RQ.load(sc, vocab, betas, args)` |
| EC:697 `load_context(args)` | ctx: ckpt, splats, pca, frozen{colors, mu=mu*}, map_views / upd_views[{tok (LSeg tokens, CPU), c2w, K, h, w, rgb (P,3), depth (P,), mask_real 192×256, index}] | runs LSeg on every map and update view: 25–66 s per scene | see section 5 |

For metadata only, use `torch.load(..., mmap=True)` as RQ:111 n_mapping_views and TF:172 load_field_file do.

## 2. Render RGB and depth at any pose and resolution
- **Rasterizer:** usplat/gsplat_adapter.py:74 `GSplatFeatureRasterizer(means, quats, scales, opacities, viewmat (w2c 4×4), K, width, height, near_plane=0.01, far_plane=1e10, channel_chunk=32)`. Build it with `:92 from_splats(splats, viewmat, K, w, h)`.
  - `:112 __call__(buf (N,C))` → (P = H·W row-major, C). It alpha-composites with no background (black) and is affine in buf.
  - `:117 render_image(colors)` → (H,W,k) and alpha (H,W).
  - `:123 render_depth()` → expected depth (H,W) and alpha.
  - `:140 gaussian_depths()` → (N,).
  - It calls `gsplat.rasterization(packed=True, render_mode="RGB"/"ED")` with raw colors (no SH).
- **Factory:** ES:245 `make_rasterizer(backend, splats, fr, args)` takes `fr = dict(viewmat, K, w, h)` and `args.channel_chunk`. For a new resolution, scale K rows 0 and 1 by w/W and h/H, as ES:186 load_frames and usplat/datasets.py:665 downscale do.
- **RGB:** RQ:126 `render_rgb(splats, ras, h, w)` → (h,w,3) on CPU in [0,1], composited on black (one call with [colors | 1]). For a white background, add (1−α)·bg, with α from the ones channel.
- **Example:** `ras = GSplatFeatureRasterizer.from_splats(splats, torch.linalg.inv(c2w.double()).float().cuda(), K.cuda(), 640, 480); rgb = RQ.render_rgb(splats, ras, 480, 640)`
- **Speed:** a render alone was never timed. The closest numbers are mapping updates (render + backward, 64 channels), from `ck["timing"]`: 50 ms per keyframe for room_0 at 320×240 on an RTX 5070 Ti, and 108 ms for nor_001 at 360×270 on the 4070 Ti.

## 3. Per-pixel labels, probabilities and u_epi / u_ale for a row
- **usplat/seg_eval.py:109 `render_row(row, ras, alpha_eps=1e-6, undef_frac=0.5)`** → RowRender: feat (P,C) not divided by α, alpha (P,), unc{name: (P,), NaN = undefined}. It is one call on [buf | 1 | u…] and applies the LEGaussians decoder.
- **:161 `class_scores(rr, text)`** → (P,K): F·t for feature rows, log q for probability rows (CSS, P3).
- **:185 `pixel_beta(row, beta, rr.unc)`** → a scalar, or the per-pixel P2 temperature β0·exp(−a·u_epi − b·u_ale − g·1[undefined]). The probabilities are `softmax(beta * scores, 1)`.
- **Full probability vector:** call those three functions. `evaluate_view` returns only conf, entropy and pred.
- **:223 `evaluate_view(rows, ras, text, gt, betas, tau_cov, undef_frac=, raw_probs=)`** → per row: pix, gt, pred (−1 = uncovered or tie), conf, entropy, nll, brier, correct, covered (α ≥ 0.5), tied, alpha, u{}, on labelled pixels. Pass `gt = zeros(P)` to get every pixel; RQ does this.
- **RQ:135 `frame_maps(S, i, row, betas, args, with_rgb)`** → (h,w) maps: gt, pred, covered, labelled, correct, u{u_epi, u_ale}, rgb. It accepts any row, baselines included.
- **Our rows:** usplat/rows.py:133 `field_rows(field)` → ours_P0 (mu*), ours_P1 (A_d(κ)·mu*), ours_P2 (the P0 buffer with the adaptive temperature). Freeze with `:338 freeze(row)`. Channels come from `:85 _field_channel`: u_epi, u_ale, spread, kish_epi, change.
- **Baseline rows:** set `args.row_factory="usplat.baselines:row_factory,usplat.distill:row_factory"` and `args.rows="ours_P2,averaging,f3dgs,legs_l4,css"`.
  - usplat/baselines.py:733 `row_factory` gives:
    - averaging
    - normlift, with u channels normlift_1-R, normlift_epi, normlift_1-Cinter
    - viewcount, with viewcount and viewcount_a0
    - css and css_a{0.01,0.1,10}, with css_var_top, css_var, css_exp_entropy, css_mi, yager_conflict
  - usplat/distill.py:225 `row_factory` gives f3dgs and legs_l<λ> (channel legs_u). The files sit next to the field as `<stem>_f3dgs.pt` and `<stem>_legs_l4.pt` (Replica test) or `_legs_l1.pt` (ScanNet test).
- **Temperatures:** ES:555 `load_betas(f"{D}/uags_results/tab1/eval_val.json")` for Replica, `scannet_tab1/eval_val.json` for ScanNet. Replica P2 = {beta0 45.56, a 0.361, b 11.83, g 0.055}. The CSS read-out is the one with the lowest validation NLL: `python3 scripts/make_tables.py choose <eval_val.json> --key css`.
- **CLI:** `scripts/gpu_gate.sh 4000 python3 scripts/render_qual.py --vocab configs/vocab_replica.json --extractor lseg --scenes configs/replica_test.json --scene room_0 --betas $D/uags_results/tab1/eval_val.json --text-cache <COPY of text_cache_lseg.pt> --frame 29 --out <dir>/x.pdf`
  - `--kfield` adds the k-view row.
  - `--list-frames N` ranks frames by e_red / e_irr, a rule written before rendering. Example output: D/uags_results/qual/frames_room_0.json.
- **Measured cost:**
  - RQ, two-map figure: 9.6 s wall (D/uags_results/qual/qual_room_0_f29.json).
  - ES, Replica test, 90 frames × 17 rows: 9–20 s per scene, peak GPU 2.51 GiB, peak RAM 12.8 GB. Measured on an RTX 5070 Ti; source R/docs/reports/2026-09-29-replica-tab1/logs/eval_test_224fe1e.json. The batch scripts gate it at 7000 MiB.

## 4. Open-vocabulary text-query relevance map
**What exists:** the text side only. There is no pixel render of a query.
- ES:141 `text_embeddings(vocab, idx, args)`, with cache key `"lseg|<5 templates>|<prompt>"`. Caches: D/uags_fields/text_cache_lseg.pt (51 Replica prompts) and D/uags_fields/scannet/text_cache_lseg.pt (18 prompts).
- usplat/extractors/__init__.py:18 `get_extractor("lseg", device="cpu")` → usplat/extractors/lseg.py:282 `LSegDense.text_embedding(prompts, templates)` → (K,512) unit vectors. It loads the whole 1.73 GB D/weights/lseg/demo_e200_state_dict.pt, which is fine on the CPU.
- usplat/feature_pca.py:59 `project_text(t)` (no renormalisation).
- usplat/vmf_field.py:773 `field.query(t)` → per-Gaussian (N,K) cosine. R/scripts/run_mapping.py:689 `--query` only prints statistics.

**Minimal code to add (about 6 lines):**
```python
t = pca.project_text(lseg.text_embedding([q], templates=tuple(vocab["templates_by_extractor"]["lseg"])).to(dev))  # (1,64)
row = freeze(field_rows(field)[2]); rr = render_row(row, ras)          # ours_P2: F = sum_n w mu*, u channels
cos = (rr.feat @ t.T)[:, 0] / rr.feat.norm(dim=1).clamp_min(1e-6) / t.norm()        # heat map
s = class_scores(rr, torch.cat([t, S["text"]]))                                      # query vs scene classes
p = torch.softmax(pixel_beta(row, betas["ours_P2"], rr.unc)[:, None] * s, 1)[:, 0]   # P2-calibrated p(query)
# mask pixels with rr.alpha < 0.5 as not covered
```

## 5. NOR update replay (EC) and a hook for per-view renders
**Where the quantities live in EC:**
- :269 `rasterizer(ctx, v)` → (ras, Gaussian depths); :278 `view_inputs(ctx, v)` → (z, ras, gz, depth kwargs).
- :288 `map_render(ras, gz, frozen)` → the frozen map's rgb (on black), feat (mu*) and depth.
- :302 `covered(field, stats)` → coverage mask.
- :423 `replay_mapping`: its only use is to set the baselines' thresholds.
- :472 `run_update`, with the view loop at :499–532:
  - ours: `calibrate_change_threshold(q)` from the saved field's own c_hist (:479)
  - `record("ours", last_tested, last_fired)`
  - it returns scores (ours = c_max), flags, changed_end, robust_end, robust_map.
- :659 `pixel_level` renders per-Gaussian scores into each view with `TensorRow` + `render_row`. Reuse that pattern.

**Where they live in usplat/vmf_field.py `_ingest` (:472):**
- score c = 1 − cos(Rs, Rf) at :511; persistence `c_p = c_ring.min` at :517
- threshold :526, fire :527, `last_tested`/`last_fired` :528
- reset block through :542 (`changed |= fire`)
- re-admission (views ≥ 3 and depth residual ≤ 5 cm) at :563–578; `robust_mask` at :765

**Measured cost:** peak alloc 4.56 GB, gate 6500 MiB. Per scene: replay 7–25 s, update 10–32 s for 22–79 update views, grid 270×360 (D/uags_results/nor/tab2_relmax/{val,test}/0XX.json).

**Hook:** write a new script rather than editing EC. EC:241 `code_state()` hashes eval_change_nor.py into the Table III provenance, so an edit would change that hash. Our change path reproduces Table III up to GPU float noise, and skipping `replay_mapping` is fine when baselines are not shown.
```python
ctx = EC.load_context(argparse.Namespace(field=F, align="none", ply="", data="", update_stride=0,
                      max_update_keyframes=0, device="cuda", backend="gsplat", channel_chunk=0))
ours = DSF.from_state(ctx["ckpt"]["field"], device="cuda"); ours.calibrate_change_threshold(0.999); r0 = ours.n_resets.clone()
for v in ctx["upd_views"]:
    z, ras, gz, kw = EC.view_inputs(ctx, v); ren = EC.map_render(ras, gz, ctx["frozen"])
    st = ours.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw); nan = torch.full_like(ours.c_last, float("nan"))
    unc = dict(score=torch.where(ours.last_tested, ours.c_last, nan), fired=ours.last_fired.float(),
               reset=(ours.n_resets > r0).float(), robust=ours.robust_mask().float(), u_epi=ours.epistemic())
    rr = render_row(TensorRow("v", "feature", buf=ctx["frozen"]["colors"], unc=unc), ras)   # rr.unc[...] (P,)
    # save per view: v["rgb"] (observed, after removal), ren["rgb"] (map, before), rr.unc, rr.alpha, v["mask_real"]
torch.save(dict(field=ours.state_dict(), pca=ctx["ckpt"]["pca"], config=ctx["cfg"]), OUT)  # post-update field
```
- Binary masks render as an α-weighted share per pixel.
- NOR has no semantic labels, so `ES.load_scene` cannot read the saved post-update field. Render its semantics with FieldRow → freeze → render_row and text queries instead.
- For a mapping-progress video, copy the loop of TF:295 `map_pass` or EC:423 `replay_mapping` (a fresh field fed keyframe by keyframe; neither has a callback) and render u_epi after each keyframe.

## 6. Camera trajectories
| data | loader (usplat/datasets.py) | poses / K | frames |
|---|---|---|---|
| Replica | :389 `ReplicaSemNeRFDataset(D/replica_semnerf/Replica_Dataset/<scene>/Sequence_{1,2})` | `.poses` (900,4,4) numpy c2w in OpenCV convention; fixed K, fx = fy = 320, 640×480 | 900 continuous frames per sequence. Seq_1 is mapping (stride 5, 179 keyframes); Seq_2 is evaluation (stride 10, 90 frames). |
| ScanNet | :126 `ScanNetDataset(D/scannet/uags_f2/<scene>, stride, offset)` | pose/<i>.txt c2w; K from intrinsic_color, fx 584.8 at 648×484. Lost tracking gives −inf poses: skip non-finite ones. | 5578 / 730 / 1322 frames (scenes 0000 / 0062 / 0070). Mapping stride 10; evaluation stride 20, offset 5. |
| NOR | :475 `NeRFObjectRemovalDataset(root, stride, part="map"/"update"/"all", align=None)` | `.c2w` (n,4,4) float64; `.K` (n,3,3) per frame (the principal point drifts up to 12 px); 720×540. Table III used align "none" everywhere. | meta.json split, e.g. 001: map 0–256, update 322–509. Per-frame `mask_real`. Object box: usplat/box3d.py:77 `read_box_corners(ds.box_path)`. |

**Helpers:**
- TF:136 `select_keyframes(ds, 0.05, 5.0, max_kf)` and TF:157 `evenly(idx, n)`
- R/scripts/run_mapping.py:421 `_pose_gap(a, b)` → (m, deg)
- R/scripts/make_synthetic_scene.py:47 `orbit_poses(n, radius, height)` (looks at the origin)
- R/tests/test_gs_fit.py:70 `_look_at(eye, target)`

**There is no pose interpolation in the repo.** Use `scipy.spatial.transform.Slerp` (scipy 1.15.3) for rotation and linear interpolation for translation. For a NOR frame that is not in the capture, take K from the nearest frame.

## 7. Existing figure code and TikZ sources
- **RQ helpers:** :176 class_palette, :187 scene_palette, :202 label_image, :211 u_image (viridis; undefined grey, uncovered black), :221 error_image, :240 nice_up, :248 u_max, :258 legend_classes. :320 `compose(rows, col_titles, classes, errors, cbars)` builds a fixed 6-column figure 7.16 in wide. :363 `save(fig, out, dpi=600, png_dpi=300)` writes .pdf and .png. `RC` sets serif TrueType fonts for IEEE.
- **R/scripts/plot_views.py:** :258 collect, :527 plot, :470 draw. 2×2 panels of AUSE ratio against M, Okabe-Ito colours, 3.5×2.6 in, from the views JSON and the Table I JSON.
- **R/scripts/make_tables.py:** LaTeX rows only (tab1, tab1pair, tab3, tab2, perclass, choose/select). No plotting.
- **TikZ (standalone):** R/papers/figs/pipeline.tex (190 lines, 7.0 in, figure*), vmf_sphere.tex, gradient_trick.tex, uncertainty_plane.tex. qual.pdf comes from RQ and views.pdf from plot_views.
  - Build from papers/ with `pdflatex -output-directory <dir> figs/<name>.tex`. pdflatex is TinyTeX in ~/.local/bin. tectonic, named in figs/README.md, is not installed.
  - I compiled pipeline.tex into scratch (OK) and rasterised it with `pdftoppm -r 100 -png` (OK). gs and pdftocairo are present; ImageMagick and inkscape are not.

## 8. Packages in ~/venvs/uags and video-writer test (CPU, scratch videotest/)
| package | version | note |
|---|---|---|
| imageio | 2.38.1 | |
| imageio-ffmpeg | 0.6.0 | bundled static ffmpeg 7.0.2 with libx264, libx265, libvpx-vp9, libaom-av1, mpeg4, gif. No system ffmpeg or ffprobe. |
| opencv-python (+ headless) | 4.11.0.86 | FFMPEG backend |
| matplotlib | 3.11.2 | default backend tkagg: use Agg |
| Pillow | 12.3.0 | |
| scipy | 1.15.3 | |
| scikit-image | 0.26.0 | |
| kornia | 0.8.3 | |
| numpy | 1.26.4 | |
| torch | 2.11.0+cu128 | |
| torchvision | 0.26.0 | `io.write_video` needs PyAV, which is absent |
| gsplat | 1.5.3 | |
| open_clip_torch | 3.3.0 | |
| timm | 1.0.30 | |
| plyfile | 1.1.5 | |

Fonts available: Nimbus Roman, Liberation Serif, STIX, DejaVu Serif.

| writer | result |
|---|---|
| `imageio.get_writer(p, fps=30, codec="libx264", quality=8, pixelformat="yuv420p")` | PASS (h264 yuv420p, 30/30 frames read back) |
| `cv2.VideoWriter` mp4v .mp4 | PASS |
| `cv2.VideoWriter` XVID .avi, MJPG .avi, VP90 .webm | PASS |
| `cv2.VideoWriter` avc1 / H264 | FAIL: no encoder in OpenCV's ffmpeg |
| matplotlib `FFMpegWriter` (libx264) | PASS only after `rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()` |
| `imageio.mimsave` GIF | PASS |

## 9. Pitfalls
- **Text cache writes:** `ES.text_embeddings` adds new prompts to the `--text-cache` file. Always pass a copy, as R/scripts/replica_views.sh does.
- **NOR field names:** Table III used `D/uags_fields/nor_0XX_lseg_relmax.pt` (change_weight relmax, fast_min 0.5). configs/nor_*.json still name `nor_0XX_lseg.pt`, which does not exist.
- **k-view fields:** only `D/uags_results/qual/kfields/room_0_lseg_k30.pt` is kept; the others were deleted (KEEP=0). Rebuild them with `run_mapping.py --max-keyframes k --pca-from <full field>`.
- **Resolution:** evaluation renders at downscale 2 (Replica 320×240, ScanNet 324×242, NOR 360×270). Video can render at full resolution with K scaled. libx264 needs sides that are multiples of 16, or imageio resizes; for example, pad 360×270 to 368×272.
- **Hardware for the numbers above:** the Replica Table I timings come from an RTX 5070 Ti. The NOR numbers and the current card are an RTX 4070 Ti with 12 GB, about 6.7 GB free at 23:15. Every GPU job goes through `scripts/gpu_gate.sh <MiB>`.