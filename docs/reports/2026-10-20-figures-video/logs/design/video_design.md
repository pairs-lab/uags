# One anonymous file, supplementary_video.mp4: 1920x1080, 30 fps, H.264 High, yuv420p, no audio, captions burned in, about 136 s (2:16). It is encoded in two passes at about 2.5 Mbit/s to stay at or under 44 MB, and zipped with an anonymous README.txt and SUMMARY.txt into the single RA-L multimedia archive (limit 50 MB). Also kept: an exact-1080p CRF-12 master (about 100-200 MB) for the camera-ready version, and every per-segment dump, storyboard JSON and check log under /home/thanhnc/gauss_ws/data/uags_results/video/.

The video runs 2:16 at 1920x1080, 30 fps, silent, with captions in the frame. It has 16 shots: title, the problem as three questions, the paper's pipeline figure with build-time spotlights on each step, three static-scene results and the object-removal results. The static-scene results are: a mapping replay seen from a held-out Replica view, the qualitative figure animated along a held-out path (full map vs M=30), and a ScanNet validation clip. The object-removal timeline replays the Table III update of validation scene 002 (chair) and held-out test scene 014 (bicycle) frame by frame. Each frame shows the camera image, the old map render, our change score c-bar/theta(h), p(object) for averaging and for ours with the flagged-changed overlay, and two measured curves. The replay must match each scene's Table III operating point (Rec, FA, Kept, flagged at the end) within 0.5 points. Every render reuses code paths I read and cite (EC.load_context/view_inputs/covered/make_normlift/ground_truth, DSF.update_view, seg_eval.render_row, RQ.frame_maps, the TF.map_pass loop) through three new scripts and a CPU compositor, so eval_change_nor.py and render_qual.py stay unchanged. Scenes and frames come from rules fixed before rendering, captions quote only numbers parsed from the paper's tables, and encoding was tested on the CPU (exact 1080p master, two-pass delivery on target). The GPU work is about 15 minutes, all through gpu_gate.sh.

# Supplementary video: design (read-only phase, 2026-10-20)

Paths: R = /home/thanhnc/gauss_ws/src/uags, D = /home/thanhnc/gauss_ws/data, OUT = D/uags_results/video, SCR = /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/video (scratch evidence from this phase; copy into the report's logs/ before citing). Nothing in R or D was changed (`git status` clean) and no GPU job ran.

## 0 Constraints (checked)
| Item | Rule | Source |
|---|---|---|
| Package | One video. One zip of at most 50 MB with README.txt and SUMMARY.txt (ASCII). The paper must be understandable without it. | RA-L Information for Authors (ieee-ras.org) |
| Video | mp4/mpeg with a standard codec. Aim for 1-3 min. Subtitles recommended. Any speech in English. | RAS video submission guidelines |
| Anonymity | No author, affiliation or lab names or logos. Blur faces. No links that reveal identity. | RAS rules for double-anonymous review |
| Project | No tuning on test scenes. Every scene and frame chosen by a rule written before rendering, or taken from validation. GPU only through scripts/gpu_gate.sh. Captions quote only table numbers. Overlays show only measured values. Claude is never named. | task |

## 1 Format and style
- **Canvas:** 1920x1080, 30 fps, white background as in the paper figures. Ink #0B0B0B and #52514E. Safe margin 40 px. Silent. Cuts with 0.3 s fades.
- **Fonts:** DejaVu Sans (/usr/share/fonts/truetype/dejavu). Titles 40 px bold, panel titles 27 px bold, captions 28 px, minimum 24 px. Math is pre-rendered once per segment with matplotlib mathtext.
- **Colours follow the paper:**
  - Ours #0072B2, averaging #D55E00, NormLift #E69F00, grey #898781 (scripts/plot_views.py:71-85).
  - Classes use RQ.scene_palette (render_qual.py:187), the palette of the qualitative figure.
  - u maps use viridis with that figure's fixed tops: u_epi 0.81, u_ale 0.053 (D/uags_results/qual/qual_room_0_f29.json, "vmax").
  - Undefined u_ale is grey, uncovered is black, errors are white/red/blue (render_qual.py:62-67).
  - p(query) uses magma on 0..1, blended over the greyscale map.
  - The change ratio c-bar/theta(h) uses coolwarm on 0..2, where 1 is the threshold. Flagged-changed pixels are hatched in blue.
- **Figure numbers:** resolve them from the .aux at build time, or name the figures. In the current main.tex the caption order is pipeline=1, plane=2, qual=3, views=4. papers/main.pdf is stale and still shows "Fig. 4 Qualitative results [TODO]".

## 2 Storyboard (136 s)
| # | t | dur | What is shown | Data (rule) | Producer |
|---|---|---|---|---|---|
| S0 | 0:00 | 4 s | Title card: paper title, "Supplementary video", "Anonymous submission" | - | compose |
| S1a | 0:04 | 6 s | 3DGS RGB render next to Ours P2 labels with legend. Caption: "A robot reuses a semantic 3D map and queries it in language." | Replica room_0 (test), Sequence_2 frames 0-179 at 30 fps (R2) | seg_dump path |
| S1b | 0:10 | 6 s | Triptych of three questions: "How much evidence?" (u_epi, M=30, frame 290); "Do the views agree?" (u_ale, full map, frame 290); "Is it still there?" (NOR 002 photos 58 -> 335) | R1, R8 | seg_dump + dataset JPGs |
| S2a | 0:16 | 4 s | Pipeline figure, whole | papers/figs/pipeline.tex | compose figures |
| S2b | 0:20 | 21 s | Six spotlights, 3.5 s each: steps 1-3, 4, 5, 6, 7, 8. One plain-English caption each. | build-time copy of pipeline.tex | compose figures |
| S2c | 0:41 | 5 s | Uncertainty plane figure. Caption: "little evidence (left) vs conflicting evidence (bottom); a reset moves a Gaussian back left" | figs/uncertainty_plane.pdf | compose |
| S3a | 0:46 | 13 s | Evidence accumulates: the field is fed 179 mapping keyframes. Panels: keyframe image with a top-down minimap, P2 labels, u_epi, u_ale, all at held-out view 290. Counter t/179, 2 video frames per keyframe. | R1, R3 | seg_dump progress |
| S3b | 0:59 | 10 s | Qualitative figure animated as a 2x5 grid. Rows: full map, M=30. Columns: render or GT, labels, u_epi, u_ale, errors or e_red/e_irr. Frames 230-350 at 15 fps, then a 2 s hold on 290. | R1, R2, R4 | seg_dump path |
| S3c | 1:09 | 8 s | Real scans: render, GT, P2 labels, u_ale, errors | ScanNet scene0000_00 (val) window (R5) | seg_dump path |
| S3d | 1:17 | 9 s | Views figure still (4 s), then a Table I card (5 s) | main.tex tab:replica, figs/views.pdf | compose |
| S4a | 1:26 | 4 s | NOR setup: 002 before (58) and after (335) photos | R6, R8 | dataset |
| S4b | 1:30 | 17 s | NOR timeline for 002 (val, chair): 2 s "before" hold, 156 update frames at 15 fps, 2 s end hold, 3 s 3D summary at map frame 58 | R6, R9, R10, R12 | nor_dump |
| S4c | 1:47 | 12 s | Held-out test scene 014 (bicycle): every 2nd update frame (149 frames) at 15 fps, then a 2 s hold | R7, R9, R10 | nor_dump |
| S4d | 1:59 | 8 s | Table III card: Gaussian-level AP bars for all 7 rows (N=10), operating point, Kept 4.8 vs 9.5 (Table IV) | tab:change, tab:ablation | compose |
| S5 | 2:07 | 7 s | Limitations: the paper's 4 main points | main.tex sec:limitations | compose |
| S6 | 2:14 | 3 s | End card, "Thank you". No names, no links. | - | compose |

**Cut order if the video runs over 2:45:** S3c first, then S2c.

**Optional O1 (+15 s, +about 25 GPU-min):** all 10 included test scenes, final state, 1.5 s each.

**Captions use only table numbers (parsed from main.tex):**
- **Table I:** mIoU 28.0 vs 27.9 (Replica), 68.0 vs 67.7 (ScanNet). P2 ECE 8.7 vs 9.6 and 6.1 vs 6.2. u_ale AUSE 8.5 vs NormLift 1-C_inter 8.8 vs LEGaussians 19.7. The card carries the paper's note that the spread of mIoU between scenes exceeds most gaps.
- **Table III:** fused AP 75.0 (best). NormLift's drop beats our test alone (63.8 vs 53.4) and is best at the pixel level (65.8). Rec 56.8 at FA 6.2.
- **Table IV:** Kept 4.8 vs 9.5 without the depth check.

## 3 Static-scene segments (code paths read and cited)
**Path mode (S1a, S1b, S3b, S3c)**
- Load the scene and row with `S,row = RQ.load(sc, vocab, betas, args)` (render_qual.py:101-108). It runs ES.load_scene (eval_seg.py:256-309) with --downscale 2, so the class set and palette equal the paper's figure.
- Build the video frames from `make_dataset(name, data, stride)` (datasets.py:638; Replica :389, ScanNet :126).
  - viewmat = inv(c2w), as in eval_seg.py:209.
  - Labels go through ES.map_labels (eval_seg.py:120), then into the scene's query index with the `local` table (eval_seg.py:287-291). q is rebuilt from S["class_ids"].
  - Frames with non-finite poses are skipped (eval_seg.py:196-199).
- For each frame, `RQ.frame_maps(dict(S, frames=vframes), i, row, betas, args, with_rgb=True)` (render_qual.py:134-153). This calls evaluate_view (seg_eval.py:223-276), whose coverage, tie (:264-265) and P2 temperature rules (pixel_beta :185, adaptive_beta predictive.py:212) are the ones Table I uses.
- The M=30 field: `RQ.load(..., field=D/uags_results/qual/kfields/room_0_lseg_k30.pt)` (render_qual.py:105). Errors and e_red/e_irr come from RQ.error_image (:221-237).
- Renders are 640x480 for Replica and 648x484 for ScanNet. The video says on screen that Table I scores 320x240.
- Betas: Replica tab1/eval_val.json (P2 beta0 45.56, a 0.361, b 11.83, g 0.055); ScanNet scannet_tab1/eval_val.json. Text comes from a copy of the text caches.

**Progress mode (S3a)**
- `TF.load_field_file` (time_field.py:172), then `TF.context(ckpt, ns(frames=0, warmup=0, device, backend))` (:201).
- `TF.load_extractor` and `check_extractor` (:224, :237), then `TF.extract_views(clip, ds, idx, cfg["downscale"], TF.Clock(dev))` (:266). Then `del clip`.
- The loop is copied from TF.map_pass (:295-331) without the timing wrappers.
- After each `update_view` (vmf_field.py:344): `row_t = freeze(FieldRow("ours_P2", "feature", adaptive=True, field=field, mode="P0"))` (rows.py:100, :338), then `RQ.frame_maps(S_view, 0, row_t, ...)` at frame 290.
- What the viewer sees for unseen Gaussians:
  - mu* is 0 (vmf_field.py:719-724), so the class scores tie and the label is black.
  - u_epi is 1.
  - u_ale shows grey until 3 views.
- End check: `EC.replay_check(field, ckpt["field"])` must report ok (eval_change_nor.py:455-469). The saved room_0 field was mapped on an RTX 5070 Ti, so only float noise is allowed.
- Minimap: world up is +z (measured mean -y_cam = [0, 0, 1]). Draw the path so far, the current camera, and the held-out camera.

## 4 NOR update timeline (S4b, S4c)
**Layout**
- 002 is portrait: rotate 90 degrees clockwise for display; K stays in the stored frame. Five panels of 356x475 with 12 px gaps from x=46, curves strip below.
- 014 is landscape: a 3x2 grid of 520x390 panels, with the curves in slot 6.

**Panels.** For each displayed update frame f, the field state is the one after every update keyframe with frame number at most f, applied in Table III's order (EC.run_update:499).

| Panel | Content | Code |
|---|---|---|
| 1 Camera now | Dataset image f at 720x540. White contour = masks_real, the dataset's object mask (the pixel ground truth). | datasets.py:613 |
| 2 Map (before) | Frozen 3DGS colours at pose f | one render_row call (seg_eval.py:109) of a TensorRow (rows.py:67) |
| 3 Change score | c-bar/theta(h) of the last keyframe test, over tested Gaussians only. NaN elsewhere; undefined where under half the alpha (as EC.pixel_level :658-684). | see the loop below |
| 4 Averaging p(q) | softmax(beta_avg * F_avg . t). F_avg is the render of NormLift.feature() with the saved mapping sums plus the update views. | baselines.py:201-210, EC.make_normlift :349-358 |
| 5 Ours p(q) + flagged | softmax(beta_P0 * F . t), with F = mu* after resets. Hatched where the alpha-share of ours.changed is at least 0.5. | vmf_field.py:542, :578 |
| Curves | A: share flagged now, stale vs static, among Gaussians seen in the update. B: stale Gaussians whose unique top class is q, averaging vs ours. A cursor marks the current keyframe; final values are printed and labelled "measured on this scene". | stale = D2 truth |

**Loop.** This goes in a new script, because EC.code_state (eval_change_nor.py:241-248) hashes eval_change_nor.py into the Table III provenance.
```
ctx  = EC.load_context(ns(field, align="none", ply="", data="", update_stride=0, max_update_keyframes=0,
                          device, backend, channel_chunk=0))                      # :697-742, LSeg on all views
gt   = EC.ground_truth(ctx, ns(inflate=usplat.nor_align.INFLATE, near=EC.NEAR))   # :745-761 (D2)
ours = DSF.from_state(ctx["ckpt"]["field"], dev); ours.calibrate_change_threshold(0.999)  # vmf_field.py:811,:581 = EC:477-480
r0 = ours.n_resets.clone(); nl = EC.make_normlift(ctx["ckpt"], ours, saved=True)  # hooks only read ViewStats (vmf_field.py:390-392)
for v in ctx["upd_views"] (up to the display frame):
    ns_pre = ours.ns + ours.nf
    z, ras, gz, kw = EC.view_inputs(ctx, v); st = ours.update_view(ras, z, num_pixels=v["h"]*v["w"], **kw)
    cov = EC.covered(ours, st); observed |= cov                                    # :301-309 = EC:503-507
    thr = ours.thr_by_hist[torch.bucketize(ns_pre, ours._hist_edges)]              # vmf_field.py:497,:521,:526
    assert torch.equal(ours.last_fired, ours.last_tested & (ours.c_last > thr))    # :519,:527-528
    record: ratio=c_last/thr (tested), reset_ever=n_resets>r0, changed, robust_mask() (:765), top class of mu*@T and nl.feature()@T (bit-packed per keyframe)
```
- `ns_pre + nf_pre` is exact because fast_memory = 0 and slow_decay = 1 in the saved 002 and 014 configs (checked).
- One render per displayed frame: [colours 3 | f_avg 64 | mu* 64 | 1 | ratio + defined | changed | fired], about 137 channels at 720x540.
- The "before" state is rendered once at the first update pose.
- The 3D summary uses map frame 58 (002), two panels:
  - Averaging: stale Gaussians with top class q in vermillion.
  - Ours: the same, plus flagged stale Gaussians in blue.
  - Everything else greyscale.

**Query.**
- Vocabulary: the 19 ScanNet classes (configs/vocab_scannet.json) plus the noun if it is missing: 002 "chair" (present), 014 "bicycle" (added).
- Text: the LSeg 5-template ensemble through ES.text_embeddings (eval_seg.py:141), always on a **copy** of the text cache, then `ctx["pca"].project_text` (feature_pca.py:59).
- Temperatures from the ScanNet validation fit: ours_P0 39.21, averaging 38.66.

**Check against Table III.** The end values of Rec and FA (operating_point.ours), Kept (ours.kept) and changed_end must be within 0.005 of the per-scene records:
- nor/tab2_relmax/val/002.json: 0.536 / 0.140 / 0.090, changed_end 0.506 / 0.027
- test/014.json: 0.833 / 0.068 / 0.034, changed_end 0.824 / 0.023

**Caption rule (fixed now).** If averaging's end share of stale Gaussians with top class q is at least 2x ours, use: "Averaging keeps the removed chair's old evidence; ours resets it and keeps it out of the robust map." Otherwise use: "Both lose the 'chair' label; only ours flags these Gaussians and keeps them out of the robust map." The scene is never swapped.

002 has many false alarms: 14.0% of static Gaussians are reset at some point, but only 2.7% are still flagged at the end. The caption should say these are re-admitted after the depth check.

## 5 Code plan
| New file | Kind | CLI (env as nor_tab2.sh:41-45: UAGS_GPU_HEADROOM_MIB=1024, UAGS_GPU_USGLOC_EXTRA_MIB=99999, UAGS_GPU_POLL_S=5) |
|---|---|---|
| scripts/video_seg_dump.py | GPU; Replica/ScanNet path and progress dumps (npz with float16 and uint8 maps, plus JSON meta) | `scripts/gpu_gate.sh 5000 python3 scripts/video_seg_dump.py path --vocab configs/vocab_replica.json --extractor lseg --scenes configs/replica_test.json --scene room_0 --betas $D/uags_results/tab1/eval_val.json --text-cache $OUT/text_cache_lseg.copy.pt --kfield $D/uags_results/qual/kfields/room_0_lseg_k30.pt --frames 0:180,230:351 --kframes 230:351 --check-frame 290 --out $OUT/replica_room_0_path` ; `... 6000 ... progress --view-frame 290 --out $OUT/replica_room_0_progress` ; `... 5000 ... path --vocab configs/vocab_scannet.json --scenes configs/scannet_val.json --scene scene0000_00 --betas $D/uags_results/scannet_tab1/eval_val.json --text-cache <copy of uags_fields/scannet/text_cache_lseg.pt> --window-rule labels240 --stride 2 --out $OUT/scannet_scene0000_00` |
| scripts/video_nor_dump.py | GPU; replay of Table III's update with per-frame renders, timeline series and Table III check | `scripts/gpu_gate.sh 6500 python3 scripts/video_nor_dump.py --field $D/uags_fields/nor_002_lseg_relmax.pt --align none --query chair --vocab configs/vocab_scannet.json --betas $D/uags_results/scannet_tab1/eval_val.json --text-cache $OUT/text_cache_lseg.copy.pt --display-stride 1 --summary-frame rule --check $D/uags_results/nor/tab2_relmax/val/002.json --out $OUT/nor_002` (014: `--query bicycle --display-stride 2 --check .../test/014.json`) |
| scripts/video_compose.py | CPU; subcommands `figures`, `compose`, `encode` | `figures`: spotlight pages from a build-time copy of pipeline.tex, with the overlay wrapped in pgfinterruptboundingbox; pdflatex, then pdftoppm -scale-to-x 1920; assert equal page sizes. `compose --storyboard configs/video_storyboard.json --dumps $OUT --paper papers/main.tex --out $OUT/master.mp4`: one CRF-12 clip per segment, encoded in parallel, then joined with the ffmpeg concat demuxer (-c copy); table numbers regex-parsed from main.tex and asserted against their JSON; panels asserted inside the safe area. `encode --master $OUT/master.mp4 --max-mb 44 --out $OUT/submission/` |
| configs/video_storyboard.json | config | segments, durations, sources (with sha256 in the output JSON), rules, caption templates with `{tab:change/fused/G_AP}`-style placeholders |
| tests/test_video.py | CPU, mock backend | toy NOR scene of tests/test_eval_change.py::build (patch EC.load_extractor -> Fake; text via --text-file): the dump's final rec/fa/kept equal ec.main's record, ratio>1 iff fired, reset bits monotone, p in [0, 1]. Toy of tests/test_seg_eval.py (as test_render_qual.py): path maps equal RQ.frame_maps. Tiny compose and encode: 1920x1080, frame count, durations. |

**Encoder (CPU-tested in SCR/enc_out):**
- **Master:** `imageio_ffmpeg.write_frames(path, (1920, 1080), fps=30, codec="libx264", pix_fmt_out="yuv420p", quality=None, macro_block_size=8, output_params=["-preset", "slow", "-crf", "12", "-map_metadata", "-1"])`. The default macro_block_size=16 silently rescales 1080 to 1088 (measured), so 8 is required.
- **Delivery:** two-pass with the bundled ffmpeg 7.0.2 (`imageio_ffmpeg.get_ffmpeg_exe()`): `-c:v libx264 -preset slow -profile:v high -pix_fmt yuv420p -b:v {floor(44e6*8/dur/1000)}k -pass 1|2 -movflags +faststart -map_metadata -1 -fflags +bitexact -flags:v +bitexact -an`.
- **Not usable:** cv2 avc1/H264 has no encoder; mp4v works but is MPEG-4 Part 2.

| CPU test (6 s, 180 frames, dataset-photo panels) | Result |
|---|---|
| CRF 23 at 1088p (default block size) | 3.09 Mbit/s, about 58 MB per 150 s |
| CRF 26 at 1088p | 2.22 Mbit/s, about 42 MB per 150 s |
| CRF 23 at 720p | 1.74 Mbit/s |
| CRF 12 master at true 1920x1080 | 11.5 Mbit/s, 180/180 frames, 8.7 fps encode |
| Two-pass, 2200k target | 2191 kbps actual, 1920x1080, 180/180 frames, metadata stripped, text legible |
| PIL composition at 1080p | 154 ms per frame on one core |

## 6 GPU estimate (4070 Ti, gpu_gate.sh, waiting time excluded)
| Job | Basis | Estimate |
|---|---|---|
| Micro-benchmark first: 20 renders of 137 channels at 720x540 | render time alone was never measured | under 1 min |
| seg path room_0: 301 frames full field + 121 frames M=30 | loading the 1.3 GB and 0.57 GB fields; renders assumed 0.1-0.2 s per frame | about 3 min |
| seg progress room_0: 179 keyframes | LSeg 67-110 ms per keyframe and field 37-41 ms per keyframe (saved timing), plus 179 fixed-view renders | about 3 min |
| seg path ScanNet: 120 frames | one field of about 1.2 GB | about 2 min |
| nor 002: 48 keyframes, 157 renders | EC load 66 s and ground truth 15 s measured (val/002.json timing) | about 3 min |
| nor 014: 79 keyframes, 151 renders | load 31 s and ground truth 15 s measured | about 2.5 min |
| **Total** | | **about 15 GPU-min**. Optional O1 adds about 25 min. |

**Gates:** 5000-6500 MiB plus 1024 MiB headroom. EC peaked at 4.56 GB with two fields; the new dump holds one.

## 7 Verification
| ID | Check | Pass when |
|---|---|---|
| V1 | NOR dump vs the Table III per-scene JSON | Rec, FA, Kept and changed_end all within 0.005 |
| V2 | Progress replay vs the saved room_0 field | `EC.replay_check` ok |
| V3 | room_0 frame 290 at 320x240 vs qual_room_0_f29.json counts (labelled 76800, right 55748, wrong 21052, e_red 4170, e_irr 18007) | equal |
| V4 | Caption numbers | equal the main.tex tables and their JSON source |
| V5 | Layout | inside the safe area; font at least 24 px |
| V6 | Anonymity | contact sheet of every 30th video frame reviewed for names, logos and faces (including ScanNet mirror frames); mp4 tags show no title, artist or encoder; neutral file names; README/SUMMARY with no contact details |
| V7 | Delivery file | 1920x1080, 30 fps, H.264 High, frame count equal to the master, 60-180 s, at most 44 MB, zip at most 50 MB, decodes fully |
| V8 | tests/test_video.py | passes on the CPU |
| V9 | Repo changes | `git diff` shows only new files; eval_change_nor.py and render_qual.py unchanged |

## 8 Evidence from this phase (SCR)
- `enc_test.py`, `enc_test2.py`, `enc_out/enc_test.json`, `enc_out/enc_test2.json`, `enc_out/deliver_f90_crop.png`: encoder tests.
- `make_pipeline_steps.py`, `pipeline_steps/pipeline_steps.pdf`: 9 spotlight pages built with pdflatex. Page 2 grew by 0.3 pt, which is why the overlay needs pgfinterruptboundingbox.
- `mockup_layouts.py`, `mockups/*.png`: the first layouts overflowed by 2-78 px; the panel sizes above are fixed for that.
- `sheet.py`, `sheets/*.jpg`: contact sheets. No faces seen. 002 frames from 393 on show a finger. ScanNet 0062 has a mirror; 0000 has a bathroom mirror around frames 3800-4000.

RULES:['R0: No method change and no tuning. Every field, map and threshold is the saved one from Table I or III, and temperatures come from validation fits. No choice in the video depends on test results.', "R1: Replica scene room_0 (test) and Sequence_2 frame 290 are the scene and frame of the paper's qualitative figure. That frame is evaluation frame 29, picked by the rule 'most e_red pixels' (render_qual.py --list-frames, D/uags_results/qual/frames_room_0.json). No new choice.", 'R2: Replica paths. S1a uses Sequence_2 frames 0-179 at 30 fps (the first 6 s). S3b uses frames 230-350 (290 +- 60) at 15 fps, then holds frame 290 for 2 s.', 'R3: The S3a progress shows all 179 mapping keyframes of the saved room_0 field in mapping order (the TF.select_keyframes rule), 2 video frames each, from the fixed held-out view 290.', 'R4: The M=30 field is the existing D/uags_results/qual/kfields/room_0_lseg_k30.pt (the one the qualitative figure uses). No remapping.', 'R5: ScanNet uses validation scene scene0000_00, the validation scene with more classes (14 vs 6). The window is the 240 consecutive frames with finite poses whose label-filt maps (every 10th frame, 19 classes) hold the most distinct classes; ties go to the earliest. This label-only rule is computed before any render. Shown at stride 2, 15 fps.', 'R6: The main NOR timeline uses validation scene 002, the validation scene with the largest annotated box (517 L vs 54/30/367 L). The rule uses box sizes only.', 'R7: The held-out NOR clip uses test scene 014, the included test scene with the largest box (660 L), by box sizes only. It is disclosed as a second, qualitative look at a test scene: a replay of the Table III run on which no choice or number depends.', 'R8: NOR stills are the map frame and the update frame with the largest masks_real area (002: 58/335; 014: 274/600; selected_frames.json, rule fixed before looking). The 3D summary renders from that map frame (002: 58).', "R9: NOR display frames are every update frame in order (002 at stride 1: 156 frames; 014 at stride 2: 149 frames). None are dropped, including 002's finger-occluded tail from frame 393. Played at 15 fps.", "R10: The NOR query is the removed object's noun from the data inventory, written before rendering: 002 'chair', 014 'bicycle'. Vocabulary is the 19 ScanNet classes plus the noun if missing, with the frozen LSeg templates and the ScanNet validation temperatures (ours_P0 39.21, averaging 38.66).", "R11: Colour scales are fixed before rendering: u_epi [0, 0.81] and u_ale [0, 0.053] (the qualitative figure's), p(query) [0, 1], c-bar/theta(h) [0, 2] with 1 as the threshold.", "R12: The S4b caption is chosen by a rule fixed now. If averaging's end share of stale Gaussians whose top class is the query is at least 2x ours, use caption A ('averaging keeps the old evidence'); otherwise use caption B ('both lose the label; only ours flags and excludes them'). The scene is never swapped.", "R13: Captions quote only numbers parsed at build time from the main.tex tables (tab:replica, tab:change, tab:ablation). The build fails on a parse error or a mismatch with the JSON source. Per-scene values appear only as overlays labelled 'measured on this scene'.", 'R14 (optional O1): all 10 included test scenes in scene-id order, 1.5 s each, at the update frame with the largest masks_real, ours vs averaging p(noun). Nouns from the data inventory: book, box, jacket, backpack, laundry hamper, fan heater, backpack, paper bag, bicycle, bicycle.']
SCRIPTS:["scripts/video_seg_dump.py: GPU dumps for Replica and ScanNet. Mode 'path' renders RQ.frame_maps along dataset poses for the full field and the k-field: labels, u_epi, u_ale, coverage, correctness, RGB, plus an exact-count check at room_0 frame 290 at 320x240. Mode 'progress' replays the mapping pass keyframe by keyframe (TF.context, extract_views and the map_pass loop) and renders a fixed held-out view after each keyframe, then runs EC.replay_check. Run with: scripts/gpu_gate.sh 5000|6000 python3 scripts/video_seg_dump.py {path|progress} --vocab ... --scenes ... --scene room_0 --betas $D/uags_results/tab1/eval_val.json --text-cache <copy> [--kfield ...] --frames 0:180,230:351 --view-frame 290 --out $OUT/...", "scripts/video_nor_dump.py: GPU replay of the Table III update with EC.load_context, EC.ground_truth, a calibrated field, and NormLift (holding averaging's feature) attached as a hook. For each displayed update frame it renders the map colours, averaging and ours p(query), c-bar/theta(h) and the changed/fired shares. It saves bit-packed per-keyframe states and the timeline series, renders the 3D summary at the rule's map frame, and asserts that the final Rec/FA/Kept/changed_end are within 0.005 of the per-scene Table III JSON. Run with: scripts/gpu_gate.sh 6500 python3 scripts/video_nor_dump.py --field $D/uags_fields/nor_002_lseg_relmax.pt --align none --query chair --vocab configs/vocab_scannet.json --betas $D/uags_results/scannet_tab1/eval_val.json --text-cache <copy> --display-stride 1 --check $D/uags_results/nor/tab2_relmax/val/002.json --out $OUT/nor_002", "scripts/video_compose.py: CPU only, three subcommands. 'figures' builds pipeline spotlight pages from a build-time copy of papers/figs/pipeline.tex (pdflatex, pdftoppm) and rasterizes the plane and views PDFs. 'compose' turns the dumps, figures and main.tex table numbers (parsed and asserted) into 1920x1080 segments, writes one CRF-12 master clip per segment with macro_block_size=8, joins them with the concat demuxer, and writes storyboard_out.json with sources and sha256. 'encode' does the two-pass delivery at or under 44 MB, strips metadata, verifies the result, and writes README.txt, SUMMARY.txt and the zip of at most 50 MB.", 'configs/video_storyboard.json: segment list (id, start, duration, layout, sources, selection rule, caption templates with table placeholders) and the R12 caption rule.', "tests/test_video.py: CPU tests with the mock backend. The NOR dump on the toy scene of tests/test_eval_change.py must agree with eval_change_nor.py's record, and ratio > 1 must hold exactly when a Gaussian fired. The path dump on the toy of tests/test_seg_eval.py must equal RQ.frame_maps. Tiny compose and encode runs check 1920x1080, frame counts and durations."]
GPU:15
RISKS:["Averaging may lose the 'chair'/'bicycle' label about as fast as ours, so 'averaging keeps them' may not be visible. Handled by the caption rule fixed in advance (R12). The scene must not be swapped after seeing the result.", 'Validation scene 002 has many false alarms: 14.0% of static Gaussians are reset at some point, but only 2.7% are still flagged at the end (val/002.json). The timeline will show resets on static content. The caption must explain re-admission after the depth check, not hide it.', 'The GPU is not bit-exact, so the replayed update can differ slightly from the Table III run. The check tolerance is 0.005 on Rec, FA, Kept and changed_end; a larger gap stops the build for investigation (wrong field, keyframes or alignment).', 'Render-only GPU time was never measured, so the 15-minute estimate rests on loading and mapping timings. A first micro-benchmark of 20 renders (under 1 minute) replaces the estimate.', 'The GPU is shared. The gate waits until free memory is at least need + 1024 MiB (6500 -> about 7.5 GB), and it never runs beside USGLoc under the NOR environment, so jobs may queue.', "imageio-ffmpeg's default macro_block_size=16 silently rescales 1920x1080 to 1920x1088 (measured). Always pass macro_block_size=8 and check the output dimensions.", 'File size: CRF 23 measured about 3.1 Mbit/s on photo-like content (about 58 MB per 150 s), which is over budget. Use the two-pass delivery at about 2.5 Mbit/s, which hit its target within 0.5% in the test, and keep the zip at or under 50 MB.', 'A spotlight overlay can grow the pipeline PDF page (page 2 grew by 0.3 pt in the test). Wrap overlays in pgfinterruptboundingbox and assert that all pages have the same size.', 'Layout overflow: the first mock-ups overflowed by 2-78 px. Use the fixed panel sizes and an automated in-bounds assert with a 40 px margin and a 24 px minimum font.', 'Anonymity: possible faces in mirrors (ScanNet bathrooms), people, third-party logos and file metadata. Review contact sheets of the rendered video, blur any face, keep the bitexact flags, use neutral file names, and keep contact details out of README and SUMMARY. The RAS rule that videos carry title, authors and affiliations applies only to the camera-ready version, with names the user provides. Never credit Claude.', 'Dataset terms for showing Replica, ScanNet and NeRF-Object-Removal images in a publication video were not checked here (no licence files on disk).', "Text cache: ES.text_embeddings writes new prompts (for example 'bicycle') into the cache it is given, so always pass a copy.", 'Portrait scenes (002) are stored sideways and must be rotated 90 degrees clockwise for display only; K stays in the stored frame. This was checked on the contact sheet; the build should assert orientation per scene.', 'Renders are at full resolution (640x480 / 720x540) while evaluation used 320x240 / 360x270. State this on screen, and run the exact-count check V3 at 320x240.', 'The saved room_0 field was mapped on an RTX 5070 Ti. If EC.replay_check fails on the 4070 Ti, stop rather than show the progress segment.', "papers/main.pdf is stale: it still says 'Fig. 4 Qualitative results [TODO]', while main.tex now numbers qual=3 and views=4. Resolve figure numbers from the .aux at build time or name the figures. Re-run compose if table numbers change.", 'Length creep: if the video runs over 2:45, cut S3c and S2c first. Optional O1 adds about 25 GPU-minutes.', 'Extra looks at test data (014 and the Replica room_0 renders) are qualitative only. Disclose them in the report and README; no number from them enters the paper.']