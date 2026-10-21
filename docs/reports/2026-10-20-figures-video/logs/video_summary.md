# Supplementary video: summary (2026-10-21)

Verdict: DONE. The video runs 2:24 (144.2 s) and the delivery mp4 is 37.9 MB (zip also 37.9 MB). Every caption number is parsed from papers/main.tex, and the build fails if one is not. Both object-removal replays reproduce their scene's Table III record exactly (delta 0). The anonymity checks pass. GPU time was 4.6 min.

D = /home/thanhnc/gauss_ws/data, V = D/uags_results/video, R = /home/thanhnc/gauss_ws/src/uags. Environment as in the task prompt. Nothing was committed. `git status` shows only the 5 new files.

## 1 Deliverables

| What | Path | Size / sha256 (first 12) |
|---|---|---|
| Submission zip (mp4 + README.txt + SUMMARY.txt, ASCII, fixed zip dates) | V/supplementary_video.zip | 37.9 MB / 3a1069175a03 |
| Delivery mp4 (two-pass H.264 High, 2152 kbit/s target) | V/submission/supplementary_video.mp4 | 37.9 MB / fa9fd1b59286 |
| Master (CRF 12, per-segment clips joined with -c copy) | V/master.mp4, V/clips/S*.mp4 | 33.6 MB / 81791d794fbc |
| Storyboard as built (captions, sources, numbers, input sha256) | V/storyboard.json | - |
| Stills, one per segment | V/stills/S*.png (15) | - |
| Contact sheets, every 30th frame of the delivery file | V/contact/sheet_00..09.jpg | - |
| Checks | V/deliver_checks.json, V/nor/{002,014}/meta.json, V/static/room_0_path.json | - |
| Rules fixed before any video render | V/decisions.json | - |
| Logs | V/logs/ (GPU jobs, compose, deliver, contact, pytest) | - |

New code (untracked, not committed):

| File | Kind | Role |
|---|---|---|
| scripts/video_nor_timeline.py | GPU | Replays the Table III update through dump_nor_qual.py (imported). Its `capture_fields(per_view=...)` hook only reads the fields. After each update keyframe it renders p(query) for ours and averaging, top class = query, and the reset share, all at the scene's fixed rule camera. Then it checks against the record and the NOR dump (exit 2 on failure). |
| scripts/video_static_path.py | GPU | fig:qual panels (RQ.load / ES.load_frames / RQ.frame_maps) along room_0 Sequence_2 frames 230-350, for the full map and the M=30 field. Checks the counts at frame 290. |
| scripts/video_gpu.sh | driver | The 3 GPU jobs through gpu_gate.sh (6500 / 4500 MiB, headroom 1024, poll 5 s). |
| scripts/make_video.py | CPU | `numbers` (main.tex parser), `compose`, `deliver`, `contact` |
| tests/test_video.py | CPU tests | 6 tests (see section 5) |

## 2 Storyboard as built (V/storyboard.json)

| # | Start s | Dur s | Content | Source |
|---|---|---|---|---|
| S0 | 0.0 | 4.0 | Title, "Anonymous submission" | main.tex \title |
| S1 | 4.0 | 7.0 | Three questions: u_epi (M=30), u_ale (full map) at room_0 frame 290; 002 map vs photo after removal | static dump; NOR 002 dump |
| S2a | 11.0 | 22.0 | pipeline_v2.pdf whole (4 s), then 6 spotlights of 3 s: (1-2), (3-5), (6), (threshold+7), (8+robust set), (9) | fig:pipeline + caption, Sec. III |
| S2b | 33.0 | 6.0 | Uncertainty-plane figure | fig:plane caption |
| S3a | 39.0 | 11.1 | fig:qual animated: 121 frames at 15 fps, then a 3 s hold on frame 290. Rows: full map and M=30. Panels: render, GT, P2 labels, u_epi, u_ale, errors | V/static/room_0_path.npz |
| S3b | 50.1 | 7.0 | Views figure | fig:views, Sec. V-C |
| S3c | 57.1 | 8.0 | qual_compare in two halves (Replica, then ScanNet with the column titles repeated) | fig:qual-compare caption |
| S3d | 65.1 | 8.0 | Table I card: Averaging, F3DGS, LEGaussians, Ours (mIoU, ECE, NLL on both datasets) | tab:replica |
| S4a | 73.1 | 6.0 | 002 map render vs photo at frame 373 | NOR 002 dump |
| S4b | 79.1 | 18.1 | 002 timeline (validation, 'chair', camera = frame 373): 1.5 s after mapping, 48 keyframes x 6 video frames, 7 s end hold | V/nor/002 |
| S4c | 97.2 | 19.0 | 014 timeline (test, 'bike', camera = frame 748): 1.5 s, 79 keyframes x 4 video frames, 7 s hold | V/nor/014 |
| S4d | 116.2 | 8.0 | Bars: stale-answer rate 48/34/21; Kept 9.5 vs 4.8 | abstract, Sec. V-D, tab:ablation |
| S4e | 124.2 | 9.0 | Bars: G-AP of the 5 Table III rows, ours, and 4 fused rows (dispersion and its fused row left out) | tab:change, tab:fused |
| S5 | 133.2 | 8.0 | Limitations (6 bullets) | sec:limitations |
| S6 | 141.2 | 3.0 | "Thank you / Anonymous submission" | - |

Timeline panels: camera now (masks_real outline) | map render at the fixed camera | averaging p(q) (vermillion frame) | ours p(q) (blue frame) | reset by our test (orange #E69F00). p(q) is drawn in #CC79A7 over the grey map, as in the teaser. Curves are measured on the scene at Gaussian level:
- top class = query, for averaging (#D55E00) and ours (#0072B2), plus static top class changed (dashed);
- reset share, stale vs static, plus static still flagged.

002 is shown rotated (1 quarter turn, from the dump), 014 upright.

Deviations from the design (logs/design/video_design.md):

| Change | Reason |
|---|---|
| Dropped S3a (mapping-progress replay) and S3c (ScanNet clip) | They needed extra GPU and code; ScanNet is shown through the paper's comparison figure. Allowed by task item 4. |
| No "fused best" claim; captions follow the current paper | Task item 1 |
| pipeline_v2.pdf, 6 spotlights | Task item 2 |
| Vocabulary: 101 Replica classes with Replica validation temperatures (ours_P0 35.98, averaging 36.00), not 19 ScanNet classes | Same setup as the NOR dumps, the teaser and the paper's stale-answer rate. The queries are the dump's rule queries ('chair', 'bike'), not 'bicycle'. |
| NOR display: every update keyframe at one fixed rule camera, not every update frame at its own pose | Task item 3 (002 "at the rule frame 373"). The field changes only at keyframes. |
| No 3D summary render at map frame 58; no O1 grid | Time; not needed for the claims |

## 3 Captions and their source in main.tex

Every number below was parsed by `make_video.py numbers` from main.tex (sha256 5d5aa184cfc9). `check_caption_numbers` fails the build if any caption number is missing from main.tex. The only exceptions are declared data values: frame ids, and "this scene" values read from the Table III record or the replay's curves.json.

| Seg | Caption (abridged) | Source |
|---|---|---|
| S1 | A robot that reuses a semantic map must know how much evidence supports each part and must stop trusting content that has changed. | abstract, sentence 1 |
| S2a | (1-2)...(9): one plain sentence per step, with no numbers except "64 dimensions" | fig:pipeline caption, Sec. III-B (d=64), abstract |
| S2b | Little and conflicting evidence stay separate; a fired test resets the support to at most one view. | fig:plane caption |
| S3a | Frame 290: with 30 frames u_epi rises, also where the reducible errors are, while u_ale is still undefined on most pixels; with the full map, u_ale is highest at the rug. | fig:qual caption (frame id from qual_room_0_f29.json) |
| S3b | u_epi vs u_ale on reducible errors 0.34 vs 0.90 at M=10; u_ale vs u_epi on irreducible errors 0.48 vs 0.83 at M=90; N=6 | Sec. V-C text, fig:views caption |
| S3c | On scene0140_00 ours (68.5) is below F3DGS (69.6), 2D (69.3), averaging (69.1) | fig:qual-compare caption |
| S3d | Ours ties averaging (sign test, 14 scenes, no difference); lower NLL than F3DGS in 13 of 14; std of mIoU 2.3-4.9, 9.7-10.5; table cells | Sec. V-C, abstract, tab:replica (cells and caption) |
| S4b/c | "This scene": reset 53.6% / 14.0% (002), 83.3% / 6.8% (014); still flagged 2.7% / 2.3%; top class = query at the end: averaging 44% vs ours 15% (002), 20% vs 1% (014) | Table III per-scene records (val/002.json, test/014.json); curves.json of the replay. 53.6 and 14.0 also appear in the teaser caption. |
| S4d | Stale answers 34% (averaging) -> 21% (ours), after mapping 48%; Kept 4.8 vs 9.5 | abstract + Sec. V-D (cross-checked equal by the parser), tab:ablation |
| S4e | Ours alone 53.4 vs NormLift 63.8; fused: feature 81.1, NormLift 79.9, ours 75.0; Rec 56.8 at FA 6.2 (feature residual 75.6 at 13.7) | tab:change, tab:fused |
| S5 | Fixed geometry; LSeg confusions; measured rate 4.0x10^-3 vs nominal 10^-3; ScanNet P2 sign; 5 cm depth check; re-implementations | sec:limitations |

R12 caption rule (fixed in the design before rendering): if averaging's end share is at least 2x ours, use caption A. The shares are 0.443 vs 0.152 (002) and 0.196 vs 0.013 (014), so A applies in both scenes. The A wording was softened from the design: it now gives the two shares and no longer says "keeps it out of the robust map", because Kept is 9.0% in 002.

## 4 Commands (cwd R, env as in the prompt)

```bash
scripts/video_gpu.sh nor002 static nor014          # GPU, through gpu_gate.sh; logs V/logs/<job>.20261021-155451.log
python3 scripts/make_video.py numbers              # parsed main.tex numbers (also in storyboard.json)
python3 scripts/make_video.py compose --workers 15 # clips, master, stills, storyboard.json (651 s CPU wall)
python3 scripts/make_video.py deliver --max-mb 40  # two-pass mp4, README/SUMMARY, zip, checks (79 s)
python3 scripts/make_video.py contact --video V/submission/supplementary_video.mp4 --every 30 --sheet-dir V/contact
python3 -m pytest -q tests/test_video.py tests/test_dump_nor_qual.py tests/test_render_qual.py
```

## 5 Checks

| ID | Check | Result | Status |
|---|---|---|---|
| V1 | 002 replay vs Table III record (val/002.json): FA, Rec, Kept, changed_end FA/Rec | 0.1400 / 0.5364 / 0.0903 / 0.0266 / 0.5064, every delta 0.0; Gaussian AP/AUROC max delta 1.5e-8 | PASS |
| V1 | 014 replay vs test/014.json | 0.0681 / 0.8328 / 0.0339 / 0.0233 / 0.8236, every delta 0.0; max delta 2.0e-8 | PASS |
| V1b | Replay end state vs the NOR dump (gauss.pt): flags, changed_end, robust_end, observed, stale, top classes (before/after, ours/avg); last hook state = run_update's end state | all bit-equal (both scenes) | PASS |
| V1c | Renders vs the dump's frame npz (full_pq ours/avg after and before, flag share), tolerance 2e-3 | max 2.4e-4; NaN patterns equal | PASS |
| V1d | Per-keyframe curves vs the dump's curves.json (002) | equal | PASS |
| V1e | replay_check (mapping replay vs saved field); keyframe lists; rule frame re-derived (pick_frame) = 373 / 748 | ok | PASS |
| V3 | room_0 frame 290 at 320x240 vs fig:qual meta counts (labelled 76800, right 55748, e_red 4170, e_irr 18007, ale_defined 0.2362); the path-frame builder gives the same gt/pose/K as the eval frame | equal (full and M=30) | PASS |
| V4 | Caption numbers parsed from main.tex; abstract vs results stale rate cross-check; every caption number present in main.tex or declared | parser + check_caption_numbers in every compose | PASS |
| V5 | Layout: every panel and text inside the 40 px safe area; captions at most 3 lines and below all content; font at least 24 px | asserted per frame (paste/text/caption) | PASS |
| V6 | Anonymity: no "thanhnc", "/home/", "gauss_ws", "claude", "anthropic", user e-mail, lab name or "uags_results" in the mp4, master, zip, README, SUMMARY bytes. Metadata limited to handler_name VideoHandler, vendor_id 0, encoder "Lavc libx264" (no version, no names). README/SUMMARY are ASCII with no contact details. | 0 hits | PASS |
| V6b | Visual review of the 10 contact sheets and the 002 update photos at display size | no faces or people; small marks (monitor logo, mouse pad, a printed box, lettering on the bike frame) are not readable at display size, so nothing was blurred | PASS (visual) |
| V7 | Delivery: h264 High, 1920x1080, 30 fps, 4326 frames = master = storyboard, 144.2 s (60-180), no audio, 37.9 MB <= 40, zip 37.9 MB <= 50 | deliver_checks.json ok=true | PASS |
| V8 | `pytest tests/test_video.py tests/test_dump_nor_qual.py tests/test_render_qual.py` | 17 passed (V/logs/pytest_final.log) | PASS |
| V9 | Frozen files untouched: eval_change_nor.py sha 88bda977 (= record), dump_nor_qual.py 053d66ad (= test dumps), main.tex not edited; `git status` = 5 new untracked files | ok | PASS |
| - | Text-cache copies unchanged during the runs (sha checked in-script) | ok | PASS |

The test_video.py toy test runs the timeline script on the toy NOR scene, after its dump_nor_qual dump. It checks: delta 0 against the record, bit-equality with the dump, the curves equal to curves.json, monotone reset bits, the patches undone, and the cache not written. Other tests: the parser on the real main.tex, the caption-number check, the template refusing unknown keys, and a tiny 1080p encode (no 1088 padding).

## 6 Test looks

| Scene | Stage | What | When | Purpose |
|---|---|---|---|---|
| 014 (NOR) | test | one GPU replay (video_nor_timeline.py, sha 80bb8fe2) | 15:57-15:59 | qualitative timeline; reproduces the record |
| room_0 (Replica) | test | one GPU path render, frames 230-350 (video_static_path.py, sha 1bc23c83) | 15:56-15:57 | animated fig:qual |
| CPU compositions of the dumps above | - | layout changes after seeing the 014 render: grid sizes (a panel overlapped the caption), legend transparency, current-point markers, shorter durations | 16:00-17:05 | display only; no selection, threshold or number changed |

No number from these looks enters the paper. The scene, frame and query of each segment come from rules fixed before the renders (V/decisions.json, written 15:54, before the first GPU job at 15:54:51).

## 7 GPU time

| Job | Wall under the gate | Peak alloc |
|---|---|---|
| nor002 | 79 s | 4571 MiB |
| static room_0 | 70 s | 871 MiB |
| nor014 | 126 s | 4571 MiB |
| Total | 275 s (4.6 min); the gate ran every job without waiting | - |

CPU: each compose took about 9-11 min on 15 workers (4 builds in total while iterating). The final build is the one in V/.

## 8 Open issues

1. **Caption tone vs the teaser.** The teaser's pre-registered rule C1 (averaging's mean p(chair) >= 0.5) failed, so the paper's teaser caption is neutral. The video's R12 rule uses a different measure, the Gaussian top-class share (44% vs 15%). By that measure 002 gets "averaging keeps more of the old evidence". Both measures are shown honestly (the p maps show little pink for either method at the end). The lead may prefer the neutral caption B.
2. **False alarms are only partly in words.** In 002, ours changes the top class of more static Gaussians than averaging (dashed curves; teaser: 33% vs 20%). The video shows this as curves with the note "Dashed: static Gaussians whose top class changed", but gives no numbers.
3. **Typed setup constants** (N = 6/8/10, M = 10/30/90, 180 frames, 5 cm, 10^-3, 64 dimensions) are written in the templates. They are all present in main.tex, and the build checks for that.
4. **Dataset licences** for showing Replica, ScanNet and NeRF-Object-Removal images in a video were not checked (as in the design).
5. **Camera-ready version:** add title, authors and affiliations to S0/S6 (RAS rule). The master can be re-encoded with `deliver`.
6. **Encoder SEI string:** the delivery mp4 holds the standard x264 settings string in its bitstream. It is not identifying.
