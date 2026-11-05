# Teaser layout B, best-case comparison figure, unused figure files removed
2026-11-02 | root: /home/thanhnc/gauss_ws/src/uags | git: master@242f03c | [all reports](../INDEX.md)
Follows: [Figure 2 v6 and cleanup](../2026-11-02-figure2-v6-and-cleanup/README.md)
Verdict: DONE -- new Fig. 1 (6 panels + curves, one row), fig:qual-compare on our best scenes (disclosed), 17 unused files removed
Key result: Fig. 1 2.91 -> 1.68 in tall, Figs. 1-2 share page 2; qual-compare: office_4 (+0.50 mIoU) and scene0070_00 (+2.70)
Attempts: n/a | Time: 17:50-18:05 JST (`date` stamps)

## 0 Context
- Asks (user, 2026-11-02): "check in figs, remove some un-use figure"; "we also need to improve teaser figure";
  "for qual_compare figure, please choose our best case to show".

## 1 Changes
| Item | Before | Now | Evidence |
|---|---|---|---|
| papers/figs | 17 files not used by main.tex or any script: gradient_trick, vmf_sphere, pipeline_v3, 5 PNG previews, 6 caption drafts | removed (git history keeps them); README lists each used PDF and its generator | `grep` over scripts/usplat/tests/main.tex |
| Fig. 1 (teaser) | layout A: 10 panels in 2 rows, p(query) overlays, fragmented dashed outlines, 4 colour bars, 2.91 in | layout B (`plot_nor_qual.py teaser --layout B`): after mapping (map with top class 'chair', $u^{\rm epi}$, $u^{\rm ale}$) and after removal (camera with one outline, averaging, ours with reset Gaussians in orange) + the same curves; 1.68 in | [before](figures/teaser_before.jpg), [after](figures/teaser_after.jpg) |
| Teaser panels show | probability p('chair') (faint) | top class = 'chair', the measure of the stale-answer rate; this view: 75.5% after mapping, averaging 34.2%, ours 5.1% of the removed chair's pixels (in the caption) | dump results/qual/nor/002 frame 373 |
| fig:qual-compare | lower-median scenes room_1, scene0140_00 (our worst ScanNet scene, -1.03) | our best scene per dataset by a stated rule: largest per-scene mIoU margin of ours P2 over the best other row (office_4 +0.50, scene0070_00 +2.70); caption and Sec. IV-C say it is a favorable case | [margins](logs/best_case_margins.txt), [before](figures/qual_compare_before.jpg), [after](figures/qual_compare_after.jpg) |
| scripts/qual_compare.py | opened the Table I record by its pre-move path (crash) | path read through `usplat.paths.moved` (sha256 check unchanged) | missed in the 2026-10-31 move |
| scripts/make_video.py (compare segment) | scene0140_00, "ours is below ..." | scene0070_00, "ours is above averaging, Feature 3DGS and the 2D features" (per-scene values) | `compose --dry` ok |

Layout A of the teaser stays available (`--layout A`, default) for reproducibility.

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Paper build | pdflatex + bibtex + 3x pdflatex | 24 pages, 17.6 MB; 0 warnings, 0 undefined, 0 "??", no Type 3 | PASS |
| Anonymity | pdfinfo; strings + decoded streams | 0 identity hits | PASS |
| Video parsing | `make_video.py numbers` (teaser.rec/fa = 53.6/14.0), `compose --dry` | ok | PASS |
| Tests | pytest test_plot_nor_qual, test_qual_compare, test_video, tab2, tab3 | 29 + 19 passed ([tests](logs/tests.txt)) | PASS |

## 7 Open issues
- The delivered video zip still shows the old comparison scenes and the v2 method figure; rebuild it (CPU) to match.
- Other figure scripts may also open paths stored before the move; qual_compare was found only by running it.
- The Replica best case wins by 0.5 mIoU only; the per-scene numbers in the figure show this.

## 8 Files
- Commit 242f03c. Changed: scripts/plot_nor_qual.py (teaser layout B), scripts/qual_compare.py, scripts/qual_compare.sh,
  scripts/make_video.py, papers/main.tex, papers/main.pdf, papers/figs/{teaser,qual_compare}.pdf, papers/figs/README.md.
- This folder: [figures/](figures/), [logs/](logs/).
