# Reports: uags
## Current state
Working draft (2026-11-02, [report](2026-11-02-merge-and-overview-figure/README.md)): papers/main.tex is ONE merged paper (24 pages) with the supplement items as marked SUPP-CANDIDATE blocks; the user decides which go to a supplement. The 8-page submission version is commit bb8dc0c. RA-L allows no supplementary text or figures (video zip only, 41.6 MB).
Claims: the two uncertainty channels separate reducible from irreducible error (Replica 6/6, ScanNet 8/8 test scenes); the reset lowers stale answers 34% -> 21% (N=10), 19.5% -> 4.8% on the 8 object-query test scenes (post hoc, disclosed; [report](2026-11-06-stale-split-and-cusum/README.md)).
Table I is a parity check: ours ties averaging the same features (mIoU Replica 28.0 vs 27.9, ScanNet 68.0 vs 67.7); 6 registered attempts to improve did not hold (paper Secs. IV-C, IV-D, Appendix D; latest: CUSUM change test, 2026-11-06).
Fused baselines rank stale Gaussians above ours + depth (Gaussian AP 81.1, 79.9 vs 75.0).
Data and results live inside the repo since 2026-10-31 ([report](2026-10-31-move-data-into-uags/README.md)): uags/data (datasets, weights), uags/results (records, fields, maps, video).
Figure 2 = papers/figs/pipeline_v6.tex (v3 + equations; v2 kept for the video, v3 kept).
Open: user picks the supplement items; the video's method segment still shows v2. Last updated 2026-11-06.
## Runs (newest first)
| Date | Report | Verdict | Main result |
|---|---|---|---|
| 2026-11-06 | [Attempt 07: whole-update change scores](2026-11-06-whole-update-score/README.md) | IN PROGRESS | registered; validation waits for the GPU |
| 2026-11-06 | [Stale answers by query type; attempt 06](2026-11-06-stale-split-and-cusum/README.md) | NEGATIVE | object-query test scenes: averaging 19.5% -> ours 4.8% (8/8, post hoc split); attempt 06 CUSUM: stale 25.4 vs 24.8% on validation, stopped before the test |
| 2026-11-02 | [Teaser and best-case comparison](2026-11-02-teaser-and-best-case/README.md) | DONE | Fig. 1 one row (2.91 -> 1.68 in), Figs. 1-2 on page 2; fig:qual-compare on our best scenes (disclosed); 17 unused figure files removed |
| 2026-11-02 | [Figure 2 v6 and cleanup](2026-11-02-figure2-v6-and-cleanup/README.md) | DONE | Fig. 2 = v3 + 4 equations (page 2); deleted 2.5 GB CLIP cache, merged supplement, old figure versions; third_party checked only |
| 2026-11-02 | [Figure 2 v4](2026-11-02-figure2-v4/README.md) | DONE | v2's blocks and key equations (8 numbered + threshold) in v3's clean style; replaces v3 in the paper |
| 2026-11-02 | [One merged paper and a lighter Figure 2](2026-11-02-merge-and-overview-figure/README.md) | DONE | supplement merged into main.tex (24 pages; 9 marked SUPP-CANDIDATE blocks, Appendices A-D); Figure 2 redrawn: 5 stages, one picture each |
| 2026-10-31 | [Move data and results inside uags](2026-10-31-move-data-into-uags/README.md) | DONE | gauss_ws/data (105 GB) -> uags/data + uags/results (git-ignored); old paths in records read via usplat/paths.py; CPU tests 423 passed / 1 pre-existing fail |
| 2026-10-27 | [RA-L rules: paper without supplement](2026-10-27-ral-self-contained/README.md) | DONE | RA-L allows no supplementary text/figures: 10 supplement references removed (still 8 pages); ablation rows explained; video zip 41.6 MB with ReadMe.txt/Summary.txt |
| 2026-10-27 | [Final paper: 8 pages and supplement](2026-10-27-final-8-pages/README.md) | DONE | 8 pages (3.8 MB) + 16-page supplement; abstract 245 words; neutral video caption (scene 002: both 44/15% and 33/20%); push left to user |
| 2026-10-20 | [Figures, video and stronger results](2026-10-20-figures-video/README.md) | PARTIAL | teaser, pipeline v2, 3 qualitative figs, calibration appendix, 2:24 video; fused baselines beat ours+depth (81.1/79.9 vs 75.0), claim removed; stale-answer 21% vs 34% |
| 2026-10-17 | [Complete the paper](2026-10-17-paper-completion/README.md) | DONE | all tables filled: NOR Table II (test N=10) ours+depth best G AP 75.0 vs 63.8; ScanNet test (N=8) in Table I, ours best or tied on every mean; Table I v2 negative (P2 kept); 8 pages |
| 2026-09-29 | [Replica Table I](2026-09-29-replica-tab1/README.md) | DONE | test (N=6): u_ale AUSE 8.50 ~ NormLift 8.76, < CSS 11.2, << LEGaussians 19.7; tuned CSS best calibrated; Table I in paper; views + Tab III running |
| 2026-09-29 | [Paper revision 1](2026-09-29-paper-revision-1/README.md) | DONE | text matches code, ieeeconf, anonymous; 29 review fixes; 8 pages |
| 2026-09-29 | [Bibliography moved to ref.bib](2026-09-29-bibliography/README.md) | DONE | 49 entries verified; compiles with BibTeX, 0 errors/undefined |
| 2026-09-29 | [Data, baselines and eval infrastructure](2026-09-29-data-and-baselines/README.md) | IN PROGRESS | Replica ready; CLIP bug fixed; LSeg chosen (val worst-scene mIoU 13.4 -> 23.2); change score = residual on proxy |
| 2026-09-28 | [Method completion: fixes F1-F9](2026-09-28-method-completion/README.md) | DONE | F1-F9 in code, 41 CPU + GPU tests pass, 39/40 review fixes; change AUROC 0.739 -> 0.824 INVALID (broken CLIP extractor, corrected 2026-09-29) |
| 2026-09-28 | [UAGS paper completion plan](2026-09-28-paper-completion-plan/README.md) | DONE | plan + 29-gap critique; pivot main claim to stale-semantics change (RIO10) |
## Older reports (never move them)
- none (no docs/ existed before 2026-09-28)
