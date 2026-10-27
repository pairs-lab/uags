# Reports: uags
## Current state
Submission set (2026-10-27, [report](2026-10-27-ral-self-contained/README.md)): papers/main.pdf 8 pages, 3.8 MB, self-contained; video zip 41.6 MB (RA-L max 50 MB). RA-L allows no supplementary text or figures: do NOT upload papers/supplementary.pdf.
Claims: the two uncertainty channels separate reducible from irreducible error (Replica 6/6, ScanNet 8/8 test scenes); the reset lowers stale answers 34% -> 21% (N=10).
Table I is a parity check: ours ties averaging the same features (mIoU Replica 28.0 vs 27.9, ScanNet 68.0 vs 67.7); 5 registered attempts to improve did not hold (paper Secs. IV-C, IV-D).
Fused baselines rank stale Gaussians above ours + depth (Gaussian AP 81.1, 79.9 vs 75.0).
Open: push (user; master 42 commits ahead); video shows 2 figures not in the paper (S2b, S3c): keep or cut. Last updated 2026-10-27.
## Runs (newest first)
| Date | Report | Verdict | Main result |
|---|---|---|---|
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
