# Reports: uags
## Current state
Every table and figure of the paper is measured and filled ([report](2026-10-17-paper-completion/README.md)); 8 pages.
Table I: ours best in 6/7 open-vocabulary columns (P2; the pre-registered v2 predictives did not beat it on validation).
Table II (object removal, N=10): ours + depth best at the Gaussian level (AP 75.0); NormLift drop best at the pixel level.
ScanNet held-out test (N=8, run once 2026-10-20): ours best or tied on every Table I mean (mIoU 68.0 vs averaging 67.7), small margins (3-5 of 8 scenes); in the paper, 8 pages.
Running: figures, video and stronger results (no page limit in this step). Open: push from this machine. Last updated 2026-10-20.
## Runs (newest first)
| Date | Report | Verdict | Main result |
|---|---|---|---|
| 2026-10-20 | [Figures, video and stronger results](2026-10-20-figures-video/README.md) | IN PROGRESS | design workflow running (teaser, pipeline, qualitative, video, quantitative) |
| 2026-10-17 | [Complete the paper](2026-10-17-paper-completion/README.md) | DONE | all tables filled: NOR Table II (test N=10) ours+depth best G AP 75.0 vs 63.8; ScanNet test (N=8) in Table I, ours best or tied on every mean; Table I v2 negative (P2 kept); 8 pages |
| 2026-09-29 | [Replica Table I](2026-09-29-replica-tab1/README.md) | DONE | test (N=6): u_ale AUSE 8.50 ~ NormLift 8.76, < CSS 11.2, << LEGaussians 19.7; tuned CSS best calibrated; Table I in paper; views + Tab III running |
| 2026-09-29 | [Paper revision 1](2026-09-29-paper-revision-1/README.md) | DONE | text matches code, ieeeconf, anonymous; 29 review fixes; 8 pages |
| 2026-09-29 | [Bibliography moved to ref.bib](2026-09-29-bibliography/README.md) | DONE | 49 entries verified; compiles with BibTeX, 0 errors/undefined |
| 2026-09-29 | [Data, baselines and eval infrastructure](2026-09-29-data-and-baselines/README.md) | IN PROGRESS | Replica ready; CLIP bug fixed; LSeg chosen (val worst-scene mIoU 13.4 -> 23.2); change score = residual on proxy |
| 2026-09-28 | [Method completion: fixes F1-F9](2026-09-28-method-completion/README.md) | DONE | F1-F9 in code, 41 CPU + GPU tests pass, 39/40 review fixes; change AUROC 0.739 -> 0.824 INVALID (broken CLIP extractor, corrected 2026-09-29) |
| 2026-09-28 | [UAGS paper completion plan](2026-09-28-paper-completion-plan/README.md) | DONE | plan + 29-gap critique; pivot main claim to stale-semantics change (RIO10) |
## Older reports (never move them)
- none (no docs/ existed before 2026-09-28)
