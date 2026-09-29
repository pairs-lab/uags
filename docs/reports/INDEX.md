# Reports: uags
## Current state
Method complete in code ([method run](2026-09-28-method-completion/README.md)): view-level evidence, Kish aleatoric on a
centred channel, two-bucket change test with a per-scene calibrated threshold, depth-gated re-admission, P0/P1/P2 predictives.
The 7-Scenes replay numbers of 2026-09-28 are INVALID (dense CLIP extractor bug: no token attention); being re-measured.
Open: RIO10 access form + decisions D1-D7 ([plan](2026-09-28-paper-completion-plan/1-plan.md) §8); NormLift overlap; paper text not yet updated.
Last updated 2026-09-29.
## Runs (newest first)
| Date | Report | Verdict | Main result |
|---|---|---|---|
| 2026-09-29 | [Paper revision 1](2026-09-29-paper-revision-1/README.md) | IN PROGRESS | text to implemented method, ieeeconf, anonymous |
| 2026-09-29 | [Bibliography moved to ref.bib](2026-09-29-bibliography/README.md) | DONE | 49 entries verified; compiles with BibTeX, 0 errors/undefined |
| 2026-09-29 | [Data, baselines and eval infrastructure](2026-09-29-data-and-baselines/README.md) | IN PROGRESS | Replica ready; CLIP bug fixed; LSeg chosen (val worst-scene mIoU 13.4 -> 23.2); change score = residual on proxy |
| 2026-09-28 | [Method completion: fixes F1-F9](2026-09-28-method-completion/README.md) | DONE | F1-F9 in code, 41 CPU + GPU tests pass, 39/40 review fixes; change AUROC 0.739 -> 0.824 INVALID (broken CLIP extractor, corrected 2026-09-29) |
| 2026-09-28 | [UAGS paper completion plan](2026-09-28-paper-completion-plan/README.md) | DONE | plan + 29-gap critique; pivot main claim to stale-semantics change (RIO10) |
## Older reports (never move them)
- none (no docs/ existed before 2026-09-28)
