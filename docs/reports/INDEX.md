# Reports: uags
## Current state
Method in code; paper text now matches it ([paper revision 1](2026-09-29-paper-revision-1/README.md), 8 pages, all numbers
placeholders). Extractor: LSeg (Replica val 2D worst-scene mIoU 23.2, [attempt 05](2026-09-29-data-and-baselines/attempts/05-extractor-study.md)).
Running: Replica Table I validation stage ([report](2026-09-29-replica-tab1/README.md)); Fig views and Tab III pipelines ready.
Open: RIO10 access form (user); 7-page target; support = summed view weight (Kish never binds on u_epi).
Last updated 2026-09-29.
## Runs (newest first)
| Date | Report | Verdict | Main result |
|---|---|---|---|
| 2026-09-29 | [Replica Table I](2026-09-29-replica-tab1/README.md) | IN PROGRESS | val (N=2): ours P2 ECE 4.89 vs CSS 4.85, best AURC 6.15; SOTA baselines added; test chained |
| 2026-09-29 | [Paper revision 1](2026-09-29-paper-revision-1/README.md) | DONE | text matches code, ieeeconf, anonymous; 29 review fixes; 8 pages |
| 2026-09-29 | [Bibliography moved to ref.bib](2026-09-29-bibliography/README.md) | DONE | 49 entries verified; compiles with BibTeX, 0 errors/undefined |
| 2026-09-29 | [Data, baselines and eval infrastructure](2026-09-29-data-and-baselines/README.md) | IN PROGRESS | Replica ready; CLIP bug fixed; LSeg chosen (val worst-scene mIoU 13.4 -> 23.2); change score = residual on proxy |
| 2026-09-28 | [Method completion: fixes F1-F9](2026-09-28-method-completion/README.md) | DONE | F1-F9 in code, 41 CPU + GPU tests pass, 39/40 review fixes; change AUROC 0.739 -> 0.824 INVALID (broken CLIP extractor, corrected 2026-09-29) |
| 2026-09-28 | [UAGS paper completion plan](2026-09-28-paper-completion-plan/README.md) | DONE | plan + 29-gap critique; pivot main claim to stale-semantics change (RIO10) |
## Older reports (never move them)
- none (no docs/ existed before 2026-09-28)
