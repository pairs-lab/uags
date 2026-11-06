# Attempt 07: whole-update change scores from our field

## Idea
Our test scores each view against the evidence before it (Table III G AP 53.4). NormLift's drop compares the field
before and after the whole update (63.8). Read our own field the same way, from a copy that is never reset (M =
after mapping, E = after the update):

| Variant | Score | Uses |
|---|---|---|
| dir | 1 - cos(R_M, R_E - R_M) | direction of the update's evidence |
| agr | ‖R_M‖/W1_M - ‖R_E‖/W1_E (drop of the mean resultant length r̄) | agreement |
| kish | ρ̂_M - ρ̂_E (drop of the Kish agreement) | agreement |

Fair baselines, registered with it: each per-view residual (NCC, feature, depth) averaged over the covered update
views (`ncc_mean`, `feat_mean`, `depth_mean`). Their purpose: a gain must not come only from the aggregation.

Rules ([prereg](../../../../configs/whole_update_prereg.json), committed in efb6b8d before any real-data run):
- Selection: on validation, the variant with the highest mean G AP; accepted if it is >= ours + 3.0 points.
- Adoption: on the test (run once), >= ours + 3.0 points and higher in >= 7 of 10 scenes.

## Result
| Split | ours | dir | agr | kish | NormLift | feat | feat_mean | agr + depth | feat_mean + depth |
|---|---|---|---|---|---|---|---|---|---|
| validation (001-004), G AP | 60.3 | 60.9 | **67.4** | 66.0 | 61.1 | 62.8 | 67.0 | 86.1 | 87.4 |
| test (10 scenes, once), G AP | 53.4 | - | **64.6** | - | 63.8 | 56.6 | 64.7 | 81.0 | 84.9 |

- Selection: agr chosen and accepted (+7.0 >= 3.0) ([selection](../logs/val_selection.txt)).
- Adoption: ADOPT. agr is 11.2 points above ours and higher in 10/10 scenes ([test table](../logs/test_table.md)).
- Sources: [val rows](../logs/val_rows.md), [test rows](../logs/test_rows.md).

## Critic
- Good against ours: G AP 53.4 -> 64.6 (10/10 scenes), G AUROC 87.2 -> 91.4 (10/10). The score has no history
  gate: it is NaN on 1.1% of the observed Gaussians, against 7.0% for ours.
- No win against the fair baseline: feat_mean ties it (64.7 vs 64.6; agr higher in 6/10). Fused with depth,
  feat_mean ranks above it (84.9 vs 81.0; agr higher in 2/10).
- The gain is the aggregation, not our field: averaging over the update raises the feature residual 56.6 -> 64.7
  (8/10 scenes). It does not help NCC (30.4 -> 27.2) or depth (57.4 -> 55.8).
- Pixel level is worse: agr P AUROC 71.7 vs ours 77.8 (lower in 8/10); P AP 59.8 vs NormLift 65.8.
- dir alone does not help (60.9 vs 60.3 on validation): the agreement terms carry the gain.
- Lesson: compare any "whole-update" score of our field with the same aggregation of the strongest residual. Our
  field ties it; it does not beat it.
