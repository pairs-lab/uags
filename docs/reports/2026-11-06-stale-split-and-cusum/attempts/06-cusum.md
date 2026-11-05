# Attempt 06: a CUSUM change test fires our reset
Registration: [configs/cusum_reset_prereg.json](../../../../configs/cusum_reset_prereg.json), committed in 5f87c81
before any real-data run. Code: [usplat/cusum_reset.py](../../../../usplat/cusum_reset.py),
[scripts/cusum_reset_nor.py](../../../../scripts/cusum_reset_nor.py),
[scripts/cusum_reset_select.py](../../../../scripts/cusum_reset_select.py),
[scripts/cusum_reset.sh](../../../../scripts/cusum_reset.sh), tests [tests/test_cusum_reset.py](../../../../tests/test_cusum_reset.py).

## Idea
Our test fires when the 2-view persistent score c_p passes a per-view threshold. A moderate change seen over many
views never passes it. Page's CUSUM sums the excess of c_p over a reference level and fires when the sum is large:
`S <- max(0, S + c_p - delta)` where c_p is defined, `S <- 0` after a firing. Everything else stays ours (reset,
depth re-admission, robust set).

| Setting | Value |
|---|---|
| delta (variants) | 0.90 / 0.97 / 0.99 quantile of the mapping c_p (C90 / C97 / C99) |
| threshold theta_S | 0.999 quantile of S over the static mapping replay, per scene (Table III's rule) |
| selection (validation 001-004) | eligible: mean FA <= 2 x ours'; lowest mean stale-answer rate (tie 0.1 point -> higher G AP) |
| accept for the test | >= 2.0 points below ours' stale-answer rate on validation, else stop (no test run) |
| adopt (test, once) | >= 2.0 points below ours, worse in <= 2 of 10 scenes, FA <= 2 x ours' |

## Result
Validation, 4 scenes, ours replayed in the same runs ([val_rows](../logs/val_rows.md), [selection](../logs/val_selection.txt)).
Stale answers: lower is better. G AP: Gaussian-level AP of the firing score, higher is better.

| Row | stale-answer % | FA % | Rec % | G AP | G AP +depth | eligible (FA <= 11.6) |
|---|---|---|---|---|---|---|
| ours | 24.8 | 5.8 | 53.2 | 60.3 | 84.0 | - |
| C90 | 25.4 | 2.8 | 54.7 | 65.8 | 83.7 | yes, chosen (tied with C97, higher G AP) |
| C97 | 25.4 | 2.4 | 50.5 | 58.2 | 79.9 | yes |
| C99 | 25.8 | 2.3 | 47.2 | 51.3 | 75.2 | yes |

Accept needs <= 24.8 - 2.0 = 22.8: no variant -> the attempt stops, no test run (check-test exit 1).

| Scene (query) | ours stale % | C90 stale % | ours Rec % | C90 Rec % | ours G AP | C90 G AP |
|---|---|---|---|---|---|---|
| 001 (wall) | 96.3 | 96.3 | 20.3 | 32.4 | 22.7 | 62.2 |
| 002 (chair) | 2.0 | 3.6 | 53.6 | 49.7 | 64.0 | 56.7 |
| 003 (book) | 0.0 | 0.0 | 79.7 | 84.9 | 86.8 | 76.5 |
| 004 (chair) | 0.8 | 1.8 | 59.1 | 51.8 | 67.9 | 67.8 |

## Critic
Bad (registered rule: not 2.0 points below ours).
- CUSUM halves the false alarms (2.8 vs 5.8%) and ranks the stale Gaussians better on the hardest scene (001: G AP
  62.2 vs 22.7, Rec 32.4 vs 20.3). There the stale-answer measure is saturated (wall query, 96.3% for every row).
- Where the measure can move (002, 004), C90 resets fewer stale Gaussians (Rec 49.7 vs 53.6, 51.8 vs 59.1) and
  leaves more stale answers (3.6 vs 2.0, 1.8 vs 0.8). On 003 every row is at 0.0%.
- Lesson: summing evidence helps ranking and false alarms, not the outcome. The outcome needs the sharply changed
  Gaussians reset at once; theta_S sits in the long tail of the static S (S > 0 in 14-19% of mapping
  Gaussian-views for C90), so some of them never pass it.
- The decision rests on 2 of 4 validation scenes (001 saturated, 003 at 0). Registered as is; not changed after.
- Untested next idea: fire when either test fires (ours OR CUSUM). Needs a new registration; not run.
