# Attempt 04: per-view, history-conditioned calibration of the change threshold

## Idea
With beta = 0 the per-Gaussian MAX over the mapping pass is inflated by the first views (tiny
history), so the threshold is unreachable later (unit test: null median 1.13 vs change scores
0.98). Instead record the per-view scores of the mapping pass in histograms per history-size bucket
and test each Gaussian against the q-quantile of ITS history size (q = per-view false-alarm rate).

## Result
Unit tests (tests/test_change.py): pooled thr equals the pooled quantile; defaults + calibration
flag 100% of changed / 1% of static and re-admit 100% (q = 0.999). Replay with the field's own
calibration, null from half of the static points, false alarms on the other half
([../logs/](../logs/) g1_*_final_frac*.json):
| scene, change fraction | AUROC ours / residual / 1 - r_bar | flagged changed / held-out static, q = 0.999 |
|---|---|---|
| chess 0.5 | 0.809 / 0.810 / 0.822 | 41.7% / 3.2% |
| chess 0.2 | 0.802 / 0.803 / 0.742 | 43.5% / 3.2% |
| office 0.5 | 0.866 / 0.867 / 0.892 | 52.6% / 1.8% |
| office 0.2 | 0.861 / 0.861 / 0.833 | 48.5% / 1.8% |

## Critic
Good: a label-free, per-scene operating point that behaves the same on two scenes and at two
change fractions (q = 0.999: 42-53% detected at 1.8-3.2% false alarms). One seed, simulated change.
