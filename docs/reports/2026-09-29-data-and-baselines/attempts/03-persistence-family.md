# Attempt 03: a persistence filter and the (beta, k) family

## Idea
The residual baseline is the member beta = 0 (recent bucket = current view) with a "persist k
views" filter, which our test lacked. Add change_persist = k to the field; grid beta in {0, 0.5,
0.8} x k in {1, 2, 3} on chess; rule fixed before running: best mean c_max AUROC over change
fractions 0.5 and 0.2. Then office once, three fractions.

## Result
Chess grid ([../logs/g1_chess_grid_frac0.5.json](../logs/g1_chess_grid_frac0.5.json), [frac 0.2](../logs/g1_chess_grid_frac0.2.json)): best beta = 0, k = 2
(mean 0.805; residual baseline 0.806); every beta > 0 is worse (beta 0.8: 0.741-0.763).
Unit test: beta = 0, k = 2 equals the residual to 1e-5 (tests/test_change.py).
Office once ([../logs/](../logs/) g1_office_frozen_b0k2_frac*.json): ours = residual at every fraction
(0.866 / 0.867, 0.872 / 0.873, 0.861 / 0.861).

## Critic
Neutral-to-Bad for novelty: the best detector in our family IS the known feature residual with
persistence; the two-bucket memory adds nothing on this proxy. What remains ours is around the
detector: view-level evidence, calibration, reset and depth-gated re-admission. Defaults now
beta = 0, k = 2.
