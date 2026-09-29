# Attempt 03: normalise c by its expected value under "no change"

## Idea
A raw threshold on c = 1 - cos(slow, fast) did not transfer from chess to office (attempt 02).
Divide c by E0[1 - cos], computed from the slow bucket's unbiased agreement and the Kish sizes
of both buckets (exact recursions, `usplat/vmf_field.py` `_null_dissimilarity`), so one threshold
would mean "k times what noise gives" in every scene.

## Result
Synthetic i.i.d. noise (unit test `test_normalized_score_is_scale_free`): no-change mean of the
normalised score 0.56 / 0.52 at noise 0.3 / 0.9 while raw c is 0.047 / 0.257 -> scale-free there.
Real CLIP, chess, beta = 0.9, no resets ([../logs/g1_chess_norm.json](../logs/g1_chess_norm.json)):
| score | AUROC | strong subset | static median |
|---|---|---|---|
| raw c_max | 0.750 | 0.752 | 0.285 |
| normalised c_max | 0.603 | 0.641 | 1.94 |

## Critic
Bad. The null assumes independent views; consecutive real views are strongly correlated, so the
Kish bucket sizes overstate the evidence and the normalisation inflates well-agreeing points
(static median 1.94 instead of ~0.55). Same root cause as the pixel over-count (F1), one level up.
Kept in code as `change_score="normalized"` for the ablation table only; default stays "raw".

## Retry (same attempt, 2026-09-29): null with the shared-view term
The code review (logs/code_review.md #1) found the null E0 omitted the term (q - rho^2) sum a_i b_i
(each view is split between the buckets). Fixed; on i.i.d. synthetic data the no-change mean is now
in [0.85, 1.15] for beta 0.5/0.8/0.9 and 10/30 views (tests/test_change.py). Real CLIP, beta = 0.9,
no resets ([../logs/g1_chess_postreview.json](../logs/g1_chess_postreview.json), [../logs/g1_office_postreview.json](../logs/g1_office_postreview.json)):
| scene | raw c_max | normalised c_max | normalised static median |
|---|---|---|---|
| chess | 0.750 | 0.623 | 3.95 |
| office | 0.772 | 0.644 | 2.76 |
Critic: still Bad after the fix. The i.i.d. null is right for i.i.d. views, but real consecutive
views are correlated, so the null underestimates the no-change dissimilarity by ~3-4x. Verdict of
this attempt unchanged; the option stays ablation-only.


CORRECTION 2026-09-29: every real-CLIP number in this run (attempts 01-04, finding rows 4-7, the key result) was measured with a BROKEN extractor: usplat/clip_dense.py fed open_clip's batch-first transformer a sequence-first tensor, so no attention across tokens happened (zero-shot pixel accuracy ~1% on Replica). The numbers are kept as measured; they describe near-context-free patch features, not CLIP. Re-measured with the fixed extractor in [2026-09-29-data-and-baselines](../../2026-09-29-data-and-baselines/README.md).
