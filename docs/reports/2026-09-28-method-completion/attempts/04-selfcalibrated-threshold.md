# Attempt 04: per-scene threshold from the mapping pass (no labels)

## Idea
The mapping sequence is static by assumption, so every Gaussian's max c during mapping is a
null sample. Set thr = q-quantile of those values for THIS scene; q is the only tuned number.
Implemented as `DirectionalSemanticField.calibrate_change_threshold(q)`.

## Result
Replay: first half of each point's sequence = mapping (null), second half = update
([../logs/g1_chess_selfcal.json](../logs/g1_chess_selfcal.json), [../logs/g1_office_selfcal.json](../logs/g1_office_selfcal.json)); centred c, beta = 0.9.
| q | chess thr | chess fired changed / static | office thr | office fired changed / static |
|---|---|---|---|---|
| 0.90 | 0.435 | 59.2% / 23.0% | 0.300 | 63.3% / 23.6% |
| 0.95 | 0.517 | 45.5% / 17.1% | 0.355 | 47.8% / 16.1% |
| 0.99 | 0.673 | 27.2% / 6.0% | 0.453 | 27.4% / 5.7% |
Unit test `test_calibrated_threshold`: thr equals the q99 of the mapping c_max; flags > 90% of
changed and < 10% of static Gaussians.

## Critic
Good for transfer: the thresholds differ by scene (0.67 vs 0.45) but the operating points agree
within 2.4 points. Two limits: (1) the static false-alarm rate is above nominal (6% at q = 0.99)
because the update window has more views than the null window; match the window lengths in the
real protocol; (2) detection power on this proxy is modest (27% at 6% FA) because swapped
partners are often similar content (median raw cosine 0.87-0.91). Real RIO10 changes are needed
to judge power (gate G3). One seed.

Re-checked 2026-09-29 after the code-review fixes: identical operating points at q = 0.99
(chess 27.2% / 6.0%, office 27.4% / 5.7%) ([../logs/g1_chess_postreview.json](../logs/g1_chess_postreview.json),
[../logs/g1_office_postreview.json](../logs/g1_office_postreview.json)). The pipeline now uses this flow by default
(change_thr = inf while mapping, `calibrate_change_threshold()`, then `--update-data`).


CORRECTION 2026-09-29: every real-CLIP number in this run (attempts 01-04, finding rows 4-7, the key result) was measured with a BROKEN extractor: usplat/clip_dense.py fed open_clip's batch-first transformer a sequence-first tensor, so no attention across tokens happened (zero-shot pixel accuracy ~1% on Replica). The numbers are kept as measured; they describe near-context-free patch features, not CLIP. Re-measured with the fixed extractor in [2026-09-29-data-and-baselines](../../2026-09-29-data-and-baselines/README.md).
