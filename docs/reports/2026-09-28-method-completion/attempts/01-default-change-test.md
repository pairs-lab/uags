# Attempt 01: change test with the plan's defaults (centred channel, beta = 0.5, thr = 0.5)

## Idea
Plan F3/F4: centre CLIP features (remove the scene mean), split the centred evidence into
a recent (fast) and an older (slow) bucket, and flag a Gaussian when c = 1 - cos(slow, fast) > 0.5.

## Result
Real MaskCLIP features, 7-Scenes chess seq-01, simulated change (partner swap), CPU replay
`scripts/cpu_check_change.py --scene chess` -> [../logs/g1_chess_seed0.json](../logs/g1_chess_seed0.json).
| score | AUROC | strong subset |
|---|---|---|
| legacy 1 - r_bar (old code) | 0.704 | 0.746 |
| resid_2 (feature residual, 2 views) | 0.646 | 0.727 |
| ours c_max | 0.614 | 0.601 |
| ours u_ale (centred, Kish) | 0.481 | 0.439 |
At the default threshold 90% of STATIC points fire (99% of changed).

## Critic
Bad. Two causes, measured in attempt 02 ([../logs/g1_chess_variants.json](../logs/g1_chess_variants.json)):
(1) beta = 0.5 averages only ~2 views, too noisy on real features; (2) thr = 0.5 is far below
the static noise of centred features (static c_max median 0.68), so resets fire everywhere and
wreck all later statistics (u_ale becomes chance). Lesson: the unit tests (orthogonal content,
low noise) could not show this; a real-feature replay was needed before any dataset run.


CORRECTION 2026-09-29: every real-CLIP number in this run (attempts 01-04, finding rows 4-7, the key result) was measured with a BROKEN extractor: usplat/clip_dense.py fed open_clip's batch-first transformer a sequence-first tensor, so no attention across tokens happened (zero-shot pixel accuracy ~1% on Replica). The numbers are kept as measured; they describe near-context-free patch features, not CLIP. Re-measured with the fixed extractor in [2026-09-29-data-and-baselines](../../2026-09-29-data-and-baselines/README.md).
