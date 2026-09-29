# Attempt 02: longer fast memory, threshold from the static noise (chess), frozen check on office

## Idea
Keep the centred two-bucket test, raise beta, and set the threshold at the 95th percentile
of c on STATIC points of a tuning scene (chess); then run an unseen scene (office) once.

## Result
Chess (tuning), no resets, c_max AUROC by beta: 0.7 -> 0.728, 0.8 -> 0.738, 0.9 -> 0.750
([../logs/g1_chess_beta_grid.json](../logs/g1_chess_beta_grid.json)); raw-channel c_max 0.648-0.658
([../logs/g1_chess_variants.json](../logs/g1_chess_variants.json)). Chosen: centred, beta = 0.9;
static q95 of c_max = 0.707 -> thr = 0.70 ([../logs/g1_chess_b09_thr.json](../logs/g1_chess_b09_thr.json)).
Office, run once with that frozen config ([../logs/g1_office_frozen.json](../logs/g1_office_frozen.json)):
| score | AUROC | strong subset |
|---|---|---|
| ours c_last | 0.820 | 0.846 |
| ours c_max | 0.772 | 0.794 |
| ours u_ale | 0.752 | 0.719 |
| legacy 1 - r_bar | 0.739 | 0.817 |
| resid_2 | 0.692 | 0.794 |
Fired at thr 0.70: changed 1.8%, static 0.2%. Office static c_max q95 = 0.466 (chess 0.707).

## Critic
Mixed: Good for ranking (c beats the old dispersion and the residual baselines on an unseen
scene; the old score still wins on the strong subset for c_max), Bad for the operating point:
an absolute threshold on c does not transfer between scenes because the static noise level
differs (0.71 vs 0.47). Next: normalise c by its expected value under "no change" (attempt 03).
Caveats: one seed, simulated change (partner swap), N = 1007 / 882 points; AUROC standard
error ~0.016, so differences below ~0.03 are not resolved.

Re-checked 2026-09-29 after the code-review fixes (metrics no longer drop NaN; faithful legacy preset):
no score was undefined on either scene, and the AUROCs above reproduce exactly (centred beta = 0.9:
chess c_max 0.750, office c_max 0.772 / c_last 0.824 without resets)
([../logs/g1_chess_postreview.json](../logs/g1_chess_postreview.json), [../logs/g1_office_postreview.json](../logs/g1_office_postreview.json)).
New variant there: the change test on the RAW channel ranks lower (chess 0.667, office 0.679), so
centring helps the change test.


CORRECTION 2026-09-29: every real-CLIP number in this run (attempts 01-04, finding rows 4-7, the key result) was measured with a BROKEN extractor: usplat/clip_dense.py fed open_clip's batch-first transformer a sequence-first tensor, so no attention across tokens happened (zero-shot pixel accuracy ~1% on Replica). The numbers are kept as measured; they describe near-context-free patch features, not CLIP. Re-measured with the fixed extractor in [2026-09-29-data-and-baselines](../../2026-09-29-data-and-baselines/README.md).
