# NOR figures and fused baselines: decisions fixed before any test dump

Written 2026-10-21 10:00, after the validation dump of 001 and before any test-scene dump.
Validation dumps 002-004 were running; no test scene had been touched by the new scripts.

| # | Decision | Where it is enforced |
|---|---|---|
| D1 | Tolerance = max(0.002, 3 x largest Table III cell difference of the 001 repeat) = 0.002 (largest difference 4.5e-8, pixel/dispersion/ap). Never changed after a test look. | `qual/nor/noise/tolerance.json`; `dump_nor_qual.py` exit 2 |
| D2 | Stop rule per dump: keyframe lists identical, per-frame n_covered/n_pos/n_mask identical, G/P AP and AUROC of all 7 Table III rows within 0.002, ours FA/Rec within 0.5 points. A failure stops the loop (`nor_qual.sh`). | `dump_nor_qual.py reproduction()` |
| D3 | F2 primary comparison: re-run values of all 5 fused rows from the same dump pass (paired, same noise). Secondary: the published Table III fused cells against the re-run new rows. Both reported. | `plot_nor_qual.py fused` |
| D4 | F2 = 8 exact two-sided sign tests (4 new fused rows x {Gaussian AP, pixel AP}), ties dropped, Holm over the 8; x/10 and median paired difference (ours+depth minus X). Claim rule exactly as `configs/nor_fused_prereg.json`. | `plot_nor_qual.py fused` |
| D5 | N1/N2 scenes from the published records only: highest, lower-median (5th of 10 ascending), lowest test pixel AP of ours -> 015, 013, 006. Frames by the masks_real rule (checked for all 14 scenes, `qual/nor/select.json`). | `plot_nor_qual.py selection_rule_n1`, `select` |
| D6 | N2 numbers (stale-answer rate, delta cos) over all update keyframes at 360x270, sets before / change test off / ours / averaging after; reported for all 10 test scenes as mean +- std and x/10, whatever they show. | `dump_nor_qual.py contrast_and_fa`, `plot_nor_qual.py numbers` |
| D7 | Display: full upright frames (no crop), clockwise quarter turns from the camera; N1 ranks against the pooled Table III pixel population; N2 one viridis scale per row (p5-p99 of panel (b)); teaser p(query) overlay in Okabe-Ito pink with strength 0.85 p. Frozen after the validation renders. | `plot_nor_qual.py` |
| D8 | Test dumps without `--curves` (no new per-view test quantity); `--save-features` only for 006, 013, 015 (and val 002). | `nor_qual.sh` |
| D9 | Printed numbers in figures/captions come only from the published records (Table III cells, FA/Rec, keyframe counts). Re-run values (teaser mean p, curves) only gate caption words (teaser rules C1-C3) and are never printed. | `plot_nor_qual.py` captions |
| D10 | 2026-10-21 14:40, after validation 001-004 and before any test dump: per the lead, flagged / reset Gaussians use orange #E69F00 (the "reset by test" colour of figs/pipeline_v2.pdf). Dump TINT: flagged+stale #E69F00, stale not flagged #56B4E9, flagged+static #D55E00, no update view #404040; teaser (i) overlay #E69F00. Display constant only (no number changes); the validation dumps keep the old tint (only their supplementary grid would show it). | `dump_nor_qual.py TINT`, `plot_nor_qual.py RESET` |
