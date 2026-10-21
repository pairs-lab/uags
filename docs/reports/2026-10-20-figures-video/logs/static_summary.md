# Static comparison figure (fig:qual-compare) and supp_compare_all

2026-10-21 08:56 to 15:05 JST | git af70b11 (new files untracked, not committed) | spec: [design/qual_review.md](design/qual_review.md) section 3 "S"
Verdict: DONE. Both figures are drawn from the frozen code. Every test render reproduces its published Table I numbers, and every check below passes.

## Outputs
| File | What |
|---|---|
| papers/figs/qual_compare.pdf / .png | Paper figure: room_1 f66 and scene0140_00 f57. 7.16 x 5.85 in, 0.88 MB |
| papers/figs/qual_compare.txt | Proposed caption, one line, generated from the published JSON |
| papers/figs/supp_compare_all.pdf / .txt | 14 test scenes, 2 per page, 7 pages, 5.5 MB |
| D/uags_results/qual/compare/test/*.pt, *.json | Saved blocks (panels and meta). D = /home/thanhnc/gauss_ws/data |
| D/uags_results/qual/compare/confident_errors.json | Fig. F static confident-error listing |
| D/uags_results/qual/select/static.json, static_val.json | Outputs of the selection rules |
| logs/static/ | Trimmed logs and metas. Paths are shortened to D/ and R/ |

## New files (not committed)
| File | Role |
|---|---|
| scripts/pick_frames.py | `static` subcommand: scene rule and frame rule. `--check` compares with the review and exits 2 on a mismatch |
| scripts/qual_compare.py | Stages: `tokens` (GPU, LSeg), `render` (GPU), `figure` (CPU), `confident` (CPU) |
| scripts/qual_compare.sh | Driver: select, val-replica, val-scannet, test-main, test-supp, figures. Writes one log per run with a header (time, git, script sha256, purpose) |
| tests/test_pick_frames.py, tests/test_qual_compare.py | CPU mock-backend tests |

Commands, run from R:
```
scripts/qual_compare.sh select
scripts/qual_compare.sh val-replica; scripts/qual_compare.sh val-scannet
scripts/qual_compare.sh test-main;   scripts/qual_compare.sh test-supp
scripts/qual_compare.sh figures
```

## Selection (rules from review section 2; CPU, labels and published JSON only)
| Rule | Outcome | Expected | Status |
|---|---|---|---|
| Replica scene: lower-median ours_P2 mIoU | room_1 (26.79; ranks 3rd of 6) | room_1 | PASS |
| ScanNet scene: same rule | scene0140_00 (68.52; ranks 4th of 8) | scene0140_00 | PASS |
| Frame: most GT classes with >= 0.5% of the labelled pixels each | room_1 -> 66 (13 classes), scene0140_00 -> 57 (7 classes), plus the 12 supplementary frames | review table | PASS, 16/16 ([log](static/pick_frames_static_test.log)) |
| Validation frames, same rule | office_0 29, room_2 14, scene0000_00 150, scene0062_00 16 | none | [log](static/pick_frames_static_val.log) |

## Checks
| Check | Command / source | Result | Status |
|---|---|---|---|
| Tests | `pytest tests/test_pick_frames.py tests/test_qual_compare.py tests/test_render_qual.py tests/test_seg_eval.py` | 41 passed ([log](static/pytest_final.log)) | PASS |
| Toy panels equal eval_seg | test_render_equals_eval_seg_and_ranks: labels, coverage and correctness of all 5 rows (2D row included) vs evaluate_view with the real labels | equal | PASS |
| Toy ranks are correct | Brute-force mid-rank over the pooled population | equal (atol 1e-6) | PASS |
| Repro on real data (V3) | Render pass re-scores per-scene mIoU and own-channel AUSE of the 4 shown 3D rows. Tolerance 0.002 (0.2 points), fixed before the first run | 18/18 runs (4 val, 14 test), max abs delta 2.2e-6 (test office_1) | PASS |
| Averaging = NormLift predictive | Table I: averaging mIoU == normlift mIoU | equal for every scene | PASS |
| Numbers printed in figures | `figure` stage reads them from the table JSON and checks its sha256 against the one recorded at render | never typed or recomputed | PASS |
| Text caches | Copies `qual/text_cache_lseg_{replica,scannet}.pt`; sha256 before = after (d583386e..., a9fb49d6...); no new prompts | unchanged | PASS |
| LSeg settings | No UAGS_LSEG_* env (as in Table I); the stage refuses to run if one is set | none set | PASS |
| Frozen files | `git diff --quiet HEAD` on the 16 files in the brief | unchanged | PASS |
| Fonts | `pdffonts` on both PDFs | STIX and NimbusRoman, CID TrueType, embedded; no Type 3 | PASS |
| PDF metadata | `pdfinfo` | qual_compare: Creator/Producer "Matplotlib", no author; supp: no Creator/Producer | PASS |
| Paths and names | `strings *.pdf \| grep -ciE "home\|thanhnc"` = 0; captions contain no paths | none | PASS |
| Photos | Visual check of every reference column (14 scenes) | No faces. Small product items: bottles in scene0590_00, a note on the fridge in scene0200_00, unreadable at print size | PASS (listed) |
| GPU rules | gpu_gate 4500 MiB, headroom 1024, poll 5 s; tokens and render are separate gated processes; gpu_reserve.apply first; no empty_cache; never killed anything | peak allocated 0.7-1.9 GiB; no out-of-memory error, so the gate was never raised to 7000 | PASS |

## Test looks (every test render; all use script sha256 a7ee664be4df, frozen 10:40, git af70b11)
| Time | Scenes | Purpose |
|---|---|---|
| 10:40-10:53 | tokens room_1 f66, scene0140_00 f57; renders of both | main figure |
| 14:40-14:59 | tokens and renders of the other 12 test scenes at the review's frames | supplementary grid. Its by-product is the Fig. F listing |

- Display changes after a test render: none.
- Layout changes made on validation only (dev1-dev5 in D/uags_results/qual/compare/val/fig):
  - legend columns 7 -> 8, error legend 2 columns
  - channel names added to the AUSE notes
  - caption compares values as printed
- The two Replica validation renders ran on pre-freeze script versions (869046b1, c4e03be2). Only the re-scoring code changed between them (full metrics -> mIoU and AUSE only), and toy test test_rescore_equals_summarize proves the two give identical values. The ScanNet validation renders used the frozen version.

## GPU time (wall time inside the gate)
| Job | Wall |
|---|---|
| Validation tokens (2 runs) | 2.9 + 3.4 min |
| Validation renders | office_0 11.7, room_2 13.5, scene0000_00 23.2, scene0062_00 3.7 min |
| Test tokens (4 runs) | about 0.6 min (main) + 7.2 min (supp, includes loading frames) |
| Test renders (14) | 13.5 min total; 19-155 s each |
| Total | about 1.3 h |

The validation renders were slow because the CPU load average was 38 on 24 threads, from other users' jobs. The test renders ran at a lower load. The gate waits, for the NOR dump of another agent, are not included above.

## Key numbers
| Figure | Source | Values |
|---|---|---|
| Per-scene mIoU and own-unc. AUSE, shown in the panels | tab1/eval_test.json, scannet_tab1/eval_test.json | room_1 mIoU: 2D 23.4, F3DGS 26.4, Avg 26.7, LEGS 21.0, ours 26.8. AUSE: 1-C_inter 11.0, u 21.8, u_ale 10.3. scene0140_00 mIoU: 69.3 / 69.6 / 69.1 / 67.3 / 68.5. AUSE: 3.5 / 8.7 / 3.6 |
| Fig. F confident errors | confident_errors.json | pick: scene0097_00 frame 15 (index 305), 31507 confident wrong px of 45680 covered wrong (72993 covered). Scene medians are over covered labelled pixels with a defined value |

## Decisions, deviations and open issues
1. **Rounding conflict, needs a decision.** The brief and the review say the 2D features score "69.4" on scene0140_00. The JSON holds 69.348, which prints as 69.3. The review rounded twice (69.35 -> 69.4). The generated caption prints 69.3, and the figure panel shows 69.3. The other disclosure values match the brief: ours 68.5, averaging 69.1, F3DGS 69.6, AUSE 3.5 vs 3.6.
2. **Layout deviates from "one RQ.compose call per block".** compose cannot stack two blocks, write per-panel numbers or fill a "no own uncertainty" slot. qual_compare.compose_blocks therefore copies its geometry and fonts and reuses RQ.RC, RQ._legend, RQ.label_image, RQ.u_image and RQ.save.
   - Slots: row 3, slot 0 holds a rank key. The 2D and F3DGS slots say "no own uncertainty". The class legend sits in a strip under each block (all classes with at least 0.2% of the pixels, summed over GT and predictions). Classes left out are recorded in the figure meta.
3. **Rank population.** The rank population is the covered, labelled pixels with a defined value. That is the finite part of Table I's AUSE population, which includes only labelled pixels. The spec says "covered pixels"; I chose the labelled set because it is exactly the AUSE population. The caption says "covered labeled pixels".
4. **Error colour.** Errors use Okabe-Ito vermillion #D55E00 for wrong (render_qual uses red). Class colours stay render_qual.scene_palette, as required. That palette is tab20-based and not colour-blind safe.
5. **Caption sentences.** They are generated, and they compare values as printed. "slightly" is written when the AUSE gap is below 0.5 points.
6. **Repro check.** Its numbers are a check only and are never printed. The 2D row has no per-scene repro check: that would need LSeg on every frame. On the shown frame, its path is tested against eval_seg on the toy.
7. **scripts/pick_frames.py has only `static`.** The NOR rules (`nor`) belong to the other agent. The output file is select/static.json, not selection.json, so the two agents do not collide.
8. **No CSS column in supp.** The review allowed one, but it was left out.
9. **Not done.** papers/figs/README.md is not updated (the rule is new files only). Proposed row: `qual_compare.pdf` and `supp_compare_all.pdf` come from `scripts/qual_compare.sh figures`.
10. **Figure size.** The paper figure is 5.85 in tall as a figure*, which is large. It could drop to one block if space is tight.
