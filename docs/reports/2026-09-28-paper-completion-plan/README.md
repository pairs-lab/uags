# UAGS paper completion plan
2026-09-28 | root: /home/thanhnc/gauss_ws/src/uags | git: master@ab28fb4 (clean) | [all reports](../INDEX.md)
Verdict: DONE -- plan written and critiqued; 7 decisions left for the user (D1-D7 in [1-plan.md](1-plan.md) §8)
Key result: n/a (planning run; no experiment)
Attempts: n/a | Time: 2026-09-28 22:15 -> 23:32 +07 (`date` stamps; workflow 71 min)

## 0 Context
- Ask: a plan to finish the RA-L paper [papers/main.tex](../../../papers/main.tex)
  ("Uncertainty-Aware Open-Vocabulary Semantic Gaussian Splatting for Robust Mapping in Dynamic Scenes").
- Success test: every placeholder maps to an experiment with data, baselines, splits, compute and dates.
- Scope: planning only. No project code or paper text changed in this run.
- Method: workflow wf_7bbcd724-e8d, 14 agents: 5 audits (code, math, related work, data, venue) ->
  3 plans (minimal / rigorous / reuse) -> 2 judges -> synthesis -> 2 critics (29 gaps) -> final plan.
  All agent outputs: [logs/workflow/](logs/workflow/). Their scripts and logs: [logs/wf/](logs/wf/).
- Machine: RTX 5070 Ti 16 GB shared with running USGLoc jobs; 48 GB disk free. [logs/machine_state.txt](logs/machine_state.txt)

## 1 Findings
| # | Finding | Reason | Evidence |
|---|---|---|---|
| 1 | Every experiment in the draft is a placeholder (3 tables, qual. figure, implementation, headline) | not run yet | [main.tex:389-505](../../../papers/main.tex#L389) |
| 2 | Small-sample shrinkage r̃ does nothing at r̄=1 | 1-r̃ = (1-r̄)(n+2)/(n+1) | [usplat/vmf_field.py:74-76](../../../usplat/vmf_field.py#L74); [logs/workflow/report_math.md](logs/workflow/report_math.md) E7 |
| 3 | n_eff counts pixels, ~50x over-count; every Gaussian "stable" after 1 view | correlated pixels of one view | report_math E8 (real MaskCLIP on 7-Scenes chess) |
| 4 | Raw r̄ barely separates change (AUROC 0.59); r_unstable=0.55 never fires | shared CLIP component, cos ~0.65 between random pixels | report_math E9, E12 (simulated change) |
| 5 | Decay leaves r̄ and μ* unchanged; forgotten Gaussians re-attract EFE | λ scales R and n alike | [usplat/consolidation.py:39-40](../../../usplat/consolidation.py#L39); report_math E12 |
| 6 | κ has no effect on any decision; μ* = weighted mean feature | κ0=0 | [logs/workflow/report_code.md](logs/workflow/report_code.md) mismatch 6 |
| 7 | PCA-64 keeps 12% of text energy; 13% of labels flip | uncentred image-only basis | report_math E15 |
| 8 | Gradient trick (C3) already published | Joseph et al. SIGGRAPH Asia 2025, arXiv 2411.15193 | [logs/arxiv_check.txt](logs/arxiv_check.txt); [logs/workflow/report_sota.md](logs/workflow/report_sota.md) |
| 9 | NormLift (arXiv 2609.18898, 2026-09-16) = view-level Kish + reliability ‖f‖; our 1-u_ale reduces to it on static scenes | concurrent work | [logs/arxiv_check.txt](logs/arxiv_check.txt); logs/wf/judge/estim_equiv.log |
| 10 | Open-vocab continuous BKI already exists (LatentBKI, RA-L 2025) on voxels | Gaussian/NIW model | report_sota §2 |
| 11 | No changing-scene data on disk; localization table overlaps the same group's USGLoc | scope | [logs/workflow/report_venue.md](logs/workflow/report_venue.md), [logs/workflow/report_data.md](logs/workflow/report_data.md) |

CORRECTION 2026-09-29: the audit's real-CLIP measurements (math audit C3/C3b: pixel correlation, r_bar of static points, simulated-change AUROC; the centring check) used the broken extractor found in [2026-09-29-data-and-baselines](../2026-09-29-data-and-baselines/README.md) (no token attention). Findings 3, 4 and 7 rest on those numbers and must be re-measured; the analytic findings (2, 5, 6, 8-11) do not depend on them.

## 3 Plan summary (full plan: [1-plan.md](1-plan.md))
- Main claim moves to C4: detect and forget stale open-vocabulary semantics after real rearrangement (RIO10),
  go/no-go gate G3 on RIO10 val. Replica becomes a parity + calibration block (gate G2).
- Cut: localization table, EFE section; gradient trick shrinks to one cited paragraph.
- Method fixes F1-F11 before any experiment (view-level evidence, Kish/joint-prior estimator, centred channel,
  two-timescale change test, re-admission check, posterior predictive, query fix, token cache, gsplat identity test, faithful NormLift baseline).
- Target: submit Fri 2026-11-27 (31.75 person-days in 41 workdays); IROS 2027 window needs submission by ~Nov 30.

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| CPU unit tests | `PYTHONPATH=. python3 tests/<t>.py` | 5/6 pass; test_gs_io needs `plyfile` | PASS (5/6), see [logs/cpu_tests.log](logs/cpu_tests.log) |
| Key competitor papers exist | `curl https://arxiv.org/abs/<id>` | NormLift 2026-09-16, GradBP 2024-11-19, LatentBKI found | PASS [logs/arxiv_check.txt](logs/arxiv_check.txt) |
| Plan attacked by independent critics | workflow critics | 29 gaps; each answered in 1-plan.md §10 | PASS |
| Numbers in the plan re-run by me | — | only the arXiv check and tests above | NOT RUN for the agents' other numbers (their commands are in logs/wf/) |

## 7 Open issues
- Decisions D1-D7 (date, dataset forms, GPU sharing, length, USGLoc order, supervisor load, extra forms) need the user.
- RIO10 needs an access form; instance-ID consistency between rescans is not checked.
- Many plan estimates are marked [A] (assumption) in 1-plan.md.
- The data report's RIO10 stats file (2.3 MB) is not copied: see logs/wf/SKIPPED.txt.

## 8 Files
- [1-plan.md](1-plan.md) final plan; [logs/workflow/](logs/workflow/) all agent reports, plans, judges, critiques
- [logs/wf/](logs/wf/) agents' scripts and logs; [logs/cpu_tests.log](logs/cpu_tests.log); [logs/machine_state.txt](logs/machine_state.txt); [logs/arxiv_check.txt](logs/arxiv_check.txt)
- No project file changed.
