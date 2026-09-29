# Paper revision 1: text matches the implemented method
2026-09-29 | root: /home/thanhnc/gauss_ws/src/uags | git: master@6cebc7d | [all reports](../INDEX.md)
Follows: [Bibliography moved to ref.bib](../2026-09-29-bibliography/README.md)
Verdict: DONE -- text rewritten to the code, ieeeconf, anonymous; 29 review findings applied; 8 pages (RA-L max), not the planned 7
Key result: n/a (text edit; no result number is in the paper yet, all are \ph placeholders)
Attempts: n/a | Time: workflow 3134 s of agent wall time (task notification); started 2026-09-29 ~10:18, done 14:05 +07

## 0 Context
- Ask (user, 2026-09-29): "continue to improve the paper", then "continue to complete our paper".
- Scope: rewrite papers/main.tex so every formula matches the code (usplat/), port to the RAS conference class
  (ieeeconf, required for the first RA-L submission), anonymise (double-anonymous review), apply the audits'
  honesty rules, remove the localization claims/table (overlap with the group's localization paper) and the EFE
  section (no experiment) into papers/parked_sections.tex, and restructure the experiments to the plan with
  placeholders. No result number is added (none exists yet).
- Input for every drafter: [logs/fact_sheet.md](logs/fact_sheet.md) (updated 13:52 with the LSeg choice before the
  fix step). Original draft: [logs/main_tex_before.tex](logs/main_tex_before.tex).
- Venue rules: [report_venue.md](../2026-09-28-paper-completion-plan/logs/workflow/report_venue.md) (6 pages + up to 2 paid;
  ieeeconf for the first submission; double-anonymous since 2025-01-01).
- Workflow: [logs/workflow_paper_revision.js](logs/workflow_paper_revision.js), run wf_17b72234-86b: 6 section
  drafters -> integrator -> 3 reviewers (math, claims, form) -> 1 fixer. 11 agents, 0 errors.

## 1 Findings
Review findings (29: 1 critical, 10 major, 18 minor). Full text: [math](logs/review_math.json),
[claims](logs/review_claims.json), [form](logs/review_form.json). What the fixer did: [logs/fix_report.md](logs/fix_report.md).

| Finding | Reason | Evidence | Action |
|---|---|---|---|
| 9 pages after integration (max 8) | 4 tables, repeated statements (3-6 times), intro repeated related work | [review_form.json](logs/review_form.json), [integrate_report.md](logs/integrate_report.md) | cut to 8 pages: intro paragraph, contribution 4, Sec. III-H, Table IV, P0/P1/P2 rows of Tab III, "et al." after 6 authors |
| Support s = min(n_K, W1) is always W1 | view weight <= 1 gives W2 <= W1, so n_K >= W1: Kish never binds on the epistemic channel | [review_math/check_support.log](logs/review_math/check_support.log) | paper: support = summed view weight; Kish kept only in rho_hat; code docstring noted ([vmf_field.py:18](../../../usplat/vmf_field.py#L18)) |
| "Pixels instead of views" row as described did not change u_epi | evidence="pixels" alone keeps a Kish view count | [review_math/check_ablation_switches.log](logs/review_math/check_ablation_switches.log) | row defined as evidence="pixels", epistemic="pixels" ([ablation_rows.py](../../../usplat/ablation_rows.py)) |
| "the bracket is unbiased" too strong | only the ratio inside the clamp is unbiased; clamp + sqrt leave ~+0.1 at rho = 0, n = 3 (r_bar: +0.58) | [review_math/mc_rho.log](logs/review_math/mc_rho.log) | wording fixed, Fig. 2 legend fixed |
| RIO10 claimed as done | no data yet (access form pending) | [review_claims.json](logs/review_claims.json) | visible \todo in abstract and intro |
| Threshold "sets" a false-alarm rate | mapping-pass scores are in-sample; only a nominal rate | review_claims.json | "targets a nominal rate; we measure the actual rate"; limitation added |
| One-pass trick listed as a contribution | prior work (joseph2025gradbp) | review_claims.json | now an implementation note that credits it |
| NormLift described unfairly | it already forms per-view means and factors C_intra x C_inter | review_claims.json | credited; new Table I row NormLift 1 - C_inter (code: [baselines.py](../../../usplat/baselines.py), commit 4829617) |
| Code name calib_mode="pooled" was the history-conditioned rule | naming | fix_report.md | code: "history" (default), "pooled" = one quantile, "max"; old states load as "history" (commit 4829617) |
| No "uncorrected r_bar" estimator | Tab III row had no code | fix_report.md | estimator="rbar" added (commit 4829617) |
| Yager rule uncited | -- | fix_report.md | yager1987dempster added, DOI checked (commit 6cebc7d) |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| paper compiles, page count | `cp -r papers/. build/ && tectonic --keep-logs main.tex` at 6cebc7d | rc 0, 8 pages letter, 0 undefined refs/cites, 3 underfull boxes ([logs/compile_6cebc7d.log](logs/compile_6cebc7d.log)) | PASS (8 pages = RA-L max; planned 7 not met) |
| math matches code | review:math agent (Monte Carlo + code checks) | 7 findings, all applied ([review_math.json](logs/review_math.json)) | PASS |
| claims, anonymity, scope | review:claims agent | 13 findings, all applied; PDF author "Anonymous Authors" ([integrate_report.md](logs/integrate_report.md)) | PASS |
| I3 new reference | Crossref API for 10.1016/0020-0255(87)90007-7 | Yager, Inf. Sci. 41(2):93-137, 1987: matches the bib entry | PASS |
| I3 other references | -- | checked in the [bibliography run](../2026-09-29-bibliography/README.md) | (copied from that report, not re-checked) |
| claims about other papers (CSS, LatentBKI, LEGaussians, LAMP, MIF) | -- | taken from the 2026-09-28 audit, PDFs not re-read | NOT RUN |
| I2 eval/tests not changed to pass | -- | no eval code changed by this run; tests changed only for the new options | PASS |

## 7 Open issues
- 8 pages, not 7: reaching 7 needs ~0.8 page more (remove Alg. 1, shrink Fig. 2, drop keywords or citations, or cut
  RIO10 if access fails). Decide when the numbers are in.
- 24 \todo markers remain (placeholders for results, RIO10, hardware, qualitative figure).
- The fixer closed some \todos without sign-off: tuning scene = 7-Scenes chess; hardware = RTX 5070 Ti + i5-14600KF
  (keep only if the timing runs on this machine); baselines get the same 1e-3 per-view false-alarm rate.
- RIO10 access forms still not sent by the user; the whole change-detection block depends on it.
- Unused figures vmf_sphere.pdf, gradient_trick.pdf still carry old notation.
- Symbol clashes left: c (class, change score), d (dimension, depth superscript), N (Gaussians, scenes).

## 8 Files
- Changed: papers/main.tex, papers/figs/pipeline.{tex,pdf}, papers/figs/uncertainty_plane.{tex,pdf}, new
  papers/parked_sections.tex (commit 514dd92); papers/ref.bib + citation (6cebc7d); comment block (3b84c1d).
- This folder: [logs/fact_sheet.md](logs/fact_sheet.md), [logs/main_tex_before.tex](logs/main_tex_before.tex),
  [logs/workflow_paper_revision.js](logs/workflow_paper_revision.js), [logs/integrate_report.md](logs/integrate_report.md),
  [logs/review_math.json](logs/review_math.json), [logs/review_claims.json](logs/review_claims.json),
  [logs/review_form.json](logs/review_form.json), [logs/fix_report.md](logs/fix_report.md),
  [logs/fix_main_tex.diff](logs/fix_main_tex.diff), [logs/fix_pipeline.diff](logs/fix_pipeline.diff),
  [logs/fix_plane.diff](logs/fix_plane.diff), [logs/review_math/](logs/review_math/) (check scripts + logs),
  [logs/compile_6cebc7d.log](logs/compile_6cebc7d.log).
