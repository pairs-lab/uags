# One merged paper and a lighter Figure 2
2026-11-02 | root: /home/thanhnc/gauss_ws/src/uags | git: master@e085873 | [all reports](../INDEX.md)
Follows: [Move data and results inside uags](../2026-10-31-move-data-into-uags/README.md)
Verdict: DONE -- supplement merged into papers/main.tex (24 pages, marked blocks); Figure 2 redrawn with 5 stages and little text
Key result: Figure 2 from 9 formula boxes (2.93 in tall) to 5 picture stages (2.17 in) ([before](figures/fig2_before.jpg), [after](figures/fig2_after.jpg))
Attempts: n/a | Time: 16:25-16:45 JST (`date` stamps)

## 0 Context
- Ask (user, 2026-11-02): "merge the main and supplement to one; then I will decide which things will go to the
  supplement later" and "improve the overview figure (Figure 2), now it has too much text. You can check RAL- or
  TRO-related papers".
- Note: RA-L allows no supplementary text or figures (report 2026-10-27); a later supplement can only be for arXiv
  or another venue.

## 1 Merge
Every supplement block is in papers/main.tex between `%% [SUPP-CANDIDATE Sx]` and `%% [/SUPP-CANDIDATE Sx]`,
so moving one out again is one cut. Placement follows the pre-split paper (eeb65ea). No number or sentence with a
number was reworded; references M-x became x, and the 10 pointers removed on 2026-10-27 point to the merged items.

| Block | Content | Now in |
|---|---|---|
| S4 | agreement estimator, fig:plane | Sec. III-E |
| S3 | per class and per method: fig:qual, tab:perclass, fig:qual-compare | Sec. IV-C |
| S2a, S2c | fused baselines tab:fused; qual_change, qual_query and the stale-answer definition | Sec. IV-D |
| S1 | paired tests and calibration: tab:paired, tab:calib, fig:calibration | Appendix A |
| S2b | measured false-alarm rate, tab:rstatic | Appendix B |
| S5 | P2 coefficients, build cost, run time per keyframe | Appendix C |
| S6 | registered later attempts, tab:attempts | Appendix D |
| S3g | all-scene grid (7 float pages) | end, after the appendix floats (else it holds back later figures) |

Layout: text and main floats on pages 1-13, appendix tables 14-16, grid 17-23, references 24
([pages](figures/merged_pages.jpg), [float pages](logs/float_pages.txt)). tab:fused repeats the fused rows of Table III
(as in the supplement); drop one copy when deciding.

## 2 Figure 2
| Reference (venue checked on arXiv) | What it does |
|---|---|
| E2-BKI, arXiv 2509.11964, "Accepted to IEEE RA-L" | 4 stage boxes, one picture each, one formula, a dashed recursive-update loop |
| ConvBKI, arXiv 2310.16020, T-RO (DOI 10.1109/TRO.2024.3453771) | 4 icons with 2-word labels and one loop |
| GSFusion (2408.12677), Clio (2404.13696) | images with short arrow labels; venue not in their arXiv records |

New figure papers/figs/pipeline_v3.tex (v2 kept): Keyframe -> Lifting -> Change test -> Evidence and uncertainty -> Query,
the order the code runs (the test reads the earlier evidence before the view is added). Each stage: title, one
picture, at most one formula; two dashed arrows (earlier evidence, next keyframe); colours as the teaser. The details
moved to the caption (6 lines). "calibrated P(y=c)" was removed from a label: P2 is not better calibrated everywhere.

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Build | pdflatex + bibtex + 2x pdflatex (scratch copy) | 24 pages; 0 warnings, 0 undefined, 0 "??", 0 overfull | PASS ([build_checks](logs/build_checks.txt)) |
| Fonts | pdffonts main.pdf, pipeline_v3.pdf | all Type 1 embedded, no Type 3 | PASS |
| Anonymity | pdfinfo; strings + decoded streams | "Anonymous Authors"; 0 identity strings | PASS |
| Numbers | blocks copied verbatim; only refs and 4 bold paragraph leads added | no number changed | PASS |
| Paper parsers | after the merge: `pytest` on the 4 test files that read papers/ (37 passed); after Figure 2: test_video, test_make_tables_tab2/tab3 (24 passed); `make_video.py numbers` (84), `compose --dry` (ok) | all pass | PASS |

## 7 Open issues
- The video's method segment (S2a) still shows the old Figure 2 (pipeline_v2.pdf); updating it is a CPU rebuild.
- papers/supplementary.tex and supplementary.pdf are now duplicates of the merged blocks (kept, not deleted).
- pipeline_v3.pdf is 0.7 MB (PNG thumbnails); scripts/shrink_figs.py can JPEG them later.
- The merged paper is 24 pages, 18.6 MB: a working draft until you pick the supplement items.

## 8 Files
- Commits: 505e462 (merge), e085873 (Figure 2 + caption + main.pdf).
- Changed: [papers/main.tex](../../../papers/main.tex), papers/main.pdf; new [papers/figs/pipeline_v3.tex](../../../papers/figs/pipeline_v3.tex), papers/figs/pipeline_v3.pdf.
- This folder: [figures/](figures/) (fig2_before, fig2_after, merged_pages), [logs/](logs/) (build_checks, float_pages).
