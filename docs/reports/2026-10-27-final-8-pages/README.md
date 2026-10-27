# Final paper: 8 pages, supplement, smaller PDFs, neutral video caption
2026-10-27 | root: /home/thanhnc/gauss_ws/src/uags | git: master@ad9f893 | [all reports](../INDEX.md)
Follows: [Figures, video and stronger results](../2026-10-20-figures-video/README.md)
Verdict: DONE -- 8-page paper + supplement + video built and checked; push left to the user (no credentials here)
Key result: main paper 9 -> 8 pages, 6.2 -> 3.8 MB, abstract 258 -> 245 words ([build_checks](logs/build_checks.txt))
Attempts: n/a | Time: 15:16-16:01 JST (`date` stamps)

## 0 Context
- Ask (user, 2026-10-27): "yes, continue" to: finalize around the measured contributions (two uncertainty
  channels, forgetting), trim the paper to 8 pages, shrink the PDF, make the scene-002 video caption neutral, push.
- Limits: RA-L 8 pages including references; IEEE abstract 250 words. What does not fit moves to an anonymous
  supplement (papers/supplementary.tex). Nothing deleted. No new experiment; no measured number changed.

## 1 Findings
| What | Why it matters | Fix |
|---|---|---|
| Abstract: 258 words in the PDF | IEEE limit 250 | cut to 245 |
| "fused row has the highest Gaussian AP and AUROC of Table II" became false once the fused baselines joined Table II | wrong claim | now: fusing raises AP 57.4 -> 75.0 (Sec. IV-D) |
| "NormLift drop is the best row at the pixel level" became false for pixel AP (NormLift + depth 66.9) | wrong claim | now "best unfused row" |
| Text said "Tables I and III" (old numbering) | change table is now Table II | `\ref` (Sec. IV-A) |
| Video number parser and caption check read only main.tex | 4 numbers moved to the supplement; build failed | both read supplementary.tex too ([make_video.py](../../../scripts/make_video.py)) |
| Scene-002 video caption only said averaging keeps more of the chair (44% vs 15%) | ours also changes the top class of more static Gaussians there (33% vs 20%) | caption gives both, for both scenes ([video_compose](logs/video_compose.txt), [frame](figures/video_002_end.jpg)) |

## 2 What moved where
| Item | Before | Now |
|---|---|---|
| Fused-baseline rows | own table | bottom block of Table II; sign tests in Supp. S2 |
| Calibration figure, paired tests | main appendix | Supp. S1 |
| Measured false-alarm rate, change and query figures | main | Supp. S2 |
| Per-class table + its sentences, method comparison, all-scene grid, room_0 uncertainty figure | main | Supp. S3 |
| Agreement-estimator figure | main | Supp. S4 |
| P2 coefficients, build cost, per-keyframe timing | main | Supp. S5 (main keeps 110 ms and build times) |
| Five registered later attempts | text | one sentence in main + Supp. S6 table |
| Limitations | own subsection | paragraph of the Conclusion |

## 3 Size
| File | Before | After | How |
|---|---|---|---|
| papers/main.pdf | 6.2 MB | 3.8 MB | JPEG q90 for large images in figure PDFs ([shrink_figs.py](../../../scripts/shrink_figs.py)) |
| papers/supplementary.pdf | 17.4 MB | 14.8 MB | same; palette heat maps stay large |
| Video zip | 37.9 MB | 41.6 MB | re-delivered; under the script's 44 MB cap ([checks](logs/video_deliver_checks.json)) |
| Figure fidelity | | PSNR 51.7-65.2 dB, 99th pct diff <= 2/255 | [shrink_fidelity](logs/shrink_fidelity.txt); one 1-px colorbar edge differs |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Page limit | pdflatex + bibtex of papers/ | 8 pages, ~7 reference lines spare | PASS ([build_checks](logs/build_checks.txt)) |
| References | main.log, supplementary.log, pdftotext | 0 undefined, 0 duplicate, 0 "??" | PASS |
| Fonts | pdffonts | all embedded, no Type 3 | PASS ([pdffonts](logs/pdffonts.txt)) |
| Anonymity (PDFs) | pdfinfo, strings, decoded streams | "Anonymous Authors"; no user name or path | PASS |
| I2 no number changed | regex, main.tex prose vs 4e8c296 | only new number 110 (= 109.9 ms rounded) | PASS ([number_check](logs/number_check.txt)) |
| Video numbers | `make_video.py numbers`, `compose --dry` | 90 numbers parsed; caption check passes | PASS |
| Video build | `make_video.py compose --only S2a,S3a,S4b,S4c,S4e`; `deliver` | 144.2 s, frames equal, anonymity hits 0 | PASS ([compose](logs/video_compose.txt), [summary](logs/video_summary.txt)) |
| Push | git push | no credentials on this machine | NOT RUN (user) |

## 7 Open issues
- Push: master is 39 commits ahead of origin; the user must push.
- RA-L size limit for the supplement (14.8 MB) and the video zip (41.6 MB) not checked here.
- supplementary.tex reads main.aux (xr-hyper): build main.tex first. Edit supplementary.tex directly.
- Optional cleanup (not done, needs approval): video caches /home/thanhnc/gauss_ws/data/uags_results/video/
  clips (33 MB), stills (7.7 MB), contact (3.7 MB); submission/ (40 MB) duplicates the zip.

## 8 Files
- Changed: [papers/main.tex](../../../papers/main.tex), new [papers/supplementary.tex](../../../papers/supplementary.tex),
  papers/main.pdf, new papers/supplementary.pdf, 6 figure PDFs in papers/figs/,
  new [scripts/shrink_figs.py](../../../scripts/shrink_figs.py), [scripts/make_video.py](../../../scripts/make_video.py).
- Commits: e7042dc figures, 9ff47a7 8 pages + supplement, a1f9f0f abstract, 5df86d1 supplement references, ad9f893 video.
- Video: /home/thanhnc/gauss_ws/data/uags_results/video/supplementary_video.zip (41.6 MB).
- This folder: [logs/](logs/), [figures/video_002_end.jpg](figures/video_002_end.jpg).
