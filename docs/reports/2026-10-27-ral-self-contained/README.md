# RA-L rules: paper without supplement, ablation note, video zip
2026-10-27 | root: /home/thanhnc/gauss_ws/src/uags | git: master@a9cbc20 | [all reports](../INDEX.md)
Follows: [Final paper: 8 pages and supplement](../2026-10-27-final-8-pages/README.md)
Verdict: DONE -- paper is self-contained (no supplement references), ablation rows explained, video zip meets RA-L rules
Key result: n/a (no new measurement)
Attempts: n/a | Time: 23:24-23:55 JST (`date` stamps; video render 16 min, encode 3 min)

## 0 Context
- Ask (user, 2026-10-27): add a sentence on the two ablation rows with equal cells ("yes, you can add with
  three"), and "do you need to run something else".
- Checking the open item "RA-L size limits" found the RA-L rules ([ral_rules](logs/ral_rules.txt)).
- No new experiment. No measured number changed.

## 1 Findings
| Finding | Evidence | Action |
|---|---|---|
| RA-L takes only one zip (max 50 MB): one video, data, code, ReadMe.txt, Summary.txt. "DO NOT submit supplemental text or figures" | [ral_rules](logs/ral_rules.txt) | the 16-page supplement cannot be submitted |
| main.tex cited the supplement 10 times | `grep Supp papers/main.tex` | all removed; each claim keeps its numbers in the text |
| "Pixels instead of views" and "Pixel-level variant" print equal cells. Both use pixel mass, so mu* and u_epi are equal (mIoU 27.9128, AUSE epi 16.5149 in both); the other switches move ECE, NLL, AUSE ale by < 0.07 (AUSE ale 8.8145 vs 8.7521) | [build_checks](logs/build_checks.txt) | one sentence in Sec. IV-E |
| Video captions cited "the supplement's" figures; 4 caption numbers (scene0140_00 mIoU) came from the supplement | [make_video.py](../../../scripts/make_video.py) | removed; the 4 numbers now come from scannet_tab1/eval_test.json, labelled "this scene" ([frame](figures/video_S3c_caption.jpg)) |
| Zip text files were README.txt / SUMMARY.txt | old zip listing | now ReadMe.txt / Summary.txt |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Page limit | pdflatex + bibtex, scratch copy | 8 pages, 0 warnings, 0 undefined, 1-2 lines spare | PASS ([build_checks](logs/build_checks.txt)) |
| No supplement reference | grep "Supp", non-comment lines of main.tex | 0 | PASS |
| Fonts, PDF version | pdffonts, pdfinfo | all embedded, no Type 3; PDF 1.7 (RA-L min 1.4) | PASS |
| Anonymity (PDF) | pdfinfo; strings + decoded streams | "Anonymous Authors"; 0 identity strings | PASS |
| Index terms | RA-L keyword list | 4 of 4 in the list | PASS ([ral_rules](logs/ral_rules.txt)) |
| I2 new numbers | sentence vs tab3/eval_test.json | 8.75, 8.81 = AUSE ale of abl_legacy_P2, abl_pixels_P2 | PASS |
| Video numbers | `make_video.py numbers`; `compose --dry` | 84 parsed; every caption number is in main.tex or labelled "this scene" | PASS |
| Video build | `compose --only S3a,S3c`; `deliver` | 144.2 s, 4326 frames equal; zip 41.6 MB (max 50); ReadMe.txt + Summary.txt; anonymity hits 0 | PASS ([compose](logs/video_compose.txt), [deliver](logs/video_deliver.txt)) |
| Push | git push | no credentials on this machine | NOT RUN (user) |

## 7 Open issues
- Do NOT upload papers/supplementary.pdf. supplementary.tex/.pdf stay in the repo (kept for arXiv later, or delete: user).
- Video segments S2b (estimator figure, 6 s) and S3c (method comparison, 8 s) show figures that are no longer in
  the paper. A strict reading of "no supplemental figures" could object; cutting them is a CPU-only rebuild.
- Page 8 has only 1-2 lines spare.
- Claim 1 (u_epi ranks the reducible error) is shown on Replica only (6/6); a ScanNet few-view run was never done (GPU).
- Dataset licences for images in the video not checked (carried over).
- video/submission/ still holds the old README.txt and SUMMARY.txt (not in the zip).
- Push: master is 41 commits ahead of origin before this report's commit.

## 8 Files
- Changed: [papers/main.tex](../../../papers/main.tex), papers/main.pdf (bb8dc0c);
  [scripts/make_video.py](../../../scripts/make_video.py) (a9cbc20).
- Video: /home/thanhnc/gauss_ws/data/uags_results/video/supplementary_video.zip (41.6 MB).
- This folder: [logs/](logs/) (ral_rules, build_checks, video_compose, video_deliver),
  [figures/video_S3c_caption.jpg](figures/video_S3c_caption.jpg).
