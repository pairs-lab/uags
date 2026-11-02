# Figure 2 v6 (v3 + equations) and folder cleanup
2026-11-02 | root: /home/thanhnc/gauss_ws/src/uags | git: master@34390d0 | [all reports](../INDEX.md)
Follows: [Figure 2 v4](../2026-11-02-figure2-v4/README.md)
Verdict: DONE -- v6 in the paper; 2.5 GB cache and 16 MB of old paper files deleted; third_party only checked
Key result: Fig. 2 = v3 + 4 short equations, back on page 2 ([v6](figures/fig2_v6.jpg)); results/ 53 -> 50.5 GB
Attempts: n/a | Time: 17:20-17:50 JST (`date` stamps)

## 0 Context
- Ask (user, 2026-11-02): after v5 (no numbers, inputs/outputs in maths) "now I think v3 is best. we can add a
  little equation to v3"; then "we also need to clean our folder, like in doc, results or third_party".
- Deletions chosen by the user (AskUserQuestion): the CLIP token cache and the old paper files; third_party: check only.

## 1 Figure 2 v6
papers/figs/pipeline_v6.tex = pipeline_v3.tex plus: $\mathbf u_v=\mathbf R_v/m_v$ (lifting),
$c=1-\cos(\mathbf R^{\rm old},\mathbf R^{\rm new})$ (change test), $u^{\rm epi}=1/(1+W_1)$ and $u^{\rm ale}=1-\hat\rho$
(evidence), $P(y_p{=}c)\propto\exp(\beta\mathbf F_p^\top\mathbf t_c)$ (query). Bug fixed on the way: v3 used `\top`
as a coordinate name, which printed $\mathbf F_p^{0.00}$; renamed to `\ytop`. Caption: v3's, with the new symbols.
v5 (built, not kept) was dropped when the user chose v3.

## 2 Cleanup
| Item | Size | Action | Why |
|---|---|---|---|
| results/cache/clip_tokens | 2.5 GB | deleted | per-frame cache, rebuilt by scripts/extract_clip.py when needed |
| papers/supplementary.tex, .pdf | 15 MB | deleted (git) | merged into main.tex (2026-11-02) |
| papers/figs/pipeline.tex/.pdf, pipeline_old_placeholder.pdf.bak, pipeline_v4.*, pipeline_v5.* | 1.7 MB | deleted | superseded figure versions |
| papers/figs/pipeline_v2.*, pipeline_v3.* | 1.3 MB | kept | v2 feeds the video's method segment (make_video.py); v3 is v6's base |
| docs/ (8.8 MB) | - | kept | report history; the largest files are the cited Table I sources |

## 3 third_party check (no change)
| Folder | Loaded by the code? | Note |
|---|---|---|
| /home/thanhnc/gauss_ws/third_party/{dino, proxyclip, radio, sam, talk2dino} | yes (usplat/extractors) | in use |
| gauss_ws/third_party/{radseg, trident} | no path in code | kept |
| src/third_party/proxyclip | no | byte-identical duplicate of the loaded copy (diff 0) |
| src/third_party/talk2dino | no | differs from the loaded copy (72 vs 7 MB) |
| src/third_party/lseg | no | reference only: LSeg code is adapted into usplat/extractors/lseg.py |
| src/third_party/dinotxt | no, but should be | dinotxt.py looks for gauss_ws/third_party/dinotxt/dinov2, which does not exist; the clone is here, so the DINOtxt extractor cannot load (its test skips) |
| src/third_party/tectonic | no | LaTeX binary; the paper now builds with pdflatex |
| src/third_party/{mast3r, dust3r, mast3r_ckpt} | USGLoc | never touched |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Paper build | pdflatex + bibtex + 2x pdflatex | 24 pages; 0 warnings, 0 undefined, no Type 3; Fig. 2 on page 2 | PASS |
| Anonymity | pdfinfo; strings + decoded streams | "Anonymous Authors"; 0 identity hits | PASS |
| Video still builds after the deletions | `make_video.py numbers`, `compose --dry` | see [checks](logs/checks.txt) | PASS |
| Tests | pytest test_video, test_make_tables_tab2/tab3, test_clip_cache | see [checks](logs/checks.txt) | PASS |

## 7 Open issues
- Possible further cleanup (needs your OK): src/third_party/proxyclip (exact duplicate, 13 MB), talk2dino copy (71 MB),
  tectonic (35 MB); results/video caches (84 MB); results/qual figure dumps (5.2 GB, GPU to rebuild).
- dinotxt path mismatch above: move the clone or fix the path if DINOtxt is ever needed again.

## 8 Files
- Commit 34390d0. New papers/figs/pipeline_v6.tex/.pdf; changed papers/main.tex, papers/main.pdf, papers/figs/README.md; deletions above.
