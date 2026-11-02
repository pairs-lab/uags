# Figure 2, v4: the v2 blocks and equations in the clean v3 style
2026-11-02 | root: /home/thanhnc/gauss_ws/src/uags | git: master@ed4cbad | [all reports](../INDEX.md)
Follows: [One merged paper and a lighter Figure 2](../2026-11-02-merge-and-overview-figure/README.md)
Verdict: DONE -- papers/figs/pipeline_v4.tex replaces v3 in the paper; v2 and v3 kept
Key result: 8 numbered blocks + threshold, 1-4 key equations each, no prose paragraphs; 7.16 x 2.82 in ([v4](figures/fig2_v4.jpg))
Attempts: n/a | Time: 16:47-17:02 JST (`date` stamps)

## 0 Context
- Ask (user, 2026-11-02): "pipeline_v3 is quite basic, right? let improve by using pipeline_v2; add some other
  equation and block".

## 1 What changed
| Block | v2 | v3 | v4 |
|---|---|---|---|
| Keyframe, features | 2 boxes, 3 lines | 1 box | 1 tall box, $\mathbf z_p\in\mathbb S^{d-1}$, PCA $d{=}64$ |
| Lifting | 5 formula lines | words only | $\mathbf F_p$, the gradient identity, $\Rightarrow\mathbf R_v, m_v$ |
| Observation | glyph + 2 formulas | merged into lifting | glyph, $\mathbf u_v=\mathbf R_v/m_v$, $\omega_v$ |
| Running sums | 4 sums + 3 prose lines | one line | 4 sums ($s=W_1$), buckets $\mathbf R^{\rm old},\mathbf R^{\rm new}$ |
| Two uncertainties | 2 small thumbs + 4 formulas | 2 thumbs, words | $\boldsymbol\mu^\star$, 2 thumbs, $u^{\rm epi}$, $u^{\rm ale}$, $\hat\rho^2$ |
| Threshold $\theta(h)$ | histogram + 3 lines | dropped | histogram, 0.999-quantile of $\bar c$ |
| Change test | thumb + 3 lines | thumb, 1 line | thumb, $c$, $\bar c$, fires if $\bar c>\theta(h)$ |
| Reset, re-admission | 6 prose lines | 1 line | 3 short lines ($\nu\ge3$, mean depth residual $\le5$ cm) |
| Read-out, robust set | 7 lines + legend | labels only | softmax, P0, P2, labels; robust set $s\ge\tau_{\min}$, $\hat\rho\ge0.6$ |

Every formula is the one of v2 / Sec. III (same symbols); prose moved to the caption. Fonts: titles 8 pt, formulas 7 pt,
notes 6 pt; all Type 1 embedded.

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Figure build | pdflatex figs/pipeline_v4.tex | 518 x 203 pt (text width), no overlap after 2 fixes | PASS ([v4](figures/fig2_v4.jpg)) |
| Paper build | pdflatex + bibtex + 2x pdflatex | 24 pages; 0 warnings, 0 undefined, 0 "??", no Type 3 | PASS |
| Anonymity | pdfinfo; strings + decoded streams | "Anonymous Authors"; 0 identity strings | PASS |
| Paper parsers | `make_video.py numbers`; pytest test_video, test_make_tables_tab2/tab3 | 84 numbers; 24 passed ([tests](logs/tests.txt)) | PASS |

## 7 Open issues
- Fig. 2 moved from page 2 to page 3 (v4 is 0.65 in taller than v3).
- The video's method segment (S2a) still shows v2.

## 8 Files
- Commit ed4cbad. New [papers/figs/pipeline_v4.tex](../../../papers/figs/pipeline_v4.tex), papers/figs/pipeline_v4.pdf; changed
  [papers/main.tex](../../../papers/main.tex) (include + caption), papers/main.pdf.
