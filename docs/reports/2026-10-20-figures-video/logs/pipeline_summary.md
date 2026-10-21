# Pipeline figure v2 (fig:pipeline): diagnosis, design, build, checks

2026-10-21 09:02-09:23 | R = /home/thanhnc/gauss_ws/src/uags, D = /home/thanhnc/gauss_ws/data, O = D/uags_results/pipeline
Verdict: DONE. The new figure builds cleanly at 7.14 in, all fonts are Type 1, every symbol matches Sec. III, and all thumbnails come from validation scenes.
Not done here: main.tex, figs/README.md and the old pipeline.tex/pdf are untouched. Nothing was committed.

## 1 Diagnosis of the old figure (papers/figs/pipeline.tex)
| # | Problem | Evidence |
|---|---|---|
| D1 | A wall of math: about 35 formulas in 9 boxes and no real image. At print size nothing anchors the eye. | old render (pipeline.pdf) |
| D2 | The per-keyframe loop is not drawn: no "next keyframe" arrow, and the mapping and update passes are not named. The two threshold arrows have no labels. | pipeline.tex:166-169 |
| D3 | The change test does not match Sec. III-F: it uses ω_v (`R_<t = Σ ω_v u_v`), but the Method counts test views by ν. Buckets R^old/R^new, retention λ, persistence k and h = W^old are all missing. | pipeline.tex:122-127 vs main.tex III-F |
| D4 | The robust set (s ≥ τ_min, ρ̂ ≥ 0.6, not changed) is missing. | main.tex III-F, last sentence |
| D5 | Re-admission says "after 3 views". The Method says the weight ν since the reset (firing view included) is ≥ 3, and the mean depth residual is ≤ 5 cm. | pipeline.tex:132 |
| D6 | `valid_p` and the per-pixel `d^obs_p` are not paper symbols. | pipeline.tex:76 |
| D7 | Named xcolor tints instead of the paper's Okabe-Ito palette. The header says "tectonic" and "ieeeconf 7.0 in", but the paper is now IEEEtran journal (7.16 in) and tectonic is not installed. | pipeline.tex:1-7 |

## 2 Design
- Six columns in two rows, in the order of the paper's sections. Row A: (1) keyframe -> (3) render + backward -> (4) one observation per view -> (5) running sums -> (6) posterior + u_epi/u_ale -> (9) predictive. Row B: (2) features under (1), threshold θ(h), (7) change test, (8) reset/re-admission, robust set.
- Update loop: a dashed frame around (1)-(5), (7), (8) labelled "per keyframe v: mapping pass (static: no reset, c̄ recorded), then update pass", plus a "next keyframe v+1" arrow from (5) back to (1). Change loop: (4) -ν_v u_v-> (7) <-R^old- (5); (7) -> (8) -reset-> (5); θ(h) -> (7); (8) and (6) -> robust set.
- Colours (Okabe-Ito): blue #0072B2 = evidence (3-5); orange #E69F00 = change handling; bluish green #009E73 = read-out. u maps use viridis (as Fig. qual); grey = undefined.
- Real thumbnails: RGB keyframe, LSeg z_p (the first 3 coordinates in basis B), u_epi, u_ale, P2 labels with a 4-class legend, the NOR change overlay with a "before" inset, and the real mapping-pass null histogram with θ(h).

## 3 Selection rules (fixed before any render; stored in O/selection.json, written 09:04:26, before the GPU runs at 09:04:33)
| Rule | Content | Result |
|---|---|---|
| R1 | Among the Replica val mapping keyframes (office_0, room_2; field's own keyframe rule), take the one with the most GT classes covering ≥ 0.5% of the pixels each. Ties: labelled share, then scene order, then index. Labels only. | office_0, Sequence_1 frame 825 (14 classes; next: office_0 830 with 14, room_2 475 with 13) |
| R2 | All Replica thumbnails at that pose on the 320x240 mapping grid. u maps and labels come from the saved full field, through the Table I code path. | see O/replica_meta.json |
| R3 | NOR val 002. t* = the update keyframe (of the 48 in the val record) with the largest masks_real area. | frame 335 (36,552/49,152 px, 38th of 48) |
| R4 | State after all 48 update keyframes through the Table III path (calibrate q = 0.999, update_view with LiDAR depth). Overlay = share of rendered mass from Gaussians reset at least once. Checks against the val record. | PASS (section 6) |
| R5 | Full resolution, rotate cw, 4:3 crop centred on the masks_real centroid; grey photo plus orange overlay at opacity 0.85 x share. | crop y0 = 189, 540x405 |
| R5b | Added after the first render; no selection changed. A "before" inset (the map render at t*, same crop), because the overlay alone did not show what was removed. | - |
| R6 | Threshold glyph: the 002 null of c̄ in the history bucket with the most samples, on [0, 1.6θ], log counts. | bucket 10 < h ≤ 20 views, θ = 0.187, 3.36 M scores, 0.092% above θ |

## 4 Self-critique before building, and fixes
| Issue found | Fix |
|---|---|
| Notation: the paper's c̄ formula (III-F) sums ω, but the test weights views by ν | Figure uses the general bucket form, c = 1 − cos(R^old, R^new), with ν, λ = 0, k = 2. Paper sentence flagged in section 7. |
| Misleading: Replica fields were mapped without depth | No Replica depth thumbnail. The label reads "posed RGB (+ depth)" and the caption says "if available". |
| Misleading: the read-outs come from the final map, not from after keyframe v | Caption: "read-outs of the final map at the keyframe's pose". |
| Misleading: the change thumbnail is from another scene and another pass | Caption names NOR 002 and the update. The inset shows the map before the update. |
| Misleading: the overlay shows resets during the whole update (Table III "flagged"), false alarms included | Shown as is (orange spill on the floor and desk). Mean reset share in the crop: 0.51 on stale pixels vs 0.12 on other covered pixels (O/meta.json change). |
| Clutter | One idea per line, symbols over prose. The tightest lines were measured in LaTeX and then fitted to the boxes. |
| Colour-blind and greyscale | Okabe-Ito groups are also told apart by position, frame and titles; viridis for the u maps; the orange overlay differs in luminance from the grey photo. The tab20 label palette is not colour-blind safe, but no claim depends on it. |
| Legibility at 7.16 in | Body text 7 pt (\scriptsize), small labels 6 pt, thumbnails 1.40-1.96 cm wide. Checked at 300 dpi. |
| Brand marks | NOR 335: the monitor and mouse-pad logos fall outside the crop. Replica 825: a clock and a world-map screen, no readable brand. Nothing blurred. |

## 5 Commands (cwd R; env: PATH=~/venvs/uags/bin:~/.local/cuda-12.8/bin, CUDA_HOME, PYTHONPATH=.)
```bash
python3 scripts/render_pipeline_thumbs.py select --out $O                                   # CPU, 32 s
UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 4500 python3 scripts/render_pipeline_thumbs.py replica --out $O   # 86 s, peak 1874 MiB
UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 4500 python3 scripts/render_pipeline_thumbs.py nor --out $O       # 89 s, peak 2715 MiB
python3 scripts/render_pipeline_thumbs.py compose --out $O                                  # CPU; copies to papers/figs/pipeline_thumbs/
cd papers && pdflatex -output-directory <scratch> figs/pipeline_v2.tex && cp <scratch>/pipeline_v2.pdf figs/
```
GPU log: O/gpu_run.log (RTX 4070 Ti; the gate ran both jobs at once, without waiting). Text cache: own copy O/text_cache_lseg_pipeline.pt (still byte-identical to the original, checked with cmp).

## 6 Checks
| Check | Result |
|---|---|
| NOR replay vs val record (D/uags_results/nor/tab2_relmax/val/002.json) | thresholds equal (pooled 0.1845, 6 buckets); resets 117,170 vs 117,170; Rec 0.5364 vs 0.5364; FA 0.1400 vs 0.1400; stale observed 84,205 vs 84,205: PASS (O/nor_meta.json) |
| Compile | pdflatex rc 0; 0 warnings, 0 overfull boxes; 514.4 x 211.4 pt (7.14 x 2.94 in) |
| Overlaps and clipped text | Viewed at 150 and 300 dpi (O/pipeline_v2_300dpi.png) after 6 layout passes. The last pass fixed 6 lines that crossed box borders. None left. |
| In the paper (scratch copy of main.tex with the new figure and caption) | figure* on page 2; still 12 pages; no float warning |
| pdffonts | 15 fonts, all Type 1 (NimbusRomNo9L, CM, MSBM). null_hist.pdf has no fonts. |
| Names, paths, logos | `strings` finds no "thanhnc", "/home", "gauss_ws" or "claude" in the PDF or the thumbnails. PDF metadata has no Author or Title. No readable brand text in the thumbnails. |
| Validation scenes only | Replica office_0 (val); NOR 002 (val) |
| Forbidden files | Not modified; main.tex's Method is unchanged since it was read (git diff 6029f04..HEAD touches only the abstract, results and appendix). |

Symbol check against main.tex Sec. III (n dropped except in w_{p,n}, f_n, μ_n, as stated in the caption):
| Figure | main.tex | OK |
|---|---|---|
| z_p ∈ S^{d−1}, d = 64, PCA B, LSeg | III-B | yes |
| F_p = Σ_n w_{p,n} f_n; ∂_{f_n} Σ_p Y_p^T F_p = Σ_p w_{p,n} Y_p; Y_p = [z_p, 1, depth, mask] | III-A, III-C | yes (the mask is named in words, as in the text) |
| R_v, m_v, u_v = R_v/m_v, ‖u_v‖ ≤ 1, ω_v = min(1, m_v/(a_0HW)) | III-C | yes |
| R = Σωu, W_1 = Σω, W_2 = Σω², Q = Σω²‖u‖² | III-C | yes |
| vMF, μ* = R/‖R‖, A_d(κ̂) = ‖R‖/(W_1 + c_0) | III-D, eq. kappa | yes |
| s = W_1, u^epi = 1/(1+s); ρ̂² = [(‖R‖² − Q)/(W_1² − W_2)]_0^1, u^ale = 1 − ρ̂ if s ≥ τ_min | III-E, eq. rho | yes |
| R^old, R^new, ν, retention λ, ν_v u_v into the recent bucket, c = 1 − cos(R^old, R^new), c̄ = min of the last k, λ = 0, k = 2 | III-F | yes |
| θ(h), history size h, q-quantile, q = 0.999, nominal 10^-3 per view | III-F | yes |
| reset ("sums restart from this view"), *changed*, ν ≥ 3, mean \|d^obs − d^ctr\| ≤ 5 cm | III-F | yes |
| robust set: s ≥ τ_min, ρ̂ ≥ 0.6, not changed | III-F | yes |
| t_c = B^T t̃_c, P(y_p = c) = softmax_c(β F_p^T t_c), P0 f_n = μ*_n, P1 E[μ_n], P2 = P0 with β_p(u^epi_p, u^ale_p) | III-B, III-G | yes |

## 7 Open issues
| Issue | Next step |
|---|---|
| main.tex III-F says "Σ_{v<t} ω_{v,n} u_{v,n}" for c̄, but the test weights views by ν (the NOR fields use relmax; the Replica fields use "view" weights, where ν = ω, fast_min 1). The figure follows the bucket definition (ν). | Lead/author: write ν_{v,n} in that formula, or say that ν = ω on Replica. |
| Text is 7 pt and small labels 6 pt (the old figure used 8 pt). The figure is 2.94 in tall (old: 2.40 in). | Accept, or drop the threshold glyph and the "before" inset to gain space. |
| The tab20 class palette in (9) is not colour-blind safe. | Optional: label 2-3 regions directly. |
| The orange overlay (change handling) differs from the teaser plan, which uses #0072B2 for "reset by test". | Pick one colour when the teaser is built. |
| papers/figs/README.md has no row for pipeline_v2 yet (rule: new files only). | Add it when the figure is adopted: swap figs/pipeline.pdf -> figs/pipeline_v2.pdf and the caption in main.tex (from figs/pipeline_v2.txt). |

## 8 Files
- New: R/scripts/render_pipeline_thumbs.py; R/papers/figs/pipeline_v2.tex, pipeline_v2.pdf, pipeline_v2.txt (caption); R/papers/figs/pipeline_thumbs/ (8 PNG, null_hist.pdf, values.tex; 656 KB)
- Data: O/selection.json (rules), O/replica.npz, O/nor.npz (7.7 MB), O/meta.json, O/replica_meta.json, O/nor_meta.json, O/gpu_run.log, O/pipeline_v2_300dpi.png, O/pipeline_v2_150dpi.png
