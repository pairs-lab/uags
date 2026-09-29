# Fix agent report (workflow wf_17b72234-86b, verbatim)

Everything from the review is applied to the paper except the page target. The copy now compiles to **8 pages, not the planned 7**. Tectonic exits 0 with no errors, no undefined references or citations and no rerun warning. The only warnings left are three underfull boxes, the worst a small gap on page 6. The PDF is at `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/paper/main_v2.pdf`.

**Page count (critical).** It went from 9 to 8 pages. The body now ends about 0.3 column into page 8, and the references fill the rest of that page. 8 pages is the RA-L limit. Reaching 7 needs about 0.8 page more, and every remaining option drops planned content, so the lead has to choose:
- remove Alg. 1 (about 0.3 column);
- shrink or drop Fig. 2;
- drop the keywords block;
- drop some citations;
- or shorten the RIO10 part if access fails.

What went to get from 9 to 8:
- The intro paragraph that repeated Related Work.
- Contribution 4 (the one-pass trick). It is now a short note in III-C that credits joseph2025gradbp.
- Section III-H, folded into III-C.
- Table IV and IV-F, replaced by one paragraph.
- The P0/P1/P2 rows of Table III.
- A shorter conclusion and shorter Fig. 1 and Fig. 2 captions, and tighter Setup and Metrics.
- Repeated statements, each now kept once in its home section.
- The reference list now uses "et al." after 6 authors. The rule lives in `ieeectl.bib`, which the preamble writes on the first pass (ref.bib is untouched).

**Math findings**
- **Kish support (major):** confirmed. Since the view weight is at most 1, the Kish count is never below the summed view weight, so the support is just that sum. The paper now says the support is the weighted number of views everywhere, including Fig. 1 and Fig. 2. Kish stays only in the agreement estimator and the upward bias of the mean length. One correction to the finding: NormLift's count is not "size-blind". It is scale-invariant, so ten tiny views count as much as ten full ones, and the text now says that. The "rank about equally" expectation is gone.
- **"Pixels instead of views" row:** the text now says what changes and what stays at view level. A source comment gives the needed settings (`evidence="pixels"`, `epistemic="pixels"`).
- **Agreement estimator bias:** only the ratio inside the clamp is called unbiased, and the clamp and square root are said to leave a small bias. The Fig. 2 caption now says the curves are root-mean-square values and the estimate "stays close to" the true value. The legend is updated.
- **CSS kernel:** fixed. The text says the counts match categorical BKI only when it is run on our per-view kernel, and that CSS uses the pixel kernel.
- **Re-admission:** the text now describes the code, not the fact sheet: the view weight since the reset, the view that fired included, must reach 3, and a weighted depth mean is used. I did not change the code.
- **Calibration mode name:** the paper's description is right, but the code's `calib_mode="pooled"` is actually the history-conditioned rule. I added a source comment. The code must get a true pooled mode before the "Pooled null" row runs.
- **Fig. 1 fixes:** the view weight is parenthesized, the support box shows the summed weight, and the agreement formula shows its clamp.

**Claims findings**
- RIO10 now has visible `\todo`s in the abstract and the intro.
- Contribution 2 says the mapping-pass threshold "targets a nominal per-view false-alarm rate … and we measure the actual rate." Limitations adds that the mapping-pass scores are in-sample.
- NormLift is now described fairly: we credit it for the per-view mean, and name the real differences.
- New references for the aleatoric channel: a NormLift "1 − C_inter" row in Table I, the same curve in Fig. 3, and an "uncorrected mean length" row in Table III. The comparison text is rewritten.
- The Yager row is now "(our design)". E2-BKI is described as using Dempster's rule and is dropped from the caption and Limitations.
- MIF is described correctly, and the feature residual is credited to CL-Splats and to MIF's semantic term.
- The stale-geometry claim is hedged: it holds only while the surface behind lies more than 5 cm away, and Limitations adds thin objects. "When depth is available" is added to the abstract.
- The tuning-sensitive claims now read "in tuning" and point to the fixed-threshold row.
- NormLift's scores are described as built from its quantities, not defined by NormLift.
- The text now says five prompt templates.
- The Fig. 1 box is renamed "threshold".

**Form findings**
- LSeg is named in all four places, and the resolution example is now "one ViT-L/16 patch of LSeg covers 16×16 pixels".
- The frame count is renamed M, the predictive is written P(y_p = c) in the text and Fig. 1, and the P2 coefficients are renamed γ_e, γ_a, γ_u.
- Brier is marked ×100 and the Table II units are fixed. The integer placeholder in Table III uses a new `\phn` macro.
- Defaults are now written "(default …)", not in square brackets. "masses alike" is gone, and no sentence starts with a symbol.
- Section references no longer break across lines, algorithm line 1 no longer wraps, and the loose lines are fixed.
- The depth validity mask is added to Fig. 1.
- TODO markers went from 32 to 25. I closed these without the lead's sign-off, so please confirm them:
  - **Tuning scene:** 7-Scenes chess, from `usplat/vmf_field.py:44` and the data-and-baselines report.
  - **Hardware:** RTX 5070 Ti and i5-14600KF, read from `nvidia-smi` and `lscpu`. Keep it only if the timing runs on this machine.
  - **Operating point:** the baselines get the same 10⁻³ per-view false-alarm rate as ours.
  - I added one TODO: "cite Yager 1987".

**Still to do outside the paper**
- `usplat/baselines.py` has to expose a NormLift "1 − C_inter" score.
- `usplat/vmf_field.py` needs an "uncorrected mean length" estimator option, and the pooled calibration mode described above.
- ref.bib needs Yager 1987, doi:10.1016/0020-0255(87)90007-7.

Files changed (all in /home/thanhnc/gauss_ws/src/uags/papers):
- main.tex
- figs/pipeline.tex and figs/pipeline.pdf (rebuilt)
- figs/uncertainty_plane.tex and figs/uncertainty_plane.pdf (rebuilt)
- parked_sections.tex (old Table IV and old contribution 4 added)

The originals are in `.../scratchpad/fix/orig/`, and the diffs are in `.../scratchpad/fix/` (`main_tex.diff`, `pipeline.diff`, `plane.diff`).