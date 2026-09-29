# Integrate agent report (verbatim)

The new `papers/main.tex` is assembled and compiles cleanly, but the PDF is 9 pages. That is over the RA-L maximum of 8 pages (6 plus 2 paid) and the planned 7.

**Safety check.** `papers/main.tex` had md5 `689e608cf0020780d3933df3774cf48d` before I wrote anything, so it had not been edited and I replaced it in place. The old draft is backed up at `main_before_integration.tex` in the scratchpad and is also in git (commit `1029dd7`). `ref.bib` is unchanged (md5 `d00fc90c0a6021b51dc93d7de2ff0f59`).

**Build result.** Command: `cp -r papers/. <scratch>/build/ && tectonic --keep-logs --keep-intermediates main.tex`.
- It exits with 0. The log has no undefined references, no undefined citations and no LaTeX warnings. BibTeX gave no warnings and formatted 37 entries.
- **Pages:** 9, US letter, PDF 1.5. The text ends on page 8. Page 9 only has references, in the left column, filled to 530 of 734 pt, so the paper is about 8.35 pages. The Experiments section with its 4 tables and 2 placeholder figures takes about 3 pages, and the Method about 2.
- **Overfull boxes:** none. There was one of 1.69 pt at the four-sums display, which I fixed by tightening its spacing.
- **Underfull boxes:** 2 horizontal ones, in the NormLift paragraph of Related Work (badness 1484) and the NormLift sentence of the aleatoric paragraph (badness 5374). There is also 1 vertical one, badness 2245, raised during page output.
- **Fonts and anonymity:** all fonts are embedded and none are Type 3. The PDF author is "Anonymous Authors". A scan of 141 streams found no identifying strings.
- **Citations and labels:** 37 `\cite` keys, all in `ref.bib` and all on the allowed list. 30 labels, no duplicates, and every `\ref` and `\eqref` resolves. There is no non-ASCII text, no "---" and no semicolon chains.

**Parked text.** `papers/parked_sections.tex` holds the old text verbatim inside `\iffalse…\fi`, so it does nothing even if someone inputs it. It covers the localization and active-perception paragraphs of Related Work, the EFE section, the localization setup and metrics sentences, the localization table, the old ablation table with its EFE row and ATE column, and the one-line localization mentions. The header comment gives the date and the reasons quoted from fact sheet C.

**How `main.tex` is built.** It is the preamble, then the title and author from the front draft, then the front, related, method and experiments drafts, then the bibliography. The full edit diff against the plain concatenation is saved as `integration_edits.diff`. The edits:
- **Fig. 1:** I added the pipeline figure after the first paragraph of the introduction, so it cannot land above the abstract. It uses the figure drafter's caption with "calibrated" replaced, and the introduction and method now refer to it.
- **"Calibration" wording (fact sheet rule C):** the method heading "Calibration." became "Threshold from the mapping pass.", "We calibrate it" became "We set it", and "its calibration" became "how its threshold is set". "Calibration" now appears only for the predictive.
- **Uncertainty-plane caption:** replaced with the figure drafter's longer caption, which also explains the estimator curves.
- **Notation:**
  - NormLift's reliability is now $\mathcal{R}$, so it does not clash with the bold $\R$.
  - Its $N/(N+1)$ is now $N_{\mathrm{eff}}/(N_{\mathrm{eff}}+1)$.
  - $\bar r_n$ is now defined in the method.
  - The $\tilde r$ ablation now divides by $(W_1+1)$, which matches the legacy estimator in `vmf_field.py`.
  - $h_n$ is now defined as the mass of the older bucket.
  - The support $s_n$ is now described as the effective number of views capped by the total view weight.
- **Acronyms:** vMF is defined at its first use, in the intro's LAMP sentence. The repeated definitions of BKI and vMF in Related Work and Method are removed.
- **Consistency:** "3DGS map" is now "3D-GS map". "labelled" and "unlabelled" are now "labeled" and "unlabeled". The ViT-B/16 sentence now reads "For example, …".
- **Preamble:** `\reimpl` and `\phs` moved into the preamble. The tuning-sequence `\todo` now notes "code notes say 7-Scenes chess".

**Open issues:**
- **Length:** 9 pages against 8 allowed and 7 planned. I did not cut any content. The lead has to decide what to shorten.
- **Pipeline figure:** a box in `papers/figs/pipeline.pdf` is still titled "calibration", which conflicts with rule C. That is a one-word change to the figure file, which I left alone.
- **Symbol clashes left in place:** $c$ is both the class index and the change score, $d$ is both the dimension and the depth superscripts, and $N$ is both the Gaussian count and the number of scenes.
- **Unused figures:** `vmf_sphere.pdf` and `gradient_trick.pdf` are no longer used and still carry the old notation.
- **Carried over from the drafts:**
  - RIO10 access is still pending.
  - The extractor is not chosen yet.
  - The MIF and CL-Splats attribution was not checked against those papers.
  - The joint vMF prior has no citation.
  - The re-implemented baseline rows ("NormLift reliability drop", "pooled null") are not in the code yet.

Files are in /tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/paper:
- main_v1.pdf
- build/main.pdf
- build/main.log
- main_before_integration.tex
- integrate/main_assembled.tex
- integrate/integration_edits.diff

In the repo:
- /home/thanhnc/gauss_ws/src/uags/papers/main.tex
- /home/thanhnc/gauss_ws/src/uags/papers/parked_sections.tex