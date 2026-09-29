The compile succeeded with no errors: exit code 0, 6 pages, and BibTeX ran cleanly. But the 6-page count is inflated because the body text is not in Times, so it is not a valid pdflatex page count. The same source built with Times metrics is 5 pages. No repo file was written: the timestamps in `/home/thanhnc/gauss_ws/src/uags/papers` are all from before the run. The `M main.tex` / `?? ref.bib` status in git was already there.

**Tectonic install**
- The musl build of v0.17.0 was 10,151,914 bytes (the gnu build was 22.7 MB).
- The unpacked binary is `/home/thanhnc/gauss_ws/third_party/tectonic/tectonic` (26 MB, statically linked). `--version` prints `Tectonic 0.17.0`. The tarball is still next to it.
- The first run downloaded about 48 MB of TeX bundle files into `~/.cache/Tectonic`.

**Main build (plain copy of papers/)**
- Command, run in `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/extr/compile/papers`: `tectonic main.tex --outdir build --keep-logs --keep-intermediates`
- Result: exit 0, 09:30:32 to 09:34:14 including the downloads. Tectonic ran TeX, then BibTeX, then TeX twice more.
- Pages: 6 (`pdfinfo`, letter size). The last page holds only references [19] onward, 23 text lines.
- Files:
  - PDF: `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/extr/compile/papers/build/main.pdf`
  - Log: `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/extr/compile/papers/build/main.log`
  - Also in `build/`: `main.blg`, `main.bbl` and `tectonic_stdout.txt`.

**Checks on the main build**

| Check | Result |
|---|---|
| Errors | 0 lines starting with `!` in the log |
| Undefined citations or references | 0. No "Citation/Reference undefined", "There were undefined", "Label(s) may have changed" or "multiply defined" lines. No `??` in the PDF text. |
| Cited keys | All 24 cited keys in `main.aux` have a `\bibitem` in `main.bbl`. `ref.bib` has 49 `@` entries, so 25 are unused, which is normal. |
| BibTeX | `main.blg` shows IEEEtran.bst v1.14 and "Done." with 0 warnings and 0 errors |
| Overfull boxes | None at all |
| Underfull boxes | `\vbox` (badness 5592) at `main.tex:216` while `\output` is active. `\hbox` (badness 2205) at `main.bbl` lines 131–135, the "K. Nagami, T. Chen, ..." entry. |
| Other warnings | `algorithmic.sty:11` has an invalid UTF-8 byte, which is inside the package itself and harmless |

**Font problem (the main issue)**
- The log has 75 LaTeX Font Warnings like "Font shape `TU/ptm/*` undefined". XeTeX has no Times in the TU encoding, so the text falls back to Latin Modern.
- `pdffonts` shows only regular Latin Modern for the body text. Bold, italic and small caps in body text and headings are printed as upright regular.
- The old repo `papers/main.pdf` (xdvipdfmx, created Jun 13, 6 pages) has the same Latin Modern fonts, so it has the same problem.

**Times-metrics build (my addition)**
- This is a second copy with one line added after `\documentclass` in the copy only: `\usepackage[OT1]{fontenc}`. XeTeX then uses the same Times (ptm) fonts as pdflatex.
- Location: `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/extr/compile_ot1/papers/build/` (`main.pdf`, `main.log`). The first try failed only because my sed dropped the backslash; the rerun is the one reported here.
- Result: exit 0, 5 pages. The body is in NimbusRomNo9L Regular, Medium and Italic.
- Checks: 0 errors, 0 undefined citations or references, 0 overfull boxes. One underfull `\hbox` (badness 1642) at `main.tex` lines 421–428.
- Only 5 font warnings remain. They come from the class loading, before fontenc takes effect.

**Paper content**
- `main.tex` has 19 `\todo{}` calls, and "TODO" appears 21 times in the PDF text of both builds, printed in red.
- Example: the Table II caption reads "[TODO: replace placeholders.]".

I did not write a report under docs/reports because this task did not name one.