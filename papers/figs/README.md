# Manuscript figures (TikZ)

Each figure is a standalone TikZ document compiled to the PDF that
`main.tex` includes via `\includegraphics`:

| Source | PDF | Used in | Width |
|---|---|---|---|
| `pipeline_v6.tex` | `pipeline_v6.pdf` | Fig. 2 (`fig:pipeline`, `figure*`) | two-column (`\textwidth`) |
| `pipeline_v3.tex` | `pipeline_v3.pdf` | earlier Fig. 2 (v6 = v3 + equations) | two-column |
| `pipeline_v2.tex` | `pipeline_v2.pdf` | the video's method segment (scripts/make_video.py) | two-column |
| `vmf_sphere.tex` | `vmf_sphere.pdf` | `fig:vmf`, Sec. III-A | single column |
| `gradient_trick.tex` | `gradient_trick.pdf` | `fig:gradtrick`, Sec. III-B | single column |
| `uncertainty_plane.tex` | `uncertainty_plane.pdf` | `fig:plane`, Sec. III-C/D | single column |

Rebuild after editing a source (from `papers/`):

```bash
tectonic figs/<name>.tex            # or: pdflatex -output-directory figs figs/<name>.tex
```

On Overleaf either upload the compiled PDFs, or add the `.tex` sources and
compile them as separate standalone projects. The drawings' natural widths
match their print widths, so fonts render at the intended ~8 pt.

