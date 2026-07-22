# Manuscript figures (TikZ)

Each figure is a standalone TikZ document compiled to the PDF that
`main.tex` includes via `\includegraphics`:

| Source | PDF | Used in | Width |
|---|---|---|---|
| `pipeline.tex` | `pipeline.pdf` | Fig. 1 (`fig:pipeline`, `figure*`) | two-column (`\textwidth`) |
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

`pipeline_old_placeholder.pdf.bak` is the pre-TikZ auto-generated placeholder,
kept for reference; safe to delete.
