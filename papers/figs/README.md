# Manuscript figures

PDFs that `main.tex` includes (`\includegraphics`), and how each is made:

| PDF | Figure | Made by |
|---|---|---|
| `teaser.pdf` | Fig. 1 (`fig:teaser`) | `scripts/plot_nor_qual.py teaser` (NeRF-Object-Removal dumps) |
| `pipeline_v6.pdf` | Fig. 2 (`fig:pipeline`) | TikZ `pipeline_v6.tex`, thumbnails in `pipeline_thumbs/` (`scripts/render_pipeline_thumbs.py`) |
| `uncertainty_plane.pdf` | `fig:plane` | TikZ `uncertainty_plane.tex` |
| `qual.pdf` | `fig:qual` | `scripts/render_qual.py` |
| `views.pdf` | `fig:views` | `scripts/plot_views.py` |
| `qual_compare.pdf`, `supp_compare_all.pdf` | `fig:qual-compare`, `fig:supp-grid` | `scripts/qual_compare.sh` |
| `qual_change.pdf`, `qual_query.pdf` | `fig:qual-change`, `fig:qual-query` | `scripts/plot_nor_qual.py` |
| `calibration.pdf` | `fig:calibration` | `scripts/plot_calibration.py` |

`pipeline_v2.tex/.pdf` is not in the paper: `scripts/make_video.py` builds the video's method segment from it.
Build a TikZ figure from `papers/`: `pdflatex -output-directory <dir> figs/<name>.tex`.
Large raster images inside the PDFs can be re-encoded with `scripts/shrink_figs.py`.
