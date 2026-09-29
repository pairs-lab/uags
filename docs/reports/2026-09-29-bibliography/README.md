# Bibliography moved to ref.bib
2026-09-29 | root: /home/thanhnc/gauss_ws/src/uags | git: master@ab28fb4 +dirty (uncommitted) | [all reports](../INDEX.md)
Verdict: DONE -- ref.bib (49 entries, 24 cited) verified entry by entry; the paper compiles with BibTeX, 0 errors, 0 undefined citations
Key result: n/a
Attempts: n/a | Time: 2026-09-29 (started after the user's request)

## 0 Context
- Ask (user, 2026-09-29): "use ref.bib for bib file".
- Scope: papers/ref.bib (new) and papers/main.tex (inline thebibliography removed; \bibliographystyle{IEEEtran}
  \bibliography{ref}; one key renamed: liu2025physgs -> chopra2026physgs because the authors were wrong).
- Source of the corrected data: the related-work audit of 2026-09-28
  ([report_sota.md](../2026-09-28-paper-completion-plan/logs/workflow/report_sota.md) section 4).

## 1 Findings
| # | Finding | Evidence |
|---|---|---|
| 1 | All 49 entries verified (Crossref by DOI, CVF, PMLR, NeurIPS, JMLR, OpenReview, arXiv, Open Library); 24 changed (full author lists, DOIs, pages, one title: VG-Mapping), 25 unchanged | [logs/verify_report.md](logs/verify_report.md) |
| 2 | PhysGS key renamed liu2025physgs -> chopra2026physgs (no author named Liu) | papers/main.tex line with \cite{chopra2026physgs} |
| 3 | Two sources disagree on the ActiveSGM pages (Crossref 37293-37320 vs proceedings.neurips.cc 33134-33161); kept the DOI's pages | [logs/verify_report.md](logs/verify_report.md) note 1 |
| 4 | Older CVF papers (ScanNet, RIO, Semantic-NeRF) carry CVF pages without a DOI because the IEEE pages differ | [logs/verify_report.md](logs/verify_report.md) note 3 |
| 5 | Compile: exit 0, BibTeX clean, 0 undefined citations/references, no overfull boxes; the body font falls back to Latin Modern under XeTeX without T1/mathptmx, which inflates the page count (6 vs 5 pages) | [logs/compile_report.md](logs/compile_report.md), [logs/main.blg](logs/main.blg) |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| every \cite key exists in ref.bib | python regex over main.tex and ref.bib | 24 cited, 49 entries, 0 missing | PASS |
| I3 references checked by DOI/arXiv | workflow agent refs:verify | 49/49 verified | PASS [logs/verify_report.md](logs/verify_report.md) |
| paper compiles with BibTeX | `tectonic main.tex --outdir build` (tectonic 0.17.0, third_party/tectonic) | exit 0, 0 errors, 0 undefined | PASS [logs/compile_report.md](logs/compile_report.md) |

## 7 Open issues
- Font: use T1 + mathptmx (done in the paper revision's preamble) so XeTeX builds use Times; the old papers/main.pdf has the same fallback.
- The paper text itself is being revised in [2026-09-29-paper-revision-1](../2026-09-29-paper-revision-1/README.md).

## 8 Files
- papers/ref.bib (new), papers/main.tex (bibliography block replaced); original main.tex: [logs/main_tex_before.tex](logs/main_tex_before.tex)
