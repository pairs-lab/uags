I checked all 49 entries in `papers/ref.bib` against the sources listed below. All 49 are verified, and no `% UNVERIFIED` comments are left. Keys and the entry count (49) are unchanged. Every key cited in `main.tex` is still in the bib, and `main.tex` was not touched. No LaTeX or BibTeX is installed on this machine, so I could not compile the paper. I checked the file with a Python parser instead: braces balance, every entry parses, and no "and others" remains.

## Changes

"Crossref" means `https://api.crossref.org/works/<DOI>`. "CVF" means the paper's page on openaccess.thecvf.com.

| key | field changed | old -> new | source URL |
|---|---|---|---|
| (file header) | comment | added a 3-line note dated 2026-09-29 on how entries were checked | n/a |
| banerjee2005vmf | number | (none) -> 46 | https://www.jmlr.org/papers/v6/banerjee05a.html |
| ye2024gsplat | author | Ye, Vickie and others -> all 11 authors (Ye, Li, Kerr, Turkulainen, Yi, Pan, Seiskari, Ye, Hu, Tancik, Kanazawa) | https://www.jmlr.org/papers/v26/24-1476.html |
| qin2024langsplat | doi | (none) -> 10.1109/CVPR52733.2024.01895 | https://openaccess.thecvf.com/content/CVPR2024/html/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.html + Crossref |
| zhou2024feature | author, doi | Zhou, Shijie and others -> all 10 authors (…, Kadambi, Achuta); doi -> 10.1109/CVPR52733.2024.02048 | https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.html + Crossref |
| wu2024opengaussian | author, doi | Wu, Yanmin and others -> all 11 authors (…, Zhang, Jian); doi -> 10.52202/079017-0604 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html |
| chen2024marepo | author, doi | Chen, Shuai and others -> Chen, Cavallari, Prisacariu, Brachmann; doi -> 10.1109/CVPR52733.2024.01953 | https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Map-Relative_Pose_Regression_for_Visual_Re-Localization_CVPR_2024_paper.html |
| brachmann2023ace | doi | (none) -> 10.1109/CVPR52729.2023.00488 | CVF CVPR2023 page + Crossref |
| chu2026semantic | author | Chu, B.-T.-H. and Kim, G.-W. -> Chu, Ba-Tuan-Hoang and Kim, Gon-Woo | https://api.crossref.org/works/10.1109/LRA.2025.3643336 |
| shi2024legaussians | doi | (none) -> 10.1109/CVPR52733.2024.00510 | CVF CVPR2024 page + Crossref |
| sheppard2026slimvdb | author | Sheppard, Anja and others -> all 9 authors (Sheppard, Ewen, Wilson, Sethuraman, Adewole, Li, Chen, Vasudevan, Skinner) | https://api.crossref.org/works/10.1109/LRA.2026.3652875 |
| jiang2026mif | author, doi | initials only -> Jiang, Peifeng; Liu, Hong; Jin, Jin; Wang, Wenshuai; Li, Xia; doi -> 10.15607/RSS.2026.XXII.023 | https://www.roboticsproceedings.org/rss22/p023.html |
| chen2025activesgm | author, doi | Yin, **Hongbo** -> Yin, **Hairong**; doi -> 10.52202/085713-1113 (page conflict, see note 1) | https://proceedings.neurips.cc/paper_files/paper/2025/hash/2fa561fb73e30a2396c268253eeca191-Abstract-Conference.html + Crossref |
| ackermann2025clsplats | author, doi | Ackermann, Jan and others -> all 8 authors (…, Guibas, Leonidas J.; Peng, Songyou); doi -> 10.1109/ICCV51701.2025.00732. The title was already correct. | https://openaccess.thecvf.com/content/ICCV2025/html/Ackermann_CL-Splats_Continual_Learning_of_Gaussian_Splatting_with_Local_Optimization_ICCV_2025_paper.html |
| zeng2025gaussianupdate | author, doi | Zeng, Lin and others -> all 7 authors (…, Cui, Zhaopeng); doi -> 10.1109/ICCV51701.2025.02393. The title was already correct. | https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_GaussianUpdate_Continual_3D_Gaussian_Splatting_Update_for_Changing_Environments_ICCV_2025_paper.html |
| he2026vgmapping | title | "VG-Mapping: Variation-Aware 3D Gaussians for Online Semi-Static Scene Mapping" -> "VG-Mapping: Variation-Aware Density Control for Online 3D Gaussian Mapping in Semi-Static Scenes" | https://api.crossref.org/works/10.1109/LRA.2026.3723741 |
| wald2020rio10 | pages, doi | (none) -> 467--487; doi -> 10.1007/978-3-030-58571-6_28 | https://api.crossref.org/works/10.1007/978-3-030-58571-6_28 |
| wald2019rio | pages | (none) -> 7658--7667 (page numbers printed in the CVF PDF) | https://openaccess.thecvf.com/content_ICCV_2019/papers/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.pdf |
| straub2019replica | author | Straub, Julian and others -> all 30 authors | https://arxiv.org/abs/1906.05797 |
| zhi2021semanticnerf | pages | (none) -> 15838--15847 (CVF numbering) | https://openaccess.thecvf.com/content/ICCV2021/html/Zhi_In-Place_Scene_Labelling_and_Understanding_With_Implicit_Scene_Representation_ICCV_2021_paper.html |
| lan2024clearclip | pages, doi | (none) -> 143--160; doi -> 10.1007/978-3-031-72970-6_9. The authors were already correct. | https://api.crossref.org/works/10.1007/978-3-031-72970-6_9 |
| zhou2022maskclip | pages, doi | (none) -> 696--712; doi -> 10.1007/978-3-031-19815-1_40 | https://api.crossref.org/works/10.1007/978-3-031-19815-1_40 |
| cherti2023reproducible | author, pages, doi | Cherti, Mehdi and others -> all 9 authors (…, Jitsev, Jenia); pages -> 2818--2829; doi -> 10.1109/CVPR52729.2023.00276 | https://openaccess.thecvf.com/content/CVPR2023/html/Cherti_Reproducible_Scaling_Laws_for_Contrastive_Language-Image_Learning_CVPR_2023_paper.html |
| ilg2018uncertainty | author, pages, doi | Ilg, Eddy and others -> all 7 authors (Ilg, Çiçek, Galesso, Klein, Makansi, Hutter, Brox); pages -> 677--693; doi -> 10.1007/978-3-030-01234-2_40 | https://api.crossref.org/works/10.1007/978-3-030-01234-2_40 |

## Verified with no change (25 entries)

- **Crossref by DOI:** kerbl2023, mildenhall2020nerf, wilson2025css, gan2020bayesian, kim2025e2bki, lee2026lamp, nagami2025vista, friston2010, parr2022active, glocker2013, wilson2025latentbki, joseph2025gradbp, mardia1976bayesian, lu2025gscd, mardia2000directional.
- **PMLR:** radford2021 (v139 p. 8748–8763), guo2017calibration (v70 p. 1321–1330).
- **NeurIPS proceedings:** vegabrown2014, kendall2017uncertainties.
- **CVF:** chopra2026physgs (CVPR 2026, p. 18980–18990), dai2017scannet (pages 5828–5839, read from the numbers printed in the CVF PDF).
- **arXiv:** maggio2025bayesian (2503.05949 v1; I found no published version), zang2026normlift (2609.18898 v1).
- **BMVC 2025 proceedings:** cheng2025occam (paper 694; there are no pages and no DOI).
- **OpenReview:** li2022lseg (forum RriDjddCLN).
- **Open Library:** kish1965survey (J. Wiley, New York, 1965).

## Still unverified

None.

## Notes for the lead

1. **chen2025activesgm pages: the two official sources disagree.** Crossref, for the DOI 10.52202/085713-1113 (which resolves to proceedings.com), gives 37293–37320. The proceedings.neurips.cc BibTeX gives 33134–33161, volume "38, Main Conference". I kept 37293–37320 so the pages match the DOI.
2. **mardia2000directional:** Crossref gives the print date as 1999-01-03. Open Library lists the Wiley print (ISBN 9780471953333, Chichester) as 2000. I kept 2000, which is the usual citation year.
3. **No DOIs for three older CVF papers.** dai2017scannet, wald2019rio and zhi2021semanticnerf have IEEE Xplore DOIs, but their IEEE page numbers differ from the CVF page numbers. The DOIs are 10.1109/CVPR.2017.261 (IEEE pages 2432–2443), 10.1109/ICCV.2019.00775 (7657–7666) and 10.1109/ICCV48922.2021.01554 (15818–15827). I kept the CVF pages and left the DOIs out. For CVPR/ICCV 2023 and later the two page sets match, so those entries now carry DOIs. chopra2026physgs has no IEEE record yet.
4. **Name spellings follow the sources.** I kept the spelling each source uses: Kulhanek (not Kulhánek) in the CVF/IEEE records, and Raul Mur-Artal and Jesus Briales (no accents) on arXiv.
5. **Very long Replica author list.** With all 30 authors, IEEEtran will print every name. If you want "et al.", set the IEEEtranBSTCTL author limit in `main.tex`; I did not touch that file.
6. **Title capitalisation only.** Some titles differ from their sources only in capital letters (for example "Through" vs "through", "Multi-Modal" vs "Multi-modal"). I left these unchanged.
7. **Keys whose year differs from the entry year.** lee2026lamp has year 2025, and kim2025e2bki / nagami2025vista have year 2026. The years are correct; the keys were kept as instructed.

The original file is backed up at `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/bib/ref.orig.bib`. As a subagent I wrote no report under `docs/reports/`.

Files:
- `/home/thanhnc/gauss_ws/src/uags/papers/ref.bib` (edited)
- `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/bib/ref.orig.bib` (backup of the original)