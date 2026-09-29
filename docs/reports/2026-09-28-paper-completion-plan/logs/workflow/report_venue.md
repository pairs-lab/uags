# Venue report (label: venue): UAGS RA-L submission

All pages were checked on 2026-09-28 with WebFetch. `curl` to ieee-ras.org returned a Cloudflare 403 block page. Scratch files are in `/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/wf/venue/`. I did not write to any project file.

## A. RA-L rules

| Item | Official rule (quoted or close paraphrase) | Source | UAGS status |
|---|---|---|---|
| Page limit | "six pages long in two-column IEEE format ..., **including figures and references**, with up to two extra pages allowed at a charge ($175 per extra page)", so 8 pages at most. The limit covers "everything ... figures, tables, and any appendices". | https://www.ieee-ras.org/publications/ra-l/information-for-authors-ra-l | Not measured. No LaTeX on the machine (`~/.local/bin/tectonic` does not exist). |
| Class for the first submission | RAS conference format: `\documentclass[letterpaper, 10 pt, conference]{ieeeconf}`. Word users use `ieeeconf_letter.dot`. | same | **Wrong now.** main.tex:13 uses `\documentclass[journal]{IEEEtran}`. It must be ported. The IEEEtran-only macros also need checking: `\IEEEPARstart` (main.tex:81), the `IEEEkeywords` environment (main.tex:73-76) and `\markboth` (main.tex:45). I have not checked whether ieeeconf supports them. |
| Class for the final version | `\documentclass[letterpaper, 10 pt, journal, twoside]{ieeetran}` | same | Keep the IEEEtran version for camera-ready. |
| Double-anonymous review | In force from **Jan 1, 2025** for RAS journals, including RA-L. No names or affiliations anywhere, including the appendix and supplementary files. Acknowledgments and funding only after acceptance. Strip file metadata. Use neutral self-citation ("In previous work…"). arXiv is allowed and reviewers should not search for it. "No links to external websites that reveal identity (e.g., YouTube, GitHub, ...)". Blur faces and remove lab logos in videos. Violations "can result in the paper being desk rejected". The RA-L page adds: "leave space for an author byline, but do not include the author information"; "Do not use a cover page". | https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process ; RA-L page above | See section E. |
| Video and supplementary | "Only one video is allowed per submission". A "single zipped archive file (max 50 MB)" with an ASCII `ReadMe.txt` and a `Summary.txt`. "DO NOT submit supplemental text or figures as multimedia materials". | RA-L information-for-authors page | No video planned yet. |
| Keywords | "minimum of 2 up to a maximum of 5 (recommended)" | same | main.tex:74-75 has 5 keywords, including "localization" (see section C). |
| Review timeline | First decision "within three months from submission". Revision "within 30 days". Final decision "within 5 months from initial submissions". Published within 6 months or rejected. Average submission-to-online time is about 4 months. | RA-L information-for-authors page; https://www.ieee-ras.org/publications/ra-l/ | — |
| Prior or dual submission | RA-L "will not consider submissions that are currently under review for a conference". IEEE: "Authors shall inform editors of multiple submissions" and "must disclose whether there are prior publications ... by the authors that are similar, whether published or submitted", with a statement of "how the new submission differs". An arXiv preprint needs the notice "This work has been submitted to the IEEE for possible publication". | RA-L page; https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/ | This applies to UAGS and USGLoc (section C). |
| Open-access fee | "US$2800 for articles submitted in 2026" | RA-L page | Not checked whether open access is optional. |

## B. Deadlines

There is **no separate "RA-L + ICRA 2027" or "RA-L + IROS 2027" submission deadline**. IROS 2026 said: "IROS no longer offers an RA-L+IROS submission option" (https://2026.ieee-iros.org/contribute/call-for-papers/). What counts now is the RA-L transfer window. "The paper must be transferred within 270 days of acceptance", and a paper can be presented at one conference only (https://www.ieee-ras.org/publications/ra-l/).

| Option | Official dates | Can UAGS make it? |
|---|---|---|
| ICRA 2027 (Seoul, May 24-28, 2027) | RA-L transfer window **Mar 1, 2026 to Dec 31, 2026**. The ICRA page gives a transfer deadline of "31st December 2026". The direct ICRA paper deadline was Sep 16, 2026 (extended) and has passed. Sources: ICRA 2027 call (https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/) and the RA-L page. | **No.** It needs acceptance by Dec 31, 2026, but the first decision alone takes up to 3 months and UAGS has no results yet. |
| IROS 2027 (Florence, Sep 26 - Oct 1, 2027) | RA-L transfer window **Aug 1, 2026 to Apr 30, 2027**. The direct IROS paper deadline is **Mar 1, 2027** (https://www.ieee-ras.org/event/call-for-papers-paper-submission-deadline-iros-2027-...). I found no IROS 2027 call page, so the "no RA-L+IROS option" rule is carried over from IROS 2026 and not checked for 2027. | **Yes, if UAGS is submitted by about Nov 30, 2026.** That date comes from the 5-month final-decision guarantee. I recommend aiming for Nov 15 to leave a buffer. USGLoc also aims at IROS 2027 (plan:315-316). |
| Fallbacks (RA-L page) | CASE 2027: Aug 1, 2026 - Apr 30, 2027. RO-MAN 2027: Sep 1, 2026 - Apr 30, 2027. AIM 2027: Sep 15, 2026 - Jun 1, 2027. Humanoids 2027: Oct 1, 2026 - Jun 30, 2027. | Possible if IROS is missed. |

## C. Overlap between UAGS and USGLoc, and the recommended boundary

A text search of the USGLoc draft for vMF, von Mises or "uags" found nothing. So the UAGS method core (vMF conjugate BKI, the gradient trick, the (n_eff, r̄) epistemic/aleatoric pair, EFE) does not appear in USGLoc. The overlap is in the claims, the experiments and the positioning.

| # | Overlap | UAGS (papers/main.tex) | USGLoc (paper/usgloc_ral_draft.tex) | Risk |
|---|---|---|---|---|
| 1 | Same headline claim: map uncertainty replaces Chu & Kim's stable-object selection, with "no dynamic-object heuristics or language model" | 67-68, 128-129, 146-148, 367-369 | 88 ("No stage requires dynamic-object heuristics or a language model"), 92, 117, 207 ("replaces the binary robust mask of [chukim]") | **High** |
| 2 | A moved object is detected from conflicting observations | Aleatoric / low r̄ with high n_eff, consolidation (~326-331, 358-366) | Yager conflict goes into vacuity, "signature of an object that moved" (150-156). Its limitation at 471 asks for "a temporal conflict model", which is exactly UAGS's idea. | Medium-high |
| 3 | Uncertainty-aware render-and-compare pose refinement on 3DGS | "robust subset used for localization" (367); `usplat/localization.py:60` `refine_pose(..., init_c2w, robust_mask)` | Vacuity-weighted refinement (196-207) | **High** |
| 4 | Same localization benchmark, baselines and metrics | 7-Scenes (395-396); Marepo, ACE, Chu & Kim (402-406, 474-476); median cm/°, 5cm/5°, 2cm/2° (408-411); tab:loc (462-480) | 7Scenes and 12Scenes (256); Marepo, ACE, Chu & Kim (258); same metrics (260) | **High** |
| 5 | The same group contradicts itself | "a Dirichlet treatment does not apply" to open vocabulary (104-105) | "Dempster–Shafer belief over open-vocabulary semantics" (72), yet it uses a closed 25-class vocabulary (`usgloc/mapping.py:47-53`; p_i ∈ Δ^{C-1} at 137) | Medium. A shared reviewer will notice. |
| 6 | Uncertainty from observation count per Gaussian | u_epi = 1/(1+n_eff) (~322) | Observation-scaled evidence; an unobserved Gaussian has vacuity 1 (262) | Medium |
| 7 | Front end and maps | CLIP ViT-B/16 and gsplat; needs a pre-trained 3DGS `--ply` (README:52-54) | Same CLIP and gsplat, plus its own map training and distillation pipeline | Low if disclosed. High if UAGS silently reuses USGLoc maps, its EDL head or its numbers. |
| 8 | Similar titles and keywords, so the same editor/reviewer pool | "Uncertainty-Aware Open-Vocabulary Semantic Gaussian Splatting …"; keyword "localization" (74) | "Uncertainty-Aware Semantic Visual Localization from 3D Gaussian Splatting" (58) | Medium |
| 9 | E2-BKI cited in two different versions | arXiv:2509.11964, marked "TODO verify" (553-556) | RA-L 11(3):2378-2385 (`paper/references.bib:36-42`) | Low. Pick one verified entry for both. |

**Boundary.** UAGS covers the map-side semantic belief: what the map knows about language semantics and how that knowledge ages when the scene changes. USGLoc covers the query-side pose belief.

**UAGS must not:**
- Claim localization accuracy or a calibrated pose. Remove "localization" from the abstract (69), the contributions, the keywords (74) and the conclusion.
- Report a 7-Scenes or 12-Scenes localization table against Marepo, ACE or Chu & Kim.
- Say it "replaces Chu & Kim's robust mask for localization". That is USGLoc's claim.
- Reuse USGLoc text, figures, its EDL head, its 25-class vocabulary, its maps' distilled semantics, or any USGLoc number (for example 1.05 cm).

**UAGS may share, with a citation or disclosure:**
- The CLIP ViT-B/16 front end, gsplat and the dataset loaders.
- 3DGS geometry fitted with USGLoc's `train_map.py`, but only if the paper says so. Fitting UAGS's own maps is better.
- Related-work statements on Chu & Kim, E2-BKI and CSS, written fresh.
- USGLoc cited in the third person as concurrent work: "[X] concurrently propagates evidential map uncertainty to a pose posterior; we instead …".

**Localization table: remove it.** Drop tab:loc (main.tex:456-480) and the ATE column of tab:abl (489-505), for these reasons:
- It collides with USGLoc's main result.
- The dynamic dataset does not exist yet (398 is a TODO).
- `refine_pose` needs an initial pose (localization.py:60), so comparing it with prior-free APR and SCR methods is unfair.
- 7-Scenes frames are colour, depth and pose only (`ls data/7scenes_source/chess/seq-01 | sed 's/frame-[0-9]*\.//' | sort | uniq -c` gives 1000 each of color.png, depth.png and pose.txt). It has no semantic ground truth for tab:seg and no change labels.
- It costs space inside a 6-page limit that includes references.

**Replace it** with a "map under scene change" table:
- Segmentation accuracy on changed vs unchanged regions, before and after consolidation.
- AUSE of the aleatoric channel against a change mask.
- The rate of stale content left in the map.

If the team wants a downstream check, add one ablation row: pose refinement from a perturbed start, with vs without the consolidated subset, on the same change data, with no state-of-the-art baselines.

**Disclosure.** Tell the RA-L editor about the other submission under the IEEE multiple-submission rule, whichever paper goes second. I have not checked which PaperCept field to use for this.

Also tell the USGLoc owners that "open-vocabulary" at usgloc_ral_draft.tex:72 conflicts with UAGS:104-105. I am read-only there.

## D. GPU schedule

**Snapshots** (from `nvidia-smi --query-gpu=... --format=csv` and `--query-compute-apps`):

| Time (09-28) | Used / free (MiB) | Processes |
|---|---|---|
| 22:19:34 | 7202 used of 16303 | PID 923057, **user `sang`**, `serve/policy_server.py --model libero90`, up 10 days 19 h, 3852 MiB, 1% utilisation. **This is not USGLoc.** It is resident all the time. |
| | | PID 2367517, USGLoc `run_7scenes.py` office, 3240 MiB |
| ~22:26 | 14074 used / 1761 free, 98% | Added: PID 2377831 `exp_floor_matcher.py` (MASt3R), 6866 MiB |
| 22:27:28 | 9182 used / 6653 free, 100% | — |

**USGLoc queue**, read from `ps` and the scripts in the other session's scratchpad:

| Batch | What it runs | Memory gate | Estimated end (my estimate) |
|---|---|---|---|
| `batch_solver.sh` (PID 2367511, 22:11) | 6 runs: office and chess × 3 arms | free ≥ 3500 MiB | About 00:00-00:15 on 09-29. Per-run times: office took 15.8 min (final2_seed2) and 16.6 min (lmret_seed0) as the sum of per-query `t_encode_s + t_localize_s`; chess took 13.1 min (final2_seed2). final2_seed2 output files are 16-20 min apart. |
| `batch_m3stage.sh` (22:17) | 3 scenes × 5 matchers, about 1 min per matcher (stairs JSONs 22:22:05 to 22:25:32) | free ≥ 7000 MiB | About 22:45 |
| `batch_tail.sh` (PID 3276401, dormant since 09-22) | Checks every 300 s for "=== MAST3R SEED 0 DONE" (count 0; log unchanged since 09-22 23:26). If the line appears, it runs the 12scenes_v2 and softmax chains. | — | Length not measured |

The USGLoc session started new batches at 21:23, 22:09, 22:11 and 22:17. Its plan said "≈1.5–2 weeks to submission" from 09-18 (plan:308-314), and it is still tuning on 09-28. Expect heavy GPU use until at least mid-October (estimate), and again during its 30-day revision round.

**UAGS rules:**
1. **Right now: do not start anything.** Free memory went down to 1761 MiB.
2. **Gate every job.** Before starting, require free ≥ (UAGS peak + 4 GB), or + 7 GB while a MASt3R batch is queued. Also require that `pgrep -f 'run_7scenes.py|exp_floor_matcher.py|train_map.py|train_encoders.py'` is empty for long jobs. USGLoc's own gates are 3500, 4000 and 7000 MiB, and they give up and launch anyway after 960 × 20 s (5.3 h). So holding memory stalls USGLoc and then risks an out-of-memory crash.
3. **Cap memory** with `torch.cuda.set_per_process_memory_fraction`, a smaller `--channel-chunk` and `--downscale 4`. The UAGS peak on a real scene is **not measured**. Measure it once, on one small scene, when the GPU is idle.
4. **Timing.** Never measure UAGS runtime while USGLoc runs, and never run UAGS during USGLoc's runtime-table runs (plan 3.5). Either way the published timings would be wrong. Say in the hardware note that `sang`'s 3.85 GB process is always present.
5. **Do CPU work now:** install `plyfile` (the only test failure), port to ieeeconf, build the scene-change protocol, and download data. Disk has 48 GB free, 95% full (`df -h /home`).
6. **Proposed calendar:** gated small jobs at night until USGLoc submits (about the first week of October), then the main runs, writing from about Nov 9, and submission by Nov 15 (hard limit Nov 30).
7. Suggest to the USGLoc session a shared `flock` lock file for the GPU. I cannot edit its scripts.

## E. Author and anonymity TODOs in papers/main.tex

| Line | Now | For the first submission | For camera-ready |
|---|---|---|---|
| 2, 13 | "IEEEtran journal class" | Switch to ieeeconf with `letterpaper, 10 pt, conference` | IEEEtran journal twoside |
| 39 | "First Author, Second Author, and Senior Author" | Remove, leave space for the byline | Real names and ORCIDs. **The author list is a team decision.** The git author is `thanhnguyencanh`. The USGLoc block (usgloc_ral_draft.tex:60-63, commented out) shows the group format and grants (JST SPRING, AOARD). Do not copy its line 61, which reads "Received 04 May 2026; accepted 04 August 2026 … Editor J. Civera" and looks like it came from another paper. |
| 40 | received date and funding TODOs | Remove. Funding only after acceptance. | Fill in |
| 41-42 | affiliation and e-mail TODOs | Remove | Fill in |
| 43 | DOI TODO | Remove | IEEE assigns it |
| 45 | `\markboth{... Month, Year}` | Remove | Set by IEEE |
| 522-523 | Acknowledgment TODO | Remove, or hide behind an `\ifanonymous` switch | Fill in |
| — | Self-citation | Add USGLoc in the third person, anonymised if it is not public | Real citation |
| — | Links | None today. Link only an anonymous.4open.science mirror, never `github.com/pairs-lab/uags`. Both repos return 404 on `curl https://api.github.com/repos/pairs-lab/{uags,usgloc}` (private or missing). | Real repo |
| — | Metadata | `pdfinfo figs/*.pdf` shows only Creator tectonic and Producer xdvipdfmx, with no Author. `grep -rniE 'thanh\|jaist\|pairs\|chong\|/home/' papers/` found nothing. Run `pdfinfo` again on the final compiled PDF. | — |
| — | Admin | Collect PaperCept PINs for every author. Get co-author sign-off. Pick 2-5 keywords. | — |
| Other TODOs | 70, 391-517 | Result TODOs. Bib entries marked "TODO verify": 553 (E2-BKI), 558-559 (PhysGS, authors missing), 587 (gsplat), 595 (Feature3DGS). | — |