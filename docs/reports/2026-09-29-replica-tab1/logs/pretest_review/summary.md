# Pre-test review (workflow wf_ce10d633-f61)
6 reviewers -> 43 raw findings -> 30 merged. Verification stopped at the subagent session limit (45 of 100 agents failed: 'session limit'); fixer, P3 and critic did not run.

| id | severity | title | votes real/done |
|---|---|---|---|
| F01 | critical | Tab III re-mapped variants are scored against text projected with a different PCA basis | 0/0 |
| F02 | critical | No pipeline produces the Tab III AUSE columns defined in the caption (e_red / e_irr averaged over 10/30/90-frame fields) | 3/3 |
| F03 | major | AUSE of u_ale ranks undefined (low-support) pixels as most uncertain, so it largely scores the tau_min gate; baseline channels get no such tier and the paper does not state the rule | 1/1 |
| F04 | major | CSS gets the classes present in the evaluation frames' GT, but the paper and baselines.py docstrings say it gets the full vocabulary | 3/3 |
| F05 | major | CSS's flat prior alpha0 = 1 is unstated and untuned, yet it is about 26-28% of the mass-weighted Dirichlet mean and likely drives CSS's ECE lead | 1/1 |
| F06 | major | Stage guards only check that eval_val.json exists; the file on disk is a failed record with no betas, and eval_seg only warns | 3/3 |
| F07 | major | eval_seg.py saves no metrics until every scene is done, so a mid-run crash loses finished scenes and forces a second test pass | 3/3 |
| F08 | major | A rerun destroys earlier results: logs and JSON are overwritten, and nothing stops a second STAGE=test | 3/3 |
| F09 | major | Mapping does not use 180 frames: the near-duplicate pose filter keeps 176-180 per scene (and fewer for k-view fields); 3D-GS map fit on 157 | 3/3 |
| F10 | major | PCA centre sample pins about 3 GiB of GPU memory for the whole mapping run | 3/3 |
| F11 | major | GPU reservation makes peak_alloc_extract_mib and peak_reserved_mib report the reservation, not the job | 3/3 |
| F12 | major | Timing fields cannot fill the paper's run-time breakdown and overstate our per-keyframe cost | 3/3 |
| F13 | major | Field files carry no code version, and existing fields are reused without a check, so val and test fields can come from different code | 3/3 |
| F14 | minor | The std in the tables is the population std (ddof=0); the paper says only 'standard deviation' | 3/3 |
| F15 | minor | Paper says tau_cov is set on the validation scenes, but it is a fixed default (0.5) | 3/3 |
| F16 | minor | Output record lists the wrong prompt templates for the LSeg run | 3/3 |
| F17 | minor | The 'never beside USGLoc' gate policy lives only in an unrecorded environment variable; gate comments are stale | 3/3 |
| F18 | minor | run_mapping sets no seed, so PCA bases differ between fits (k-view fields add basis noise to e_red / e_irr) | 1/1 |
| F19 | minor | make_tables.py does not match the tab:replica layout and does not check its input | 2/2 |
| F20 | minor | Tab III 'Full method' temperatures will differ from Tab I 'Ours' because the 2D row shifts the fitter's random subsets | 0/0 |
| F21 | minor | Ablation rows exist only for P0 and P2, but the paper leaves the full method's predictive open (P0, P1 or P2) | 1/1 |
| F22 | minor | The 2D row covers every pixel, while the 3D rows count their uncovered pixels as wrong; the paper does not say this | 0/0 |
| F23 | minor | The views figure's M=180 point (e_irr column) is not produced by replica_views.sh | 0/0 |
| F24 | minor | Field and map saves are not atomic and have no disk check, so a truncated file would be reused | 0/0 |
| F25 | minor | replica_tab3.sh hides the mapping log when a variant mapping fails | 0/0 |
| F26 | minor | STAGE is not validated, so any value other than 'val' runs the test branch | 0/0 |
| F27 | minor | A failed reservation crashes a job that would otherwise fit | 0/0 |
| F28 | minor | frozen_on never shares channels that contain NaN (u_ale), unlike its docstring says | 0/0 |
| F29 | minor | A stale LaTeX comment says normlift_1-Cinter is not exposed | 0/0 |
| F30 | minor | Near-zero-evidence Gaussians: mu* is not unit and the support is 0 instead of W1 | 0/0 |
