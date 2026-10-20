# Figures, video and stronger results for the UAGS paper
2026-10-20 | root: /home/thanhnc/gauss_ws/src/uags | git: master@c95a871 | [all reports](../INDEX.md)
Follows: [Complete the paper](../2026-10-17-paper-completion/README.md)
Verdict: IN PROGRESS -- design workflow running (inventory, 5 designs, critiques)
Key result: n/a
Attempts: n/a | Time: started 2026-10-20 14:50 JST

## 0 Context
- Ask (user, 2026-10-20): "in this step, we do not need to keep this in 8 pages, just try to improve results, qualitative
  results, teaser figure, pipeline figure, video, quantitative results".
- Rules kept: no tuning on test scenes (Replica, ScanNet and NOR test results are already seen; a method change must be chosen
  on validation and every extra test look disclosed); figure frames/scenes chosen by a rule written before rendering, or from
  validation scenes; double-anonymous (no names or lab marks in figures or video); GPU only through scripts/gpu_gate.sh.
- Machine: RTX 4070 Ti 12 GB (shared), i7-13700KF. No system ffmpeg: installed imageio-ffmpeg 0.6.0 (bundled ffmpeg 7.0.2)
  into ~/venvs/uags with `uv pip install --no-deps` (the venv has no pip); torch stays 2.11.0+cu128; H.264 mp4 write/read test PASS.

## 1 Plan
| Step | What | Status |
|---|---|---|
| 1 | Design workflow (wf_7001c8b3-071): code + data inventory; designs for teaser, pipeline, qualitative, video, quantitative; one critic per design | running |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|

## 7 Open issues
- none yet

## 8 Files
- this folder
