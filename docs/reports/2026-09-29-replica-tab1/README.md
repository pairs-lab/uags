# Replica Table I: open-vocabulary segmentation and calibration
2026-09-29 | root: /home/thanhnc/gauss_ws/src/uags | git: master@5b21781 | [all reports](../INDEX.md)
Follows: [Data, baselines and evaluation infrastructure](../2026-09-29-data-and-baselines/README.md)
Verdict: IN PROGRESS -- validation stage queued behind USGLoc on the shared GPU (fit room_2, map office_0 + room_2, fit temperatures)
Key result: n/a
Attempts: n/a | Time: started 2026-09-29 13:49 +07

Running: `UAGS_GPU_POLL_S=15 STAGE=val scripts/replica_tab1.sh` (at 5b21781, mapping started 15:15:58) | /home/thanhnc/gauss_ws/data/uags_results/tab1/stage_val.log

## 0 Context
- Ask: "continue to complete our paper" -> fill Table I (plan section 3) with measured numbers.
- Protocol (plan): Replica, Semantic-NeRF renders; map on Sequence_1 every 5th frame (180 frames), evaluate on
  Sequence_2 every 10th frame (90); validation scenes office_0, room_2 fit one temperature per row (P2: its 4
  parameters); the 6 test scenes (room_0, room_1, office_1-4) run ONCE with those temperatures.
- Features: LSeg (chosen on validation, [attempt 05](../2026-09-29-data-and-baselines/attempts/05-extractor-study.md)),
  PCA-64 fit on 20 mapping frames, 5-template ensemble; vocabulary = classes present in each scene's evaluation frames.
- Rows: ours P0/P1/P2, ours-scalar, averaging (GradBP), NormLift (Sec. 3.3), view count, CSS (Dirichlet over the
  present classes, given at mapping time), all fed by the same per-view sums.
- 3DGS: scripts/fit_gs.py, 30k iterations, max 1M Gaussians, 640x480; office_0: held-out PSNR 41.5 dB,
  SSIM 0.989, 620,290 Gaussians, 205 s (map meta in /home/thanhnc/gauss_ws/data/uags_maps/office_0_seq1.pt).
- Pipeline: [scripts/replica_tab1.sh](../../../scripts/replica_tab1.sh) (STAGE=val, then STAGE=test once).
- Machine: RTX 5070 Ti 16 GB (shared; GPU steps through scripts/gpu_gate.sh).

## 1 Findings
| Finding | Reason | Evidence |
|---|---|---|
| Mapping cannot run beside a USGLoc stage | LSeg (fp32, 640x480, one frame) needs > 3.3 GB: a 3-keyframe probe capped at 3500 MiB ran out of memory in the DPT head at the first PCA frame (14:20); USGLoc stages hold 6.5-9.2 GB and a policy server 3.8 GB of the 16 GB | [logs/memprobe_lseg_3p5gb.log](logs/memprobe_lseg_3p5gb.log) |

So the gate stays at 6000 MiB (+2048 headroom) and the stage waits for gaps between USGLoc runs.

| Mapping OOM at 15:13 (cap 6000 MiB) | LSeg activations + field + baselines + renders peaked together in the first mapping frame (5.6 GB allocated) | [logs/oom_mapping_office_0_1513.log](logs/oom_mapping_office_0_1513.log) |
| Fix: extract all tokens first (CPU, fp32, same values), free LSeg, then build the field (commit 5b21781) | 3-keyframe smoke test: extraction peak 4.9 GB, mapping peak 4.8 GB (was their sum) | [logs/smoke_mapping_3kf.log](logs/smoke_mapping_3kf.log) |

| Change before the first run | Reason | Commit |
|---|---|---|
| First queued run (13:48) stopped before it did any work; relaunched | add the "2D features, per frame" row of Table I (LSeg on each evaluation frame, same token -> upsample -> PCA path as mapping, own temperature, covers every pixel) so all rows come from one evaluation | 01f39a7 |
| All-zero statistics saved as None | disk has 25 GB free (98% used); a field with its centred channel and change buckets is ~1 GB | 01f39a7 |
| NormLift 1 - C_inter channel, rbar estimator, flat pooled null | paper review (Table I / III rows) | 4829617 |

## 2 Baseline
3DGS maps (Sequence_1, every 5th frame = 180 frames, 157 train / 23 held out, i % 8 == 0; 30k iterations; 640x480):

| Scene | Held-out PSNR | SSIM | Gaussians | Train time | Peak GPU | Source |
|---|---|---|---|---|---|---|
| office_0 | 41.53 dB | 0.989 | 620,290 | 205 s | 756 MiB | fitter agent run (copied from its report, not re-checked; map meta in /home/thanhnc/gauss_ws/data/uags_maps/office_0_seq1.pt) |
| room_2 | 37.05 dB | 0.977 | 954,119 | 268 s | 900 MiB | [logs/fit_room_2.log](logs/fit_room_2.log) |

Held-out frames sit between training frames of the same trajectory: the PSNR measures interpolation, not new viewpoints.
The semantic baselines (averaging, NormLift, view count, CSS, 2D per frame) are rows of Table I, below.

## 5 Verification
(filled at the end)

## 7 Open issues
(filled at the end)

## 8 Files
(filled at the end)
