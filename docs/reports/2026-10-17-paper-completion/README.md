# Complete the UAGS paper
2026-10-17 | root: /home/thanhnc/gauss_ws/src/uags | git: master@3eb0e86 | [all reports](../INDEX.md)
Follows: [Replica Table I](../2026-09-29-replica-tab1/README.md)
Verdict: IN PROGRESS -- new machine: environment being rebuilt; RIO10 replacement being researched
Key result: n/a
Attempts: n/a | Time: started 2026-10-17 10:44 JST

## 0 Context
- Ask (user, 2026-10-17): "now we need to complete the paper". Target: RA-L submission 2026-11-27 (8 pages, IEEEtran journal).
- Done before this run: Table I (Replica test, N=6, one shot) is in papers/main.tex; views figure and Table III validated.
- Left: views test (2 of 6 scenes missing), Table III test, qualitative figure, timing, abstract / intro / conclusion,
  and the change-detection evaluation (Table II), planned on RIO10 whose access form was never sent.
- Machine changed: RTX 4070 Ti 12 GB (was RTX 5070 Ti 16 GB), Ubuntu with Python 3.8 and no torch; the data folders
  were copied. GitHub push needs credentials on this machine (git fetch/push fail); commits stay local until then.

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Environment on the new machine | CPU suite `CUDA_VISIBLE_DEVICES="" python3 -m pytest tests/`; GPU tests test_gsplat_gpu, test_gs_fit, test_extractor_lseg, test_clip_real | 145 passed / 10 skipped (rejected-extractor weights, GPU-only); GPU 24 passed after one stale test was updated to the k = 2 default (ee7b4fe) | PASS |
| I1 cross-machine: views for room_0 re-mapped and re-evaluated on the RTX 4070 Ti vs the RTX 5070 Ti file | [repro.sh](logs/repro_room_0.log); compare script in this README | 114 AUSE values, largest change 0.0021 (x100); e_red/e_irr pixel counts within 0.01%; mIoU equal to 3 decimals ([json](logs/views_test_room_0_4070ti.json)) | PASS |

## 7 Open issues
- RIO10 access (user) or a replacement dataset (research running).
- Push: no GitHub credentials on this machine.

## 8 Files
(filled at the end)
