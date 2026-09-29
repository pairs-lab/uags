# Data, baselines and evaluation infrastructure
2026-09-29 | root: /home/thanhnc/gauss_ws/src/uags | git: master@ab28fb4 +dirty (uncommitted) | [all reports](../INDEX.md)
Follows: [Method completion: fixes F1-F9](../2026-09-28-method-completion/README.md)
Verdict: IN PROGRESS -- Replica ready; extractor chosen (LSeg, +9.8 mIoU); modules built, reviewed, wired; first Replica map being fitted
Key result: change detection on the 7-Scenes proxy (real CLIP, office unseen, q = 0.999): 49-53% of changes flagged at 1.8% false alarms; ranking equals the feature-residual baseline (0.861-0.866 vs 0.861-0.867 AUROC) ([attempts/04](attempts/04-history-conditioned-calibration.md))
Attempts: 5 (2 Good / 2 Bad / 1 mixed) | Time: 2026-09-29 07:37 -> (running) +07

## 0 Context
- Ask: "continue" = the plan items that do not wait on the user's decisions: Replica data + loader + frozen
  vocabulary + 2D CLIP capacity gate (K8), token cache (K6), 3DGS fitter (K9), baselines (K11), change
  baselines (K13), run records (K14/K15), segmentation/calibration evaluation (K10b).
- Blocked on the user: RIO10/3RScan forms (D2, D7), scope confirmation, USGLoc order (D5), GPU sharing (D3).
- Data: Replica (Semantic-NeRF renders) at /home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset
  (7.6 GB, 8 scenes x 2 sequences x 900 frames; zip 6,762,780,327 B, `unzip -t` OK, deleted after extraction).
  Splits (plan): val office_0, room_2; test room_0, room_1, office_1-4.
- 7-Scenes proxy (internal, not a paper result): chess = tuning, office = unseen, one seed.

## 1 Findings
| # | Finding | Evidence |
|---|---|---|
| 1 | Dense CLIP extractor bug: open_clip 3.3 is batch-first, the code fed (L,N,D) -> no attention across tokens; 1% pixel accuracy on Replica. Every real-CLIP number before 2026-09-29 is invalid (corrections added to the earlier reports). Fixed; ClearCLIP last block chosen on validation. | [attempts/01](attempts/01-clip-capacity-gate.md), [tests/test_clip_real.py](../../../tests/test_clip_real.py) |
| 2 | Capacity gate FAILS the pre-set bar: best worst-scene mIoU 13.4 < 15 (room_2 22.6); fallbacks (mask pooling 11.3, openai 13.4) do not help; PCA-64 + decode-first loses nothing (13.1) | [logs/check_clip2d_run2.log](logs/check_clip2d_run2.log), [logs/check_clip2d_run3_openai.log](logs/check_clip2d_run3_openai.log) |
| 3 | With correct features the centred channel hurts (u_ale 0.735 vs 0.813 raw on chess) -> centring OFF by default | [attempts/02](attempts/02-change-replay-fixed-features.md) |
| 4 | Whole-history dispersion wins when half of the views changed; temporal tests win when the change is 20% of the history | [attempts/02](attempts/02-change-replay-fixed-features.md) |
| 5 | The best member of our change-test family (beta = 0, k = 2) IS the classic feature residual; two-bucket memory adds nothing on the proxy | [attempts/03](attempts/03-persistence-family.md) |
| 6 | Max-over-mapping calibration is inflated by short early histories; a per-view, history-conditioned null gives transferable operating points | [attempts/04](attempts/04-history-conditioned-calibration.md) |
| 7 | Replica loader verified: depth reprojection agrees within 2% on 99.2% of co-visible pixels; 101 classes, identical in all scenes | [tests/test_replica_loader.py](../../../tests/test_replica_loader.py) |
| 8 | Trained extractor: LSeg reaches 23.2 worst-scene mIoU (office_0 23.2, room_2 33.5) on validation vs 13.4 for ClearCLIP; PCA-64 loses nothing; chosen with the 5-template ensemble | [attempts/05](attempts/05-extractor-study.md) |
| 9 | Pipeline wiring: run_mapping.py uses the extractor registry and feeds NormLift/averaging, view count and CSS from the same per-view sums (CSS gets the scene's present classes at mapping time); eval_seg.py embeds text with the map's extractor and passes the scene's class ids to CSS | scripts/run_mapping.py, scripts/eval_seg.py |

## 3 Attempts
| # | idea | result | critic | link |
|---|---|---|---|---|
| 01 | 2D CLIP capacity gate | bug found and fixed; 13.4 / 22.6 mIoU (val) | Bad for the gate (< 15), Good for the pipeline | [01](attempts/01-clip-capacity-gate.md) |
| 02 | re-run change replay with correct features | dispersion/residual >= ours; centring hurts | Bad for the method as designed | [02](attempts/02-change-replay-fixed-features.md) |
| 03 | persistence k and the (beta, k) family | best = beta 0, k 2 = residual | Neutral-to-Bad for novelty | [03](attempts/03-persistence-family.md) |
| 04 | history-conditioned per-view calibration | 42-53% detected at 1.8-3.2% FA, both scenes | Good | [04](attempts/04-history-conditioned-calibration.md) |
| 05 | trained / stronger extractors (LSeg, dino.txt, Talk2DINO, ProxyCLIP) | LSeg worst-scene 23.2 (PCA-64 23.2) vs ClearCLIP 13.4 | Good: gate passes; LSeg chosen | [05](attempts/05-extractor-study.md) |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| CPU tests | `PYTHONPATH=. python3 -m pytest -q tests --ignore=tests/test_gsplat_gpu.py` | 87 passed, 2 skipped | PASS (log at the end of the run) |
| Token cache = direct extraction | `pytest tests/test_clip_cache.py` | min cosine 0.999999 | PASS |
| Dense CLIP is semantic | `pytest tests/test_clip_real.py` | 69.1% pixel accuracy (room_2, 2 frames) | PASS |
| CLIP token cache, all Replica frames (K6) | `scripts/extract_clip.py --scenes <8> --sequences Sequence_1:5,Sequence_2:10 --device cpu` | 2160 frames, 2.5 GB, 1271 s (0.59 s/frame), no partial files | PASS [logs/extract_clip_replica.log](logs/extract_clip_replica.log) |
| Module build + adversarial review | workflow wf_1c914ce7-25c | 4/5 built; fitter waiting for GPU | IN PROGRESS |

## 7 Open issues
- Main-claim risk: on the proxy our detector equals the residual baseline; plan gate G3 on RIO10 decides.
- Capacity gate: failed with ClearCLIP (13.4), PASSES with LSeg (23.2 >= 15) after the user asked for trained extractors.
- GPU: user allowed parallel runs with USGLoc (gate defaults relaxed 2026-09-29).
- LSeg token cache not built (4.9 MB/frame at stride 8 = 10.6 GB for all frames); the driver extracts on the GPU instead.

## 8 Files
(filled at the end)
