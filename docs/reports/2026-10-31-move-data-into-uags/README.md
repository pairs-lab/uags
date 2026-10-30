# Move all UAGS data and results inside the repo
2026-10-31 | root: /home/thanhnc/gauss_ws/src/uags | git: master@0123c72 | [all reports](../INDEX.md)
Verdict: DONE -- gauss_ws/data moved into uags/data + uags/results; code, configs and tests follow; tables and video rebuild from there
Key result: 105 GB moved, file counts identical ([before](logs/move_before.tsv), [after](logs/move_after.tsv)); CPU tests 398 -> 423 passed ([final](logs/tests_final.txt))
Attempts: n/a | Time: 00:33-01:22 JST (`date` stamps; three CPU test runs 6.5, 10.5, 10.1 min)

## 0 Context
- Ask (user, 2026-10-31): "move the results of uags inside uags, we no need data folder like the current".
  Layout chosen by the user (AskUserQuestion): move everything; outputs to results/, inputs to data/; remove the
  empty /home/thanhnc/gauss_ws/data. Both folders git-ignored (105 GB).
- Checked first: no USGLoc / third_party file and no running process used gauss_ws/data; same disk, so `mv` is a rename.

| Old (/home/thanhnc/gauss_ws/data/...) | New (/home/thanhnc/gauss_ws/src/uags/...) | Size |
|---|---|---|
| uags_results/ | results/ | 6.5 GB |
| uags_fields/ | results/fields/ | 45.6 GB |
| uags_maps/ | results/maps/ | 1.5 GB |
| uags_cache/ | results/cache/ | 2.7 GB |
| replica_semnerf/, scannet/, nerf_object_removal/, weights/ | data/<same name>/ | 54.9 GB |

Old reports keep the old paths as text (history): read them through this table. The video zip is now
/home/thanhnc/gauss_ws/src/uags/results/video/supplementary_video.zip.

## 1 Findings
| Finding | Evidence | Action |
|---|---|---|
| 1246 ScanNet symlinks had absolute targets under the old root | `find -type l` | re-pointed; 0 broken |
| The 3 frozen *_prereg.json and the result records store old absolute paths, and their sha256 is checked (selection records, `config_sha256`) | scripts/extractor_select.py, fusion_select.py, records in results/ | kept byte-identical; old paths translated when read: [usplat/paths.py](../../../usplat/paths.py) `moved()` / `moved_tree()` at 26 Python call sites (11 files) and 24 shell snippets |
| Scene configs are live inputs and must match the scripts (test_scannet_loader checks it) | tests/test_scannet_loader.py | rewritten with the same mapping: 203 path rewrites in 52 files |
| 2 script tests passed only while the real maps were absent (the ScanNet dry run skips existing maps; the static-FA run makes runs/ before the gate) | [tests_after_edit](logs/tests_after_edit.txt) | tests made independent of results/ (fixture with empty map/field folders; `exist_ok`) |
| test_make_tables_tab2 failed since 2026-10-27: Table II gained the fused block, the test was not updated (my miss then) | [tests_before_edit](logs/tests_before_edit.txt) | test now checks the first 7 rows as before plus the 4 fused rows (inputs, cells) against their pre-registered rows file |
| test_css_fullvocab fails before and after; it reads no data | [tests_before_edit](logs/tests_before_edit.txt) | not fixed (unrelated to the move) |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Nothing lost | `find -type f / -type l | wc -l`, `du --apparent-size` per folder | counts identical; bytes identical except the links (+9 B each: longer target) | PASS ([after](logs/move_after.tsv)) |
| Old folder gone, not recreated | `ls /home/thanhnc/gauss_ws` after every run | src, third_party | PASS |
| Paper tables from results/ | `make_tables.py tab1pair / tab2 / tab3` | Table I 10/10 and Table III 7/7 rows verbatim; Table II 7/7 equal without bold/underline (marks span all 12 rows) | PASS ([tab1](logs/regen_tab1pair.tex), [tab2](logs/regen_tab2.tex), [tab3](logs/regen_tab3.tex)) |
| Video from results/ | `make_video.py numbers`; `compose --dry` | 84 numbers; dry run ok | PASS ([video_dry](logs/video_dry.txt)) |
| I2 records and preregs unchanged | no JSON in results/ or *_prereg.json edited | their sha256 checks still hold | PASS |
| CPU tests | `pytest -q tests --ignore=tests/test_gsplat_gpu.py`, CUDA hidden | before edits 398 passed / 2 failed / 34 skipped; final 423 / 1 / 10 | PASS except test_css_fullvocab (pre-existing) ([before](logs/tests_before_edit.txt), [final](logs/tests_final.txt)) |
| GPU tests and GPU pipelines | scripts/gpu_gate.sh | not needed for a move | NOT RUN |

The 10 final skips: 4 slow tests off (UAGS_SLOW_TESTS), DINOv2 repo not cloned, 3 extractors need open_clip
weights from the internet, 2 need CUDA (hidden on purpose).

## 7 Open issues
- test_css_fullvocab (bit-for-bit CSS tensors on the CPU) still fails; unrelated, not investigated.
- 24 shell snippets now import usplat (torch, ~1 s each): they need the venv python, as their GPU steps already did.
- Code paths not exercised by the tests or the checks above may still open an old path from a record; that fails
  with a missing-file error and cannot recreate the old folder.

## 8 Files
- Commits: 7078a83 (move: new usplat/paths.py, .gitignore /data/ /results/, 7 scene configs, 33 scripts,
  9 usplat files, 17 tests, papers/supplementary.tex comment), 0123c72 (Table II test).
- This folder: [logs/](logs/) (move_before/after, tests_before_edit/after_edit/final, regen_*, video_dry).
