"""
paths.py — where UAGS keeps its data and results.

Since 2026-10-31 everything lives inside the repository (both folders are git-ignored):

  data/      datasets and model weights (inputs): replica_semnerf, scannet,
             nerf_object_removal, weights
  results/   records, figures and video (was uags_results), with fields/, maps/ and cache/
             (were uags_fields, uags_maps, uags_cache)

Before that they were under OLD_ROOT = /home/thanhnc/gauss_ws/data. Records, field checkpoints
and the pre-registered configs written before the move still name files by those old paths, and
they stay byte-identical (their sha256 is checked), so code that opens a path read from one of
them passes it through moved() (one path) or moved_tree() (a whole JSON object).
"""
from __future__ import annotations
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
RESULTS = os.path.join(REPO, "results")

OLD_ROOT = "/home/thanhnc/gauss_ws/data"
# first folder under OLD_ROOT -> its new place; any other folder moved to DATA unchanged
_RENAMED = {"uags_results": RESULTS,
            "uags_fields": os.path.join(RESULTS, "fields"),
            "uags_maps": os.path.join(RESULTS, "maps"),
            "uags_cache": os.path.join(RESULTS, "cache"),
            "uags_swap": os.path.join(RESULTS, "swap")}


def moved(path):
    """The current place of a path under OLD_ROOT; any other value is returned unchanged."""
    if not isinstance(path, str) or not path.startswith(OLD_ROOT + "/"):
        return path
    head, sep, tail = path[len(OLD_ROOT) + 1:].partition("/")
    if head in _RENAMED:
        return _RENAMED[head] + sep + tail
    return os.path.join(DATA, head) + sep + tail


def moved_tree(obj):
    """moved() applied to every string of a JSON-like object, keys included (extractor_prereg
    keys its weight files by path); dicts and lists are copied."""
    if isinstance(obj, dict):
        return {moved(k): moved_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(moved_tree(v) for v in obj)
    return moved(obj)


if __name__ == "__main__":
    # python3 -m usplat.paths <path>...   prints the current place of each path
    import sys
    for p in sys.argv[1:]:
        print(moved(p))
