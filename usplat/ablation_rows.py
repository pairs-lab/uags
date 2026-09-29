"""
ablation_rows.py — the rows of Tab III (plan §3), for scripts/eval_seg.py --row-factory
usplat.ablation_rows:row_factory. Each variant gives a P0 row (mu*, channels u_epi, u_ale:
the AUSE columns) and a P2 row (adaptive beta from the same channels: the calibration
columns), named abl_<variant>_P0 / abl_<variant>_P2.

The Replica block of the paper's Table III (sec:ablation). Read-out variants use the SAME
per-view sums as the main field (a copy of the field with another FieldConfig read-out
option; no second mapping pass):
    est_rbar       "Uncorrected mean length": aleatoric from r_bar = ||R|| / W1
    est_joint      "Prior-shrunk agreement":  ||R|| / (W1 + c0)
    est_legacy     "Shrunk mean length":      r_tilde = r_bar - (1 - r_bar)/(W1 + 1)
(An epistemic read-out ablation is not listed: with view weights <= 1 the support
min(n_K, W1) is W1, so epistemic="views" equals the default.)
Re-mapped variants need their own mapping pass on the same 3DGS map; they are read from
files next to the main field, <main>_<variant>.pt (run_mapping.py --field-config), and are
skipped with a warning when the file is missing:
    pixels   "Pixels instead of views" (FieldConfig(evidence="pixels", epistemic="pixels"))
    centre   "Centered features" (FieldConfig(centre=True))
    legacy   "Pixel-level variant", the original draft's method (FieldConfig.legacy())
The main field path is the mapping run's --out (ckpt["config"]["out"]) unless the caller
gives field_path.
"""
from __future__ import annotations
import copy
import dataclasses
import os
import warnings

import torch

from .rows import FieldRow
from .vmf_field import DirectionalSemanticField

READOUT = {
    "est_rbar": dict(estimator="rbar"),
    "est_joint": dict(estimator="joint"),
    "est_legacy": dict(estimator="legacy"),
}
REMAPPED = ("pixels", "centre", "legacy")
REMAPPED_CONFIG = {                    # --field-config of each re-mapped variant
    "pixels": {"evidence": "pixels", "epistemic": "pixels"},
    "centre": {"centre": True},
    "legacy": "legacy",                # FieldConfig.legacy()
}


def field_config_json(variant):
    """The run_mapping.py --field-config JSON of a re-mapped variant."""
    import json
    from .vmf_field import FieldConfig
    c = REMAPPED_CONFIG[variant]
    return json.dumps(FieldConfig.legacy().to_dict() if c == "legacy" else c)


def readout_copy(field, **cfg_overrides):
    """A shallow copy of the field that shares its statistics but reads them out with
    another FieldConfig (read-out options only: no statistic is recomputed)."""
    f = copy.copy(field)
    f.cfg = dataclasses.replace(field.cfg, **cfg_overrides)
    f.cfg.validate()
    return f


def variant_rows(field, name, p2=True):
    rows = [FieldRow(f"abl_{name}_P0", "feature", field=field, mode="P0")]
    if p2:
        rows.append(FieldRow(f"abl_{name}_P2", "feature", adaptive=True, field=field,
                             mode="P0"))
    return rows


def variant_path(main_path, variant):
    stem = main_path[:-3] if main_path.endswith(".pt") else main_path
    return f"{stem}_{variant}.pt"


def row_factory(ckpt, field, device, class_ids=None, field_path=None, variants=None):
    rows = []
    want = set(variants) if variants is not None else None
    for name, over in READOUT.items():
        if want is None or name in want:
            rows += variant_rows(readout_copy(field, **over), name)
    main = field_path or (ckpt.get("config") or {}).get("out")
    for name in REMAPPED:
        if want is not None and name not in want:
            continue
        p = variant_path(main, name) if main else None
        if not p or not os.path.exists(p):
            warnings.warn(f"ablation variant {name!r}: no field at {p}; row skipped")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        f = DirectionalSemanticField.from_state(ck["field"], device=device)
        if f.N != field.N:
            raise ValueError(f"variant {name!r} has {f.N} Gaussians, the main field {field.N}")
        rows += variant_rows(f, name)
    return rows
