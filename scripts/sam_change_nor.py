"""
sam_change_nor.py — the NeRF-Object-Removal tables for the SAM-mask baselines (usplat/sam_lift.py,
configs/sam_baselines_prereg.json): runs one of the two scripts of the Feature 3DGS / LEGaussians
rows with its row loader extended to the sam_lift files.

    change: scripts/eval_change_nor_distilled.py (Table III, tab:change: the feature residual
            on the files trained on the mapping keyframes, <field>_langsplat_<l>.pt, _opengs.pt)
    stale:  scripts/stale_baselines_nor.py (Table IV, tab:stale: the stale-answer rate of the
            files re-trained on the mapping and update keyframes, <field>_<name>_upd.pt)

    scripts/gpu_gate.sh 6000 python3 scripts/sam_change_nor.py change \
        --field results/fields/nor_001_lseg_relmax.pt --align none --update-stride 1 --tau-cov 0.5 \
        --distilled results/fields/nor_001_lseg_relmax_langsplat_w.pt,results/fields/nor_001_lseg_relmax_opengs.pt \
        --reference results/nor/tab2_relmax/val/001.json --out results/nor/sam/val/001.json

Both scripts are imported and never edited. Their load_sources (eval_change_nor_distilled.py) is
wrapped: a Feature 3DGS or LEGaussians file loads as before, a sam_lift file gives the row
"<name>_feat" whose feature map is the file's query feature (usplat.sam_lift.render_fn:
LangSplat renders its latent and decodes it, OpenGaussian renders its instance features).
The code hashes the records carry include this file and usplat/sam_lift.py.
"""
from __future__ import annotations
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import eval_change_nor_distilled as ECD                                     # noqa: E402
import stale_baselines_nor as SB                                            # noqa: E402
from usplat import distill as DI                                            # noqa: E402
from usplat import sam_lift as SL                                           # noqa: E402

EXTRA_CODE = ("scripts/sam_change_nor.py", "usplat/sam_lift.py")
_LOAD_SOURCES = ECD.load_sources


def load_sources(ctx, paths, control=True):
    """eval_change_nor_distilled.load_sources, with sam_lift files read by usplat.sam_lift."""
    dev, N, ckpt = ctx["device"], ctx["N"], ctx["ckpt"]
    ours, rest = {}, []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if SL.is_sam_file(d):
            ours[p] = d
        else:
            rest.append(p)
    src, prov = _LOAD_SOURCES(ctx, rest, control)
    want = DI.basis_sha256(ckpt["pca"])
    for p, d in ours.items():
        if d.get("pca_basis_sha256") != want:
            raise SystemExit(f"{p}: trained in another PCA basis than {ctx['cfg'].get('out')}")
        n = int(SL.gaussian_features(d).shape[0])
        if n != N:
            raise SystemExit(f"{p}: {n} Gaussians, the map has {N}")
        row = f"{d['name']}_feat"
        if row in src:
            raise SystemExit(f"two files give the row {row!r}")
        fn = SL.render_fn(d, dev)
        src[row] = lambda ras, ren, fn=fn: fn(ras)
        prov[row] = dict(ECD.ECN.file_entry(p), name=d["name"], identity=d.get("identity"),
                         config=d.get("config"), main_field=d.get("main_field"),
                         n_frames=d.get("n_frames"), iters_run=d.get("iters_run"),
                         train_s=(d.get("timing") or {}).get("train_s"),
                         masks=d.get("masks"))
    return src, prov


def install():
    """Wrap the loader (idempotent) and add this code to both scripts' code hashes."""
    ECD.load_sources = load_sources
    for mod in (ECD, SB):
        if EXTRA_CODE[0] not in mod.CODE:
            mod.CODE = tuple(mod.CODE) + EXTRA_CODE


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("change", "stale"):
        raise SystemExit("usage: sam_change_nor.py change|stale <arguments of the wrapped script>")
    install()
    return (ECD.main if argv[0] == "change" else SB.main)(argv[1:])


if __name__ == "__main__":
    main()
