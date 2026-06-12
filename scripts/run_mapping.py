"""
run_mapping.py — orchestration of the uncertainty-aware semantic mapping loop.

Pipeline per frame:
    1. dense CLIP feature map  z_obs : (P, d)
    2. accumulate vMF stats through the (differentiable) feature rasterizer
    3. read posterior + uncertainty
    4. (optional) EFE next-best-view planning
    5. (optional) consolidation / forgetting step

Backends:
    --backend mock   : pure-CPU, no CUDA, uses MockFeatureRasterizer + MockDenseCLIP
    --backend gsplat : real (needs GPU + your dataset loader + camera poses)

The mock backend executes the entire control flow so you can validate logic and
the calibration harness before touching CUDA. The gsplat path is the same code
with the two adapters swapped in (see TODOs).
"""
from __future__ import annotations
import argparse
import torch

from usplat.vmf_field import DirectionalSemanticField
from usplat.efe_planner import EFEPlanner
from usplat.consolidation import Consolidator
from usplat.evaluation import sparsification_curve, semantic_error_vs_text


def run_mock(num_gaussians=200, feat_dim=32, num_frames=15, num_protos=6, seed=0):
    torch.manual_seed(seed)
    from usplat.mock_rasterizer import MockFeatureRasterizer
    from usplat.clip_dense import MockDenseCLIP

    clip = MockDenseCLIP(feat_dim=feat_dim, num_protos=num_protos, noise=0.12, seed=seed)
    field = DirectionalSemanticField(num_gaussians, feat_dim, kappa_lik=10.0)

    # fixed mock screen geometry for the Gaussians
    means2d = torch.rand(num_gaussians, 2) * 64
    sigmas = 1.0 + torch.rand(num_gaussians) * 2.0
    opacities = 0.3 + 0.6 * torch.rand(num_gaussians)
    depths = torch.rand(num_gaussians)

    # assign each Gaussian a hidden "true" semantic prototype, SPATIALLY COHERENT
    # (neighbors share a prototype, as in a real scene), some are DYNAMIC.
    true_proto = (means2d[:, 0] / 64.0 * num_protos).long().clamp(0, num_protos - 1)
    dynamic = torch.rand(num_gaussians) < 0.2          # 20% change identity over time

    H = W = 48
    pix = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy"), -1)
    pixels = pix.reshape(-1, 2).float()                # (P,2)

    for t in range(num_frames):
        # observed per-pixel labels: each pixel takes the proto of its dominant Gaussian
        ras = MockFeatureRasterizer(means2d, sigmas, opacities, depths, pixels)
        W_mat = ras.weights()                          # (P, N)
        dom = W_mat.argmax(dim=1)                       # dominant gaussian per pixel
        proto_t = true_proto.clone()
        # dynamic Gaussians flip prototype on later frames -> conflicting evidence
        if t > num_frames // 2:
            proto_t[dynamic] = (proto_t[dynamic] + 1) % num_protos
        pixel_labels = proto_t[dom]                     # (P,)
        z_obs = clip.extract_from_labels(pixel_labels.reshape(H, W)).reshape(-1, feat_dim)

        field.update_from_rasterizer(ras, z_obs)        # <-- gradient-trick accumulation

    # ---- uncertainty readout -------------------------------------------------
    epi = field.epistemic()
    ale, valid = field.aleatoric(min_obs=5.0)
    rbar = field.mean_resultant_length()
    print(f"gaussians: {num_gaussians} | mean n_eff: {field.n_eff.mean():.1f}")
    print(f"mean epistemic: {epi.mean():.3f} | mean aleatoric (valid): "
          f"{ale[valid].mean():.3f}")

    # ---- does aleatoric flag the dynamic Gaussians? --------------------------
    dyn_valid = dynamic & valid
    sta_valid = (~dynamic) & valid
    if dyn_valid.any() and sta_valid.any():
        print(f"aleatoric  dynamic={ale[dyn_valid].mean():.3f}  "
              f"static={ale[sta_valid].mean():.3f}   "
              f"(dynamic should be higher)")

    # ---- consolidation -------------------------------------------------------
    cons = Consolidator(field, min_obs=5.0, r_unstable=0.55, r_stable=0.8)
    masks = cons.step()
    print(f"robust(stable) Gaussians: {int(cons.robust_mask().sum())} | "
          f"flagged unstable: {int(masks['unstable'].sum())}")

    # ---- EFE planning over a few candidate viewpoints ------------------------
    V = 5
    visibility = torch.rand(V, num_gaussians)          # mock expected blending mass
    planner = EFEPlanner(field, lam=1.0)
    query = clip.text_embedding(0)                      # "find prototype-0 stuff"
    efe, best = planner.score(visibility, query_embed=query)
    print(f"EFE per candidate view: {[round(x,3) for x in efe.tolist()]} | best=v{best}")

    # ---- calibration: aleatoric vs semantic error ----------------------------
    gt_dirs = clip.protos[true_proto]                   # (N, d) ground-truth text dir
    err = semantic_error_vs_text(field.mu_post(), gt_dirs)
    out = sparsification_curve(err[valid], ale[valid], num_bins=10)
    print(f"AUSE(aleatoric -> semantic error): {out['ause']:.4f}  (lower=better)")
    print("done.")


def run_gsplat(args):
    """Real pipeline: pre-trained 3DGS checkpoint + posed RGB stream + dense CLIP."""
    import os
    from tqdm import tqdm
    from usplat.gs_io import load_gaussians
    from usplat.datasets import make_dataset, downscale
    from usplat.clip_dense import OpenCLIPDense
    from usplat.feature_pca import StreamingPCA
    from usplat.gsplat_adapter import GSplatFeatureRasterizer

    device = "cuda"
    splats = load_gaussians(args.ply, device=device)
    N = splats["means"].shape[0]
    print(f"loaded {N} Gaussians from {args.ply}")

    ds = make_dataset(args.dataset, args.data, stride=args.stride)
    print(f"dataset {args.dataset}: {len(ds)} frames (stride {args.stride})")

    clip = OpenCLIPDense(args.clip_model, args.clip_pretrained, device=device)

    # ---- pass 1 (first few frames): fit the PCA basis for compression --------
    pca = StreamingPCA(clip.d, args.feat_dim, device=device)
    for i in range(min(args.pca_frames, len(ds))):
        fr = downscale(ds[i], args.downscale)
        pca.accumulate(clip.extract(fr["image"]))
    explained = pca.fit()
    print(f"PCA {clip.d} -> {args.feat_dim} dims "
          f"(explained second moment: {explained:.3f})")

    field = DirectionalSemanticField(N, args.feat_dim, kappa_lik=args.kappa,
                                     device=device)
    cons = Consolidator(field, min_obs=args.min_obs,
                        r_unstable=args.r_unstable, r_stable=args.r_stable,
                        forget_lambda=args.forget_lambda)

    # ---- mapping loop ---------------------------------------------------------
    for i in tqdm(range(len(ds)), desc="mapping"):
        fr = downscale(ds[i], args.downscale)
        c2w = fr["c2w"]
        if not torch.isfinite(c2w).all():
            continue
        h, w = fr["image"].shape[:2]
        z = pca.project(clip.extract(fr["image"])).reshape(-1, args.feat_dim)
        viewmat = torch.linalg.inv(c2w.double()).float().to(device)
        ras = GSplatFeatureRasterizer.from_splats(
            splats, viewmat, fr["K"].to(device), w, h,
            channel_chunk=args.channel_chunk)
        field.update_from_rasterizer(ras, z)
        if args.consolidate_every > 0 and (i + 1) % args.consolidate_every == 0:
            cons.step()

    # ---- readout --------------------------------------------------------------
    epi = field.epistemic()
    ale, valid = field.aleatoric(min_obs=args.min_obs)
    rbar = field.mean_resultant_length()
    print(f"mean n_eff {field.n_eff.mean():.1f} | observed (n_eff>0): "
          f"{(field.n_eff > 0).float().mean() * 100:.1f}%")
    print(f"epistemic mean {epi.mean():.3f} | aleatoric mean (valid) "
          f"{ale[valid].mean():.3f} | r_bar mean {rbar[valid].mean():.3f}")
    print(f"robust subset: {int(cons.robust_mask().sum())}/{N}")

    if args.query:
        prompts = [q.strip() for q in args.query.split(",")]
        t = pca.project(clip.text_embedding(prompts))          # (K, feat_dim)
        sims = field.query(t)                                   # (N, K)
        for k, p in enumerate(prompts):
            s = sims[valid, k]
            print(f"query '{p}': max {s.max():.3f} | "
                  f">0.5 sim on {(s > 0.5).float().mean() * 100:.2f}% of valid Gaussians")

    out = dict(R=field.R.cpu(), n_eff=field.n_eff.cpu(),
               mu_post=field.mu_post().cpu(), epistemic=epi.cpu(),
               aleatoric=ale.cpu(), robust_mask=cons.robust_mask().cpu(),
               pca=pca.state_dict(), config=vars(args))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"saved field -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "gsplat"], default="mock")
    # gsplat backend args
    ap.add_argument("--ply", help="pre-trained 3DGS checkpoint (.ply or .pt)")
    ap.add_argument("--data", help="dataset root")
    ap.add_argument("--dataset", default="replica",
                    choices=["replica", "scannet", "7scenes"])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--clip-pretrained", default="openai")
    ap.add_argument("--feat-dim", type=int, default=64,
                    help="PCA-compressed semantic dimension on the splats")
    ap.add_argument("--pca-frames", type=int, default=10)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--min-obs", type=float, default=5.0)
    ap.add_argument("--r-unstable", type=float, default=0.55)
    ap.add_argument("--r-stable", type=float, default=0.85)
    ap.add_argument("--forget-lambda", type=float, default=0.9)
    ap.add_argument("--consolidate-every", type=int, default=10,
                    help="frames between consolidation steps; 0 disables")
    ap.add_argument("--channel-chunk", type=int, default=32)
    ap.add_argument("--query", default="",
                    help="comma-separated open-vocab text queries to demo")
    ap.add_argument("--out", default="outputs/field.pt")
    args = ap.parse_args()
    if args.backend == "mock":
        run_mock()
    else:
        if not (args.ply and args.data):
            raise SystemExit("--backend gsplat requires --ply and --data")
        run_gsplat(args)


if __name__ == "__main__":
    main()
