"""Compare last-block variants of dense CLIP on Replica validation frames (office_0, room_2)."""
import json, os, sys, numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
from usplat.clip_dense import OpenCLIPDense
from usplat.datasets import ReplicaSemNeRFDataset
torch.set_num_threads(16)
ROOT = "/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset"
vocab = json.load(open("/home/thanhnc/gauss_ws/src/uags/configs/vocab_replica.json"))
clip = OpenCLIPDense("ViT-B-16", "laion2b_s34b_b88k", device="cpu")
v = clip.visual

@torch.no_grad()
def tokens(image, mode):
    H, W = image.shape[:2]; P = clip.patch
    Hp, Wp = (H + P - 1) // P * P, (W + P - 1) // P * P
    x = image.float().permute(2, 0, 1)[None] / 255.0
    x = Fn.interpolate(x, size=(Hp, Wp), mode="bilinear", align_corners=False)
    x = (x - clip._mean) / clip._std
    x = v.conv1(x); gh, gw = x.shape[-2:]
    x = x.flatten(2).transpose(1, 2)
    x = torch.cat([v.class_embedding.expand(1, 1, -1), x], 1) + clip._interp_pos_embed(gh, gw)
    x = v.ln_pre(x)                                # batch-first (N, L, D)
    for blk in v.transformer.resblocks[:-1]:
        x = blk(x)
    blk = v.transformer.resblocks[-1]
    y = blk.ln_1(x); D = y.shape[-1]; a = blk.attn
    q, k, val = Fn.linear(y, a.in_proj_weight, a.in_proj_bias).split(D, dim=-1)
    ls1 = getattr(blk, "ls_1", torch.nn.Identity()); ls2 = getattr(blk, "ls_2", torch.nn.Identity())
    if mode == "res_ffn":                      # current code (value path + residual + FFN)
        o = x + ls1(a.out_proj(val)); o = o + ls2(blk.mlp(blk.ln_2(o)))
    elif mode == "value":                      # MaskCLIP value path, no residual/FFN
        o = a.out_proj(val)
    elif mode == "clearclip":                  # q-q attention, no residual/FFN
        B, L, _ = q.shape; h = a.num_heads; dh = D // h
        qh = q.reshape(B, L, h, dh).transpose(1, 2)        # (B, h, L, dh)
        vh = val.reshape(B, L, h, dh).transpose(1, 2)
        att = torch.softmax(qh @ qh.transpose(-1, -2) / dh ** 0.5, -1)
        o = a.out_proj((att @ vh).transpose(1, 2).reshape(B, L, D))
    o = v.ln_post(o) @ v.proj
    return o[0, 1:].reshape(gh, gw, -1)

res = {}
for scene in ("office_0", "room_2"):
    ds = ReplicaSemNeRFDataset(os.path.join(ROOT, scene, "Sequence_1"))
    idx = [int(i) for i in np.linspace(0, 855, 8)]
    present = sorted(set().union(*[set(torch.unique(ds[i]["labels"]).tolist()) for i in idx]) - {0})
    cls = torch.tensor(present)
    text = clip.text_embedding([vocab["classes"][str(c)]["prompt"] for c in present])
    for mode in ("res_ffn", "value", "clearclip"):
        inter = np.zeros(len(present)); union = np.zeros(len(present)); gtn = np.zeros(len(present)); cor = tot = 0
        for i in idx:
            fr = ds[i]; gt = fr["labels"]
            t = tokens(fr["image"], mode)
            f = Fn.normalize(Fn.interpolate(t.permute(2, 0, 1)[None], size=gt.shape, mode="bilinear",
                                            align_corners=False)[0].permute(1, 2, 0), dim=-1)
            pred = cls[(f @ text.t()).argmax(-1)]
            valid = gt > 0; cor += int((pred[valid] == gt[valid]).sum()); tot += int(valid.sum())
            for j, c in enumerate(present):
                p_, g_ = pred == c, gt == c
                inter[j] += int((p_ & g_).sum()); union[j] += int((p_ | g_).sum()); gtn[j] += int(g_.sum())
        has = gtn > 0
        m = float(np.mean(inter[has] / np.maximum(union[has], 1)))
        res[f"{scene}/{mode}"] = dict(miou=m, acc=cor / tot)
        print(f"{scene:9s} {mode:10s} mIoU {m*100:5.1f} acc {cor/tot*100:5.1f}", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "variants.json"), "w"), indent=1)
