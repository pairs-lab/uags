"""Semantic-swap benchmark (usplat.swap, scripts/make_swap_bank.py, scripts/replica_swap.sh), CPU.

  spec       configs/swap_replica.json loads and validates; the factory arg string parses
  rule       resolve_pairs: sources by update pixels (ties: lower id), near / far targets in
             text space, exclusions, and a loud failure instead of a relaxed rule
  bank       nearest fill against brute force; build_bank with a fake extractor keeps the
             REAL features of the target class
  transform  replaces exactly the source pixels (labels downsampled by pixel-centre nearest),
             with the projected segment stretched per connected component; deterministic per
             (seed, frame, pair); logs its frames; refuses frames without labels and banks of
             another spec / scene
  GT         SwapGT equals the brute-force W^T one-hot sums; labels at 0.5 / 0.05, targets
             fold the source into its target, objects are classes
  end to end a tiny Replica-format scene on disk: bank (fake extractor) -> fields with and
             without reset (mock rasterizer, the shared-interface call; no reset = the change
             test with an infinite threshold) -> gt-swap (mock) -> eval (the compare field's
             features checked) -> criterion: the reset field is the most accurate on swapped
             Gaussians
  script     replica_swap.sh DRY_RUN: the commands and their flags (run_mapping's own argparse
             plus the T2 flags: the two shared-interface flags, --fixed-change-thr and
             --change-baselines; eval_change / make_swap_bank parsers), map and refusal rules
             (dirty tree, no validation record, other spec, finished test)
"""
import importlib.util
import json
import os
import re
import shlex
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

from usplat import swap as SW

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(REPO, "configs", "swap_replica.json")


def _load_cli(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ spec / rule
def test_spec_and_arg_parsing(tmp_path):
    spec, sha = SW.load_spec(SPEC)
    assert len(sha) == 64 and spec["gt"]["swapped_min_frac"] == 0.5
    assert spec["gt"]["static_max_frac"] == 0.05 and spec["criterion"]["min_pass"] == 5
    assert len(spec["scenes"]["test"]) == 6 and spec["scenes"]["val"] == ["office_0", "room_2"]
    assert spec["pair_rule"]["kinds"] == ["near", "far"]
    names = {c["name"] for c in json.load(open(os.path.join(REPO, spec["vocab"])))["classes"].values()}
    assert set(spec["pair_rule"]["excluded_names"]) <= names        # no typo in the exclusions
    assert SW.parse_arg("bank=/a/b.pt, spec=c.json,scene=room_0,log=/x/y.json") == dict(
        bank="/a/b.pt", spec="c.json", scene="room_0", log="/x/y.json")
    with pytest.raises(ValueError, match="needs bank"):
        SW.parse_arg("spec=c.json")
    with pytest.raises(ValueError, match="unknown swap arg"):
        SW.parse_arg("bank=a,seed=3")
    with pytest.raises(ValueError, match="key=value"):
        SW.parse_arg("bank=a,oops")
    bad = dict(spec, pair_rule=dict(spec["pair_rule"], kinds=["near"]))
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="kinds"):
        SW.load_spec(str(tmp_path / "bad.json"))


def _rule(**kw):
    r = dict(excluded_names=["wall"], source_min_update_share=0.05,
             source_min_update_frames_frac=0.5, source_min_mapping_share=0.05,
             target_min_mapping_share=0.02, n_sources=2, kinds=["near", "far"])
    r.update(kw)
    return r


def test_resolve_pairs_rule():
    names = {1: "wall", 2: "chair", 3: "sofa", 4: "table", 5: "lamp", 6: "cup", 7: "bed"}
    st_map = dict(px={1: 500, 2: 100, 3: 100, 4: 60, 5: 40, 6: 1, 7: 100}, n_frames=10,
                  frames={c: 10 for c in names}, total_px=901)
    st_upd = dict(px={1: 800, 2: 200, 3: 200, 4: 50, 5: 30, 6: 2, 7: 150}, n_frames=10,
                  frames={1: 10, 2: 9, 3: 8, 4: 10, 5: 10, 6: 1, 7: 2}, total_px=1432)
    e = torch.eye(7)
    text = {1: e[0], 2: e[1], 3: e[2], 4: e[1] + 0.3 * e[3], 5: -e[2] + 0.1 * e[4], 6: e[1],
            7: e[1] + e[2]}
    segs = {c: 1 for c in names}
    p = SW.resolve_pairs(st_map, st_upd, list(names), names, text, _rule(), segs)
    # wall is excluded; bed is in too few update frames; chair and sofa tie on update pixels:
    # lower id first. chair -> nearest (cup has too little mapping share, so table); sofa ->
    # farthest (lamp, cosine < 0)
    assert [(x["source_name"], x["target_name"], x["kind"]) for x in p] == [
        ("chair", "table", "near"), ("sofa", "lamp", "far")]
    assert p[0]["text_cos"] > 0.9 and p[1]["text_cos"] < 0
    # a target needs a bank segment
    p = SW.resolve_pairs(st_map, st_upd, list(names), names, text, _rule(), {**segs, 4: 0})
    assert p[0]["target_name"] == "bed"                 # next nearest to chair
    with pytest.raises(ValueError, match="eligible source"):
        SW.resolve_pairs(st_map, st_upd, list(names), names, text, _rule(n_sources=3,
                         kinds=["near", "far", "near"]), segs)


# ------------------------------------------------------------------ bank pieces
def test_fill_nearest_and_resize():
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(7, 9, 4, generator=g)
    mask = torch.zeros(7, 9, dtype=torch.bool)
    mask[1:3, 2:4] = True
    mask[5, 7] = True
    out = SW.fill_nearest(feats, mask)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    for y in range(7):
        for x in range(9):
            d = (ys - y) ** 2 + (xs - x) ** 2
            best = d == d.min()
            cands = [feats[ys[k], xs[k]] for k in torch.nonzero(best).flatten()]
            assert any(torch.equal(out[y, x], c) for c in cands)
    assert torch.equal(out[mask], feats[mask])
    r = SW.resize_nearest(torch.arange(100.0).reshape(10, 10, 1), 4)
    assert r.shape == (4, 4, 1) and r[0, 0, 0] == 11.0     # pixel-centre sampling
    assert SW.resize_nearest(torch.zeros(3, 2, 5), 64).shape == (3, 2, 5)
    with pytest.raises(ValueError, match="empty"):
        SW.fill_nearest(feats, torch.zeros(7, 9, dtype=torch.bool))


def _manual_bank(d=6, sha="s" * 64, scene="toy"):
    g = torch.Generator().manual_seed(1)
    segs = {7: [dict(frame=0, component=1, area=4, bbox=[0, 1, 0, 1],
                     feats=Fn.normalize(torch.randn(2, 3, d, generator=g), dim=-1).half())
                for _ in range(3)]}
    return dict(schema=SW.BANK_SCHEMA, scene=scene, spec_sha256=sha, seed=5,
                pairs=[dict(source_id=3, source_name="a", target_id=7, target_name="b",
                            kind="near", text_cos=0.5)],
                segments=segs, classes=dict(ids=[3, 7], names=["a", "b"]),
                text_raw=torch.eye(2, d))


def test_transform_replaces_source_pixels(tmp_path):
    from usplat.feature_pca import StreamingPCA
    bank = _manual_bank()
    pca = StreamingPCA(6, 4)
    pca.basis = torch.linalg.qr(torch.randn(6, 4, generator=torch.Generator().manual_seed(2)))[0]
    xf = SW.SwapTransform(bank, pca, "cpu", log_path=str(tmp_path / "log" / "f.json"))
    h, w = 6, 8
    full = torch.zeros(12, 16, dtype=torch.long)          # labels at 2x the accumulation grid
    full[0:4, 0:6] = 3                                    # component 1 of the source
    full[8:12, 10:16] = 3                                 # component 2
    full[6:8, 0:4] = 7
    z = Fn.normalize(torch.randn(h * w, 4), dim=-1)
    out = xf(z, dict(labels=full, index=40), 4, (h, w))
    lab = SW.nearest_labels(full, h, w)
    src = (lab == 3).reshape(-1)
    assert torch.equal(out[~src], z[~src]) and not torch.equal(out[src], z[src])
    seg = xf.pick(4, 0, 7)
    proj = {tuple(v.tolist()) for v in seg.reshape(-1, 4)}
    assert all(tuple(v.tolist()) in proj for v in out[src])    # real (projected) features
    assert torch.allclose(out[src].norm(dim=-1), torch.ones(int(src.sum())), atol=1e-5)
    # each component is stretched over its own bounding box: its top-left pixel takes the
    # segment's top-left value, its bottom-right pixel the bottom-right value
    for (y0, x0, y1, x1) in ((0, 0, 1, 2), (4, 5, 5, 7)):
        assert torch.equal(out[y0 * w + x0], seg[0, 0]) and torch.equal(out[y1 * w + x1], seg[-1, -1])
    assert torch.equal(xf(z, dict(labels=full), 4, (h, w)), out)     # deterministic
    picks = {id(xf.pick(i, 0, 7)) for i in range(20)}
    assert len(picks) > 1                                 # the segment changes with the frame
    none = xf(z, dict(labels=torch.zeros(12, 16, dtype=torch.long)), 5, (h, w))
    assert none is z                                      # nothing to swap: untouched
    log = json.load(open(tmp_path / "log" / "f.json"))
    assert [f["i"] for f in log["frames"]] == [4, 4, 5] and log["frames"][0]["index"] == 40
    assert log["frames"][0]["n_px"] == {"3": int(src.sum())} and log["scene"] == "toy"
    with pytest.raises(ValueError, match="no labels"):
        xf(z, dict(labels=None), 6, (h, w))
    with pytest.raises(ValueError, match="rows"):
        xf(z[:-1], dict(labels=full), 6, (h, w))
    with pytest.raises(ValueError, match="PCA expects"):
        SW.SwapTransform(bank, StreamingPCA(5, 4), "cpu")


def test_swap_factory_checks(tmp_path):
    from usplat.feature_pca import StreamingPCA
    _, sha = SW.load_spec(SPEC)
    torch.save(_manual_bank(sha=sha, scene="room_0"), tmp_path / "bank.pt")
    pca = StreamingPCA(6, 6)
    pca.basis = torch.eye(6)
    xf = SW.swap_factory(f"bank={tmp_path / 'bank.pt'},spec={SPEC},scene=room_0", pca, "cpu")
    assert isinstance(xf, SW.SwapTransform) and xf.log_path is None
    with pytest.raises(ValueError, match="scene"):
        SW.swap_factory(f"bank={tmp_path / 'bank.pt'},scene=room_1", pca, "cpu")
    torch.save(_manual_bank(sha="0" * 64, scene="room_0"), tmp_path / "other.pt")
    with pytest.raises(ValueError, match="another spec"):
        SW.swap_factory(f"bank={tmp_path / 'other.pt'},spec={SPEC}", pca, "cpu")
    b = _manual_bank()
    b["segments"] = {7: []}
    torch.save(b, tmp_path / "empty.pt")
    with pytest.raises(ValueError, match="no segment"):
        SW.swap_factory(f"bank={tmp_path / 'empty.pt'}", pca, "cpu")


# ------------------------------------------------------------------ GT
def test_swap_gt_equals_brute_force():
    from usplat.mock_rasterizer import MockFeatureRasterizer
    g = torch.Generator().manual_seed(0)
    H, W, N = 10, 14, 40
    means = torch.stack([torch.rand(N, generator=g) * W, torch.rand(N, generator=g) * H], -1)
    py, px = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    ras = MockFeatureRasterizer(means, torch.full((N,), 1.2), torch.full((N,), 0.8),
                                torch.rand(N, generator=g), torch.stack([px.flatten(), py.flatten()], -1))
    Wm = ras.weights().double()
    lab = torch.zeros(H, W, dtype=torch.long)             # 0 = unlabelled
    lab[:, :5] = 3
    lab[:, 5:9] = 4
    lab[:5, 9:] = 9
    classes = [3, 4, 9]
    acc = SW.SwapGT(N, classes, [(3, 9)])
    acc.add_view(ras, lab)
    acc.add_view(ras, lab)
    onehot = torch.stack([(lab.reshape(-1) == c).double() for c in classes], 1)
    assert torch.allclose(acc.C, 2 * Wm.T @ onehot, atol=1e-5)
    assert torch.allclose(acc.m, 2 * Wm.sum(0), atol=1e-5)
    r = acc.result(0.5, 0.05, min_mass=0.05)
    frac = (Wm.T @ onehot[:, 0]) / Wm.sum(0)
    seen = 2 * Wm.sum(0) >= 0.05
    want = torch.full((N,), -1)
    want[seen & (frac >= 0.5)] = 1
    want[seen & (frac <= 0.05)] = 0
    assert torch.equal(r["label"].long(), want)
    assert torch.allclose(r["frac"], frac, atol=1e-5)
    post = (Wm.T @ onehot).clone()
    post[:, 2] += post[:, 0]
    post[:, 0] = 0
    has = (Wm.T @ onehot).sum(1) > 0
    assert torch.equal(r["target"][has], post.argmax(1)[has])     # class index: 9 now, not 3
    assert not bool((r["target"] == 0).any())                     # nobody shows the source
    assert torch.equal(r["object_id"][has], (Wm.T @ onehot).argmax(1)[has])
    with pytest.raises(ValueError, match="class list"):
        SW.SwapGT(N, classes, [(3, 5)])


# ------------------------------------------------------------------ fake scene on disk
CLS = dict(wall=93, chair=20, sofa=76, table=80, lamp=47, plant=44)
D = 16


def _protos():
    g = torch.Generator().manual_seed(7)
    p = {c: Fn.normalize(torch.randn(D, generator=g), dim=0) for c in CLS.values()}
    p[CLS["table"]] = Fn.normalize(p[CLS["chair"]] + 0.4 * torch.randn(D, generator=g), dim=0)
    p[CLS["lamp"]] = Fn.normalize(-p[CLS["sofa"]] + 0.3 * torch.randn(D, generator=g), dim=0)
    return p


class FakeExtractor:
    """Tokens from the class id stored in the image's red channel (+ seeded noise)."""
    d = D

    def __init__(self, noise=0.05):
        self.p = _protos()
        self.noise = noise
        self.vocab = {}

    def extract_tokens(self, image):
        cls = torch.as_tensor(image)[4::8, 4::8, 0].long()
        g = torch.Generator().manual_seed(int(cls.sum()))
        P = torch.stack([self.p.get(int(c), torch.zeros(D)) for c in cls.reshape(-1)])
        return (P + self.noise * torch.randn(P.shape, generator=g)).reshape(*cls.shape, D)

    def text_embedding(self, prompts, templates=("a photo of a {}.",)):
        v = json.load(open(os.path.join(REPO, "configs", "vocab_replica.json")))["classes"]
        by_prompt = {c.get("prompt", c["name"]): int(k) for k, c in v.items()}
        return torch.stack([self.p[by_prompt[p]] for p in prompts])


def _labels():
    lab = np.full((480, 640), CLS["wall"], np.uint8)
    lab[160:420, 40:260] = CLS["chair"]
    lab[160:420, 380:600] = CLS["sofa"]
    lab[300:470, 280:360] = CLS["table"]
    lab[20:120, 500:600] = CLS["lamp"]
    lab[20:120, 40:140] = CLS["plant"]
    return lab


def _write_replica(seq_dir, n):
    import cv2
    for sub in ("rgb", "depth", "semantic_class"):
        os.makedirs(os.path.join(seq_dir, sub), exist_ok=True)
    lab = _labels()
    poses = []
    for i in range(n):
        img = np.zeros((480, 640, 3), np.uint8)
        img[..., 2] = lab                                 # red channel (BGR on disk)
        cv2.imwrite(os.path.join(seq_dir, "rgb", f"rgb_{i}.png"), img)
        cv2.imwrite(os.path.join(seq_dir, "depth", f"depth_{i}.png"), np.full((480, 640), 2000, np.uint16))
        cv2.imwrite(os.path.join(seq_dir, "semantic_class", f"semantic_class_{i}.png"), lab)
        c2w = np.eye(4)
        c2w[0, 3] = 0.01 * i
        poses.append(c2w.reshape(-1))
    np.savetxt(os.path.join(seq_dir, "traj_w_c.txt"), np.stack(poses))


def _toy_spec(tmp_path):
    spec = json.load(open(SPEC))
    spec["scenes"] = dict(val=["toy"], test=["toy_t"])
    spec["mapping"]["stride"] = 1
    spec["update"]["stride"] = 1
    spec["downscale"] = 16
    spec["vocab"] = os.path.join(REPO, "configs", "vocab_replica.json")
    spec["bank"] = dict(frame_step=1, min_segment_px=20, max_segments_per_class=3, max_side_px=16)
    spec["criterion"]["min_pass"] = 1
    spec["criterion"]["split"] = "val"
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    return str(p), spec


def _map_fields(tmp_path, root, spec_path, bank_path, reset, log_name):
    """run_mapping.py's loop on the CPU: mapping pass, calibration (reset) or none (the
    threshold stays inf, as --fixed-change-thr inf: c is scored, nothing resets), update pass
    with the shared-interface transform call."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from eval_seg import PinholeMockRasterizer
    from usplat.datasets import make_dataset, downscale
    from usplat.clip_dense import upsample_tokens
    from usplat.feature_pca import StreamingPCA
    from usplat.vmf_field import DirectionalSemanticField, FieldConfig
    from usplat.baselines import NormLift, baselines_state
    splats = torch.load(tmp_path / "gs.pt")
    N = splats["means"].shape[0]
    pca = StreamingPCA(D, D)
    pca.basis = torch.eye(D)
    ext = FakeExtractor()
    cfg = FieldConfig(readmit_depth=None)
    cfg.change_thr = float("inf")
    field = DirectionalSemanticField(N, D, cfg)
    nl = NormLift.for_field(field, with_averaging=True)
    arg = f"bank={bank_path},spec={spec_path},scene=toy,log={tmp_path / log_name}"
    xform = SW.swap_factory(arg, pca, "cpu")

    def run_sequence(seq, transform):
        for i in range(len(seq)):
            fr_full = seq[i]
            fr = downscale(fr_full, 16)
            h, w = fr["image"].shape[:2]
            z = pca.project(upsample_tokens(ext.extract_tokens(fr_full["image"]), (h, w))).reshape(-1, D)
            if transform:
                z = xform(z, fr, i, (h, w))
            ras = PinholeMockRasterizer(splats, torch.linalg.inv(fr_full["c2w"].double()).float(),
                                        fr["K"], w, h)
            field.update_view(ras, z, num_pixels=h * w)
    run_sequence(make_dataset("replica_semnerf", os.path.join(root, "toy", "Sequence_1")), False)
    if reset:
        field.calibrate_change_threshold(0.999)
    run_sequence(make_dataset("replica_semnerf", os.path.join(root, "toy", "Sequence_2")), True)
    out = tmp_path / f"field_{int(reset)}.pt"
    torch.save(dict(field=field.state_dict(), pca=pca.state_dict(), baselines=baselines_state([nl]),
                    config=dict(feature_transform="usplat.swap:swap_factory",
                                feature_transform_arg=arg)), out)
    return out, field


def test_end_to_end_cpu(tmp_path):
    root = str(tmp_path / "Replica")
    _write_replica(os.path.join(root, "toy", "Sequence_1"), 6)
    _write_replica(os.path.join(root, "toy", "Sequence_2"), 4)
    spec_path, _ = _toy_spec(tmp_path)
    xs, ys = torch.arange(-1.9, 1.95, 0.1), torch.arange(-1.45, 1.5, 0.1)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    N = means.shape[0]
    torch.save(dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                    scales=torch.full((N, 3), 0.06), opacities=torch.full((N,), 0.9)),
               tmp_path / "gs.pt")
    # 1. bank: pairs by the rule, real target features
    bank_cli = _load_cli("make_swap_bank")
    bank_path = tmp_path / "banks" / "toy.pt"
    bank = bank_cli.main(["--spec", spec_path, "--scene", "toy", "--root", root,
                          "--extractor", "lseg", "--device", "cpu", "--out", str(bank_path)],
                         extractor=FakeExtractor())
    assert [(p["source_name"], p["target_name"], p["kind"]) for p in bank["pairs"]] == [
        ("chair", "table", "near"), ("sofa", "lamp", "far")]
    assert set(bank["segments"]) == {CLS["table"], CLS["lamp"]}
    p = _protos()
    for b, segs in bank["segments"].items():
        for s in segs:
            cos = Fn.normalize(s["feats"].float(), dim=-1) @ p[b]
            assert float(cos.mean()) > 0.9 and float(cos.min()) > 0.5   # its own features
                                                          # (token upsampling blurs the edges)
    rec = json.load(open(f"{bank_path}.json"))
    assert rec["status"] == "ok" and rec["metrics"]["pairs"][0]["target_id"] == CLS["table"]
    # 2. fields with and without the reset, through the shared-interface call
    f1, field = _map_fields(tmp_path, root, spec_path, bank_path, True, "f1.frames.json")
    f0, field0 = _map_fields(tmp_path, root, spec_path, bank_path, False, "f0.frames.json")
    assert int((field.n_resets >= 1).sum()) > 0 and int(field0.n_resets.sum()) == 0
    log = json.load(open(tmp_path / "f1.frames.json"))
    assert [f["i"] for f in log["frames"]] == [0, 1, 2, 3]
    # 3. ground truth (mock backend, the frames the transform logged)
    cli = _load_cli("eval_change")
    gt_path = tmp_path / "gt" / "toy.pt"
    gt = cli.main(["gt-swap", "--ply", str(tmp_path / "gs.pt"), "--data",
                   os.path.join(root, "toy", "Sequence_2"), "--update-stride", "1",
                   "--frames-log", str(tmp_path / "f1.frames.json"), "--bank", str(bank_path),
                   "--spec", spec_path, "--scene", "toy", "--downscale", "16", "--backend", "mock",
                   "--device", "cpu", "--out", str(gt_path)])
    sh = gt["shares"]
    assert sh["n_swapped"] > 100 and sh["n_static"] > 100 and sh["n_views"] == 4
    assert json.load(open(f"{gt_path}.json"))["status"] == "ok"
    # 4. evaluation and the criterion
    out = tmp_path / "rec" / "eval_val_toy.json"
    cli.main(["eval", "--field", str(f1), "--gt", str(gt_path), "--config", spec_path,
              "--scene", "toy", "--split", "val", "--compare-field", f"ours_noreset={f0}",
              "--out", str(out)])
    m = json.load(open(out))["metrics"]
    assert m["compare_fields"]["ours_noreset"]["n_resets_total"] == 0
    acc = {k: v["acc_changed"] for k, v in m["seg"]["rows"].items()}
    assert acc["ours"] > acc["ours_noreset"] and acc["ours"] > acc["averaging"], acc
    assert m["reset"]["recall"] > 0.5 and 0.0 <= m["reset"]["static_reset_mass"] <= 1.0
    assert m["rows"]["ours_c_max"]["gaussian"]["auroc"] > 0.8
    cr = tmp_path / "rec" / "criterion_val.json"
    cli.main(["criterion", "--config", spec_path, "--split", "val", "--records", str(out),
              "--out", str(cr)])
    c = json.load(open(cr))["metrics"]
    assert c["criteria"]["swap_reset_v1"]["binding"]
    assert c["mapping_codes"] == [f"None|{SW.code_sha256()}"]          # field + transform code
    conds = c["criteria"]["swap_reset_v1"]["scenes"]["toy"]["conditions"]
    assert conds[0]["ok"] and conds[1]["ok"]
    # 5. a frames log of another spec is refused
    bad = json.load(open(tmp_path / "f1.frames.json"))
    bad["spec_sha256"] = "0" * 64
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="another spec"):
        cli.main(["gt-swap", "--ply", str(tmp_path / "gs.pt"), "--data",
                  os.path.join(root, "toy", "Sequence_2"), "--update-stride", "1",
                  "--frames-log", str(tmp_path / "bad.json"), "--bank", str(bank_path),
                  "--spec", spec_path, "--scene", "toy", "--downscale", "16", "--backend", "mock",
                  "--device", "cpu", "--plain-json", "--out", str(tmp_path / "x.pt")])


class _CpuGSplat:
    """GSplatFeatureRasterizer stand-in on the CPU (eval_seg.PinholeMockRasterizer) with the
    calls run_mapping.run_gsplat makes: render_image, render_depth, gaussian_depths."""

    @classmethod
    def from_splats(cls, splats, viewmat, K, w, h, **kw):
        import sys
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from eval_seg import PinholeMockRasterizer
        o = cls()
        o.r = PinholeMockRasterizer(splats, viewmat, K, w, h)
        o.h, o.w = h, w
        o.width, o.height = w, h
        o.num_gaussians = o.r.num_gaussians
        o.z = splats["means"].float() @ viewmat[2, :3] + viewmat[2, 3]
        return o

    def __call__(self, buf):
        return self.r(buf)

    @torch.no_grad()
    def render_image(self, colors):
        W = self.r._r.weights()
        return (W @ colors).reshape(self.h, self.w, -1), W.sum(1).reshape(self.h, self.w)

    @torch.no_grad()
    def render_depth(self):
        W = self.r._r.weights()
        a = W.sum(1)
        d = torch.where(a > 1e-6, W @ self.z / a.clamp_min(1e-12), torch.zeros_like(a))
        return d.reshape(self.h, self.w), a.reshape(self.h, self.w)

    def gaussian_depths(self):
        return self.z


def _toy_scene(tmp_path, n1=8, n2=6):
    root = str(tmp_path / "Replica")
    _write_replica(os.path.join(root, "toy", "Sequence_1"), n1)
    _write_replica(os.path.join(root, "toy", "Sequence_2"), n2)
    spec_path, _ = _toy_spec(tmp_path)
    xs, ys = torch.arange(-1.9, 1.95, 0.1), torch.arange(-1.45, 1.5, 0.1)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    N = means.shape[0]
    torch.save(dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                    scales=torch.full((N, 3), 0.06), opacities=torch.full((N,), 0.9)),
               tmp_path / "gs.pt")
    return root, spec_path


def _run_mapping_cpu(monkeypatch):
    """scripts/run_mapping.py with its gsplat backend on the CPU (--device cpu)."""
    import usplat.extractors as EX
    import usplat.gsplat_adapter as GA
    monkeypatch.setattr(GA, "GSplatFeatureRasterizer", _CpuGSplat)
    monkeypatch.setattr(EX, "get_extractor", lambda name, device="cpu", **kw: FakeExtractor())
    return _load_cli("run_mapping")


def _swap_fields(tmp_path, monkeypatch, root, spec_path, reset_extra=(), noreset_extra=()):
    """The reset and no-reset fields of scripts/replica_swap.sh (its run_mapping.py arguments,
    scaled to the toy scene), mapped by the REAL run_gsplat."""
    rm = _run_mapping_cpu(monkeypatch)
    bank = tmp_path / "banks" / "toy.pt"
    if not bank.exists():
        _load_cli("make_swap_bank").main(["--spec", spec_path, "--scene", "toy", "--root", root,
                                          "--extractor", "lseg", "--device", "cpu",
                                          "--out", str(bank)], extractor=FakeExtractor())
    fd = tmp_path / "fields"
    common = ["--backend", "gsplat", "--device", "cpu", "--dataset", "replica_semnerf",
              "--data", f"{root}/toy/Sequence_1", "--stride", "1", "--downscale", "16",
              "--ply", str(tmp_path / "gs.pt"), "--extractor", "lseg", "--feat-dim", "8",
              "--pca-frames", "4", "--seed", "0", "--use-depth",
              "--update-data", f"{root}/toy/Sequence_2", "--update-stride", "1",
              "--feature-transform", "usplat.swap:swap_factory", "--kf-trans", "0.005"]
    arg = f"bank={bank},spec={spec_path},scene=toy,log={fd}/toy_swap%s.frames.json"
    F, F0 = str(fd / "toy_swap.pt"), str(fd / "toy_swap_noreset.pt")
    rm.main(common + ["--baselines", "--change-baselines", "--oracle-depth",
                      "--feature-transform-arg", arg % "", "--out", F] + list(reset_extra))
    rm.main(common + ["--feature-transform-arg", arg % "_noreset", "--fixed-change-thr", "inf",
                      "--pca-from", F, "--out", F0] + list(noreset_extra))
    return str(bank), F, F0


def test_run_mapping_checkpoint_to_eval_change(tmp_path, monkeypatch):
    """From the checkpoint run_mapping.py's real run_gsplat saves (CPU stand-ins for gsplat and
    LSeg) with replica_swap.sh's arguments to eval_change gt-swap / eval / criterion: the
    residual rows the run maps (ncc, feature, oracle depth) and normlift_drop are scored, not
    missing; the saved scores cover the update views only (equal to a ChangeBaselines fed
    only those views); the no-reset field carries no baselines and is still accepted."""
    root, spec_path = _toy_scene(tmp_path)
    bank, F, F0 = _swap_fields(tmp_path, monkeypatch, root, spec_path)
    ck, ck0 = (torch.load(p, map_location="cpu", weights_only=False) for p in (F, F0))
    assert ck["change_baselines_scope"] == "update" and ck["change_baselines_views"] == 6
    assert ck["change_baselines_config"]["oracle_depth"] is True
    assert {k.split("/")[0] for k in ck["change_scores"]} == {
        "ncc", "feat", "depth_gauss_oracle", "depth_render_oracle"}
    assert ck["normlift_R_before_update"].shape == (ck["field"]["c_last"].numel(),)
    assert ck0["baselines"] == [] and "change_scores" not in ck0
    assert ck0.get("normlift_R_before_update") is None
    cli = _load_cli("eval_change")
    gt = tmp_path / "gt" / "toy.pt"
    cli.main(["gt-swap", "--ply", str(tmp_path / "gs.pt"), "--data", f"{root}/toy/Sequence_2",
              "--update-stride", "1", "--frames-log", str(tmp_path / "fields" / "toy_swap.frames.json"),
              "--bank", bank, "--spec", spec_path, "--scene", "toy", "--downscale", "16",
              "--backend", "mock", "--device", "cpu", "--out", str(gt)])
    out = tmp_path / "rec" / "eval_val_toy.json"
    cli.main(["eval", "--field", F, "--gt", str(gt), "--config", spec_path, "--scene", "toy",
              "--split", "val", "--compare-field", f"ours_noreset={F0}", "--out", str(out)])
    m = json.load(open(out))["metrics"]
    st = {k: v["status"] for k, v in m["rows"].items()}
    for r in ("ours_c_max", "ncc", "feature_residual", "depth_oracle", "normlift_drop"):
        assert st[r] == "ok", (r, st[r])
        assert m["rows"][r]["gaussian"]["coverage"] > 0, r
    assert m["rows"]["depth_oracle"]["source"].startswith("change_baselines:depth_render_oracle")
    for r in ("depth_residual", "fused", "css_var", "yager_conflict"):      # not applicable
        assert st[r].startswith("missing"), (r, st[r])
    assert m["seg"]["rows"]["averaging"].get("status") is None          # from the reset field
    cli.main(["criterion", "--config", spec_path, "--split", "val", "--records", str(out),
              "--out", str(tmp_path / "rec" / "criterion_val.json")])
    # the saved scores are those of the update views alone
    import usplat.change_eval as CE
    sc, why = CE._cb_scores(ck)
    assert why is None and set(sc) == set(ck["change_scores"])
    assert any(bool(torch.isfinite(v).any()) for v in sc.values())
    # a ground truth rendered over other frames or another transform code is refused
    g = torch.load(gt, weights_only=False)
    assert len(g["logged_frames"]) == 6 and g["logged_frames"][0][:2] == [0, 0]
    ev = lambda path: cli.main(["eval", "--field", F, "--gt", str(path), "--config", spec_path,
                                "--split", "val", "--plain-json", "--out", str(tmp_path / "x.json")])
    for name, bad, msg in (("fewer", dict(logged_frames=g["logged_frames"][::2]), "other frames"),
                           ("px", dict(logged_frames=[[i, j, [[c, n + 1] for c, n in px]]
                                                      for i, j, px in g["logged_frames"]]),
                            "other frames"),
                           ("code", dict(transform_code_sha256="0" * 64), "transform code"),
                           ("old", dict(logged_frames=None), "no logged frames")):
        torch.save(dict(g, **bad), tmp_path / f"gt_{name}.pt")
        with pytest.raises(ValueError, match=msg):
            ev(tmp_path / f"gt_{name}.pt")


# ------------------------------------------------------------------ the shell script
def _run_sh(tmp_path, stage, **env):
    e = dict(os.environ, DRY_RUN="1", STAGE=stage, ROOT=str(tmp_path / "Replica"),
             MAPS=str(tmp_path / "maps"), SWAP=str(tmp_path / "swap"), OUT=str(tmp_path / "out"),
             PYTHONDONTWRITEBYTECODE="1")
    e.update(env)
    return subprocess.run(["bash", os.path.join(REPO, "scripts", "replica_swap.sh")],
                          capture_output=True, text=True, env=e, timeout=300)


def _commands(stdout):
    return [shlex.split(line[2:]) for line in stdout.splitlines() if line.startswith("+ ")]


def _run_mapping_flags():
    src = open(os.path.join(REPO, "scripts", "run_mapping.py")).read()
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))


def _tracked_dirty():
    r = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no", "--",
                        "scripts", "usplat", "configs"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def test_replica_swap_sh_dry_run(tmp_path):
    spec, sha = SW.load_spec(SPEC)
    (tmp_path / "maps").mkdir()
    r = _run_sh(tmp_path, "val")
    assert r.returncode == 1 and "refused: no map" in r.stdout
    for s in spec["scenes"]["val"] + spec["scenes"]["test"]:
        (tmp_path / "maps" / f"{s}_seq1.pt").write_bytes(b"")
    r = _run_sh(tmp_path, "val")
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = _commands(r.stdout)
    assert len(cmds) == 5 * 2 + 1                         # 5 steps per val scene + criterion
    mapping_flags = _run_mapping_flags()
    shared = {"--feature-transform", "--feature-transform-arg",     # the T2 shared interface
              "--fixed-change-thr", "--change-baselines"}           # and T2's change flags
    ec = _load_cli("eval_change").build_parser()
    mb = _load_cli("make_swap_bank").build_parser()
    root = str(tmp_path / "Replica")
    for k, s in enumerate(spec["scenes"]["val"]):
        bank, reset, noreset, gt, ev = cmds[5 * k: 5 * k + 5]
        assert bank[:2] == ["scripts/gpu_gate.sh", "5000"] and bank[3] == "scripts/make_swap_bank.py"
        a = mb.parse_args(bank[4:])
        assert a.scene == s and a.root == root and a.out.endswith(f"banks/{s}.pt")
        for cmd in (reset, noreset):
            assert cmd[:4] == ["scripts/gpu_gate.sh", "10000", "python3", "scripts/run_mapping.py"]
            flags = {t for t in cmd[4:] if t.startswith("--")}
            assert flags - mapping_flags <= shared, flags - mapping_flags
            v = dict(zip(cmd[4:], cmd[5:]))
            assert v["--data"] == f"{root}/{s}/Sequence_1" and v["--update-data"] == f"{root}/{s}/Sequence_2"
            assert v["--stride"] == "5" and v["--update-stride"] == "5" and v["--downscale"] == "2"
            assert v["--feature-transform"] == "usplat.swap:swap_factory"
            arg = SW.parse_arg(v["--feature-transform-arg"])
            assert arg["scene"] == s and arg["spec"] == "configs/swap_replica.json"
            assert arg["bank"] == a.out and arg["log"].endswith(".frames.json")
        v0 = dict(zip(noreset[4:], noreset[5:]))
        assert v0["--fixed-change-thr"] == "inf" and float(v0["--fixed-change-thr"]) == float("inf")
        assert v0["--pca-from"] == dict(zip(reset[4:], reset[5:]))["--out"]
        assert "--field-config" not in reset and "--field-config" not in noreset
        assert "--fixed-change-thr" not in reset
        assert "--change-baselines" in reset and "--change-baselines" not in noreset
        assert gt[:4] == ["scripts/gpu_gate.sh", "7000", "python3", "scripts/eval_change.py"]
        g = ec.parse_args(gt[4:])
        assert g.cmd == "gt-swap" and g.downscale == 2 and g.backend == "gsplat"
        assert g.frames_log == SW.parse_arg(dict(zip(reset[4:], reset[5:]))["--feature-transform-arg"])["log"]
        assert ev[:2] == ["python3", "scripts/eval_change.py"]
        e = ec.parse_args(ev[2:])
        assert e.cmd == "eval" and e.split == "val" and e.scene == s and e.reference == ""
        assert e.compare_field == [f"ours_noreset={v0['--out']}"]
    c = ec.parse_args(cmds[-1][2:])
    assert c.cmd == "criterion" and len(c.records) == 2 and c.reference == ""
    # existing banks / fields / GT are reused
    sw = tmp_path / "swap"
    for sub, name in (("banks", "office_0.pt"), ("fields", "office_0_swap.pt"),
                      ("fields", "office_0_swap_noreset.pt"), ("gt", "office_0.pt")):
        (sw / sub).mkdir(parents=True, exist_ok=True)
        (sw / sub / name).write_bytes(b"")
    assert len(_commands(_run_sh(tmp_path, "val").stdout)) == 1 + 5 + 1
    # the test stage
    r = _run_sh(tmp_path, "test")
    assert r.returncode == 1
    if _tracked_dirty():
        assert "uncommitted changes" in r.stdout
        return
    assert "run STAGE=val first" in r.stdout + r.stderr
    out = tmp_path / "out"
    rec = dict(status="ok", metrics=dict(split="val", config_sha256=sha, scenes=spec["scenes"]["val"]))
    (out / "criterion_val.json").write_text(json.dumps(dict(rec, metrics=dict(rec["metrics"],
                                                                                 config_sha256="0"))))
    r = _run_sh(tmp_path, "test")
    assert r.returncode == 1 and "another" in r.stdout + r.stderr
    (out / "criterion_val.json").write_text(json.dumps(rec))
    r = _run_sh(tmp_path, "test")
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = _commands(r.stdout)
    assert len(cmds) == 5 * 6 + 1
    c = ec.parse_args(cmds[-1][2:])
    assert c.split == "test" and c.reference.endswith("criterion_val.json") and len(c.records) == 6
    evs = [ec.parse_args(cmd[2:]) for cmd in cmds if cmd[:3] == ["python3", "scripts/eval_change.py", "eval"]]
    assert len(evs) == 6 and all(e.split == "test" and e.reference == c.reference for e in evs)
    (out / "criterion_test.json").write_text(json.dumps(dict(status="ok")))
    r = _run_sh(tmp_path, "test")
    assert r.returncode == 1 and "finished test run" in r.stdout
    r = _run_sh(tmp_path, "test", FORCE_TEST="1")
    assert r.returncode == 0
    ft = [cmd for cmd in _commands(r.stdout) if cmd[:3] in (
        ["python3", "scripts/eval_change.py", "eval"], ["python3", "scripts/eval_change.py", "criterion"])]
    assert len(ft) == 6 + 1 and all("--force-test" in cmd for cmd in ft)
    # a finished per-scene test record of another commit is refused
    (out / "criterion_test.json").unlink()
    (out / "eval_test_room_0.json").write_text(json.dumps(dict(status="ok", git=dict(commit="0" * 40))))
    r = _run_sh(tmp_path, "test")
    assert r.returncode == 1 and "another commit" in r.stdout
