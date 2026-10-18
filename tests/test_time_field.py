"""scripts/time_field.py (the run-time paragraph of the paper) on the CPU, mock backend.

A tiny synthetic 7-Scenes-layout scene: 8 frames of a plane of Gaussians 2 m away (frame 3 is a
near-duplicate pose), a 3D-GS map with fit meta, a second map whose fit time is only in a fit
log, and a main field file in run_mapping.py's format with NormLift, view count and CSS. A fake
extractor stands in for LSeg (the GPU is busy, and LSeg is too slow on the CPU for a test).
 * the CLI writes the JSON: keyframes (duplicate dropped, evenly spaced subset, warm-up not
   counted), every part timed once per keyframe, total = sum of the parts per keyframe, no
   GPU numbers on the CPU, fit time from the meta and from the log, the LaTeX sentence (one
   result, and two results as 'first (second)');
 * the timed pass leaves the results unchanged: its field and baselines equal a plain update
   loop exactly, and the patched accumulation and _ingest are restored afterwards;
 * the keyframe rule (finite pose; a gap below kf_trans AND kf_rot is dropped; max_keyframes);
 * the state size of a small field by hand (with and without the change-test buckets);
 * lscpu parsing, a missing command, and the refusals (gsplat on the CPU, too few frames,
   another extractor configuration).
"""
import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest
import torch

from usplat import baselines as B
from usplat import bki_accumulate, vmf_field
from usplat.clip_dense import upsample_tokens
from usplat.feature_pca import StreamingPCA
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = pathlib.Path(__file__).resolve().parents[1]
D, d, K = 16, 8, 4                      # token size, field size (PCA), CSS classes


def _load():
    spec = importlib.util.spec_from_file_location("time_field", REPO / "scripts" / "time_field.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Fake:
    """Stands in for LSeg: a (H/16, W/16, D) token grid; the red band of a 16x16 block picks
    one of K class directions, the block colour fills three more channels."""
    d = D

    def extract_tokens(self, image):
        img = torch.as_tensor(image).float()
        H, W = img.shape[:2]
        blocks = img.reshape(H // 16, 16, W // 16, 16, 3).mean((1, 3))
        tok = torch.zeros(H // 16, W // 16, D)
        band = (blocks[..., 0] / 64).long().clamp(0, K - 1)
        tok.scatter_(-1, band[..., None], 1.0)
        tok[..., K:K + 3] = blocks / 255.0
        return tok


def _scene(tmp, n=8):
    """Writes the scene; returns (time_field module, common CLI argv)."""
    import cv2
    data, maps, logs = tmp / "data", tmp / "maps", tmp / "logs"
    for p in (data, maps, logs):
        p.mkdir()
    xs = [0.06 * i for i in range(n)]
    xs[3] = xs[2] + 0.01                      # frame 3: a near-duplicate of frame 2 (dropped)
    for i, x in enumerate(xs):
        base = str(data / f"frame-{i:06d}")
        img = np.zeros((480, 640, 3), np.uint8)
        img[..., 0] = (np.arange(640) // 160 * 64)[None, :]
        img[..., 1] = (np.arange(480) * 255 // 480)[:, None]
        cv2.imwrite(base + ".color.png", np.ascontiguousarray(img[..., ::-1]))   # RGB on disk
        cv2.imwrite(base + ".depth.png", np.full((480, 640), 2000, np.uint16))
        c2w = np.eye(4)
        c2w[0, 3] = x
        np.savetxt(base + ".pose.txt", c2w)
    gx, gy = torch.meshgrid(torch.arange(-1.2, 1.6, 0.1), torch.arange(-0.9, 0.91, 0.1),
                            indexing="xy")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    N = means.shape[0]
    splats = dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                  scales=torch.full((N, 3), 0.05), opacities=torch.full((N,), 0.9))
    meta = dict(time_init_s=2.0, time_train_s=58.0, wall_time_s=61.0, gpu="Test GPU",
                n_gaussians=N, iters=10)
    torch.save(dict(splats, meta=meta), maps / "scene_seq1.pt")
    torch.save(dict(splats), maps / "other_seq1.pt")          # no fit meta: its log has it
    (logs / "fit_other.log").write_text(
        "[fit] step 10/10 loss 0.01\n"
        + json.dumps(dict(time_train_s=120.0, wall_time_s=125.0,
                          out=str(maps / "other_seq1.pt"))) + "\n")
    pca = StreamingPCA(D, d)
    pca.basis = torch.eye(D)[:, :d]
    f = DirectionalSemanticField(N, d, FieldConfig())
    f.set_centre(torch.zeros(d))
    bs = B.default_baselines(f, text=pca.project_text(torch.eye(D)[:K]), class_ids=[1, 2, 3, 4])
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        feat = torch.nn.functional.normalize(torch.randn(N, d, generator=g), dim=-1)
        f.update_sparse(torch.arange(N), torch.ones(N), feat, num_pixels=100)
    cfg = dict(backend="gsplat", ply=str(maps / "scene_seq1.pt"), data=str(data),
               dataset="7scenes", stride=1, downscale=16, extractor="fake", feat_dim=d,
               kf_trans=0.05, kf_rot=5.0, max_keyframes=0, use_depth=True, channel_chunk=32,
               baselines=True, field_config="")
    tf = _load()
    torch.save(dict(field=f.state_dict(), pca=pca.state_dict(), config=cfg,
                    timing=dict(keyframes=n - 1), field_config=f.cfg.to_dict(),
                    extractor="fake", extractor_options={}, code=tf.RM.mapping_code_state(),
                    baselines=B.baselines_state(bs), css_class_ids=[1, 2, 3, 4]),
               tmp / "field.pt")
    return tf, ["--field", str(tmp / "field.pt"), "--device", "cpu", "--backend", "mock",
                "--fit-maps", str(maps / "*_seq1.pt"), "--fit-logs", str(logs / "*.log")]


def test_cli_cpu_mock(tmp_path, monkeypatch):
    tf, argv = _scene(tmp_path)
    monkeypatch.setattr(tf, "load_extractor", lambda cfg, device: Fake())
    path = tmp_path / "out" / "time.json"
    tf.main(argv + ["--frames", "6", "--warmup", "2", "--out", str(path)])
    rec = json.loads(path.read_text())
    k = rec["keyframes"]
    assert k["mapping"] == 7 and k["saved_run"] == 7 and 3 not in k["indices"], k
    assert k["timed"] == 6 and k["counted"] == 4 and k["indices"][0] == 0, k
    assert rec["grid_hw"] == [30, 40] and rec["image_hw"] == [480, 640]
    for p in ("extract", "to_cpu", "project", "render_backward", "stats", "other", "total"):
        st = rec["ms"][p]
        assert st["n"] == 4 and st["mean"] >= 0 and st["std"] >= 0, (p, st)
    for f in rec["per_keyframe_s"]["ours"]:                     # the parts add up per keyframe
        parts = f["extract"] + f["project"] + f["render_backward"] + f["stats"] + f["other"]
        assert math.isclose(f["total"], parts, rel_tol=1e-12) and f["other"] > -1e-9, f
    assert len(rec["per_keyframe_s"]["ours"]) == 6
    assert rec["ms_added"]["css"]["n"] == 4 and rec["ms_added"]["hooks"]["n"] == 4
    assert rec["memory_mib"]["peak_allocated"] is None and rec["memory_mib"]["map"] is None
    assert rec["change_test"]["n_resets"] == 0 and rec["change_test"]["n_scored"] > 0
    assert rec["code"]["same"] and rec["hardware"]["device"] == "cpu"
    assert "gpu" not in rec["hardware"]
    s = rec["state_bytes_per_gaussian"]
    assert s["posterior"] == (d + 4) * 4                         # R (d) + W1, W2, Q, Wr, float32
    assert math.isclose(sum(s["by_tensor"].values()), s["total"])
    assert math.isclose(s["total"] - s["without_change_buckets"], (2 * d + 2) * 4)
    # in memory, every tensor: raw and centred channels 2 (d + 4), buckets 2 d + 2, weight sums
    # 6, c_ring 2, eight per-Gaussian scalars (m_max included): 58 float32; changed,
    # last_tested, last_fired (bool) 3 B, n_resets 4 B; per scene: c_hist and its 4001 bin
    # edges (float64), 5 history edges, zbar (d floats), n_tests and n_fires (6 int64 each)
    fixed = 6 * 4000 * 8 + 4001 * 8 + 5 * 4 + d * 4 + 2 * 6 * 8
    assert math.isclose(s["in_memory"], 58 * 4 + 3 + 4 + fixed / rec["n_gaussians"])
    fit = rec["fit"]
    assert fit["scene"]["fit_s"] == 60.0 and fit["scene"]["parts"] == ["time_init_s",
                                                                       "time_train_s"]
    other = [m for m in fit["maps"] if m["map"].endswith("other_seq1.pt")][0]
    assert other["fit_s"] == 120.0 and other["source"].endswith("fit_other.log")
    assert fit["minutes"] == dict(mean=1.5, std=0.5, n=2)
    assert any("mock backend" in w for w in rec["warnings"])
    assert any("fit times come from" in w for w in rec["warnings"])
    tex = rec["latex"]
    ms = rec["ms"]
    assert f"${ms['total']['mean']:.1f}\\pm{ms['total']['std']:.1f}$\\,ms in total" in tex
    assert f"${ms['stats']['mean']:.1f}\\pm" in tex and "On 7-Scenes, a keyframe" in tex
    assert "Peak GPU memory is \\ph{}\\,GB." in tex              # no GPU number on the CPU
    assert f"takes {s['total']:.0f} bytes per Gaussian" in tex
    assert "takes $1.5\\pm0.5$\\,min per scene" in tex
    assert "%%   WARNING: mock backend" in tex
    assert "no other job running" not in tex.splitlines()[-1]   # not checked on the CPU
    assert fit["gpus"] == ["Test GPU", "unknown"]               # the log line has no GPU
    two = tf.main(["--latex-from", str(path), str(path)])        # Replica (RIO10) form
    tot = f"${ms['total']['mean']:.1f}\\pm{ms['total']['std']:.1f}$"
    assert "On 7-Scenes (7-Scenes)" in two and f"{tot} ({tot})\\,ms in total" in two
    # a GPU record: "no other job running" only when nvidia-smi listed nothing but this process
    me = str(rec["hardware"]["pid"])
    rec["hardware"].update(gpu="Test GPU", cpu="Intel Core i5-14600KF",
                           gpu_processes_start=[], gpu_processes_end=[dict(pid=me, name="python")])
    assert ("on one Test GPU with an Intel Core i5-14600KF and no other job running."
            in tf.latex([rec]))
    rec["hardware"]["gpu_processes_end"].append(dict(pid="1", name="other"))
    busy = tf.latex([rec])
    assert "no other job running" not in busy.splitlines()[-1]
    assert "%% WARNING: other GPU compute processes ran" in busy
    print(f"PASS time_field CLI (CPU, mock): total {ms['total']['mean']:.1f} ms, "
          f"state {s['total']:.0f} B per Gaussian")


def _same(a, b):
    assert set(a) == set(b)
    for key in a:
        x, y = a[key], b[key]
        if torch.is_tensor(x) or torch.is_tensor(y):
            torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True, msg=key)
        else:
            assert x == y, key


def test_timed_pass_equals_plain_updates(tmp_path):
    tf, argv = _scene(tmp_path)
    args = tf.build_parser().parse_args(argv + ["--out", str(tmp_path / "x.json")])
    ckpt = tf.load_field_file(str(tmp_path / "field.pt"))
    ctx = tf.context(ckpt, args)
    views = tf.extract_views(Fake(), ctx["ds"], ctx["idx"], 16, tf.Clock("cpu"))
    field, bs, clock = tf.map_pass(ctx, views, baselines=True)
    assert vmf_field.bki_grad_accumulate_groups is bki_accumulate.bki_grad_accumulate_groups
    assert "_ingest" not in vars(field)                          # the wrapper is gone
    assert clock.calls["render_backward"] == clock.calls["stats"] == len(views) == 7
    assert clock.calls["hooks"] == 2 * len(views) and clock.calls["css"] == len(views)
    # the same keyframes through a plain loop: no clock, no patch
    ref = DirectionalSemanticField(ctx["N"], ctx["d"], tf.field_config(ckpt))
    ref.set_centre(ctx["zbar"])
    rbs = tf.make_baselines(ref, ckpt, "cpu")
    raster = tf.rasterizer_factory("mock")
    for v in views:
        z = ctx["pca"].project(upsample_tokens(v["tok"], (v["h"], v["w"]))).reshape(-1, d)
        viewmat = torch.linalg.inv(v["c2w"].double()).float()
        ras = raster(ctx["splats"], viewmat, v["K"], v["w"], v["h"])
        gd = ctx["splats"]["means"] @ viewmat[2, :3] + viewmat[2, 3]
        ref.update_view(ras, z, num_pixels=v["h"] * v["w"], depth_obs=v["depth"].reshape(-1),
                        gauss_depth=gd)
        for c in rbs:
            if isinstance(c, B.CSS):
                c.update_view(ras, z)
    assert ref.depth_seen and bool((ref.raw.W1 > 0).any()) and bool((ref.c_max > 0).any())
    _same(field.state_dict(), ref.state_dict())
    assert [type(b).__name__ for b in bs] == ["NormLift", "ViewCount", "CSS"]
    for b, r in zip(bs, rbs):
        _same(b.state_dict(), r.state_dict())
    assert float(bs[2].counts.sum()) > 0
    print("PASS timed pass = plain update loop (field, NormLift, view count, CSS)")


class _Poses:
    def __init__(self, poses):
        self.poses = poses

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, i):
        return dict(c2w=self.poses[i])


def _pose(x, deg=0.0):
    a = math.radians(deg)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                                [-math.sin(a), 0, math.cos(a)]])
    c2w[0, 3] = x
    return c2w


def test_keyframe_rule():
    tf = _load()
    nan = torch.full((4, 4), float("nan"))
    ds = _Poses([_pose(0), _pose(0.01), _pose(0.10), nan, _pose(0.10, 10.0), _pose(0.12, 10.0)])
    # 1: 1 cm and 0 deg from 0 (both below: dropped); 3: not finite; 4: 0 cm but 10 deg (kept);
    # 5: 2 cm and 0 deg from 4 (dropped)
    assert tf.select_keyframes(ds, 0.05, 5.0) == [0, 2, 4]
    # max_keyframes = 3 looks at frames 0, 2, 5 only; 5 is 10 deg from 2: kept
    assert tf.select_keyframes(ds, 0.05, 5.0, max_keyframes=3) == [0, 2, 5]
    assert tf.evenly([0, 2, 4], 2) == [0, 4]
    assert tf.evenly([0, 2, 4], 0) == tf.evenly([0, 2, 4], 10) == [0, 2, 4]
    print("PASS keyframe rule")


def test_state_bytes_by_hand():
    tf = _load()
    N, dd = 10, 4
    f = DirectionalSemanticField(N, dd, FieldConfig())
    f.set_centre(torch.zeros(dd))
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        f.update_sparse(torch.arange(N), torch.ones(N), torch.randn(N, dd, generator=g),
                        num_pixels=100)
    s = tf.state_bytes(f.state_dict())
    # float32 per Gaussian: raw R (d) + W1 W2 Q Wr = 8; buckets Rf Rs (2d) + nf ns = 10;
    # weight sums fB2 fC fCq fBq2 T2 TQ = 6; c_ring (k = 2) + c_last + c_max = 4; pixel_mass,
    # views_since_reset, depth_last = 3; last_tested (bool) 1 B. Per scene: zbar (d floats),
    # c_hist 6 x 4000 float64, n_tests 6 int64 (counted with the read-outs by state_bytes).
    per_scene = (dd * 4 + 6 * 4000 * 8 + 6 * 8) / N
    assert s["posterior"] == 8 * 4
    assert math.isclose(s["total"], (8 + 10 + 6 + 4 + 3) * 4 + 1 + per_scene)
    assert math.isclose(s["total"] - s["without_change_buckets"], 10 * 4)
    assert math.isclose(s["without_change_test"], (8 + 3) * 4 + 1 + (dd * 4 + 6 * 8) / N)
    assert {"cen_R", "depth_abs", "depth_w", "changed", "n_resets"} <= set(s["saved_as_none"])
    print(f"PASS state bytes: {s['total']:.1f} B per Gaussian (N = {N}, d = {dd})")


def test_cpu_model_and_missing_command():
    tf = _load()
    text = "Architecture: x86_64\nModel name:   13th Gen Intel(R) Core(TM) i7-13700KF\nCPU(s): 24\n"
    assert tf.cpu_model(text) == "Intel Core i7-13700KF"
    assert tf.cpu_model("Model name: AMD Ryzen 9 7950X 16-Core Processor") == \
        "AMD Ryzen 9 7950X 16-Core Processor"
    assert tf.cpu_model("") is None
    assert tf._run(["no-such-command-for-uags-tests"]).startswith("error")
    assert tf.other_gpu_jobs(dict(pid=7, gpu_processes_start=[dict(pid="7"), dict(pid="9")],
                                  gpu_processes_end="error: no nvidia-smi")) == [dict(pid="9")]
    print("PASS machine record")


def test_refusals(tmp_path, monkeypatch):
    tf, argv = _scene(tmp_path)
    monkeypatch.setattr(tf, "load_extractor", lambda cfg, device: Fake())
    out = ["--out", str(tmp_path / "r.json")]
    with pytest.raises(SystemExit, match="needs a CUDA device"):
        tf.main(argv + out + ["--backend", "gsplat"])
    with pytest.raises(SystemExit, match="nothing left to count"):
        tf.main(argv + out + ["--frames", "3", "--warmup", "3"])
    ck = torch.load(tmp_path / "field.pt", weights_only=False)
    ck["extractor_options"] = {"token_stride": 4}
    torch.save(ck, tmp_path / "field.pt")
    with pytest.raises(SystemExit, match="extractor options"):
        tf.main(argv + out)
    assert not (tmp_path / "r.json").exists()
    print("PASS refusals")
