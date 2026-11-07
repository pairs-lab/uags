"""NormLift's Sec. 3.4 KNN mode voting (usplat.baselines.knn_mode_vote, NormLiftVote, the row
"normlift_vote"): exact KNN, equality with a line-by-line port of the official chunk loop
(kaust-violab/NormLift_release @ e0982e2, pipeline/knn_vote.py), chunk invariance, the row, and
eval_seg passing the Gaussian means to the factory. CPU only."""
import argparse
import importlib.util
import pathlib

import torch
import torch.nn.functional as F

from usplat import baselines as B

REPO = pathlib.Path(__file__).resolve().parents[1]


def _official(means, feats, conf, K=36, SIGMA_D=0.3, TAU_COS=0.55, GAMMA=0.1, DELTA=0.05,
              CHUNK_SIZE=4096):
    """The official main() steps 1-3, line by line, with a brute-force KNN in place of
    faiss.IndexFlatL2 (K + 1 nearest, first column = self, dropped)."""
    N = means.shape[0]
    D_sq = torch.cdist(means.double(), means.double()) ** 2
    D_sq, knn_idx = D_sq.sort(dim=1, stable=True)
    knn_idx, knn_dsq = knn_idx[:, 1:K + 1], D_sq[:, 1:K + 1].float()
    feats_n_dev = F.normalize(feats, dim=1, p=2)
    conf_dev = conf
    knn_dw_dev = torch.exp(-knn_dsq / (2 * SIGMA_D ** 2))
    target_idx_all = torch.arange(N)
    new_feats = feats.clone()
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        chunk_targets = target_idx_all[start:end]
        M = chunk_targets.shape[0]
        ext_idx = torch.cat([chunk_targets.unsqueeze(1), knn_idx[chunk_targets]], dim=1)
        ext_feats_n = feats_n_dev[ext_idx]
        ext_conf = conf_dev[ext_idx]
        ext_dw = torch.cat([torch.ones(M, 1, dtype=knn_dw_dev.dtype), knn_dw_dev[chunk_targets]], dim=1)
        cos_mat = torch.bmm(ext_feats_n, ext_feats_n.transpose(1, 2))
        g_jk = torch.sigmoid((cos_mat - TAU_COS) / GAMMA)
        weight_k = ext_conf * ext_dw
        weighted_g = (g_jk * weight_k.unsqueeze(1)).sum(dim=-1)
        S = ext_conf * weighted_g
        S_ii = S[:, 0]
        S_neighbors = S[:, 1:]
        max_S, argmax_in_nbrs = S_neighbors.max(dim=1)
        chosen_global_idx = ext_idx.gather(1, (argmax_in_nbrs + 1).unsqueeze(1)).squeeze(1)
        replace_mask = max_S > (S_ii + DELTA)
        if int(replace_mask.sum()) > 0:
            new_feats[chunk_targets[replace_mask]] = feats[chosen_global_idx[replace_mask]]
    return new_feats


def _scene(n=400, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * 2.0                        # a 2 m box: neighbours vary
    centers = torch.randn(5, d, generator=g)
    lab = torch.randint(0, 5, (n,), generator=g)
    feats = centers[lab] + 0.6 * torch.randn(n, d, generator=g)        # noisy class modes
    feats = feats / feats.norm(dim=1, keepdim=True) * torch.rand(n, 1, generator=g)   # ||f|| <= 1
    conf = torch.rand(n, generator=g)
    feats[:7] = 0.0                                                    # unseen: f = 0, R = 0
    conf[:7] = 0.0
    return means, feats, conf


def test_knn_is_exact_sorted_and_excludes_self():
    means, _, _ = _scene()
    idx, dsq = B.knn_indices(means, 36)
    D = torch.cdist(means.double(), means.double()) ** 2
    D.fill_diagonal_(float("inf"))
    ref_d, ref_i = D.sort(dim=1, stable=True)
    assert torch.equal(idx, ref_i[:, :36])
    assert torch.allclose(dsq.double(), ref_d[:, :36], atol=1e-6)
    assert not (idx == torch.arange(len(means))[:, None]).any()


def test_knn_with_an_exact_duplicate_keeps_the_twin_not_self():
    means, _, _ = _scene(n=100)
    means[1] = means[0]
    idx, dsq = B.knn_indices(means, 5)
    assert 0 not in idx[0].tolist() and idx[0, 0] == 1 and float(dsq[0, 0]) == 0.0
    assert 1 not in idx[1].tolist() and idx[1, 0] == 0


def test_vote_equals_the_official_loop():
    means, feats, conf = _scene()
    idx, dsq = B.knn_indices(means, 36)
    new, rep, src = B.knn_mode_vote(feats, conf, idx, dsq)
    ref = _official(means, feats, conf)
    assert torch.equal(new, ref)
    assert 0 < int(rep.sum()) < len(means)                              # the test has teeth
    assert torch.equal(rep, (src >= 0)) and torch.equal(new[rep], feats[src[rep]])
    assert torch.equal(new[~rep], feats[~rep])                          # kept as they were
    # chunking does not matter (one pass, simultaneous: every read is from the original)
    new7, rep7, _ = B.knn_mode_vote(feats, conf, idx, dsq, chunk=7)
    assert torch.equal(new7, new) and torch.equal(rep7, rep)
    assert torch.equal(_official(means, feats, conf, CHUNK_SIZE=7), ref)


def _normlift(n=60, d=4, views=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    nl = B.NormLift(n, d)
    for _ in range(views):
        m = torch.rand(n, generator=g) * 5.0
        m[:3] = 0.0                                                     # three unseen Gaussians
        R = torch.randn(n, d, generator=g) * m[:, None] * 0.3
        nl.on_view(type("V", (), dict(R=R, m=m))())
    return nl


def test_row_buffer_and_uncertainties():
    nl = _normlift()
    means = torch.rand(nl.N, 3, generator=torch.Generator().manual_seed(2))
    v = B.NormLiftVote(nl, means, params=dict(k=8))
    (row,) = v.rows()
    assert row.name == "normlift_vote" and row.kind == "feature"
    f = row.buffer()
    norms = f.norm(dim=1)
    assert torch.all(((norms - 1).abs() < 1e-5) | (norms == 0))         # unit or zero
    u, u_nl = row.uncertainties(), nl.uncertainties()
    assert set(u) == set(u_nl) and all(torch.equal(u[k], u_nl[k]) for k in u)   # R unchanged
    # the lifted feature is NormLift's Eq. 3 (unnormalized), 0 for the unseen
    lifted = v.lifted()
    assert torch.allclose(lifted[3:], nl.F[3:] / nl.S1[3:, None]) and torch.all(lifted[:3] == 0)
    # the voted buffer = the normalized official vote on (lifted, R)
    idx, dsq = B.knn_indices(means, 8)
    new, _, _ = B.knn_mode_vote(lifted, nl.reliability(), idx, dsq)
    want = F.normalize(new, dim=1)
    assert torch.allclose(f, want, atol=1e-6)
    assert v.stats["n"] == nl.N and v.stats["k"] == 8


def test_eval_seg_passes_the_means_to_the_factory():
    spec = importlib.util.spec_from_file_location("eval_seg", REPO / "scripts" / "eval_seg.py")
    ES = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ES)
    from usplat.vmf_field import DirectionalSemanticField
    nl = _normlift()
    field = DirectionalSemanticField(nl.N, nl.d)
    ckpt = dict(baselines=B.baselines_state([nl]))
    means = torch.rand(nl.N, 3, generator=torch.Generator().manual_seed(3))
    args = argparse.Namespace(row_factory="usplat.baselines:normlift_vote_factory",
                              rows="normlift_vote", no_p3=True, v2_rows=False, with_2d=False)
    rows = ES.candidate_rows(field, ckpt, "cpu", args, means=means)
    assert [r.name for r in rows] == ["normlift_vote"]
    assert rows[0].buffer().shape == (nl.N, nl.d)
    try:
        ES.candidate_rows(field, ckpt, "cpu", args)                     # no means: refused
    except ValueError as e:
        assert "means" in str(e)
    else:
        raise AssertionError("normlift_vote_factory without means must be refused")
