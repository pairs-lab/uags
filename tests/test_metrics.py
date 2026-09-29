"""K10a/F6: ranking, detection and calibration metrics; the predictive fits.

 * AUSE is invariant to the order of tied uncertainties, 0 for the oracle, and equal to
   the random gap for a constant uncertainty;
 * AUROC/AP agree with brute-force definitions (ties counted as 1/2 / grouped);
 * multiclass_calibration: a perfectly calibrated synthetic predictive has ECE ~ 0;
 * fit_temperature recovers the beta that generated the labels;
 * P1 (posterior mean direction) shrinks weakly supported Gaussians, P0 does not.
"""
import torch
import torch.nn.functional as Fn
from usplat.evaluation import (sparsification_curve, auroc, average_precision,
                               multiclass_calibration, aurc)
from usplat.predictive import fit_temperature, fit_adaptive, gaussian_buffer
from usplat.vmf_field import DirectionalSemanticField, FieldConfig


def test_ause_ties_and_limits():
    torch.manual_seed(0)
    M = 1000
    e = (torch.rand(M) < 0.3).double()
    u = torch.randint(0, 5, (M,)).double()               # heavy ties
    a1 = sparsification_curve(e, u)["ause"]
    perm = torch.randperm(M)
    a2 = sparsification_curve(e[perm], u[perm])["ause"]
    assert abs(a1 - a2) < 1e-12, (a1, a2)
    assert abs(sparsification_curve(e, e)["ause"]) < 1e-12
    out = sparsification_curve(e, torch.zeros(M))
    assert abs(out["ause"] - out["random_gap"]) < 1e-12
    print(f"PASS AUSE: tie-invariant ({a1:.4f}), oracle 0, constant u = random gap")


def test_auroc_ap_bruteforce():
    torch.manual_seed(1)
    s = torch.randint(0, 6, (300,)).double()
    y = (torch.rand(300) < 0.4).double()
    pos, neg = s[y == 1], s[y == 0]
    bf = ((pos[:, None] > neg[None]).double().mean() + 0.5 * (pos[:, None] == neg[None]).double().mean()).item()
    assert abs(auroc(s, y) - bf) < 1e-12
    # AP brute force over distinct thresholds
    ap, prev_r = 0.0, 0.0
    for thr in torch.unique(s).flip(0):
        sel = s >= thr
        p = y[sel].mean().item(); r = (y[sel].sum() / y.sum()).item()
        ap += (r - prev_r) * p; prev_r = r
    assert abs(average_precision(s, y) - ap) < 1e-12
    print(f"PASS AUROC {bf:.4f} and AP {ap:.4f} equal brute force with ties")


def test_calibration_and_temperature():
    torch.manual_seed(2)
    M, K, beta = 20000, 6, 7.0
    scores = Fn.normalize(torch.randn(M, 16), dim=-1) @ Fn.normalize(torch.randn(K, 16), dim=-1).t()
    p = torch.softmax(beta * scores, dim=1)
    y = torch.multinomial(p, 1).squeeze(1)
    cal = multiclass_calibration(p, y)
    assert cal["ece"] < 0.02, cal
    b = fit_temperature(scores, y)
    assert abs(b - beta) / beta < 0.08, b
    ada = fit_adaptive(scores, y, torch.zeros(M), torch.zeros(M))
    assert abs(ada["beta0"] - beta) / beta < 0.08, ada
    assert aurc(torch.zeros(10), torch.ones(10)) == 1.0
    print(f"PASS calibration: ECE {cal['ece']:.4f} for a calibrated model; beta fit {b:.2f} (true {beta})")


def test_posterior_predictive_shrinks():
    from usplat.vmf_math import a_d
    N, d = 3, 16
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False, view_mass_frac=0.01))
    z = Fn.normalize(torch.randn(d), dim=0)
    zs = Fn.normalize(z + 0.3 * torch.randn(200, d), dim=-1)
    # Gaussian 0: 1 view; Gaussian 1: 12 views; Gaussian 2: 200 views
    f.update_sparse(torch.tensor([0, 1, 2]), torch.ones(3), zs[:3], num_pixels=100)
    for v in range(3, 200):
        ids = torch.tensor([1, 2]) if v < 13 else torch.tensor([2])
        f.update_sparse(ids, torch.ones(len(ids)), zs[v:v + 1].expand(len(ids), d),
                        num_pixels=100)
    p0, p1 = gaussian_buffer(f, "P0").norm(dim=-1), gaussian_buffer(f, "P1").norm(dim=-1)
    assert torch.allclose(p0, torch.ones(N), atol=1e-5)
    assert p1[0] < p1[1] < p1[2] and p1[2] > 0.99, p1
    ref = a_d(d, f.kappa_hat() * f.raw.R.norm(dim=-1))          # A_d(kappa_hat ||R||)
    assert torch.allclose(p1, ref, atol=1e-5), (p1, ref)
    print(f"PASS P1: |E[mu]| = {p1[0]:.3f} / {p1[1]:.3f} / {p1[2]:.4f} at 1 / 12 / 200 views "
          f"= A_d(kappa_hat ||R||); P0 = 1")


def test_p2_recovers_slopes():
    torch.manual_seed(5)
    M, K = 40000, 5
    scores = Fn.normalize(torch.randn(M, 16), dim=-1) @ Fn.normalize(torch.randn(K, 16), dim=-1).t()
    u_epi = torch.rand(M)
    u_ale = torch.rand(M)
    u_ale[torch.rand(M) < 0.3] = float("nan")                  # undefined for 30%
    true = dict(beta0=12.0, a=1.5, b=0.8, g=1.0)
    from usplat.predictive import adaptive_beta
    beta = adaptive_beta(true["beta0"], true["a"], true["b"], u_epi, u_ale, true["g"])
    y = torch.multinomial(torch.softmax(beta[:, None] * scores, 1), 1).squeeze(1)
    fit = fit_adaptive(scores, y, u_epi, u_ale)
    for k in ("a", "b", "g"):
        assert abs(fit[k] - true[k]) < 0.25 * true[k] + 0.1, (k, fit, true)
    print(f"PASS P2: fitted a {fit['a']:.2f} (1.5), b {fit['b']:.2f} (0.8), g {fit['g']:.2f} (1.0)")


def test_calibration_metrics_bruteforce():
    torch.manual_seed(6)
    M, K = 50, 4
    p = torch.softmax(torch.randn(M, K) * 2, 1).double()
    y = torch.randint(0, K, (M,))
    out = multiclass_calibration(p, y)
    nll = sum(-float(torch.log(p[i, y[i]])) for i in range(M)) / M
    brier = sum(float(((p[i] - torch.eye(K, dtype=torch.float64)[y[i]]) ** 2).sum()) for i in range(M)) / M
    assert abs(out["nll"] - nll) < 1e-10 and abs(out["brier"] - brier) < 1e-10
    e = (torch.rand(200) < 0.4).double()
    orc, rnd = aurc(e, e), aurc(torch.zeros(200), e)
    anti = aurc(-e, e)
    assert orc < rnd < anti, (orc, rnd, anti)
    # AURC brute force: mean over k of the error rate of the k most certain elements
    u = torch.rand(200).double()
    order = torch.argsort(u)
    bf = sum(float(e[order[:k]].mean()) for k in range(1, 201)) / 200
    assert abs(aurc(u, e) - bf) < 1e-10
    print(f"PASS NLL/Brier/AURC = brute force; AURC oracle {orc:.3f} < random {rnd:.3f} < anti {anti:.3f}")


def test_undefined_policy():
    y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0.])
    s = torch.tensor([0.9, 0.8, float("nan"), float("nan"), 0.1, 0.2, 0.3, 0.4])
    filled = torch.where(torch.isnan(s), torch.full_like(s, -1e9), s)
    assert abs(auroc(s, y) - auroc(filled, y)) < 1e-12
    assert auroc(s, y) < 1.0 and auroc(s, y, undefined="drop") == 1.0
    assert average_precision(s, y) < 1.0
    e = torch.tensor([1.0, 0, 1, 0])
    out = sparsification_curve(e, torch.tensor([float("nan"), 0.1, 0.5, 0.2]), num_bins=4)
    assert out["n_undefined"] == 1
    print("PASS undefined scores rank last (AUROC/AP < 1 with 2 unscored positives); "
          "AUSE counts them")


if __name__ == "__main__":
    test_ause_ties_and_limits()
    test_auroc_ap_bruteforce()
    test_calibration_and_temperature()
    test_posterior_predictive_shrinks()
    test_p2_recovers_slopes()
    test_calibration_metrics_bruteforce()
    test_undefined_policy()
