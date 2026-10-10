"""
vmf_field.py — per-Gaussian directional (von Mises–Fisher) semantic field.

Evidence is counted in VIEWS, not pixels. Per keyframe v, one render + one backward
(the gradient trick, see bki_accumulate.py) gives for every Gaussian n
    R_{v,n} = sum_p w_{p,n} z_p ,   m_{v,n} = sum_p w_{p,n}      (w = T * alpha)
The view contributes ONE observation: the per-view mean feature u_{v,n} = R_{v,n}/m_{v,n}
(||u|| <= 1: its length is the within-view coherence) with a view weight
    w_{v,n} = min(1, m_{v,n} / (a0 * H * W))                      (F1)
so a Gaussian that fills a few CLIP tokens counts as one view, whatever the image
resolution, and pixels of one view are no longer treated as independent samples.

Sufficient statistics per channel (S = sum over views):
    R = S w u,  W1 = S w,  W2 = S w^2,  Q = S w^2 ||u||^2,  Wr = S w ||u||
Read-outs
    mu*        = R/||R||   (+ optional text prior kappa0 mu0)
    n_K        = W1^2 / W2                                  Kish effective views
    support    = min(n_K, W1)   = W1 with view weights: w <= 1 gives W2 <= W1, so n_K >= W1
                 and the min never binds (it binds only with pixel-mass weights > 1)
    agreement  = sqrt((||R||^2 - Q)/(W1^2 - W2))  "kish" (default): the argument of the sqrt
                 is unbiased for rho^2 = ||E u||^2 under independent views (the sqrt keeps a
                 small downward bias at small n, and the clamp at 0 an upward one); it removes
                 the ~1/sqrt(n) bias of r_bar WITHOUT making the value depend on n    (F2)
               = ||R|| / (W1 + c0)          "joint": MAP of the joint conjugate prior
                 p(mu,kappa) ∝ C_d(kappa)^c0 e^{kappa m0.mu}, m0=0; A_d(kappa_hat) = agreement.
                 It shrinks toward 0 when evidence is scarce, i.e. it mixes the amount of
                 evidence into the value (a reliability score, like NormLift's R). It is
                 therefore used for kappa_hat and the posterior predictive, and is only an
                 ablation option for the aleatoric channel.
    epistemic  = 1/(1 + support)  (default; or n_K, views, pixels, or the posterior spread
                 1 - A_d(kappa_post))
    aleatoric  = 1 - agreement, defined once support >= tau_min

Centred channel (F3, OFF by default): z_hat = (z - zbar)/||z - zbar||, zbar = scene mean
feature from the mapping frames. It was meant to remove the common component of raw CLIP
features, but with correct features it lowered both change and aleatoric AUROC on the
7-Scenes replay, so the default reads the raw channel everywhere; centre=True is an ablation.

Change test (F4): the evidence is split into two DISJOINT buckets, recent (fast) and older
(slow). Each view enters the fast bucket, whose content leaks into the slow bucket at rate
1 - beta per effective view: F <- beta F + w u, S <- lambda S + (1 - beta) F_old, so S + F is
all the evidence (lambda = 1). c = 1 - cos(S, F) compares old with recent evidence; a slow
bucket that also absorbed the newest views would hide the change it is meant to detect.
The score is the min of the last k = change_persist defined c values. The default beta = 0,
k = 2 won the chess grid and EQUALS the classic feature residual (the last view against all
earlier views, persisting two views); beta > 0 is an ablation.
change_score="normalized" divides c by an i.i.d.-views null E0[1 - cos]; MEASURED NEGATIVE
(consecutive views are correlated; AUROC 0.75 -> 0.60 on the 7-Scenes replay), ablation only.

Threshold (calibrate_change_threshold): map with change_thr = inf (the default: scores are
recorded, nothing is reset), then set the threshold from the field's OWN mapping pass, which
is static by assumption. calib_mode="history" (default) keeps a histogram of every per-view
score, bucketed by the slow-bucket history (edges 4, 6, 10, 20, 40 views), and takes its
q-quantile per bucket (q = change_quantile = 0.999, a per-view false-alarm rate); a bucket
with fewer than MIN_NULL = 200 scores borrows the nearest bucket that has enough. Short
histories are noisier, so one pooled threshold would be too tight for them and too loose for
long ones. Ablations: calib_mode="pooled" (one q-quantile of all per-view scores) and "max"
(q-quantile of the per-Gaussian max over the mapping pass). Field states saved before
2026-09-29 called the history rule "pooled"; from_state maps them to "history".
When c > threshold the Gaussian's evidence is reset and it is flagged `changed` (F5). A
changed Gaussian re-enters the robust set only after `readmit_views` new views AND a small
observed-depth residual (non-oracle depth input), so a removed object's Gaussians cannot
come back with stale geometry.

Test record: after each view last_tested (N,) marks the Gaussians whose persistent score was
defined (they were tested) and last_fired those above the threshold; n_tests / n_fires count
both per history bucket. On a static sequence n_fires / n_tests is the REALISED per-view
false-alarm rate that the quantile q targets (scripts/run_mapping.py --summary-json).

Change-stream weight (change_weight, opt-in; default "view" as above). The test is defined
only after tau_min views of mass a0*H*W (7.7 px on Replica at 320x240), and most Gaussians
collect a few pixels over the whole mapping pass, so most are never tested. Alternatives that
count a view by how much of the Gaussian it shows, not by its pixel size:
    "visibility"  w = clamp(m_v / m_unocc_v, 0, 1),  m_unocc = o 2 pi / sqrt(det conic), the
                  splat mass of the projected 2D Gaussian without occlusion (from the
                  rasterizer, ViewStats.m_unocc): the transmittance-weighted visible share
    "relmax"      w = m_v / max_{v' <= v} m_v'   (running max of the raw mass, per Gaussian)
They enter ONLY the change-test stream and views_since_reset (re-admission): the posterior
statistics R, W1, W2, Q, Wr are bit-identical across change_weight settings as long as the
same Gaussians are reset (tests/test_change_coverage.py). With fast_memory = 0 the recent
bucket is the current view alone (nf = w <= 1), so fast_min = 1 asks for a view of weight 1:
under "relmax" only a view that sets a new running max of m has w = 1, and under "visibility"
even a fully visible Gaussian gets slightly less (gsplat clamps alpha at 0.999 and cuts it
below 1/255, the mock rasterizer's pixel grid discretises the footprint). Either mode with
fast_memory = 0 and fast_min >= 1 would test almost nothing, so validate() refuses it: pair
them with fast_min < 1 (e.g. fast_min = slow_min = 0.5). The slow bucket accumulates every
older view, so slow_min = 1 does not degenerate.

Every fix has a switch in FieldConfig so the ablations can turn it off one at a time;
FieldConfig.legacy() reproduces the field at git ab28fb4 (tests/test_legacy.py).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, NamedTuple, Optional
import torch
import torch.nn.functional as Fn

from .bki_accumulate import bki_grad_accumulate_groups
from .vmf_math import a_d, a_d_inv

_EPS = 1e-12


@dataclass
class FieldConfig:
    # evidence (F1)
    evidence: str = "views"          # "views" | "pixels" (legacy: evidence = pixel mass)
    view_mass_frac: float = 1e-4     # a0: pixel mass (fraction of H*W) for a full view
    # estimator (F2)
    estimator: str = "kish"          # aleatoric: "kish" | "rbar" (uncorrected ||R||/W1) |
                                     # "joint" | "legacy" (old r_tilde)
    c0: float = 1.0                  # joint-prior pseudo-count (prior strength in views)
    tau_min: float = 3.0             # support (views) before aleatoric/change are defined
    epistemic: str = "support"       # "support" | "kish" (NormLift parity) | "views" |
                                     # "posterior" | "pixels"
    # likelihood / prior: only scale kappa* (they never change mu*, agreement or decisions)
    kappa_mode: str = "map"          # "map": per-Gaussian kappa_hat | "fixed": kappa_lik
    kappa_lik: float = 10.0
    prior_kappa: float = 0.0         # text prior strength kappa0
    # centred channel (F3)
    centre: bool = False             # centred channel (F3). Default OFF: with correct CLIP
                                     # features it lowered both change and aleatoric AUROC
                                     # on the 7-Scenes replay (report 2026-09-29)
    centre_eps: float = 0.05         # pixels with ||z - zbar|| below this are skipped
    # change test and consolidation (F4, F5)
    change_test: bool = True
    slow_decay: float = 1.0          # lambda per effective view on the slow (older) bucket
    fast_memory: float = 0.0         # beta: retention per effective view of the fast bucket.
                                     # 0 (+ change_persist=2) won the chess grid (report
                                     # 2026-09-29): it equals the classic feature residual
    fast_min: float = 1.0            # fast-bucket mass (views) needed before c is defined
    change_persist: int = 2          # k: a change must persist k consecutive defined views
                                     # (score = min of the last k c values); fast_memory = 0
                                     # with k = 2 is the classic feature-residual test
    slow_min: float = 1.0            # slow-bucket mass (views) needed before c is defined
    change_score: str = "raw"        # "raw": c | "normalized": c / E0[c] (see above)
    change_thr: float = float("inf") # c above this => changed. inf = uncalibrated: scores
                                     # are recorded, nothing resets. Set it per scene with
                                     # calibrate_change_threshold() after the mapping pass
    change_quantile: float = 0.999   # q of the mapping-pass null (per-view rate 1 - q; over V
                                     # update views the sequence-level rate is ~V (1 - q))
    calib_mode: str = "history"      # null = the per-view scores of the mapping pass (thr =
                                     # per-view false-alarm quantile, independent of sequence
                                     # length). "history": one quantile per slow-bucket history
                                     # size; "pooled": one quantile for all; "max": per-Gaussian
                                     # max over the mapping pass (inflated by short histories)
    change_weight: str = "view"      # weight of a view in the change-test stream and in
                                     # views_since_reset (never in the posterior): "view"
                                     # min(1, m/(a0 H W)); "visibility" min(1, m/m_unocc)
                                     # (rasterizer's unoccluded splat mass); "relmax"
                                     # m / running max_v m. Opt-in, needs fast_min < 1 with
                                     # fast_memory = 0 (see the module docstring)
    readmit_views: float = 3.0       # views after a reset before re-admission
    readmit_depth: Optional[float] = 0.05  # |depth residual| [m]; None = views only
    r_stable: float = 0.6            # agreement needed for the robust set

    def to_dict(self):
        return asdict(self)

    @classmethod
    def legacy(cls, **kw):
        """The field at git ab28fb4 (before F1-F9): pixel evidence, r_tilde shrinkage,
        raw channel only, no change test, epistemic 1/(1 + pixel mass), fixed kappa,
        min_obs 5. For the ablation baseline; checked by tests/test_legacy.py."""
        base = dict(evidence="pixels", estimator="legacy", centre=False, change_test=False,
                    epistemic="pixels", kappa_mode="fixed", kappa_lik=10.0, tau_min=5.0)
        base.update(kw)
        return cls(**base)

    def validate(self):
        if self.change_score == "normalized" and self.slow_decay < 1.0:
            raise ValueError("change_score='normalized' assumes slow_decay = 1")
        for k, ok in (("evidence", ("views", "pixels")),
                      ("estimator", ("kish", "rbar", "joint", "legacy")),
                      ("epistemic", ("support", "kish", "views", "posterior", "pixels")),
                      ("change_score", ("raw", "normalized")),
                      ("kappa_mode", ("map", "fixed")),
                      ("calib_mode", ("history", "pooled", "max")),
                      ("change_weight", ("view", "visibility", "relmax"))):
            if getattr(self, k) not in ok:
                raise ValueError(f"{k}={getattr(self, k)!r} not in {ok}")
        if self.change_weight != "view" and self.fast_memory == 0 and self.fast_min >= 1:
            # recent bucket = this view, w = 1 only at a record ("relmax") or a fully
            # unclamped ("visibility") view: the test would almost never be defined
            raise ValueError(f"change_weight={self.change_weight!r} with fast_memory = 0 needs "
                             f"fast_min < 1 (got {self.fast_min:g}), e.g. fast_min = slow_min "
                             "= 0.5")


class ViewStats(NamedTuple):
    """Per-view, per-Gaussian sums from one backward pass (for baselines/hooks)."""
    R: torch.Tensor                  # (N, d)  sum_p w z_p          (raw)
    m: torch.Tensor                  # (N,)    sum_p w              (raw pixel mass)
    Rc: Optional[torch.Tensor]       # (N, d)  sum_p w z_hat_p       (centred)
    mc: Optional[torch.Tensor]       # (N,)
    Ds: Optional[torch.Tensor]       # (N,)    sum_p w depth_p       (observed depth)
    Dm: Optional[torch.Tensor]       # (N,)
    ref: float                       # a0 * H * W
    m_unocc: Optional[torch.Tensor] = None   # (N,) unoccluded splat mass o 2 pi/sqrt(det conic)
                                             # (change_weight="visibility" only)


class _Channel:
    """Sufficient statistics of one directional channel."""

    def __init__(self, N, d, device, dtype):
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        self.R, self.W1, self.W2, self.Q, self.Wr = z(N, d), z(N), z(N), z(N), z(N)

    def add(self, w, u, decay=None):
        un = u.norm(dim=-1)
        if decay is not None:
            self.R *= decay[:, None]; self.W1 *= decay; self.Wr *= decay
            self.W2 *= decay * decay; self.Q *= decay * decay
        self.R += w[:, None] * u
        self.W1 += w
        self.W2 += w * w
        self.Q += w * w * un * un
        self.Wr += w * un

    def reset(self, mask):
        for t in (self.R, self.W1, self.W2, self.Q, self.Wr):
            t[mask] = 0

    def scale(self, mask, lam):
        """Legacy forgetting: scale the statistics of `mask` by lam."""
        self.R[mask] *= lam; self.W1[mask] *= lam; self.Wr[mask] *= lam
        self.W2[mask] *= lam * lam; self.Q[mask] *= lam * lam

    def kish(self):
        # W2 > 0 (not > 1e-12): a Gaussian with W1 ~ 1e-6 views still has n_K >= W1, so its
        # support is W1 (the paper's s = W1), not 0
        tiny = torch.finfo(self.W2.dtype).tiny
        return torch.where(self.W2 > 0, self.W1 ** 2 / self.W2.clamp_min(tiny),
                           torch.zeros_like(self.W1))

    def support(self):
        """min(Kish views, total view mass): many weak views count as little evidence,
        one dominant view among weak ones counts as about one."""
        return torch.minimum(self.kish(), self.W1)

    def rbar(self):
        return self.R.norm(dim=-1) / self.W1.clamp_min(_EPS)

    def agreement(self, estimator, c0):
        Rn = self.R.norm(dim=-1)
        if estimator == "joint":
            return Rn / (self.W1 + c0).clamp_min(_EPS)
        if estimator == "kish":
            den = self.W1 ** 2 - self.W2
            ok = den > 1e-6 * self.W1 ** 2 + _EPS
            rho2 = ((Rn ** 2 - self.Q) / den.clamp_min(_EPS)).clamp(0.0, 1.0)
            return torch.where(ok, rho2.sqrt(), torch.full_like(Rn, float("nan")))
        if estimator == "rbar":                   # uncorrected mean length (ablation)
            return self.rbar()
        if estimator == "legacy":
            r = self.rbar()
            return (r - (1.0 - r) / (self.W1 + 1.0)).clamp(0.0, 1.0)
        raise ValueError(f"unknown estimator {estimator!r}")

    def within(self):
        return self.Wr / self.W1.clamp_min(_EPS)


# per-Gaussian / per-bucket tensors of state_dict (after the channels). States saved before
# the test record (last_tested ... n_fires) or m_max lack them: from_state keeps the zeros
_STATE_TENSORS = ("Rf", "nf", "Rs", "ns", "fB2", "fC", "fCq", "fBq2", "T2", "TQ", "pixel_mass",
                  "depth_abs", "depth_w", "depth_last", "views_since_reset", "c_ring", "c_hist",
                  "c_last", "c_max", "changed", "n_resets", "last_tested", "last_fired",
                  "n_tests", "n_fires", "m_max")


class DirectionalSemanticField:
    HIST_EDGES = (4.0, 6.0, 10.0, 20.0, 40.0)   # slow-bucket history (views) buckets
    MIN_NULL = 200                              # null samples a bucket needs for its own thr

    def __init__(self, num_gaussians, feat_dim, config: Optional[FieldConfig] = None,
                 prior_mu=None, device="cpu", dtype=torch.float32, **overrides):
        cfg = config or FieldConfig()
        # backward-compatible keyword overrides, e.g. kappa_lik=..., prior_kappa=...
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown field option {k!r}")
            setattr(cfg, k, v)
        cfg.validate()
        self.cfg = cfg
        self.N, self.d = int(num_gaussians), int(feat_dim)
        self.device, self.dtype = torch.device(device), dtype
        N, d = self.N, self.d
        self.raw = _Channel(N, d, self.device, dtype)
        self.cen = _Channel(N, d, self.device, dtype)         # centred, all evidence
        z = lambda *s: torch.zeros(*s, device=self.device, dtype=dtype)
        self.Rf, self.nf = z(N, d), z(N)                       # centred, recent bucket
        self.Rs, self.ns = z(N, d), z(N)                       # centred, older bucket
        # fast-bucket weight sums for the no-change expectation (b_i = current fast weight
        # of view i, w_i its full weight): fB2 = S b^2, fC = S w b, fCq = S w b ||u||^2,
        # fBq2 = S b^2 ||u||^2; slow weights are a_i = w_i - b_i (lambda = 1)
        self.fB2, self.fC, self.fCq, self.fBq2 = z(N), z(N), z(N), z(N)
        self.T2, self.TQ = z(N), z(N)                          # S w^2, S w^2 ||u||^2 (stream)
        k = max(1, int(cfg.change_persist))
        self.c_ring = torch.full((N, k), float("nan"), device=self.device, dtype=dtype)
        # histograms of the per-view scores while uncalibrated (the mapping-pass null), one per
        # history-size bucket: a score computed from a short history is noisier, so each
        # Gaussian is tested against the null of ITS current history size
        self._c_edges = torch.linspace(0.0, 2.0, 4001, device=self.device, dtype=torch.float64)
        self._hist_edges = torch.tensor(self.HIST_EDGES, device=self.device, dtype=dtype)
        nb = len(self.HIST_EDGES) + 1
        self.c_hist = torch.zeros(nb, 4000, device=self.device, dtype=torch.float64)
        self.thr_by_hist = None                                # (nb,) after calibration
        # test record: last_tested = the persistent score was defined at the last view (the
        # Gaussian was tested), last_fired = it exceeded the threshold; n_tests / n_fires count
        # them per history bucket over the field's life (fires / tests = realised per-view
        # false-alarm rate on a static sequence)
        self.last_tested = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.last_fired = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.n_tests = torch.zeros(nb, dtype=torch.int64, device=self.device)
        self.n_fires = torch.zeros(nb, dtype=torch.int64, device=self.device)
        self.m_max = z(N)                                      # running max_v m_v ("relmax")
        self.pixel_mass = z(N)                                 # sum_v m_v (ablation)
        self.depth_abs, self.depth_w = z(N), z(N)              # |observed - Gaussian| depth
        self.views_since_reset = z(N)                          # view units in every mode
        self.depth_seen = False                                # any depth stream so far
        self.depth_last = torch.full((N,), float("nan"), device=self.device, dtype=dtype)
        self.c_last = torch.full((N,), float("nan"), device=self.device, dtype=dtype)
        self.c_max = torch.zeros(N, device=self.device, dtype=dtype)
        self.changed = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.n_resets = torch.zeros(N, dtype=torch.int32, device=self.device)
        self.zbar = None
        self._warned = set()
        self.view_hooks: list[Callable[[ViewStats], None]] = []
        if prior_mu is not None and cfg.prior_kappa > 0:
            pm = prior_mu.to(self.device, dtype)
            pm = pm.expand(N, d) if pm.dim() == 1 else pm
            self.prior_mu = Fn.normalize(pm, dim=-1)
        else:
            self.prior_mu = None

    # ------------------------------------------------------------------ setup
    def set_centre(self, zbar):
        """Scene mean feature (from the mapping frames only, before accumulation)."""
        self.zbar = None if zbar is None else zbar.to(self.device, self.dtype).reshape(-1)

    # ---------------------------------------------------------------- updates
    def update_view(self, rasterize_fn, z_obs, pixel_valid=None, depth_obs=None,
                    gauss_depth=None, num_pixels=None):
        """Ingest one keyframe.
        rasterize_fn : (buf:(N, C)) -> (P, C), affine in buf (geometry detached)
        z_obs        : (P, d) unit per-pixel features
        pixel_valid  : (P,) bool/float mask (e.g. padding), optional
        depth_obs    : (P,) observed depth [m] (sensor or estimated; 0 = invalid), optional
        gauss_depth  : (N,) camera-frame depth of the Gaussian centres; needed with depth_obs
        num_pixels   : H*W of the view (defaults to P)
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError("update_view runs a backward pass (the gradient trick); call "
                               "it outside torch.inference_mode() (torch.no_grad() is fine)")
        cfg = self.cfg
        z = z_obs.to(self.device, self.dtype)
        P = z.shape[0]
        a = (torch.ones(P, device=self.device, dtype=self.dtype) if pixel_valid is None
             else pixel_valid.to(self.device, self.dtype))
        groups = [(z, a)]
        if cfg.centre:
            zhat, ok = self._centred(z)
            groups.append((zhat, a * ok))
        use_depth = depth_obs is not None
        if use_depth:
            assert gauss_depth is not None, "depth_obs needs gauss_depth"
            dep = depth_obs.to(self.device, self.dtype).reshape(-1)
            good = torch.isfinite(dep) & (dep > 0)
            dep = torch.where(good, dep, torch.zeros_like(dep))    # NaN/inf never enter
            groups.append((dep[:, None], a * good.to(self.dtype)))
        with torch.enable_grad():        # the accumulation IS a backward pass
            out = bki_grad_accumulate_groups(rasterize_fn, groups, self.N,
                                             device=self.device, dtype=self.dtype)
        (R_v, m_v) = out[0]
        Rc_v, mc_v = out[1] if cfg.centre else (None, None)
        Ds_v, Dm_v = (out[-1][0][:, 0], out[-1][1]) if use_depth else (None, None)
        ref = cfg.view_mass_frac * float(num_pixels or P)
        m_unocc = None
        if cfg.change_weight == "visibility":
            fn = getattr(rasterize_fn, "unoccluded_mass", None)
            if fn is None:
                raise ValueError("change_weight='visibility' needs a rasterizer with "
                                 "unoccluded_mass() (MockFeatureRasterizer, "
                                 "GSplatFeatureRasterizer)")
            with torch.no_grad():
                m_unocc = fn().to(self.device, self.dtype).reshape(-1)
        stats = ViewStats(R_v, m_v, Rc_v, mc_v, Ds_v, Dm_v, ref, m_unocc)
        self._ingest(stats, gauss_depth)
        for hook in self.view_hooks:
            hook(stats)
        return stats

    def update_from_rasterizer(self, rasterize_fn, z_obs, **kw):
        """Backward-compatible name for update_view."""
        return self.update_view(rasterize_fn, z_obs, **kw)

    @torch.no_grad()
    def update_sparse(self, gauss_ids, weights, z, num_pixels, normalize_z=True, m_unocc=None):
        """One view from explicit (gaussian_id, weight, feature) triples (m_unocc: (N,)
        unoccluded splat mass, needed only with change_weight="visibility")."""
        gid = gauss_ids.to(self.device).long()
        w = weights.to(self.device, self.dtype)
        z = z.to(self.device, self.dtype)
        if normalize_z:
            z = Fn.normalize(z, dim=-1)
        R_v = torch.zeros(self.N, self.d, device=self.device, dtype=self.dtype)
        m_v = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        R_v.index_add_(0, gid, w[:, None] * z)
        m_v.index_add_(0, gid, w)
        Rc_v = mc_v = None
        if self.cfg.centre:
            zhat, ok = self._centred(z)
            wc = w * ok
            Rc_v = torch.zeros_like(R_v).index_add_(0, gid, wc[:, None] * zhat)
            mc_v = torch.zeros_like(m_v).index_add_(0, gid, wc)
        stats = ViewStats(R_v, m_v, Rc_v, mc_v, None, None,
                          self.cfg.view_mass_frac * float(num_pixels),
                          None if m_unocc is None else m_unocc.to(self.device, self.dtype))
        self._ingest(stats, None)
        for hook in self.view_hooks:
            hook(stats)
        return stats

    def _centred(self, z):
        """(z - zbar)/||z - zbar|| and the mask ||z - zbar|| > eps (shared by both paths)."""
        if self.zbar is None and "zbar" not in self._warned:
            import warnings
            warnings.warn("centre=True but set_centre() was not called: the centred channel "
                          "equals the raw channel (set zbar from the mapping frames)")
            self._warned.add("zbar")
        zc = z - self.zbar if self.zbar is not None else z
        nrm = zc.norm(dim=-1)
        return zc / nrm.clamp_min(_EPS)[:, None], (nrm > self.cfg.centre_eps).to(self.dtype)

    def _view_weights(self, m, ref):
        """w = min(1, m / ref): one view = one observation (F1)."""
        seen = m > _EPS
        w = (m / ref).clamp(max=1.0)
        return torch.where(seen, w, torch.zeros_like(w)), seen

    def _change_weights(self, m, s: ViewStats):
        """Weight of a view in the change-test stream and in views_since_reset
        (FieldConfig.change_weight): w = clamp(m / den, 0, 1) with den = a0*H*W ("view", the
        default: _view_weights), the unoccluded splat mass s.m_unocc ("visibility") or the
        running max of the raw mass m_max, this view included ("relmax"). The posterior
        channels never see it."""
        mode = self.cfg.change_weight
        if mode == "view":
            return self._view_weights(m, s.ref)
        if mode == "visibility":
            if s.m_unocc is None:
                raise ValueError("change_weight='visibility' needs ViewStats.m_unocc (a "
                                 "rasterizer with unoccluded_mass())")
            den = s.m_unocc.to(self.device, self.dtype)
        else:                                                  # "relmax"
            den = self.m_max
        seen = m > _EPS
        w = (m / den.clamp_min(_EPS)).clamp(0.0, 1.0)
        return torch.where(seen, w, torch.zeros_like(w)), seen

    def _weights(self, m, ref):
        """Evidence weights of the posterior channels: views (F1) or pixel mass (legacy).
        The change test, re-admission and depth bookkeeping always use view weights."""
        if self.cfg.evidence == "views":
            return self._view_weights(m, ref)
        seen = m > _EPS
        return torch.where(seen, m, torch.zeros_like(m)), seen

    @torch.no_grad()
    def _ingest(self, s: ViewStats, gauss_depth):
        cfg = self.cfg
        w, seen = self._weights(s.m, s.ref)
        if cfg.change_weight == "relmax":
            self.m_max = torch.maximum(self.m_max, s.m)       # this view included: w <= 1
        wv, _ = self._change_weights(s.m, s)                   # views_since_reset
        u = s.R / s.m.clamp_min(_EPS)[:, None]
        self.pixel_mass += s.m
        if cfg.centre:
            wc, seen_c = self._weights(s.mc, s.ref)
            uc = s.Rc / s.mc.clamp_min(_EPS)[:, None]
            # change-test stream: centred features, view weights
            wt, seen_t = self._change_weights(s.mc, s)
            ut = uc
        else:
            wt, seen_t, ut = wv, seen, u

        # ---- change test: recent (fast) vs older (slow) evidence, in view units
        if cfg.change_test:
            wc_, uc_ = wt, ut
            keep = cfg.fast_memory ** wc_                      # 1 where unseen
            leak = 1.0 - keep
            if cfg.slow_decay < 1.0:
                ds = cfg.slow_decay ** wc_                     # per view unit of the stream
                self.Rs.mul_(ds[:, None]); self.ns.mul_(ds)
            self.Rs.add_(leak[:, None] * self.Rf); self.ns.add_(leak * self.nf)
            un2 = (uc_ * uc_).sum(-1)
            w2 = wc_ * wc_
            self.fB2.mul_(keep * keep).add_(w2)
            self.fC.mul_(keep).add_(w2)
            self.fCq.mul_(keep).add_(w2 * un2)
            self.fBq2.mul_(keep * keep).add_(w2 * un2)
            self.Rf.mul_(keep[:, None]).add_(wc_[:, None] * uc_)
            self.nf.mul_(keep).add_(wc_)
            tot2 = self.T2 + w2                                # sum w^2 incl. this view
            totq = self.TQ + w2 * un2
            defined = (seen_t & (torch.minimum(self.ns + self.nf, (self.ns + self.nf) ** 2
                                               / tot2.clamp_min(_EPS)) >= cfg.tau_min)
                       & (self.nf >= cfg.fast_min) & (self.ns >= cfg.slow_min))
            c = 1.0 - Fn.cosine_similarity(self.Rs, self.Rf, dim=-1, eps=1e-8)
            if cfg.change_score == "normalized":
                c = c / self._null_dissimilarity(tot2, totq)
            # persistence: keep the last k defined c values; the score is their minimum
            ring = torch.cat([self.c_ring[:, 1:], c[:, None]], dim=1)
            self.c_ring = torch.where(defined[:, None], ring, self.c_ring)
            c_p = self.c_ring.min(dim=1).values                # NaN until k values exist
            ok = defined & torch.isfinite(c_p)
            self.c_last = torch.where(ok, c_p, self.c_last)
            self.c_max = torch.where(ok, torch.maximum(self.c_max, c_p), self.c_max)
            hb = torch.bucketize(self.ns, self._hist_edges)     # history-size bucket
            if self.thr_by_hist is None and cfg.change_thr == float("inf") and bool(ok.any()):
                v = c_p[ok].double().clamp(0.0, 2.0 - 1e-9)     # mapping pass: record null
                bins = (v * 2000).long().clamp(max=3999)
                self.c_hist.index_put_((hb[ok], bins), torch.ones_like(v), accumulate=True)
            thr = self.thr_by_hist[hb] if self.thr_by_hist is not None else cfg.change_thr
            fire = ok & (c_p > thr)
            self.last_tested, self.last_fired = ok, fire
            self.n_tests += torch.bincount(hb[ok], minlength=self.n_tests.numel())
            self.n_fires += torch.bincount(hb[fire], minlength=self.n_fires.numel())
            if bool(fire.any()):
                self.raw.reset(fire); self.cen.reset(fire)
                self.depth_abs[fire] = 0; self.depth_w[fire] = 0
                self.views_since_reset[fire] = 0
                self.c_ring[fire] = float("nan")
                self.Rs[fire] = 0; self.ns[fire] = 0
                self.Rf[fire] = wc_[fire, None] * uc_[fire]   # recent bucket = this view
                self.nf[fire] = wc_[fire]
                self.fB2[fire] = w2[fire]; self.fC[fire] = w2[fire]
                self.fCq[fire] = (w2 * un2)[fire]; self.fBq2[fire] = (w2 * un2)[fire]
                self.T2[fire] = 0; self.TQ[fire] = 0
                self.changed |= fire
                self.n_resets += fire.to(torch.int32)
            self.T2 += w2
            self.TQ += w2 * un2

        # ---- add the view
        self.raw.add(w, u)
        if cfg.centre:
            self.cen.add(wc, uc)
        self.views_since_reset += wv

        # ---- observed-depth residual (non-oracle depth input) and re-admission
        if s.Ds is not None and gauss_depth is not None:
            self.depth_seen = True
            wd, seen_d = self._view_weights(s.Dm, s.ref)
            res = s.Ds / s.Dm.clamp_min(_EPS) - gauss_depth.to(self.device, self.dtype)
            res = torch.where(seen_d, res, torch.zeros_like(res))
            self.depth_last = torch.where(seen_d, res, self.depth_last)
            self.depth_abs += wd * res.abs()
            self.depth_w += wd
        if cfg.change_test and bool(self.changed.any()):
            enough = self.views_since_reset >= cfg.readmit_views
            if cfg.readmit_depth is None:
                ok = enough
            elif not self.depth_seen:
                # no depth stream at all: the depth check cannot run; fall back to views
                if "nodepth" not in self._warned:
                    import warnings
                    warnings.warn("readmit_depth is set but no depth was ever supplied: "
                                  "re-admission falls back to views only")
                    self._warned.add("nodepth")
                ok = enough
            else:
                # a depth stream exists: fail closed for Gaussians without depth evidence
                dres = self.depth_abs / self.depth_w.clamp_min(_EPS)
                ok = enough & (self.depth_w > 0) & (dres <= cfg.readmit_depth)
            self.changed &= ~ok

    @torch.no_grad()
    def calibrate_change_threshold(self, q=None):
        """Set change_thr from the field's OWN mapping pass (no labels): the mapping
        sequence is static by assumption, so its scores are null samples. Map with
        change_thr = inf, call this, then run the update sequence. A raw threshold on c does
        not transfer between scenes (feature noise differs); a quantile of the scene's own
        null does. calib_mode "history" (default): every per-view score, one q-quantile per
        slow-bucket history bucket (thr_by_hist; q is a per-view false-alarm rate); the
        returned change_thr is the quantile of all buckets pooled. "pooled": that pooled
        quantile for every Gaussian (no thr_by_hist). "max": the q-quantile of each
        Gaussian's max c over the mapping pass."""
        q = self.cfg.change_quantile if q is None else q
        if self.cfg.calib_mode in ("history", "pooled"):
            counts = self.c_hist.sum(1)
            if float(counts.sum()) == 0:
                raise RuntimeError("no defined change score was recorded during mapping")

            def quant(h):
                cdf = torch.cumsum(h, 0) / h.sum()
                i = int(torch.searchsorted(cdf, torch.tensor([q], dtype=cdf.dtype,
                                                             device=cdf.device)).clamp(max=3999))
                return float(self._c_edges[i + 1])            # upper edge of the q bin
            thr = quant(self.c_hist.sum(0))                    # pooled (reported, fallback)
            if self.cfg.calib_mode == "pooled":
                self.thr_by_hist = None
                self.cfg.change_thr = thr
                self.c_max.zero_()
                self.c_last.fill_(float("nan"))
                return thr
            good = [b for b in range(len(counts)) if counts[b] >= self.MIN_NULL]
            per = []
            for b in range(len(counts)):                       # nearest bucket with data
                src = min(good, key=lambda g: abs(g - b)) if good else None
                per.append(quant(self.c_hist[src]) if src is not None else thr)
            self.thr_by_hist = torch.tensor(per, device=self.device, dtype=self.dtype)
        else:
            null = self.c_max[torch.isfinite(self.c_last)]
            if null.numel() == 0:
                raise RuntimeError("no Gaussian had a defined change score during mapping")
            thr = float(torch.quantile(null.float(), q))
        self.cfg.change_thr = thr
        self.c_max.zero_()
        self.c_last.fill_(float("nan"))
        return thr

    @torch.no_grad()
    def set_change_threshold(self, thr):
        """One fixed threshold for every history size (the fixed-threshold ablation: a value
        carried over from the validation scenes instead of the scene's own null). Like
        calibrate_change_threshold it starts a fresh c_max / c_last record."""
        self.thr_by_hist = None
        self.cfg.change_thr = float(thr)
        self.c_max.zero_()
        self.c_last.fill_(float("nan"))
        return self.cfg.change_thr

    def _null_dissimilarity(self, W2, Q):
        """E0[1 - cos(S, F)] when both buckets sample the same distribution
        (slow_decay = 1). Each view is split between the buckets (a_i = w_i - b_i in S,
        b_i in F), so E[S.F] = rho^2 A B + (q - rho^2) sum_i a_i b_i, with
        sum_i a_i b_i = fC - fB2."""
        A, B = self.ns, self.nf
        A2 = (W2 - 2.0 * self.fC + self.fB2).clamp_min(_EPS)
        QS = (Q - 2.0 * self.fCq + self.fBq2).clamp_min(0.0)
        den = (A * A - A2)
        rho2 = ((self.Rs.pow(2).sum(-1) - QS) / den.clamp_min(_EPS)).clamp(0.0, 1.0)
        rho2 = torch.where(den > 1e-6 * A * A + _EPS, rho2, torch.zeros_like(rho2))
        q = (Q / W2.clamp_min(_EPS)).clamp(0.0, 1.0)
        nS = A * A / A2
        nF = B * B / self.fB2.clamp_min(_EPS)
        x = q - rho2
        shared = (self.fC - self.fB2).clamp_min(0.0) / (A * B).clamp_min(_EPS)
        cos0 = (rho2 + x * shared) / torch.sqrt((rho2 + x / nS.clamp_min(_EPS)) *
                                                (rho2 + x / nF.clamp_min(_EPS))).clamp_min(1e-12)
        return (1.0 - cos0).clamp_min(1e-3)

    # ------------------------------------------------------------ read-outs
    @property
    def n_eff(self):
        """Evidence mass of the raw channel (views, or pixel mass in pixel mode)."""
        return self.raw.W1

    def kish_views(self, channel="raw"):
        return self._ch(channel).kish()

    def support(self, channel="raw"):
        return self._ch(channel).support()

    def _ch(self, channel):
        if channel == "raw":
            return self.raw
        if channel == "centred":
            if not self.cfg.centre:
                raise ValueError("centred channel disabled (centre=False)")
            return self.cen
        raise ValueError(channel)

    def _dispersion_channel(self):
        return "centred" if self.cfg.centre else "raw"

    def _support(self, channel):
        ch = self._ch(channel)
        if self.cfg.evidence == "pixels" or self.cfg.estimator == "legacy":
            return ch.W1                     # n_eff >= min_obs, as at git ab28fb4
        return ch.support()

    def mean_resultant_length(self, channel="raw"):
        return self._ch(channel).rbar()

    def agreement(self, channel=None):
        channel = channel or self._dispersion_channel()
        return self._ch(channel).agreement(self.cfg.estimator, self.cfg.c0)

    def within_view(self, channel=None):
        return self._ch(channel or self._dispersion_channel()).within()

    def across_view(self, channel=None):
        ch = self._ch(channel or self._dispersion_channel())
        return (ch.rbar() / ch.within().clamp_min(_EPS)).clamp(max=1.0)

    def kappa_hat(self):
        """Per-Gaussian MAP concentration of the joint prior (raw channel)."""
        r = self.raw.agreement("joint", self.cfg.c0).clamp(max=1 - 1e-9)
        return a_d_inv(self.d, r)

    def _kappa(self):
        if self.cfg.kappa_mode == "map":
            return self.kappa_hat()
        return torch.full((self.N,), self.cfg.kappa_lik, device=self.device, dtype=self.dtype)

    def eta(self):
        e = self._kappa()[:, None] * self.raw.R
        if self.prior_mu is not None:
            e = e + self.cfg.prior_kappa * self.prior_mu
        return e

    def kappa_post(self):
        return self.eta().norm(dim=-1)

    def mu_post(self):
        """mu* = eta/||eta|| (unit for any nonzero eta: F.normalize's eps = 1e-12 would leave
        Gaussians with ~1e-6 views of evidence shorter than 1); 0 where eta = 0 (unseen)."""
        e = self.eta()
        n = e.norm(dim=-1, keepdim=True)
        return torch.where(n > 0, e / n.clamp_min(torch.finfo(e.dtype).tiny), torch.zeros_like(e))

    def mean_direction(self):
        """E[mu] = A_d(kappa_post) mu*: shrinks weakly supported Gaussians to 0."""
        return a_d(self.d, self.kappa_post())[:, None] * self.mu_post()

    def angular_spread(self):
        """Exact posterior spread E[1 - mu*^T mu] = 1 - A_d(kappa_post)."""
        return 1.0 - a_d(self.d, self.kappa_post())

    def epistemic(self):
        mode = self.cfg.epistemic
        if mode == "support":
            return 1.0 / (1.0 + self.raw.support())
        if mode == "kish":
            return 1.0 / (1.0 + self.raw.kish())
        if mode == "views":
            return 1.0 / (1.0 + self.raw.W1)
        if mode == "pixels":
            return 1.0 / (1.0 + self.pixel_mass)
        if mode == "posterior":
            return self.angular_spread()
        raise ValueError(f"unknown epistemic mode {mode!r}")

    def aleatoric(self, min_obs=None):
        """(u_ale (N,) with NaN where undefined, valid mask)."""
        ch = self._dispersion_channel()
        tau = self.cfg.tau_min if min_obs is None else min_obs
        a = 1.0 - self.agreement(ch)
        valid = (self._support(ch) >= tau) & torch.isfinite(a)
        return torch.where(valid, a, torch.full_like(a, float("nan"))), valid

    def change_score(self):
        """Last defined c = 1 - cos(mu_slow, mu_fast) (NaN if never defined)."""
        return self.c_last

    def depth_residual(self):
        """Mean |observed depth - Gaussian depth| since the last reset (NaN if none)."""
        return torch.where(self.depth_w > 0, self.depth_abs / self.depth_w.clamp_min(_EPS),
                           torch.full_like(self.depth_abs, float("nan")))

    def robust_mask(self):
        ch = self._dispersion_channel()
        agree = self.agreement(ch)
        return ((self._support(ch) >= self.cfg.tau_min) & (agree >= self.cfg.r_stable)
                & ~self.changed)

    # --------------------------------------------------------- query / state
    @torch.no_grad()
    def query(self, text_embeds):
        """Cosine scores (N, K). Pass text already mapped into the field's feature
        space WITHOUT renormalising (see StreamingPCA.project_text)."""
        t = text_embeds.to(self.device, self.dtype)
        return self.mu_post() @ t.t()

    @torch.no_grad()
    def reset(self):
        self.raw.reset(slice(None)); self.cen.reset(slice(None))
        for t in (self.Rf, self.nf, self.Rs, self.ns, self.fB2, self.fC, self.fCq, self.fBq2,
                  self.T2, self.TQ, self.pixel_mass, self.depth_abs, self.depth_w,
                  self.views_since_reset, self.c_max, self.m_max):
            t.zero_()
        self.depth_last.fill_(float("nan")); self.c_last.fill_(float("nan"))
        self.c_ring.fill_(float("nan")); self.c_hist.zero_(); self.thr_by_hist = None
        self.changed.zero_(); self.n_resets.zero_()
        for t in (self.last_tested, self.last_fired, self.n_tests, self.n_fires):
            t.zero_()

    def state_dict(self):
        s = dict(config=self.cfg.to_dict(), N=self.N, d=self.d, dtype=str(self.dtype),
                 calib_names=2,                  # 2: "history" = history-conditioned rule
                 thr_by_hist=None if self.thr_by_hist is None else self.thr_by_hist.cpu(),
                 zbar=None if self.zbar is None else self.zbar.cpu(),
                 prior_mu=None if self.prior_mu is None else self.prior_mu.cpu(),
                 depth_seen=self.depth_seen)
        # tensors that are all zero (the centred channel with centre=False, the change-test
        # buckets with change_test=False) are saved as None: from_state keeps its zeros
        keep = lambda x: x.cpu() if bool(x.any()) else None
        for name in ("R", "W1", "W2", "Q", "Wr"):
            s[f"raw_{name}"] = getattr(self.raw, name).cpu()
            s[f"cen_{name}"] = keep(getattr(self.cen, name))
        for name in _STATE_TENSORS:
            x = getattr(self, name)
            s[name] = x.cpu() if name in ("depth_last", "c_ring", "c_last") else keep(x)
        return s

    @classmethod
    def from_state(cls, s, device="cpu"):
        dtype = getattr(torch, s.get("dtype", "torch.float32").replace("torch.", ""))
        cfg = dict(s["config"])
        if s.get("calib_names", 1) < 2 and cfg.get("calib_mode") == "pooled":
            cfg["calib_mode"] = "history"       # old name of the history-conditioned rule
        f = cls(s["N"], s["d"], FieldConfig(**cfg), prior_mu=s.get("prior_mu"),
                device=device, dtype=dtype)
        f.set_centre(s["zbar"])
        f.depth_seen = s.get("depth_seen", False)
        if s.get("thr_by_hist") is not None:
            f.thr_by_hist = s["thr_by_hist"].to(device, dtype)
        for name in ("R", "W1", "W2", "Q", "Wr"):
            getattr(f.raw, name).copy_(s[f"raw_{name}"])
            if s.get(f"cen_{name}") is not None:          # None: all zero when saved
                getattr(f.cen, name).copy_(s[f"cen_{name}"])
        for name in _STATE_TENSORS:
            if s.get(name) is not None:
                getattr(f, name).copy_(s[name])
        return f
