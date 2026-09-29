"""
change_baselines.py — per-Gaussian change-detection baselines (plan K13, Tab II rows).

Each baseline is a per-pixel residual r_p between what the camera sees in an update view
and what the map renders at that pose, lifted to the Gaussians with the splat weights
w_{p,n} = T * alpha (the gradient trick, see bki_accumulate.py):
    r_{v,n} = sum_p w_{p,n} a_p r_p / sum_p w_{p,n} a_p            (a_p = pixel mask)
All residuals of one view come out of ONE render + ONE backward pass
(bki_grad_accumulate_groups: one target channel and one mass channel per residual).
Per-pixel inputs are detached first: only the splat weights enter the backward pass.

Residuals (higher = more likely changed)
  ncc           1 - NCC of the observed vs rendered grey value in a 7x7 window. Each window
                is normalised to zero mean and unit variance, so a GLOBAL gain/offset of the
                grey value (exposure) gives NCC = 1. Per-channel gains (white balance),
                clipping and gamma are not affine in grey and do change NCC. Windows whose
                std is below ncc_min_std in EITHER image are masked: NCC is undefined there
                (known limit: an object removed in front of a blank wall leaves a flat
                observed window and this row cannot see it).
  feat          1 - cos(rendered feature, observed feature) per pixel (MIF / CL-Splats).
  depth_gauss   |sum_p w d_p / sum_p w - z_n|: observed depth against the depth z_n of the
                Gaussian centre; the per-view term of DirectionalSemanticField.depth_residual.
  depth_render  |d_p - D_p| per pixel: observed against rendered (expected) depth, lifted.
  The depth rows are NON-ORACLE when depth_obs is sensor or estimated depth (Tango, MASt3R).
  When the caller passes rendered MESH depth as depth_obs, it builds the accumulator with
  oracle_depth=True and the rows are named depth_gauss_oracle / depth_render_oracle
  (upper-bound rows; never used in G3).

Observation (coverage) rule. Residual g has masked pixel mass m_g = sum_p w a in view v and
view weight w_v = min(1, m_g / (a0 H W)) (a0 as FieldConfig.view_mass_frac).
  default   observed iff m_g > EPS: the field's own "seen" rule (vmf_field._view_weights),
            i.e. min_view_weight = 0.
  gate      min_view_weight > 0 also needs w_v >= min_view_weight. A gate decides which
            Gaussians get a score at all, and with NaN ranked last it moves AP as much as
            the signal does; so a gate is set on val and applied to EVERY compared row.
  shared    update(..., observed=mask) replaces the gate with one per-view (N,) bool mask
            built by the caller (e.g. from the field's ViewStats in a view hook), so that
            every row, Ours c included, uses one coverage rule. m_g > EPS is still needed
            (no residual without mass in the residual's valid pixels).
Unobserved Gaussians keep their state unchanged.

Temporal filter (per residual and Gaussian; a change must persist k consecutive views,
counted over the views that observe the Gaussian):
  ring  = the last k observed per-view residuals
  kmin  = max over time of min(ring)            defined after k observations
  max   = max over time of r_v                  one-shot, no persistence
  ewma  = e <- e + min(1, rate * w_v) * (r_v - e)   starts at the first r_v
  last  = the latest r_v
A score that was never defined is NaN (evaluation.py ranks NaN last and counts it). Rows
with different coverage have different NaN sets: report AP/AUROC with NaN last AND on the
common defined set (common_defined), with the NaN counts.

Fusion: fuse_rank averages normalised ranks with equal fixed weights (ties get their mean
rank, NaN the lowest rank 0). It is a pre-registered row: nothing is fitted. Ranks are NOT
invariant to which Gaussians are ranked, so the population is part of the definition:
pre-registered as the AP population (observed, not hidden, labelled), passed as `mask`.
"""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn.functional as Fn

from .bki_accumulate import bki_grad_accumulate_groups

_EPS = 1e-12
_LUMA = (0.299, 0.587, 0.114)                  # Rec. 601 grey value
FILTERS = ("kmin", "max", "ewma", "last")


# ------------------------------------------------------------ per-pixel residuals
def to_grey(img: torch.Tensor) -> torch.Tensor:
    """(P, 3) RGB, (P, 1) or (P,) grey -> (P,) grey."""
    if img.dim() == 2 and img.shape[1] == 3:
        return img @ torch.tensor(_LUMA, device=img.device, dtype=img.dtype)
    return img.reshape(img.shape[0])


@torch.no_grad()
def ncc_residual(observed, rendered, hw, window=7, min_std=0.01, valid=None,
                 min_valid_frac=0.5):
    """Per-pixel 1 - NCC of grey values in a window x window box (row-major P = H*W).
    Window statistics use only valid pixels (and only in-image pixels at the border);
    a pixel needs a share >= min_valid_frac of valid pixels in its window.
    returns residual (P,) in [0, 2] (0 where masked), mask (P,) bool."""
    assert window % 2 == 1, "window must be odd"
    H, W = int(hw[0]), int(hw[1])
    dt = observed.dtype if observed.dtype == torch.float64 else torch.float32
    # float64 inside: var = E[x^2] - E[x]^2 loses digits in flat windows otherwise
    o, r = to_grey(observed.to(dt)).double(), to_grey(rendered.to(dt)).double()
    if o.numel() != H * W or r.numel() != H * W:
        raise ValueError(f"images must have P = H*W = {H * W} pixels, got {o.numel()}, "
                         f"{r.numel()}")
    o, r = o.reshape(1, 1, H, W), r.reshape(1, 1, H, W)
    v = (torch.ones_like(o) if valid is None
         else valid.to(o.device).reshape(1, 1, H, W).double())
    ok_px = torch.isfinite(o) & torch.isfinite(r)
    v = v * ok_px.double()
    o = torch.where(ok_px, o, torch.zeros_like(o))
    r = torch.where(ok_px, r, torch.zeros_like(r))
    pool = lambda x: Fn.avg_pool2d(x, window, stride=1, padding=window // 2,
                                   count_include_pad=False)
    frac = pool(v)                                   # share of valid pixels in the window
    den = frac.clamp_min(_EPS)
    mo, mr = pool(v * o) / den, pool(v * r) / den
    var_o = (pool(v * o * o) / den - mo * mo).clamp_min(0.0)
    var_r = (pool(v * r * r) / den - mr * mr).clamp_min(0.0)
    cov = pool(v * o * r) / den - mo * mr
    ok = ((v > 0) & (frac >= min_valid_frac)
          & (var_o > min_std ** 2) & (var_r > min_std ** 2))
    ncc = (cov / (var_o * var_r).sqrt().clamp_min(_EPS)).clamp(-1.0, 1.0)
    res = torch.where(ok, 1.0 - ncc, torch.zeros_like(ncc))
    return res.reshape(-1).to(dt), ok.reshape(-1)


@torch.no_grad()
def feature_residual(observed, rendered, min_norm=1e-6):
    """Per-pixel 1 - cos(rendered, observed). The render may be un-normalised (T*alpha):
    the cosine ignores scale. Pixels where either norm is ~0 are masked.
    returns residual (P,) in [0, 2] (0 where masked), mask (P,) bool."""
    no, nr = observed.norm(dim=-1), rendered.norm(dim=-1)
    ok = torch.isfinite(no) & torch.isfinite(nr) & (no > min_norm) & (nr > min_norm)
    cos = (observed * rendered).sum(-1) / (no * nr).clamp_min(_EPS)
    res = torch.where(ok, 1.0 - cos.clamp(-1.0, 1.0), torch.zeros_like(cos))
    return res, ok


def _valid_depth(d):
    return torch.isfinite(d) & (d > 0)


# ------------------------------------------------------------ temporal filters
class TemporalScore:
    """Temporal filters of one per-view, per-Gaussian residual (see module docstring)."""

    def __init__(self, N, k=2, ewma_rate=0.5, device="cpu", dtype=torch.float32):
        assert k >= 1
        nan = lambda *s: torch.full(s, float("nan"), device=device, dtype=dtype)
        self.k, self.ewma_rate = int(k), float(ewma_rate)
        self.ring = nan(N, self.k)                   # oldest ... newest observed residual
        self.count = torch.zeros(N, dtype=torch.long, device=device)
        self.kmin, self.max, self.ewma, self.last = nan(N), nan(N), nan(N), nan(N)

    @torch.no_grad()
    def push(self, r, observed, weight=None):
        """r (N,) residual of this view; observed (N,) bool; weight (N,) view weight."""
        o = observed & torch.isfinite(r)
        r = torch.where(o, r, torch.zeros_like(r))
        shifted = torch.cat([self.ring[:, 1:], r[:, None]], dim=1)
        self.ring = torch.where(o[:, None], shifted, self.ring)
        self.count += o.long()
        full = o & (self.count >= self.k)            # the ring holds k real values
        cur = self.ring.min(dim=1).values
        self.kmin = torch.where(full, torch.fmax(self.kmin, cur), self.kmin)
        self.max = torch.where(o, torch.fmax(self.max, r), self.max)
        w = torch.ones_like(r) if weight is None else weight.clamp(0.0, 1.0)
        e = torch.where(torch.isnan(self.ewma), r,
                        self.ewma + (self.ewma_rate * w).clamp(max=1.0) * (r - self.ewma))
        self.ewma = torch.where(o, e, self.ewma)
        self.last = torch.where(o, r, self.last)

    def get(self, filt):
        if filt not in FILTERS:
            raise ValueError(f"filter {filt!r} not in {FILTERS}")
        return getattr(self, filt)

    _STATE = ("ring", "count", "kmin", "max", "ewma", "last")

    def state_dict(self):
        return {name: getattr(self, name).cpu() for name in self._STATE}

    def load_state_dict(self, s):
        for name in self._STATE:
            t = getattr(self, name)
            setattr(self, name, s[name].to(t.device, t.dtype))


# ------------------------------------------------------------ accumulator
class ChangeBaselines:
    """Per-Gaussian change scores of the residual baselines (Tab II), view by view.

    Scores are keyed "<residual>/<filter>", e.g. "feat/kmin", "depth_gauss/max",
    "depth_render_oracle/ewma". A residual appears once it has been supplied."""

    def __init__(self, num_gaussians, k=2, ewma_rate=0.5, view_mass_frac=1e-4,
                 min_view_weight=0.0, ncc_window=7, ncc_min_std=0.01, ncc_min_valid_frac=0.5,
                 feat_min_norm=1e-6, oracle_depth=False, device="cpu", dtype=torch.float32):
        self.N, self.k = int(num_gaussians), int(k)
        self.ewma_rate = float(ewma_rate)
        self.view_mass_frac = float(view_mass_frac)   # a0 (FieldConfig.view_mass_frac)
        self.min_view_weight = float(min_view_weight)  # 0 = the field's m > EPS rule
        self.ncc_window, self.ncc_min_std = int(ncc_window), float(ncc_min_std)
        self.ncc_min_valid_frac = float(ncc_min_valid_frac)
        self.feat_min_norm = float(feat_min_norm)
        self.oracle_depth = bool(oracle_depth)
        self.device, self.dtype = torch.device(device), dtype
        sfx = "_oracle" if self.oracle_depth else ""
        self.depth_gauss_name, self.depth_render_name = f"depth_gauss{sfx}", f"depth_render{sfx}"
        self.series: dict[str, TemporalScore] = {}
        self.num_views = 0

    def config(self):
        return dict(num_gaussians=self.N, k=self.k, ewma_rate=self.ewma_rate,
                    view_mass_frac=self.view_mass_frac, min_view_weight=self.min_view_weight,
                    ncc_window=self.ncc_window, ncc_min_std=self.ncc_min_std,
                    ncc_min_valid_frac=self.ncc_min_valid_frac,
                    feat_min_norm=self.feat_min_norm, oracle_depth=self.oracle_depth)

    def _series(self, name):
        if name not in self.series:
            self.series[name] = TemporalScore(self.N, self.k, self.ewma_rate,
                                              self.device, self.dtype)
        return self.series[name]

    # ---------------------------------------------------------------- update
    def update(self, rasterize_fn, num_pixels, observed_rgb=None, rendered_rgb=None,
               observed_feat=None, rendered_feat=None, depth_obs=None, rendered_depth=None,
               gauss_depth=None, image_hw=None, pixel_valid=None, observed=None):
        """Ingest one update view.
        rasterize_fn  : (buf (N, C)) -> (P, C), affine in buf (geometry detached)
        num_pixels    : H*W of the view (sets the view weight reference a0*H*W)
        observed_rgb, rendered_rgb   : (P, 3) in [0, 1]      -> "ncc"
        observed_feat, rendered_feat : (P, d)                -> "feat"
        depth_obs     : (P,) observed depth [m] (<= 0 or non-finite = invalid)
        gauss_depth   : (N,) camera-frame depth of the Gaussian centres -> "depth_gauss"
        rendered_depth: (P,) expected depth of the map (0 where empty)  -> "depth_render"
        image_hw      : (H, W) for the NCC window; default: rasterize_fn.height/.width
        pixel_valid   : (P,) bool/float mask (e.g. padding), optional
        observed      : (N,) bool shared coverage mask of this view; replaces the
                        min_view_weight gate (see module docstring), optional
        returns dict name -> (N,) per-view residual (NaN where not observed)."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("update runs a backward pass (the gradient trick); call it "
                               "outside torch.inference_mode() (torch.no_grad() is fine)")
        dev, dt = self.device, self.dtype
        cast = lambda x: x.detach().to(dev, dt)      # inputs never carry a caller's graph
        P = None
        for x in (observed_rgb, observed_feat):
            if x is not None:
                P = x.shape[0]
                break
        if P is None and depth_obs is not None:
            P = depth_obs.numel()
        if P is None:
            raise ValueError("no residual input: pass rgb, features or depth")
        for arg, x in (("observed_rgb", observed_rgb), ("rendered_rgb", rendered_rgb),
                       ("observed_feat", observed_feat), ("rendered_feat", rendered_feat)):
            if x is not None and x.shape[0] != P:
                raise ValueError(f"{arg} has {x.shape[0]} rows, expected P = {P}")
        for arg, x in (("depth_obs", depth_obs), ("rendered_depth", rendered_depth),
                       ("pixel_valid", pixel_valid)):
            if x is not None and x.numel() != P:
                raise ValueError(f"{arg} has {x.numel()} entries, expected P = {P}")
        for arg, x in (("gauss_depth", gauss_depth), ("observed", observed)):
            if x is not None and x.numel() != self.N:
                raise ValueError(f"{arg} has {x.numel()} entries, expected N = {self.N}")
        if observed is not None and observed.dtype != torch.bool:
            raise ValueError("observed must be a bool mask")
        a = (torch.ones(P, device=dev, dtype=dt) if pixel_valid is None
             else cast(pixel_valid.reshape(-1)))
        for pair, (x, y) in (("rgb", (observed_rgb, rendered_rgb)),
                             ("feat", (observed_feat, rendered_feat))):
            if (x is None) != (y is None):
                raise ValueError(f"pass both observed_{pair} and rendered_{pair}, or neither")
        groups, names = [], []
        if observed_rgb is not None:
            hw = image_hw
            if hw is None:
                hw = (getattr(rasterize_fn, "height", None), getattr(rasterize_fn, "width", None))
                if None in hw:
                    raise ValueError("NCC needs image_hw (H, W)")
            res, ok = ncc_residual(cast(observed_rgb), cast(rendered_rgb), hw,
                                   self.ncc_window, self.ncc_min_std,
                                   valid=a > 0 if pixel_valid is not None else None,
                                   min_valid_frac=self.ncc_min_valid_frac)
            groups.append((cast(res)[:, None], a * ok.to(dt))); names.append("ncc")
        if observed_feat is not None:
            res, ok = feature_residual(cast(observed_feat), cast(rendered_feat),
                                       self.feat_min_norm)
            groups.append((res[:, None], a * ok.to(dt))); names.append("feat")
        if depth_obs is not None:
            dep = cast(depth_obs.reshape(-1))
            good = _valid_depth(dep)
            dep0 = torch.where(good, dep, torch.zeros_like(dep))   # NaN/inf never enter
            if gauss_depth is None and rendered_depth is None:
                raise ValueError("depth_obs needs gauss_depth and/or rendered_depth")
            if gauss_depth is not None:
                groups.append((dep0[:, None], a * good.to(dt)))
                names.append(self.depth_gauss_name)
            if rendered_depth is not None:
                rd = cast(rendered_depth.reshape(-1))
                g2 = good & _valid_depth(rd)
                res = torch.where(g2, (dep0 - rd).abs(), torch.zeros_like(dep0))
                groups.append((res[:, None], a * g2.to(dt)))
                names.append(self.depth_render_name)

        with torch.enable_grad():        # the lifting IS a backward pass
            out = bki_grad_accumulate_groups(rasterize_fn, groups, self.N,
                                             device=dev, dtype=dt)
        ref = self.view_mass_frac * float(num_pixels)
        shared = None if observed is None else observed.detach().to(dev).reshape(-1)
        per_view = {}
        with torch.no_grad():
            for name, (S, m) in zip(names, out):
                seen = m > _EPS                               # the field's "seen" rule
                wv = torch.where(seen, (m / ref).clamp(max=1.0), torch.zeros_like(m))
                obs = seen & (shared if shared is not None else wv >= self.min_view_weight)
                r = S[:, 0] / m.clamp_min(_EPS)               # sum w a r / sum w a
                if name == self.depth_gauss_name:
                    r = (r - cast(gauss_depth.reshape(-1))).abs()
                r = torch.where(obs, r, torch.full_like(r, float("nan")))
                self._series(name).push(r, obs, wv)
                per_view[name] = r
        self.num_views += 1
        return per_view

    # ------------------------------------------------------------ read-outs
    @property
    def names(self):
        return list(self.series)

    def score(self, name, filt="kmin"):
        return self.series[name].get(filt)

    def scores(self, filters: Sequence[str] = FILTERS):
        """dict "<residual>/<filter>" -> (N,) score copy (higher = more likely changed)."""
        return {f"{n}/{f}": s.get(f).clone() for n, s in self.series.items() for f in filters}

    def uncertainties(self):
        """Row-compatible alias: per-Gaussian scalars, higher = less trustworthy."""
        return self.scores()

    @torch.no_grad()
    def reset(self):
        self.series.clear()
        self.num_views = 0

    def state_dict(self):
        return dict(config=self.config(), dtype=str(self.dtype), num_views=self.num_views,
                    series={n: s.state_dict() for n, s in self.series.items()})

    @classmethod
    def from_state(cls, s, device="cpu", dtype=None):
        """dtype=None restores the saved dtype (float32 for old state dicts)."""
        if dtype is None:
            dtype = getattr(torch, s.get("dtype", "torch.float32").replace("torch.", ""))
        cb = cls(device=device, dtype=dtype, **s["config"])
        cb.num_views = s["num_views"]
        for name, st in s["series"].items():
            cb._series(name).load_state_dict(st)
        return cb


# ------------------------------------------------------------ evaluation helpers
def _keep_mask(mask, N, device):
    if mask is None:
        return torch.ones(N, dtype=torch.bool, device=device)
    keep = mask.reshape(-1).to(device)
    if keep.dtype != torch.bool:
        raise ValueError("mask must be a bool tensor")
    if keep.numel() != N:
        raise ValueError(f"mask has {keep.numel()} entries, expected {N}")
    return keep


@torch.no_grad()
def common_defined(scores: Sequence[torch.Tensor], mask=None) -> torch.Tensor:
    """(N,) bool: inside `mask` (the AP population) and finite in EVERY score. Rows with
    different coverage rules have different NaN sets; AP/AUROC on this common set compare
    the signals, AP/AUROC with NaN last compare signal plus coverage. Report both."""
    assert len(scores) >= 1
    dev, N = scores[0].device, scores[0].numel()
    if any(s.numel() != N for s in scores):
        raise ValueError("all scores must have the same length")
    keep = _keep_mask(mask, N, dev)
    for s in scores:
        keep = keep & torch.isfinite(s.reshape(-1).to(dev))
    return keep


# ------------------------------------------------------------ fusion
@torch.no_grad()
def normalized_rank(score: torch.Tensor, mask=None) -> torch.Tensor:
    """(N,) -> (N,) float64: mid-rank (ties share their mean rank) among the n entries
    inside `mask` (default: all), scaled to [0, 1] by 1/(n - 1). NaN inside the mask is the
    lowest rank and gets exactly 0; entries outside the mask are NaN and never counted."""
    s = score.reshape(-1).double()
    N = s.numel()
    keep = _keep_mask(mask, N, s.device)
    out = torch.full((N,), float("nan"), dtype=torch.float64, device=s.device)
    out[keep] = 0.0
    nan = torch.isnan(s) & keep
    fin = keep & ~nan
    x = s[fin]
    if x.numel() == 0:
        return out
    order = torch.argsort(x, stable=True)
    _, inv, counts = torch.unique_consecutive(x[order], return_inverse=True,
                                              return_counts=True)
    end = torch.cumsum(counts, 0).double()
    mid = end - (counts.double() - 1.0) / 2.0            # 1-based mean rank of the block
    ranks = torch.empty_like(x)
    ranks[order] = mid[inv] + float(nan.sum())           # NaN entries occupy the bottom
    out[fin] = (ranks - 1.0) / max(int(keep.sum()) - 1, 1)
    return out


@torch.no_grad()
def fuse_rank(scores: Sequence[torch.Tensor], mask=None) -> torch.Tensor:
    """Pre-registered rank average: mean of normalized_rank over the scores, equal
    weights, nothing fitted. Ranks are taken among the entries inside `mask` (bool (N,)),
    which must be the evaluation population (observed, not hidden, labelled): the fused
    order on that set changes if other Gaussians are ranked too. Default: every entry is
    in the population. Returns (N,) float64 on the first score's device: NaN outside the
    mask and where EVERY input is NaN; otherwise a NaN input counts as rank 0."""
    assert len(scores) >= 1
    dev, N = scores[0].device, scores[0].numel()
    if any(s.numel() != N for s in scores):
        raise ValueError("all scores must have the same length")
    keep = _keep_mask(mask, N, dev)
    flat = [s.reshape(-1).to(dev) for s in scores]
    ranks = torch.stack([normalized_rank(s, keep) for s in flat], dim=1)   # (N, S)
    all_nan = torch.stack([torch.isnan(s) for s in flat], 1).all(1)
    fused = ranks.mean(dim=1)
    return torch.where(all_nan | ~keep, torch.full_like(fused, float("nan")), fused)
