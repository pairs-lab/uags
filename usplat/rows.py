"""
rows.py — the interface every segmentation / uncertainty method ("row" of a table) gives
the evaluation (usplat.seg_eval), so all methods are scored the same way.

A row is a per-Gaussian buffer that the evaluation composites with the SAME splat weights
w_{p,n} = T_{p,n} alpha_{p,n} as the field update (one render per row and view):
    kind "feature" : f_n in the field space, (N, d).  m_p = sum_n w_{p,n} f_n is NOT divided
                     by alpha, and p_p(c) = softmax_c(beta m_p . t_c): little opacity or weak
                     evidence (|f_n| < 1, e.g. the posterior mean P1) flattens the softmax.
    kind "probs"   : q_n class probabilities, (N, K) (e.g. the CSS Dirichlet mean).
                     p_p = sum_n w_{p,n} q_n / alpha_p (uniform where alpha_p ~ 0).
Per-Gaussian uncertainties u_n (higher = less certain) come from uncertainties(); they are
rendered as u_p = sum_n w_{p,n} u_n / alpha_p in the same call. NaN marks an undefined value
(e.g. u_ale before tau_min views); seg_eval.render_row says how such pixels are treated.
Buffers must be finite and u_n finite or NaN (+-inf is rejected: it cannot be averaged).

`adaptive` rows are the P2 predictive: beta_p = beta0 exp(-a u_epi,p - b u_ale,p - g 1[u_ale
undefined]) with the rendered channels named in `p2_channels` (usplat.predictive).

Table I v2 candidates (v2_rows; read-outs of the saved field, no re-mapping; selected on
validation by configs/tab1_v2_prereg.json): '<prefix>_P2p1' (P2 on the P1 base),
'<prefix>_P1_c<c0>' (P1 with the joint-prior pseudo-count c0, FieldRow.c0) and the vote
predictive P4 (FieldVoteRow: '<prefix>_P4_a<alpha0>_s<lam>' and the zero-noise
'<prefix>_P4_inf'), all with the same uncertainty channels as the other field rows.

Baselines that need only the per-view sums subscribe to field.view_hooks; baselines that
need other per-pixel targets (e.g. one-hot labels for CSS) run their own
bki_grad_accumulate_groups call. Either way they end up as a Row (subclass Row, or wrap
fixed tensors in a TensorRow).
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field, replace
from typing import Callable, Dict, Optional, Sequence, Tuple
import torch

KINDS = ("feature", "probs")


@dataclass
class Row:
    """Base row. Subclasses override buffer() and, if they have one, uncertainties()."""
    name: str
    kind: str = "feature"                              # "feature" | "probs"
    adaptive: bool = False                             # P2: per-pixel beta from p2_channels
    p2_channels: Tuple[str, str] = ("u_epi", "u_ale")
    # optional per-pixel decoder applied AFTER compositing (feature rows only): (P, C) -> (P, d)
    # e.g. LEGaussians renders an 8-d code and decodes it (usplat.distill); None = identity
    decoder: Optional[Callable] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"row {self.name!r}: kind {self.kind!r} not in {KINDS}")
        if self.adaptive and self.kind != "feature":
            raise ValueError(f"row {self.name!r}: the adaptive predictive needs a feature row")

    def buffer(self) -> torch.Tensor:
        """(N, d) features (kind 'feature') or (N, K) class probabilities (kind 'probs')."""
        raise NotImplementedError

    def uncertainties(self) -> Dict[str, torch.Tensor]:
        """{name: (N,)} per-Gaussian uncertainties, higher = less certain; may be empty."""
        return {}


@dataclass
class TensorRow(Row):
    """A row from fixed tensors (baselines that precompute their buffer)."""
    buf: Optional[torch.Tensor] = None
    unc: Dict[str, torch.Tensor] = dc_field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.buf is None:
            raise ValueError(f"row {self.name!r}: TensorRow needs buf")

    def buffer(self):
        return self.buf

    def uncertainties(self):
        return dict(self.unc)


# ---- the field's own rows (plan F6: P0 plug-in, P1 posterior mean-field, P2 adaptive)
def _field_channel(field, name):
    if name == "u_epi":                  # the configured epistemic read-out (1/(1+support))
        return field.epistemic()
    if name == "u_ale":                  # 1 - agreement on the dispersion channel, NaN undefined
        return field.aleatoric()[0]
    if name == "spread":                 # posterior spread 1 - A_d(kappa_post)
        return field.angular_spread()
    if name == "kish_epi":               # 1/(1 + Kish views): NormLift parity
        return 1.0 / (1.0 + field.kish_views("raw"))
    if name == "change":                 # last change score c (NaN if never defined)
        return field.change_score()
    raise ValueError(f"unknown field channel {name!r}")


@dataclass
class FieldRow(Row):
    """A predictive of a DirectionalSemanticField. mode 'P0' renders mu* (plug-in), 'P1'
    renders E[mu] = A_d(kappa_post) mu* (posterior mean-field). P2 = adaptive=True on top
    of either (the default in field_rows is P0). c0 (optional) reads the posterior with that
    joint-prior pseudo-count instead of FieldConfig.c0 (usplat.predictive.posterior_read_out;
    the field is not modified). Read-outs are recomputed on every call, so use
    seg_eval.freeze_rows (or evaluate_views, which does it) to snapshot a fixed field."""
    field: object = None
    mode: str = "P0"
    channels: Sequence[str] = ("u_epi", "u_ale")
    c0: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.field is None:
            raise ValueError(f"row {self.name!r}: FieldRow needs field")
        if self.mode not in ("P0", "P1"):
            raise ValueError(f"row {self.name!r}: mode {self.mode!r} not in ('P0', 'P1')")
        missing = [c for c in self.p2_channels if c not in self.channels] if self.adaptive \
            else []
        if missing:
            raise ValueError(f"row {self.name!r}: P2 needs channels {missing}")
        if self.c0 is not None and not self.c0 > 0:
            raise ValueError(f"row {self.name!r}: c0 must be > 0 (got {self.c0})")

    def buffer(self):
        from .predictive import gaussian_buffer
        return gaussian_buffer(self.field, self.mode, self.c0)

    def uncertainties(self):
        return {c: _field_channel(self.field, c) for c in self.channels}


def field_rows(field, prefix="ours", modes=("P0", "P1", "P2"), p2_base="P0",
               channels=("u_epi", "u_ale")):
    """The field's rows: '<prefix>_P0', '<prefix>_P1', '<prefix>_P2' (P2 = p2_base buffer
    with the adaptive beta). Every row carries the same uncertainty channels."""
    rows = []
    for m in modes:
        if m == "P2":
            rows.append(FieldRow(f"{prefix}_P2", "feature", adaptive=True, field=field,
                                 mode=p2_base, channels=tuple(channels)))
        else:
            rows.append(FieldRow(f"{prefix}_{m}", "feature", field=field, mode=m,
                                 channels=tuple(channels)))
    return rows


@dataclass
class FieldProbRow(Row):
    """P3: the per-Gaussian class predictive, rendered as a mixture (kind "probs").
        q_n(c) = softmax_c(beta_g * m_n . t_c),   m_n = A_d(kappa_post,n) mu*_n   (P1 direction)
    over the QUERIED classes t (K, d) in the field space, computed at query time from the
    field (open vocabulary: nothing is fixed at mapping time). The evaluator renders
    q_p = sum_n w_{p,n} q_n / alpha_p and applies its own temperature to log q_p (fit on val),
    so beta_g sets how peaked each Gaussian's distribution is BEFORE mixing and the post-hoc
    temperature rescales the mixture. A Gaussian without evidence has m_n = 0, hence the
    uniform q_n. beta_g is a hyperparameter chosen on the validation scenes."""
    field: object = None
    text: Optional[torch.Tensor] = None
    beta_g: float = 20.0
    channels: Sequence[str] = ("u_epi", "u_ale")

    def __post_init__(self):
        super().__post_init__()
        if self.kind != "probs":
            raise ValueError(f"row {self.name!r}: P3 is a probs row")
        if self.field is None or self.text is None:
            raise ValueError(f"row {self.name!r}: FieldProbRow needs field and text")
        if not self.beta_g > 0:
            raise ValueError(f"row {self.name!r}: beta_g must be > 0")

    @torch.no_grad()
    def buffer(self):
        from .predictive import gaussian_buffer
        m = gaussian_buffer(self.field, "P1")
        t = self.text.to(m.device, m.dtype)
        return torch.softmax(self.beta_g * (m @ t.t()), dim=-1)

    def uncertainties(self):
        return {c: _field_channel(self.field, c) for c in self.channels}


P3_BETAS = (10.0, 20.0, 40.0)        # beta_g grid of P3; the validation NLL picks one


def p3_rows(field, text, prefix="ours", betas=P3_BETAS):
    """'<prefix>_P3_b<beta_g>' rows (usplat.rows.FieldProbRow) for the queried text."""
    return [FieldProbRow(f"{prefix}_P3_b{b:g}", "probs", field=field, text=text, beta_g=float(b))
            for b in betas]


# ---- Table I v2 candidates (configs/tab1_v2_prereg.json fixes these grids)
P1_C0S = (0.01, 0.03, 0.1, 0.3, 1.0)  # joint-prior pseudo-count grid of P1 (1 = FieldConfig.c0)
P4_ALPHA0S = (0.01, 0.1, 1.0)         # Dirichlet pseudo-count grid of P4
P4_LAMBDAS = (0.5, 1.0, 2.0)          # tangent-noise scale grid of P4 (1 = vMF moment match)
P4_INF_ALPHA0 = 0.01                  # alpha0 of the zero-noise ablation '<prefix>_P4_inf'
P4_SAMPLES = 256                      # Monte Carlo samples (antithetic pairs), seed 0


class VoteCache:
    """Shares the Monte Carlo votes of P4 rows that differ only in alpha0 (alpha0 enters
    after the votes): one run per (lam, n_samples, seed). An entry is reused only while the
    inputs (mu*, kappa, text) are bit-identical to those it was computed from (an O(N d)
    check against the O(N S K) run), so a field updated in between never reads a stale
    entry. `runs` counts the Monte Carlo runs."""

    def __init__(self):
        self.inputs, self.votes, self.runs = None, {}, 0

    def get(self, inputs, key, compute):
        same = self.inputs is not None and all(
            a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
            and torch.equal(a, b) for a, b in zip(self.inputs, inputs))
        if not same:
            self.inputs, self.votes = tuple(inputs), {}
        if key not in self.votes:
            self.votes[key] = compute()
            self.runs += 1
        return self.votes[key]


@dataclass
class FieldVoteRow(Row):
    """P4: the per-Gaussian VOTE predictive, rendered as a mixture (kind "probs"), the
    directional counterpart of CSS's Dirichlet vote counts (usplat.predictive):
        z ~ vMF(mu*_n, kappa_n),   kappa_n = A_d^{-1}(||R_n|| / W1_n)       one pixel of n
        v_n(c) = P[argmax_k z . t_k = c]                 over the QUERIED texts t (K, d)
        q_n(c) = (alpha0 + W1_n v_n(c)) / (K alpha0 + W1_n)   (uniform where W1_n = 0)
    W1_n v_n are the expected votes of the Gaussian's W1_n views of evidence. Where P3 sharpens
    one softmax per Gaussian with a free beta_g, P4 takes the spread of the votes from the
    Gaussian's own pixel-level concentration, and alpha0 from a validated grid: a coherent,
    well-observed Gaussian votes for one class, a mixed or unseen one stays near uniform.
    lam scales the tangent noise of the Monte Carlo (1 = the vMF moment match); lam = 0 is
    the zero-noise limit (kappa -> inf): v_n = one-hot argmax_k mu*_n . t_k (mode-seeking
    ablation). The evaluator renders q_p = sum_n w q_n / alpha_p and fits its temperature on
    log q_p (val), as for P3 and CSS. Rows sharing a VoteCache run the Monte Carlo once."""
    field: object = None
    text: Optional[torch.Tensor] = None
    alpha0: float = 0.1
    lam: float = 1.0
    n_samples: int = P4_SAMPLES
    seed: int = 0
    chunk: int = 4096
    channels: Sequence[str] = ("u_epi", "u_ale")
    cache: Optional[VoteCache] = None

    def __post_init__(self):
        super().__post_init__()
        if self.kind != "probs":
            raise ValueError(f"row {self.name!r}: P4 is a probs row")
        if self.field is None or self.text is None:
            raise ValueError(f"row {self.name!r}: FieldVoteRow needs field and text")
        if not self.alpha0 > 0:                              # also rejects NaN
            raise ValueError(f"row {self.name!r}: alpha0 must be > 0 (got {self.alpha0})")
        if not self.lam >= 0:
            raise ValueError(f"row {self.name!r}: lam must be >= 0 (got {self.lam})")

    @torch.no_grad()
    def votes(self):
        """(N, K) vote fractions v_n(c) (rows sum to 1)."""
        from .predictive import argmax_votes, vote_fractions, vote_kappa
        mu = self.field.mu_post()
        t = self.text.to(mu.device, mu.dtype)
        if self.lam == 0:
            return argmax_votes(mu, t)
        kappa = vote_kappa(self.field)
        run = lambda: vote_fractions(mu, kappa, t, self.lam, self.n_samples, self.seed,
                                     self.chunk)
        if self.cache is None:
            return run()
        return self.cache.get((mu, kappa, t), (float(self.lam), int(self.n_samples),
                                               int(self.seed)), run)

    @torch.no_grad()
    def buffer(self):
        from .predictive import vote_predictive
        q = vote_predictive(self.votes(), self.field.n_eff, self.alpha0)
        return q.to(self.field.dtype)

    def uncertainties(self):
        return {c: _field_channel(self.field, c) for c in self.channels}


def p4_rows(field, text, prefix="ours", alpha0s=P4_ALPHA0S, lams=P4_LAMBDAS, inf=True,
            n_samples=P4_SAMPLES, seed=0, chunk=4096):
    """'<prefix>_P4_a<alpha0>_s<lam>' rows for the queried text, sharing one VoteCache (one
    Monte Carlo run per lam), and with inf=True the zero-noise '<prefix>_P4_inf'."""
    cache = VoteCache()
    rows = [FieldVoteRow(f"{prefix}_P4_a{a:g}_s{lam:g}", "probs", field=field, text=text,
                         alpha0=float(a), lam=float(lam), n_samples=n_samples, seed=seed,
                         chunk=chunk, cache=cache)
            for lam in lams for a in alpha0s]
    if inf:
        rows.append(FieldVoteRow(f"{prefix}_P4_inf", "probs", field=field, text=text,
                                 alpha0=P4_INF_ALPHA0, lam=0.0))
    return rows


def p1_prior_rows(field, prefix="ours", c0s=P1_C0S, channels=("u_epi", "u_ale")):
    """'<prefix>_P1_c<c0>' rows: P1 = A_d(kappa_post) mu* with kappa_hat from the joint
    estimator at pseudo-count c0 (c0 = FieldConfig.c0 = 1 reproduces '<prefix>_P1')."""
    return [FieldRow(f"{prefix}_P1_c{c:g}", "feature", field=field, mode="P1", c0=float(c),
                     channels=tuple(channels)) for c in c0s]


def v2_rows(field, text=None, prefix="ours"):
    """The Table I v2 candidates: '<prefix>_P2p1' (field_rows' P2 on the P1 base), the P1 c0
    grid and, given the queried text, the P4 grid and '<prefix>_P4_inf'. Frozen for N = 1e6,
    K = 30 (float32): 10 P4 buffers (N, K) = 1.2 GB and 4 new P1 buffers (N, 64) = 1.0 GB;
    P1_c1 and P2p1 equal P1's buffer, which eval_seg.frozen_on stores once. While the rows
    are frozen the P4 VoteCache holds mu* and 3 vote tables (~0.6 GB) plus one Monte Carlo
    chunk (~0.14 GB)."""
    p2p1 = replace(field_rows(field, prefix, modes=("P2",), p2_base="P1")[0],
                   name=f"{prefix}_P2p1")
    rows = [p2p1] + p1_prior_rows(field, prefix)
    if text is not None:
        rows += p4_rows(field, text, prefix)
    return rows


@torch.no_grad()
def freeze(row: Row) -> TensorRow:
    """Snapshot buffer() and uncertainties() once (evaluate many views of a fixed map)."""
    buf = row.buffer().detach()
    unc = {k: v.detach().reshape(-1) for k, v in row.uncertainties().items()}
    return TensorRow(row.name, row.kind, adaptive=row.adaptive,
                     p2_channels=tuple(row.p2_channels), decoder=row.decoder, buf=buf, unc=unc)
