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

Baselines that need only the per-view sums subscribe to field.view_hooks; baselines that
need other per-pixel targets (e.g. one-hot labels for CSS) run their own
bki_grad_accumulate_groups call. Either way they end up as a Row (subclass Row, or wrap
fixed tensors in a TensorRow).
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Dict, Optional, Sequence, Tuple
import torch

KINDS = ("feature", "probs")


@dataclass
class Row:
    """Base row. Subclasses override buffer() and, if they have one, uncertainties()."""
    name: str
    kind: str = "feature"                              # "feature" | "probs"
    adaptive: bool = False                             # P2: per-pixel beta from p2_channels
    p2_channels: Tuple[str, str] = ("u_epi", "u_ale")

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
    of either (the default in field_rows is P0). Read-outs are recomputed on every call, so
    use seg_eval.freeze_rows (or evaluate_views, which does it) to snapshot a fixed field."""
    field: object = None
    mode: str = "P0"
    channels: Sequence[str] = ("u_epi", "u_ale")

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

    def buffer(self):
        from .predictive import gaussian_buffer
        return gaussian_buffer(self.field, self.mode)

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


@torch.no_grad()
def freeze(row: Row) -> TensorRow:
    """Snapshot buffer() and uncertainties() once (evaluate many views of a fixed map)."""
    buf = row.buffer().detach()
    unc = {k: v.detach().reshape(-1) for k, v in row.uncertainties().items()}
    return TensorRow(row.name, row.kind, adaptive=row.adaptive,
                     p2_channels=tuple(row.p2_channels), buf=buf, unc=unc)
