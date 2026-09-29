"""usplat — uncertainty-aware open-vocabulary semantic 3D Gaussian Splatting."""
from .vmf_field import DirectionalSemanticField, FieldConfig, ViewStats
from .bki_accumulate import (bki_grad_accumulate, bki_grad_accumulate_groups,
                             bki_grad_accumulate_targets, bki_bruteforce_accumulate)
from .efe_planner import EFEPlanner
from .consolidation import Consolidator
from .feature_pca import StreamingPCA, RunningMean
from . import evaluation, predictive, vmf_math

__all__ = [
    "DirectionalSemanticField",
    "FieldConfig",
    "ViewStats",
    "bki_grad_accumulate",
    "bki_grad_accumulate_groups",
    "bki_grad_accumulate_targets",
    "bki_bruteforce_accumulate",
    "EFEPlanner",
    "Consolidator",
    "StreamingPCA",
    "RunningMean",
    "evaluation",
    "predictive",
    "vmf_math",
]
