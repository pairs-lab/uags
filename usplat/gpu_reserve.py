"""
gpu_reserve.py — hold a job's GPU memory from its first second on the shared card.

scripts/gpu_gate.sh starts a job when enough memory is free and exports
UAGS_GPU_MEM_FRACTION = need / total. Jobs spend their first seconds (and the gaps between
scenes) on the CPU; another project's batch that polls for free memory would start in such
a gap and the job would then run out of memory. apply() caps the process at the fraction
and reserves the cap minus a margin at once: one allocation that is freed at once but stays
in PyTorch's caching allocator, so the memory stays with this process (never call
torch.cuda.empty_cache() after it) and later allocations are carved out of it.
"""
from __future__ import annotations
import os

import torch

MARGIN_MIB = 512            # room for the small-block pool, which cannot use the reservation


def apply(device="cuda", margin_mib=MARGIN_MIB):
    """Set the gate's memory fraction and reserve it. Returns the reserved MiB (0 if none)."""
    frac = os.environ.get("UAGS_GPU_MEM_FRACTION")
    if not frac or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0
    frac = float(frac)
    torch.cuda.set_per_process_memory_fraction(frac)
    total = torch.cuda.get_device_properties(torch.device(device)).total_memory
    mib = int(frac * total / 2 ** 20) - int(margin_mib)
    if mib <= 0 or os.environ.get("UAGS_GPU_RESERVE", "1") == "0":
        return 0
    block = torch.empty(mib * 2 ** 20, dtype=torch.uint8, device=device)
    del block                                   # back to the cache, still reserved
    return mib
