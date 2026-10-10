"""scripts/disk_guard.py: N from a .pt map (memory-mapped) or a .ply header; a step needing more
than the free space is refused (exit 1), factor 0 turns the check off; the shell scripts call
it before every field mapping (not in their dry runs)."""
import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("disk_guard", REPO / "scripts" / "disk_guard.py")
DG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DG)


def test_num_gaussians_and_refusal(tmp_path, capsys):
    torch.save(dict(means=torch.zeros(1234, 3), quats=torch.zeros(1234, 4)), tmp_path / "m.pt")
    torch.save(dict(splats=dict(means=torch.zeros(7, 3))), tmp_path / "s.pt")
    (tmp_path / "m.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\n"
                                     b"element vertex 4321\nproperty float x\nend_header\n\0\0")
    assert DG.num_gaussians(tmp_path / "m.pt") == 1234
    assert DG.num_gaussians(tmp_path / "s.pt") == 7
    assert DG.num_gaussians(tmp_path / "m.ply") == 4321
    out = tmp_path / "not" / "yet"                         # resolved to an existing parent
    assert DG.main([str(out), str(tmp_path / "m.pt"), "1000"]) == 0
    assert "the step needs" in capsys.readouterr().out
    assert DG.main([str(out), str(tmp_path / "m.pt"), "1e18"]) == 1   # ~1e21 B: never free
    assert "refused: disk" in capsys.readouterr().out
    assert DG.main([str(out), str(tmp_path / "m.pt"), "1e18", "--factor", "0"]) == 0


def test_scripts_guard_every_field_mapping():
    swap = (REPO / "scripts" / "replica_swap.sh").read_text()
    assert swap.count('space "$SWAP/fields" "$MAPS/${s}_seq1.pt"') == 2
    assert 'python3 scripts/disk_guard.py "$@" --factor "$DISK_FACTOR"' in swap
    scan = (REPO / "scripts" / "scannet_tab1.sh").read_text()
    assert 'python3 scripts/disk_guard.py "$FIELDS" "$MAPS/$s.pt" 1600' in scan
