"""scripts/distill_features.py on a NeRF-Object-Removal field (2026-11-07): the mapping part only,
paths recorded before the 2026-10-31 move mapped to their current place."""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _df():
    spec = importlib.util.spec_from_file_location("distill_features", REPO / "scripts" / "distill_features.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_nor_reads_the_mapping_part_and_moved_paths():
    from usplat.paths import OLD_ROOT, moved
    df = _df()
    old = f"{OLD_ROOT}/nerf_object_removal/nerf-object-removal/001"
    name, root, opts = df.mapping_dataset(dict(dataset="nor", part="map", data=old))
    assert name == "nor" and opts == dict(part="map") and root == moved(old)
    with pytest.raises(SystemExit):
        df.mapping_dataset(dict(dataset="nor", part="update", data=old))
    name, root, opts = df.mapping_dataset(dict(dataset="replica_semnerf", data="/x/office_0"))
    assert (name, root, opts) == ("replica_semnerf", "/x/office_0", {})
