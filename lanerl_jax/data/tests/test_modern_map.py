"""Modern map ingestion must never reinterpret the legacy map as current."""
import struct
import hashlib

import numpy as np
import pytest

from lanerl_jax.data.modern_map import (
    BLUE_ONLY, RED_ONLY, TRANSPARENT, ModernMapGrid, load_artifact, load_patch_map,
    read_ngrid, write_artifact,
)


def tiny_ngrid():
    # Rectangular, translated grid deliberately catches transposition/origin
    # assumptions; fixture follows the documented v7.1 binary layout.
    header = struct.pack("<BH6ffII", 7, 1, -5, -10, 20, 25, 50, 40, 10, 3, 2)
    flags = np.array([[0, 1, 2], [4, 66, 0]], dtype="<u2")
    regions = np.zeros((2, 3, 4), np.uint8)
    regions[0, 1, 1] = 11 << 4  # top alcove label, no image flip
    heights = np.arange(12, dtype="<f4").reshape(3, 4)
    return (header + bytes(6*48) + flags.tobytes() + regions.tobytes()
            + bytes(8*132) + struct.pack("<IIff", 4, 3, 5, 5) + heights.tobytes())


def open_grid():
    return ModernMapGrid(np.zeros((8, 9), np.uint16), np.zeros((8, 9, 4), np.uint8),
                         np.zeros((9, 10), np.float32), (25, 25), 50,
                         (-10, -5, 20), (440, 5, 420))


def test_binary_layout_axes_and_layers():
    g = read_ngrid(tiny_ngrid())
    assert g.flags.shape == (2, 3)
    assert g.main_region[0, 1] == 11
    assert g.brush[0, 1]
    assert g.heights[2, 3] == 11
    assert g.cell(10, 25) == (1, 0)
    assert g.is_walkable(10, 25)
    assert not g.is_walkable(20, 25)
    assert not g.flags.flags.writeable


@pytest.mark.parametrize("raw", [b"", tiny_ngrid()[:100], tiny_ngrid()[:-1],
                                  bytes([5])+tiny_ngrid()[1:],
                                  tiny_ngrid()[:1]+bytes([2])+tiny_ngrid()[2:]])
def test_rejects_truncated_or_unknown_format(raw):
    with pytest.raises(ValueError):
        read_ngrid(raw)


def test_rejects_mismatched_dimensions():
    raw = bytearray(tiny_ngrid())
    struct.pack_into("<f", raw, 15, 1000)
    with pytest.raises(ValueError, match="dimensions"):
        read_ngrid(raw)


def test_artifact_roundtrip_patch_gate_and_integrity(tmp_path):
    path = tmp_path / "map"
    write_artifact(tiny_ngrid(), path, patch="historical-fixture", source="synthetic",
                   retrieved_at="2026-09-30")
    g, _ = load_artifact(path, expected_patch="historical-fixture")
    pin = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
    load_artifact(path, expected_patch="historical-fixture", expected_manifest_sha256=pin)
    with pytest.raises(ValueError, match="manifest checksum"):
        load_artifact(path, expected_patch="historical-fixture", expected_manifest_sha256="0"*64)
    np.testing.assert_array_equal(g.flags, read_ngrid(tiny_ngrid()).flags)
    with pytest.raises(ValueError, match="mismatch"):
        load_artifact(path, expected_patch="26.19")
    with pytest.raises(ValueError, match="mismatch"):
        load_artifact(path, expected_patch="historical-fixture", expected_variant="infernal")
    with pytest.raises(FileExistsError):
        write_artifact(tiny_ngrid(), path, patch="26.19", source="synthetic", retrieved_at="today")
    with (path / "grid.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(path, expected_patch="historical-fixture")


def test_world_boundary_does_not_alias_next_row():
    g = open_grid()
    assert not g.is_walkable(-10.01, 40)
    assert not g.is_walkable(440, 40)
    assert not g.is_walkable(0, 420)
    assert not g.is_walkable(float("nan"), 40)
    assert g.is_walkable(-10, 20)
    assert not g.is_walkable(40, 70, radius=50)
    assert g.is_walkable(40.1, 70.1, radius=50)


def test_reviewed_profile_rejects_relabelled_fixture_and_unknown_patch(tmp_path):
    path = tmp_path / "map"
    write_artifact(tiny_ngrid(), path, patch="26.19", source="synthetic",
                   retrieved_at="2026-09-30")
    with pytest.raises(ValueError, match="manifest checksum"):
        load_patch_map(path)
    with pytest.raises(ValueError, match="no reviewed"):
        load_patch_map(path, patch="latest")


def test_team_gates_and_structure_collision():
    f = np.array([[BLUE_ONLY | TRANSPARENT, RED_ONLY | TRANSPARENT, 4, 1]], np.uint16)
    g = ModernMapGrid(f, np.zeros((1, 4, 4), np.uint8), np.zeros((2, 5), np.float32),
                      (25, 25), 50, (0, -1, 0), (200, 1, 50))
    assert g.walkable().tolist() == [[False, False, False, True]]
    assert g.walkable(0).tolist() == [[True, False, False, True]]
    assert g.walkable(1).tolist() == [[False, True, False, True]]


def test_jit_vmap_agrees_with_independent_host_disk_queries():
    import jax
    import jax.numpy as jnp
    from lanerl_jax.sim.modern_terrain import is_walkable
    g = open_grid()
    f = g.flags.copy()
    f[2:5, 4] = 2
    g = ModernMapGrid(f, g.regions, g.heights, g.height_spacing, g.cell_size,
                      g.min_bounds, g.max_bounds)
    rng = np.random.default_rng(15)
    points = rng.uniform([-20, 0, 0], [450, 430, 150], (128, 3)).astype(np.float32)
    points = np.concatenate([points, np.array([
        [-10.01, 40, 0], [440, 40, 0], [10, 50, 0], [165, 145, 25],
        [40, 70, 50], [40.1, 70.1, 50], [165, 145, 24.9],
    ], np.float32)])
    compiled = jax.jit(jax.vmap(lambda p: is_walkable(*p, g.as_jax())))
    actual = np.asarray(compiled(jnp.asarray(points)))
    expected = [g.is_walkable(float(x), float(z), radius=float(r)) for x, z, r in points]
    np.testing.assert_array_equal(actual, expected)
    assert not bool(is_walkable(100, 100, 151, g.as_jax()))
    assert not bool(is_walkable(100, 100, -1, g.as_jax()))


def test_zero_radius_half_open_edges_and_positive_radius_contact():
    from lanerl_jax.sim.modern_terrain import is_walkable
    g = open_grid()
    f = g.flags.copy()
    f[1, 1] = 2
    g = ModernMapGrid(f, g.regions, g.heights, g.height_spacing, g.cell_size,
                      g.min_bounds, g.max_bounds)
    for x, z in [(90, 95), (65, 120), (90, 120)]:
        assert g.is_walkable(x, z)
        assert bool(is_walkable(x, z, 0, g.as_jax()))
        assert not g.is_walkable(x, z, radius=.01)
        assert not bool(is_walkable(x, z, .01, g.as_jax()))
