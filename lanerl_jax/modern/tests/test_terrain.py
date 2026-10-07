"""``is_walkable`` (two ``row_gaps`` candidates per row) against the full (k, k) cell-window test it replaces."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import collision as C
from lanerl_jax.modern.map.terrain import StaticTerrain, is_walkable, row_gaps, team_view
from lanerl_jax.modern.world import config as MW


def window_walkable(x, z, radius, terrain, *, max_radius_cells=3):
    """The (k, k) window test over the grid padded with blocked cells (reference)."""
    x, z, radius = (jnp.asarray(v, jnp.float32) for v in (x, z, radius))
    nx, nz = (x-terrain.min_x)/terrain.cell_size, (z-terrain.min_z)/terrain.cell_size
    r = radius/terrain.cell_size
    base_x, base_z = jnp.floor(nx).astype(jnp.int32), jnp.floor(nz).astype(jnp.int32)
    offsets = jnp.arange(-max_radius_cells-1, max_radius_cells+2)
    ix, iz = base_x+offsets[None, :], base_z+offsets[:, None]
    dx = jnp.maximum(jnp.abs(nx-(ix+.5))-.5, 0)
    dz = jnp.maximum(jnp.abs(nz-(iz+.5))-.5, 0)
    touched = dx*dx + dz*dz <= r*r
    m, k = max_radius_cells + 1, 2 * max_radius_cells + 3
    pad = ((0, 0),) * (terrain.walkable.ndim - 2) + ((m, m), (m, m))
    padded = jnp.pad(jnp.asarray(terrain.walkable, bool), pad)
    if terrain.layer is None:
        values = jax.lax.dynamic_slice(padded, (base_z, base_x), (k, k))
    else:
        values = jax.lax.dynamic_slice(padded, (terrain.layer, base_z, base_x), (1, k, k))[0]
    disk = jnp.all(~touched | values)
    point = values[m, m]
    point_bounds = ((x >= terrain.min_x) & (x < terrain.max_x)
                    & (z >= terrain.min_z) & (z < terrain.max_z))
    disk_bounds = ((x-radius > terrain.min_x) & (x+radius < terrain.max_x)
                   & (z-radius > terrain.min_z) & (z+radius < terrain.max_z))
    supported = jnp.isfinite(x) & jnp.isfinite(z) & jnp.isfinite(r) & (r >= 0) & (r <= max_radius_cells)
    return supported & jnp.where(radius == 0, point_bounds & point, disk_bounds & disk)


def world_radii():
    """Every terrain-check radius the world uses, incl. the segment-check inflations (routes radius 35)."""
    unit = (0.0, 35.0, *C.MINION_PATHING_RADIUS, *C.monster_pathing_radii())
    rr = sorted({min(r, 35.0) for r in unit})
    infl = [r + length / (samples - 1) / 2 for r in rr for samples, top in ((65, 600.), (33, 350.), (33, 200.))
            for length in np.linspace(0, top, 7)]
    return np.unique(np.float32([*unit, *rr, *(min(r, 50.0) for r in unit), *infl]))


def queries(rng, walk, terrain, n, radii):
    """``n`` (x, z, radius, layer): uniform over (and past) the grid, on a cs/8 lattice next to blocked cells and the
    border (exact ties), and jittered by a few ulps."""
    walk = np.asarray(walk).reshape(-1, *np.shape(walk)[-2:])
    h, w = walk.shape[-2:]
    cs, x0, z0 = terrain.cell_size, terrain.min_x, terrain.min_z
    layer = rng.integers(0, walk.shape[0], n)
    blocked = np.argwhere(np.pad(~walk, ((0, 0), (1, 1), (1, 1)), constant_values=True).any(0)) - 1
    cell = blocked[rng.integers(0, len(blocked), n)] + rng.integers(-3, 4, (n, 2))
    lattice = (cell[:, ::-1] + rng.integers(0, 9, (n, 2)) / 8) * cs + (x0, z0)
    uniform = rng.uniform((x0 - 2 * cs, z0 - 2 * cs), (x0 + (w + 2) * cs, z0 + (h + 2) * cs), (n, 2))
    pos = np.where(rng.random((n, 1)) < 0.5, lattice, uniform).astype(np.float32)
    ulps = (rng.integers(-3, 4, (n, 2)) * (rng.random((n, 1)) < 0.3)).astype(np.int32)
    pos = (pos.view(np.int32) + ulps).view(np.float32)
    pick = rng.random(n)
    r = np.where(pick < 0.5, rng.choice(radii, n), np.where(pick < 0.8, rng.integers(0, 33, n) * cs / 8,
                                                             rng.uniform(0, 3.2 * cs, n))).astype(np.float32)
    return pos[:, 0], pos[:, 1], r, layer


def compare(stacked, n, seed):
    """Both functions agree on ``n`` queries per window size on the layer view of ``stacked``."""
    rng = np.random.default_rng(seed)
    radii = world_radii()
    for cells in (1, 2, 3):
        x, z, r, layer = queries(rng, stacked.walkable, stacked, n, radii)
        args = (jnp.asarray(x), jnp.asarray(z), jnp.asarray(r), jnp.asarray(layer, jnp.int32))
        new, ref = (jax.jit(jax.vmap(lambda u, v, rr, ly: f(u, v, rr, stacked._replace(layer=ly),
                                                            max_radius_cells=cells)))(*args)
                    for f in (is_walkable, window_walkable))
        new, ref = np.asarray(new), np.asarray(ref)
        bad = np.nonzero(new != ref)[0]
        assert bad.size == 0, [(float(x[i]), float(z[i]), float(r[i]), int(layer[i]), cells) for i in bad[:5]]
        assert 0.03 < ref.mean() < 0.97                               # both outcomes well represented


@pytest.mark.parametrize("x64", [False, True])
def test_matches_window_on_random_grids(x64):
    rng = np.random.default_rng(3)
    with jax.enable_x64(x64):
        for seed in range(4):
            h, w = rng.integers(5, 40, 2)
            walk = rng.random((2, h, w)) < rng.uniform(0.85, 0.98)
            cs = float(rng.choice([50.0, 37.5]))
            x0, z0 = rng.uniform(-500, 500, 2).astype(np.float32).tolist()
            ext = (w - rng.uniform(0, 1)) * cs, (h - rng.uniform(0, 1)) * cs           # grid may overhang the bounds
            t = StaticTerrain(jnp.asarray(walk), cs, x0, z0, x0 + ext[0], z0 + ext[1])
            compare(t._replace(gaps=row_gaps(walk)), 20000, seed)
            compare(t, 2000, seed)                                                     # gaps derived per query


@pytest.mark.skipif(not MW.DEFAULT_MAP.exists(), reason="pinned 26.19 navgrid not mounted")
@pytest.mark.parametrize("x64", [False, True])
def test_matches_window_on_the_map_team_views(x64):
    from lanerl_jax.modern.data.navgrid import load_patch_map
    grid, _ = load_patch_map(MW.DEFAULT_MAP)
    with jax.enable_x64(x64):
        compare(team_view(tuple(grid.as_jax(t) for t in (0, 1)), 0), 400000, 11)


def test_rift_variant_gaps_match_their_masks():
    from lanerl_jax.modern.map import rift as R
    try:
        rt = R.load_rift_terrain()
    except FileNotFoundError:
        pytest.skip("rift variant artifact not mounted")
    assert bool(jnp.all(rt.gaps == row_gaps(rt.walkable)))
    t = team_view(R.terrain_pair(rt, R.variant_index(2, 1)), 0)
    assert bool(jnp.all(t.gaps == row_gaps(t.walkable)))
    compare(t, 100000, 5)
