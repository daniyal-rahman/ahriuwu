"""Fused CUDA visibility: keep each ray's traversal inside one GPU kernel.

Validated in PERF-005: GPU rays, vision tests and full collection/PPO outputs.
vision.clear_ray_reference remains the CPU and fidelity reference. No cell-bound, precision or brush-rule
changes. CPU interpretation is for bounded correctness checks, not throughput.
"""
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def clear_ray_fused(grid, x0, y0, x1, y1, *, enabled=True, interpret=False):
    x0, y0, x1, y1 = jnp.broadcast_arrays(
        (x0-grid.min_x)/grid.cell_size, (y0-grid.min_y)/grid.cell_size,
        (x1-grid.min_x)/grid.cell_size, (y1-grid.min_y)/grid.cell_size)
    shape = x0.shape
    count = math.prod(shape)
    ix, iy = jnp.floor(x0).astype(jnp.int32), jnp.floor(y0).astype(jnp.int32)
    ex, ey = jnp.floor(x1).astype(jnp.int32), jnp.floor(y1).astype(jnp.int32)
    dx, dy = jnp.abs(x1-x0), jnp.abs(y1-y0)
    sx, sy = jnp.sign(x1-x0).astype(jnp.int32), jnp.sign(y1-y0).astype(jnp.int32)
    error = jnp.where(sx > 0, ix+1-x0, x0-ix)*dy - jnp.where(sy > 0, iy+1-y0, y0-iy)*dx
    error = jnp.where(dx == 0, jnp.inf, error)
    error = jnp.where(dy == 0, -jnp.inf, error)
    remaining = 1+jnp.abs(ex-ix)+jnp.abs(ey-iy)
    height, width = grid.flags.shape
    block = 128

    def kernel(flags, ix_ref, iy_ref, ex_ref, ey_ref, sx_ref, sy_ref,
               dx_ref, dy_ref, error_ref, left_ref, enabled_ref, out):
        lane = pl.program_id(0)*block + jnp.arange(block)
        mask = lane < count
        def read(ref):
            return plt.load(ref.at[lane], mask=mask, other=0)
        x, y, end_x, end_y = map(read, (ix_ref, iy_ref, ex_ref, ey_ref))
        step_x, step_y, ddx, ddy = map(read, (sx_ref, sy_ref, dx_ref, dy_ref))
        err, left = read(error_ref), read(left_ref)

        def flags_at(xx, yy):
            valid = (xx >= 0) & (yy >= 0) & (xx < width) & (yy < height)
            value = plt.load(flags.at[jnp.clip(yy, 0, height-1),
                                     jnp.clip(xx, 0, width-1)], mask=mask, other=0)
            return value, valid
        start, valid0 = flags_at(x, y)
        end, valid1 = flags_at(end_x, end_y)
        start_grass, end_grass = (start & 1) != 0, (end & 1) != 0
        clear = valid0 & valid1 & (read(enabled_ref) != 0) & mask

        def cell_clear(xx, yy):
            value, valid = flags_at(xx, yy)
            transparent = ((value & 2) == 0) | ((value & (0x40 | 0x100)) != 0)
            grass = (value & 1) != 0
            return valid & transparent & jnp.where(start_grass, ~end_grass | grass, ~grass)

        def pending(carry):
            iteration, _, _, _, remaining_, clear_ = carry
            return (iteration < 64) & (jnp.max(jnp.where(clear_, remaining_, 0)) > 0)

        def advance(carry):
            iteration, xx, yy, error_, remaining_, clear_ = carry
            active = (remaining_ > 0) & clear_
            corner = jnp.abs(error_) <= 1e-3
            ok = cell_clear(xx, yy) & (~corner | (cell_clear(xx+step_x, yy) & cell_clear(xx, yy+step_y)))
            clear_ &= ~active | ok
            move_x, move_y = (error_ < 0) | corner, (error_ > 0) | corner
            xx += jnp.where(active & move_x, step_x, 0)
            yy += jnp.where(active & move_y, step_y, 0)
            error_ += jnp.where(move_x & move_y, ddy-ddx, jnp.where(move_x, ddy, -ddx))
            remaining_ -= jnp.where(active, 1+corner.astype(jnp.int32), 0)
            return iteration+1, xx, yy, error_, remaining_, clear_

        _, _, _, _, left, clear = jax.lax.while_loop(pending, advance, (0, x, y, err, left, clear))
        plt.store(out.at[lane], (clear & (left <= 0)).astype(jnp.int32), mask=mask)

    args = (ix, iy, ex, ey, sx, sy, dx, dy, error, remaining,
            jnp.broadcast_to(jnp.asarray(enabled, jnp.int32), shape))
    result = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((count,), jnp.int32),
        grid=(pl.cdiv(count, block),), interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=4))(
            grid.flags, *(x.reshape(-1) for x in args))
    return result.reshape(shape) != 0
