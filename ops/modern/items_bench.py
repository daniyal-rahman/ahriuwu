"""TOOL: CPU/GPU cost of one full modern item tick (docs/modern/ITEMS_IMPLEMENTATION.md).

    ops/login_capped.sh 8G 2 .venv-jax/bin/python -m ops.modern.items_bench [BATCH]
"""
import sys
import time

import jax
import jax.numpy as jnp

from lanerl_jax.modern.combat import item_tick
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import effects as E
from lanerl_jax.modern.items import inventory as I
from lanerl_jax.modern.items.effects import runtime as R
from lanerl_jax.modern.tests import item_harness as H


def main() -> None:
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    n = 66
    u = H.units(H.champions(x1=150.) + [dict(x=100. * k, y=200, team=k % 2) for k in range(n - 2)])
    inv = I.inventory_from_ids([[3071, 3053, 3078, 6631, 3047, 6333], [3075, 3068, 3065, 2504, 3111, 3083]])
    own, item = I.owned_counts(inv), I.inventory_stats(inv)
    ctx = H.ctx(base_ad=60., bonus_ad=item.attack_damage, max_hp=600 + item.health, in_combat=True)
    dfn = D.default_defense(n)._replace(unit_class=u.cls)
    off = D.default_offense(n)._replace(unit_class=u.cls)
    base = D.packets(jnp.ones(8, bool), jnp.arange(8) % 2, (jnp.arange(8) + 1) % 2, 100., D.PHYSICAL, D.BASIC_ATTACK)

    def f(st, hp, sh, status, now):
        c = ctx._replace(now=now, hp=hp[:2])
        return item_tick(st, own, c, u._replace(hp=hp), attack=H.attack(hit=(True, True), target=(1, 0)),
                         cast=H.cast(started=(True, False)), request=jnp.asarray([6631, 0], jnp.int32),
                         base_packets=base, base_offense=off, base_defense=dfn, hp=hp, max_hp=u.max_hp,
                         shields=sh, status=status, kills=H.kills(n), holder_stats=item)

    st, hp = E.init(2, n), jnp.full((n,), 3000., jnp.float32)
    j = jax.jit(f)
    t = time.time()
    o = j(st, hp, D.init_shields(n), R.init_status(n), jnp.float32(1.))
    jax.block_until_ready(o.hp)
    print('single compile+first %.1fs' % (time.time() - t))
    t = time.time()
    for k in range(30):
        o = j(o.state, o.hp, o.shields, o.status, jnp.float32(1. + k / 30))
    jax.block_until_ready(o.hp)
    print('single per tick %.2f ms overflow %d' % ((time.time() - t) / 30 * 1000, int(o.packet_overflow)))
    vj = jax.jit(jax.vmap(f, in_axes=(0, 0, 0, 0, None)))
    bt = lambda x: jnp.broadcast_to(x, (b,) + x.shape)                  # noqa: E731
    o = vj(jax.tree_util.tree_map(bt, st), bt(hp), jax.tree_util.tree_map(bt, D.init_shields(n)),
           jax.tree_util.tree_map(bt, R.init_status(n)), jnp.float32(1.))
    jax.block_until_ready(o.hp)
    t = time.time()
    for k in range(10):
        o = vj(o.state, o.hp, o.shields, o.status, jnp.float32(1. + k / 30))
    jax.block_until_ready(o.hp)
    dt = (time.time() - t) / 10
    print('vmap%d per tick %.1f ms (%.3f ms/env) device %s' % (b, dt * 1000, dt * 1000 / b, jax.devices()[0]))


if __name__ == "__main__":
    main()
