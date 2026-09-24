"""Invariants that hold in EVERY step configuration the trainer uses.

Each of these would have caught a recorded bug; the row ID is in the test
name or docstring. They run the sim in the TRAINING configuration --
`SimConfig.training`, the object `trainer.py` steps (`STRUCT-003`) -- not only
the default one, because the fountain-turret drift (`COLL-004`) lived
exclusively in the deferred path and no other test ran it. Unrouted
(`route_artifact=None`): routing only affects ordered champions, and these
tests issue no orders.
"""
from __future__ import annotations

import numpy as np


def test_turrets_never_move_in_the_training_step_configuration():
    """`COLL-004`. `repair_collision_terrain_batch` point-queried all 66 slots
    while the inline path repairs only champions and minions. The blue
    fountain turret spawns on an unwalkable cell by design, so the deferred
    repair spiralled it out of terrain EVERY tick: y=3,355 at 30 s, 14,590 at
    120 s, off the map by 360 s -- with 1250 range and 999 AD -- in every RL
    run since 2026-09-18. Found by the structural review (2026-09-23).
    """
    import jax

    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.sim.config import SimConfig
    from lanerl_jax.sim.init import init_lane
    from lanerl_jax.sim.state import TU_SLICE
    from lanerl_jax.sim.step import env_advance

    patch = load_patch()
    cfg = SimConfig.training(patch, route_artifact=None)
    assert (cfg.collision_terrain, cfg.defer_collision_terrain) == (False, True)
    s = init_lane(patch)
    tx, ty = np.asarray(s.x[TU_SLICE]).copy(), np.asarray(s.y[TU_SLICE]).copy()
    step = jax.jit(lambda st: env_advance(st, cfg))
    for _ in range(3):                       # 3 decisions = 6 ticks
        s = step(s)
    assert np.array_equal(np.asarray(s.x[TU_SLICE]), tx), \
        "a turret moved: " + str(np.nonzero(np.asarray(s.x[TU_SLICE]) != tx)[0])
    assert np.array_equal(np.asarray(s.y[TU_SLICE]), ty)


def test_a_reused_minion_slot_starts_from_the_template():
    """`SLOT-001`. `spawn_minion` reset a hand-picked subset of fields, so a
    slot recycled from a dead minion inherited `has_auto_attacked`,
    `aa_cooldown`, `ai_local_time`, its `ignore_until`/`help_priority` row and
    column, and a missile still flying at the corpse. The server constructs a
    fresh object per spawn. Measured over a 600 s idle lane: 8-19 inherited
    `has_auto_attacked` per minute. The `had_target` leak (`HADTGT-001`) was
    the same class fixed one field at a time.
    """
    import jax.numpy as jnp

    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.sim.init import init_lane, spawn_minion
    from lanerl_jax.sim.profiles import profile_id
    from lanerl_jax.sim.state import Kind, MI_SLICE, Team
    from lanerl_jax.sim.targeting import MinionType

    s = init_lane(load_patch())
    i = MI_SLICE.start          # the lowest free minion slot: this is the one reused
    n = s.x.shape[0]
    s = s.replace(
        alive=s.alive.at[i].set(False),
        is_attacking=s.is_attacking.at[i].set(True),
        aa_target=s.aa_target.at[i].set(3),
        has_auto_attacked=s.has_auto_attacked.at[i].set(True),
        aa_cooldown=s.aa_cooldown.at[i].set(1.2),
        aa_windup=s.aa_windup.at[i].set(0.3),
        ms_since_damaged=s.ms_since_damaged.at[i].set(12.0),
        hit_flag_by=s.hit_flag_by.at[i].set(0),
        target_priority=s.target_priority.at[i].set(3),
        ai_local_time=s.ai_local_time.at[i].set(4567.0),
        time_since_attack=s.time_since_attack.at[i].set(890.0),
        ignore_until=s.ignore_until.at[i, :].set(5.0).at[:, i].set(6.0),
        help_priority=s.help_priority.at[i, :].set(3).at[:, i].set(4),
        missile_alive=s.missile_alive.at[0].set(True),
        missile_tx=s.missile_tx.at[0].set(i))
    path = jnp.asarray([[1000.0, 1000.0], [2000.0, 2000.0]], s.x.dtype)
    out = spawn_minion(s, Team.BLUE,
                       profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.BLUE),
                       455.0, path)
    assert bool(out.alive[i]) and int(out.kind[i]) == Kind.LANE_MINION
    want = {"is_attacking": False, "aa_target": -1, "has_auto_attacked": False,
            "aa_cooldown": 0.0, "aa_windup": 0.0, "ms_since_damaged": 1e6,
            "hit_flag_by": -1, "target_priority": 14, "ai_local_time": 0.0,
            "time_since_attack": 0.0, "had_target": False, "target": -1}
    for f, v in want.items():
        assert float(getattr(out, f)[i]) == v, (f, float(getattr(out, f)[i]))
    assert float(out.ignore_until[i].max()) == 0.0 and float(out.ignore_until[:, i].max()) == 0.0
    assert int(out.help_priority[i].min()) == 14 and int(out.help_priority[:, i].min()) == 14
    assert not bool(out.missile_alive[0]), "a missile aimed at the corpse must not land on the newcomer"
    # and nothing about any OTHER unit changed
    j = i + 1
    assert float(out.ignore_until[j, j + 1]) == float(s.ignore_until[j, j + 1])


def test_a_reused_minion_slot_is_not_anyone_elses_target():
    """`SLOT-002` (`docs/PLAYTEST_SWEEP.md` (c)1). The server's `TargetUnit`
    and `CastInfo.Targets[0]` reference an OBJECT; the corpse that held a
    slot is not the minion spawned into it. `spawn_minion` cleared missiles
    aimed at the slot but left every other unit's `target`/`aa_target` on
    it, so a caster mid-windup on a dead blue minion fired at the red minion
    that took its slot, across the map; and a champion held a recycled ALLY
    as its target. A held target is dropped; a swing is marked
    `AA_TARGET_GONE` (cancelled on its next update, `AA-007`), not -1, which
    would re-resolve it against `target`."""
    import jax.numpy as jnp

    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.sim.init import init_lane, spawn_minion
    from lanerl_jax.sim.profiles import profile_id
    from lanerl_jax.sim.state import Kind, MI_SLICE, Team
    from lanerl_jax.sim.targeting import MinionType

    s = init_lane(load_patch())
    i = MI_SLICE.start          # the lowest free minion slot: this is the one reused
    j, k = i + 1, 0             # a minion mid-swing on it; a champion holding it
    other = i + 2               # a unit targeting something else: untouched
    s = s.replace(
        alive=s.alive.at[i].set(False).at[j].set(True).at[other].set(True),
        kind=s.kind.at[j].set(Kind.LANE_MINION).at[other].set(Kind.LANE_MINION),
        target=s.target.at[j].set(i + 3).at[k].set(i).at[other].set(i + 3),
        aa_target=s.aa_target.at[j].set(i).at[other].set(i + 3),
        is_attacking=s.is_attacking.at[j].set(True).at[other].set(True),
        aa_windup=s.aa_windup.at[j].set(0.2).at[other].set(0.2))
    path = jnp.asarray([[1000.0, 1000.0], [2000.0, 2000.0]], s.x.dtype)
    out = spawn_minion(s, Team.RED,
                       profile_id(Kind.LANE_MINION, MinionType.CASTER, Team.RED),
                       290.0, path)
    assert bool(out.alive[i])
    assert int(out.aa_target[j]) != i, "a swing on the corpse would land on the newcomer"
    from lanerl_jax.sim.state import AA_TARGET_GONE
    assert int(out.aa_target[j]) == AA_TARGET_GONE
    assert int(out.target[j]) == i + 3, "its current target is someone else"
    assert int(out.target[k]) == -1, "the champion's held target was the corpse"
    assert int(out.target[other]) == i + 3 and int(out.aa_target[other]) == i + 3
