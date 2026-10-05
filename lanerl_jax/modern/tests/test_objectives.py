"""Symptom tests for the 26.19 epic objectives (docs/modern/OBJECTIVES.md).

A small synthetic world (no world.tick): 2 champions, 4 lane minions, the 8 objective slots,
two turrets. ``World.apply`` writes ``SlotWrites`` the way the step is expected to.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core import types as W
from lanerl_jax.modern.jungle import objectives as O

C, SLOT0 = 2, 6
N = 16
TURRET_RED, TURRET_BLUE = 14, 15
DT = 1.0 / 30.0


def table():
    return O.load_table(SLOT0)


class World:
    def __init__(self):
        k = np.zeros(N, np.int32)
        k[:2] = W.KIND_CHAMPION
        k[2:6] = W.KIND_MINION
        k[14:16] = W.KIND_TURRET
        team = np.array([0, 1, 0, 0, 1, 1] + [W.NEUTRAL] * 8 + [1, 0], np.int32)
        alive = k != W.KIND_NONE
        f = lambda v: np.full(N, v, np.float32)
        self.a = dict(kind=k, sub=np.zeros(N, np.int32), team=team, alive=alive, targetable=alive.copy(),
                      x=f(0.0), y=f(0.0), radius=f(65.0), hp=f(1000.0), max_hp=f(1000.0), armor=f(30.0),
                      magic_resist=f(30.0), attack_damage=f(100.0), attack_range=f(175.0), attack_speed=f(1.0),
                      move_speed=f(340.0), spawn_seq=np.arange(N, dtype=np.int32), spawn_time=f(0.0))
        self.a["hp"][14:16] = self.a["max_hp"][14:16] = 5000.0
        # Champions far from everything by default.
        self.place(0, 1000.0, 1000.0)
        self.place(1, 13000.0, 13000.0)
        for i in range(2, 6):
            self.place(i, 2000.0 + 100 * i, 12000.0)
        self.place(TURRET_RED, 4000.0, 13800.0)
        self.place(TURRET_BLUE, 1000.0, 10000.0)

    def place(self, i, x, y):
        self.a["x"][i], self.a["y"][i] = x, y

    def units(self):
        return W.WorldUnits(**{k: jnp.asarray(v) for k, v in self.a.items()})

    def apply(self, out: O.StepOut):
        w = jax.tree.map(np.asarray, out.writes)
        sl = slice(SLOT0, SLOT0 + 8)
        for name, src in (("kind", "kind"), ("sub", "sub"), ("team", "team"), ("x", "x"), ("y", "y"),
                          ("hp", "hp"), ("max_hp", "max_hp"), ("armor", "armor"), ("magic_resist", "magic_resist"),
                          ("attack_damage", "attack_damage"), ("attack_range", "attack_range"),
                          ("attack_speed", "attack_speed"), ("move_speed", "move_speed"), ("radius", "radius")):
            self.a[name][sl] = np.where(w.write, getattr(w, src), self.a[name][sl])
        self.a["alive"][sl] = (self.a["alive"][sl] | w.write) & ~w.despawn
        self.a["targetable"][sl] = self.a["alive"][sl]
        self.a["kind"][sl] = np.where(w.despawn, W.KIND_NONE, self.a["kind"][sl])
        self.a["hp"][sl] = np.minimum(self.a["hp"][sl] + np.asarray(out.monster_heal), self.a["max_hp"][sl])


def info(c=C):
    z = jnp.zeros((c,), jnp.float32)
    return O.ChampInfo(level=jnp.full((c,), 10, jnp.int32), bonus_ad=z + 50.0, ap=z, bonus_hp=z + 500.0,
                       max_hp=z + 2000.0, max_mana=z + 500.0, adaptive_physical=jnp.ones((c,), bool))


def step(obj, world, now, *, levels=(10, 10), dmg=None, use_eye=None, key=0, dt=DT):
    dm = jnp.zeros((N, N), bool) if dmg is None else dmg
    obj, out = O.objectives_step(obj, table(), world.units(), now=now, dt=dt, levels=jnp.asarray(levels),
                                 damage_matrix=dm, champ=info(), last_damaged=jnp.zeros((C,), jnp.float32),
                                 use_eye=use_eye, key=jax.random.PRNGKey(key))
    world.apply(out)
    return obj, out


def kill(obj, world, slot, by, now, *, levels=(10, 10)):
    """The champion ``by`` lands the final blow on objective ``slot``."""
    u = SLOT0 + slot
    pk = D.packets(jnp.asarray([True]), by, u, 1e6, D.TRUE, D.BASIC_ATTACK)
    died = jnp.zeros((N,), bool).at[u].set(True)
    killer = jnp.full((N,), -1, jnp.int32).at[u].set(by)
    hp = jnp.asarray(world.a["hp"]).at[u].set(0.0)
    obj, rw = O.objectives_after_damage(obj, table(), world.units(), pk, jnp.asarray([1e6]), died=died, killer=killer,
                                        hp_after=hp, now=now, levels=jnp.asarray(levels), champ=info())
    world.a["alive"][u] = False
    world.a["hp"][u] = 0.0
    return obj, rw


def fresh(key=0):
    return O.init_objectives(table(), N, C, jax.random.PRNGKey(key)), World()


def mtype(obj, slot):
    return int(obj.mtype[slot])


# ---- spawn timings ----------------------------------------------------------------------------

def test_spawn_timeline_grubs_herald_baron_dragon():
    obj, w = fresh()
    obj, out = step(obj, w, 299.0)
    assert not bool(out.writes.write[O.S_DRAGON])
    obj, out = step(obj, w, 300.0)                          # first drake at 5:00
    assert bool(out.writes.write[O.S_DRAGON]) and mtype(obj, O.S_DRAGON) == O.T_DRAKE
    assert int(out.writes.sub[O.S_DRAGON]) == O.SUB_BASE + O.T_DRAKE
    obj, out = step(obj, w, 479.9)
    assert not bool(out.writes.write[O.S_PIT])
    obj, out = step(obj, w, 480.0)                          # three Voidgrubs at 8:00
    assert np.asarray(out.writes.write)[:3].all() and all(mtype(obj, s) == O.T_GRUB for s in range(3))
    obj, out = step(obj, w, 885.0)                          # despawn 14:45 when not in combat
    assert np.asarray(out.writes.despawn)[:3].all()
    obj, out = step(obj, w, 900.0)                          # Rift Herald 15:00
    assert mtype(obj, O.S_PIT) == O.T_HERALD and bool(out.writes.write[O.S_PIT])
    obj, out = step(obj, w, 1185.0)                         # Herald leaves 19:45
    assert bool(out.writes.despawn[O.S_PIT])
    obj, out = step(obj, w, 1200.0)                         # Baron 20:00
    assert mtype(obj, O.S_PIT) == O.T_BARON and int(obj.baron_form) in (0, 1, 2)
    assert np.isclose(float(out.writes.x[O.S_PIT]), table().baron_pos[0])
    obj, rw = kill(obj, w, O.S_PIT, 0, 1300.0)
    assert np.isclose(float(obj.baron_next), 1300.0 + 360.0)
    obj, out = step(obj, w, 1659.0)
    assert not bool(out.writes.write[O.S_PIT])
    obj, out = step(obj, w, 1660.0)
    assert bool(out.writes.write[O.S_PIT]) and mtype(obj, O.S_PIT) == O.T_BARON


def test_no_atakhan_and_monster_stats_scale_with_level():
    import json
    d = json.loads(O.TABLE_PATH.read_text())
    assert "atakhan" in d["objectives_absent"] and not d["rules"]["atakhan_present"]["value"]
    st = O.slot_stats(table(), O.T_BARON, 0, 11, 1200.0)
    assert np.isclose(float(st["hp"]), 17791.75, atol=1.0)              # 26.1: 17,792 at level 11
    st18 = O.slot_stats(table(), O.T_BARON, 0, 18, 1200.0)
    assert np.isclose(float(st18["hp"]), 19190.0, atol=1.0)
    dr = O.slot_stats(table(), O.T_DRAKE, O.E_INFERNAL, 6, 300.0)
    assert np.isclose(float(dr["hp"]), 5106.25, atol=1.0)               # 26.1: 5,106 at level 6
    mtn = O.slot_stats(table(), O.T_DRAKE, O.E_MOUNTAIN, 6, 300.0)
    assert float(mtn["hp"]) > float(dr["hp"]) * 1.14                    # Mountain +15% HP


# ---- dragons, soul, elder, rift -----------------------------------------------------------------

def test_drake_kill_grants_its_stack_stats():
    obj, w = fresh()
    obj, _ = step(obj, w, 300.0)
    el = int(obj.elements[0])
    dx, dy = table().dragon_pos
    w.place(0, dx + 500.0, dy)                              # local XP needs 2000 range
    obj, rw = kill(obj, w, O.S_DRAGON, 0, 320.0)
    assert int(obj.stacks[0, el]) == 1 and int(jnp.sum(obj.stacks[1])) == 0
    assert np.isclose(float(rw.gold[0]), 75.0) and float(rw.xp[0]) > 0 and float(rw.epic_takedown[0]) == 1
    st = O.team_buff_stats(obj, table(), jnp.asarray([0, 1]), jnp.asarray([True, True]),
                           jnp.asarray([200.0, 200.0]), jnp.asarray([0.0, 0.0]), now=330.0,
                           out_of_combat=jnp.asarray([1.0, 1.0]))
    expect = {O.E_INFERNAL: ("attack_damage", 6.0), O.E_MOUNTAIN: ("percent_armor", 0.05),
              O.E_OCEAN: None, O.E_CLOUD: ("slow_resist", 0.05), O.E_HEXTECH: ("ability_haste", 5.0),
              O.E_CHEMTECH: ("tenacity", 0.06)}[el]
    if expect is not None:
        field, v = expect
        assert np.isclose(float(getattr(st, field)[0]), v, atol=1e-4)
        assert float(getattr(st, field)[1]) == 0.0
    assert np.isclose(float(obj.dragon_next), 620.0)                     # 5 min respawn


def test_infernal_stack_and_mountain_stack_values():
    obj, _ = fresh()
    obj = obj._replace(stacks=obj.stacks.at[0, O.E_INFERNAL].set(2).at[0, O.E_MOUNTAIN].set(1))
    st = O.team_buff_stats(obj, table(), jnp.asarray([0, 1]), jnp.asarray([True, True]), jnp.asarray([300.0, 300.0]),
                           jnp.asarray([100.0, 100.0]), now=900.0, out_of_combat=jnp.asarray([0.0, 0.0]))
    assert np.isclose(float(st.attack_damage[0]), 18.0) and np.isclose(float(st.ability_power[0]), 6.0)
    assert np.isclose(float(st.percent_magic_resist[0]), 0.05)


def test_soul_after_four_and_elder_and_rift_transformation():
    obj, w = fresh(key=3)
    t = 300.0
    variants = []
    for i in range(4):
        obj, _ = step(obj, w, t)
        assert bool(w.a["alive"][SLOT0 + O.S_DRAGON])
        obj, rw = kill(obj, w, O.S_DRAGON, 0, t + 10.0)
        variants.append(int(O.terrain_variant(obj, t + 11.0)))
        if i == 1:
            assert bool(rw.rift_decided) and int(obj.rift_element) == int(obj.elements[2])
        if i < 3:
            assert int(obj.soul[0]) == 0
        t = float(obj.dragon_next)
    assert variants[0] == 0 and variants[1] == int(obj.rift_element) * 3     # pit form not set yet
    assert int(obj.soul[0]) == int(obj.rift_element) and bool(rw.soul_granted[0])
    assert bool(obj.elder_ready) and np.isclose(t, 300.0 * 3 + 40.0 + 360.0 + 0.0, atol=1e3)
    obj, out = step(obj, w, t)                                                # Elder spawns 6 min later
    assert mtype(obj, O.S_DRAGON) == O.T_ELDER
    obj, rw = kill(obj, w, O.S_DRAGON, 0, t + 5.0)
    assert float(obj.elder_until[0]) > t and float(obj.elder_until[1]) < 0
    assert np.isclose(float(rw.gold[0]), 250.0)                              # 100 kill + 150 global
    assert np.isclose(float(obj.dragon_next), t + 5.0 + 360.0)


def test_ancient_grudge_reduces_champion_damage_to_drake():
    obj, w = fresh()
    obj, _ = step(obj, w, 300.0)
    obj = obj._replace(stacks=obj.stacks.at[0, 1].set(2))
    pk = D.packets(jnp.asarray([True, True]), jnp.asarray([0, 1]), SLOT0 + O.S_DRAGON, 100.0, D.PHYSICAL)
    out = O.objectives_packet_mods(obj, table(), pk, w.units(), now=310.0)
    assert np.allclose(np.asarray(out.raw), [70.0, 100.0])


def test_elder_execute_threshold():
    obj, w = fresh()
    obj = obj._replace(elder_until=jnp.asarray([2000.0, -1e9], jnp.float32))
    now = 1800.0
    pk = D.packets(jnp.asarray([True]), 0, 1, 100.0, D.PHYSICAL, D.BASIC_ATTACK)

    def hit(obj, hp_left, t):
        hp = jnp.asarray(w.a["hp"]).at[1].set(hp_left)
        return O.objectives_after_damage(obj, table(), w.units(), pk, jnp.asarray([100.0]),
                                         died=jnp.zeros((N,), bool), killer=jnp.full((N,), -1, jnp.int32),
                                         hp_after=hp, now=t, levels=jnp.asarray([14, 14]), champ=info())[0]
    above = hit(obj, 0.21 * 1000.0, now)                    # 21%: burn but no execute
    assert float(above.exec_at[1]) > 1e8 and float(above.burn_until[1]) > now
    below = hit(obj, 0.19 * 1000.0, now)                    # 19%: execute armed after 0.5 s
    assert np.isclose(float(below.exec_at[1]), now + 0.5)
    w.a["hp"][1] = 190.0
    o2, out = step(below, w, now + 0.25)
    ex = out.packets
    assert not bool(jnp.any(ex.valid & D.has(ex.flags, D.PROP_EXECUTE)))
    o2, out = step(o2, w, now + 0.5)
    m = np.asarray(out.packets.valid & D.has(out.packets.flags, D.PROP_EXECUTE))
    assert m.sum() == 1 and int(out.packets.dst[m][0]) == 1 and int(out.packets.src[m][0]) == 0
    assert np.isclose(float(out.packets.raw[m][0]), 1000.0)
    # Burn ticks: 75 total at <= 25 min split in three true-damage ticks.
    burns = np.asarray(out.packets.valid & D.has(out.packets.flags, D.TAG_PERIODIC) & (out.packets.dst == 1))
    assert burns.sum() <= 1


# ---- Baron ------------------------------------------------------------------------------------------

def _baron_world(now=1200.0):
    obj, w = fresh()
    obj, _ = step(obj, w, now)
    assert mtype(obj, O.S_PIT) == O.T_BARON
    return obj, w


def test_hand_of_baron_empowers_allied_minions_near_the_champion():
    obj, w = _baron_world()
    obj, rw = kill(obj, w, O.S_PIT, 0, 1500.0)              # 25:00
    assert float(rw.gold[0]) == 250.0 and np.isclose(float(rw.xp[0]), 650.0)
    ad, ap = float(obj.baron_ad[0]), float(obj.baron_ap[0])
    assert 16.0 < ad < 19.0 and 27.0 < ap < 32.0            # 25 min: between the 24/26 min wiki points
    assert float(obj.baron_until[1]) < 0                    # enemy team gets nothing
    w.place(0, 5000.0, 5000.0)
    w.place(2, 5300.0, 5000.0)                              # allied minion within 600
    w.place(3, 6300.0, 5000.0)                              # allied minion within 1450
    w.place(4, 5200.0, 5000.0)                              # enemy minion: never empowered
    w.place(5, 9000.0, 9000.0)
    obj, mb = O.baron_minion_buffs(obj, table(), w.units(), now=1510.0)
    emp = np.asarray(mb.empowered)
    assert emp[2] and emp[3] and not emp[4] and not emp[0]
    assert float(mb.bonus_range[2]) == 75.0                 # melee +75 range
    pk = D.packets(jnp.asarray([True]), 1, 2, 100.0, D.PHYSICAL, D.BASIC_ATTACK)
    out = O.objectives_packet_mods(obj, table(), pk, w.units(), now=1500.0)
    assert np.isclose(float(out.raw[0]), 100.0 * (1 - 0.55), atol=0.5)    # 50% + 1%/min: 55% at 25 min
    # The champion leaves: minions lose it beyond 1500.
    w.place(0, 12000.0, 2000.0)
    obj, mb = O.baron_minion_buffs(obj, table(), w.units(), now=1520.0)
    assert not np.asarray(mb.empowered)[2:4].any()
    st = O.team_buff_stats(obj, table(), jnp.asarray([0, 1]), jnp.asarray([True, True]), jnp.asarray([100.0, 100.0]),
                           jnp.asarray([0.0, 0.0]), now=1520.0, out_of_combat=jnp.asarray([0.0, 0.0]))
    assert np.isclose(float(st.attack_damage[0]), ad)
    _, out = step(obj, w, 1521.0)
    assert bool(out.empowered_recall[0]) and not bool(out.empowered_recall[1])


def test_baron_attacks_apply_void_corruption_and_ability_rotation():
    obj, w = _baron_world()
    bx, by = table().baron_pos
    w.place(0, bx + 400.0, by)
    u = w.units()
    for k in range(6):
        launch = W.AttackLaunch(jnp.zeros((N,), bool).at[SLOT0].set(True), jnp.zeros((N,), jnp.int32),
                                jnp.ones((N,), bool), jnp.zeros((N,), bool), jnp.zeros((N,), jnp.int32))
        obj, raw, dtype, flags, extra, cc = O.objectives_attack(obj, table(), u, launch, now=1210.0 + k)
        assert float(raw[SLOT0]) == pytest.approx(float(u.attack_damage[SLOT0]))
    assert float(obj.void_stacks[0]) >= 6 * 3
    _, out = step(obj, w, 1216.0)
    assert float(out.armor_reduction[0]) == pytest.approx(0.5 * float(obj.void_stacks[0]))
    assert int(obj.attack_count[O.S_PIT]) == 6


# ---- Herald ----------------------------------------------------------------------------------------

def test_herald_eye_summons_mercenary_that_damages_a_turret():
    obj, w = fresh()
    obj, _ = step(obj, w, 900.0)
    obj, rw = kill(obj, w, O.S_PIT, 0, 950.0)
    assert float(obj.eye_until[0]) > 950.0 and bool(obj.recall_charge[0]) and float(rw.gold[0]) == 100.0
    assert float(rw.large_monster_kill[0]) == 1.0
    w.place(0, 4000.0, 13000.0)                             # 800 from the red turret
    obj, out = step(obj, w, 960.0, use_eye=jnp.asarray([True, False]))
    assert bool(out.writes.write[O.S_MERC]) and int(out.writes.team[O.S_MERC]) == 0
    assert float(obj.eye_until[0]) < 0
    t, leap = 960.0, None
    for _ in range(90):
        t += DT
        obj, out = step(obj, w, t)
        m = np.asarray(out.packets.valid & (out.packets.dst == TURRET_RED) & (out.packets.dtype == D.TRUE))
        if m.any():
            leap = float(np.asarray(out.packets.raw)[m].sum())
            break
    assert leap == pytest.approx(3000.0)
    assert t - 960.0 == pytest.approx(2.5, abs=0.1)
    # She loses 66% of her current HP on the hit.
    selfp = np.asarray(out.packets.valid & (out.packets.dst == SLOT0 + O.S_MERC))
    assert np.isclose(float(np.asarray(out.packets.raw)[selfp].sum()), 0.66 * w.a["hp"][SLOT0 + O.S_MERC], rtol=1e-3)


def test_monster_aggro_and_leash_reset():
    obj, w = fresh()
    obj, _ = step(obj, w, 900.0)
    hx, hy = table().pit
    w.place(0, hx + 300.0, hy)
    dm = jnp.zeros((N, N), bool).at[0, SLOT0].set(True)
    obj, out = step(obj, w, 901.0, dmg=dm)
    assert bool(obj.aggro[0]) and int(obj.target[0]) == 0
    w.place(0, hx + 5000.0, hy)                             # leaves the leash
    t = 901.0
    for _ in range(40):
        t += 0.25
        obj, out = step(obj, w, t, dt=0.25)
    assert not bool(obj.aggro[0]) and int(out.desired[0]) == -1
    assert np.allclose(np.asarray(out.goal[0]), [hx, hy])


# ---- Voidgrubs ---------------------------------------------------------------------------------------

def test_grubs_buff_structure_damage_and_summon_voidmites():
    obj, w = fresh()
    obj, _ = step(obj, w, 480.0)
    for s in range(3):
        obj, rw = kill(obj, w, s, 0, 490.0 + s)
        if s == 0:
            assert float(rw.epic_takedown[0]) == 1.0 and np.isclose(float(rw.gold[0]), 30.0)
        else:
            assert float(rw.epic_takedown[0]) == 0.0
    assert int(obj.grub_stacks[0]) == 3
    w.place(0, 4000.0, 13650.0)
    pk = D.packets(jnp.asarray([True]), 0, TURRET_RED, 120.0, D.PHYSICAL, D.BASIC_ATTACK)
    obj, _ = O.objectives_after_damage(obj, table(), w.units(), pk, jnp.asarray([80.0]), died=jnp.zeros((N,), bool),
                                       killer=jnp.full((N,), -1, jnp.int32), hp_after=jnp.asarray(w.a["hp"]),
                                       now=600.0, levels=jnp.asarray([8, 8]), champ=info())
    assert bool(obj.pending[5])                             # Hunger of the Void summon armed
    total, t = 0.0, 600.0
    spawned = False
    for _ in range(int(4.2 / DT)):
        t += DT
        obj, out = step(obj, w, t)
        m = np.asarray(out.packets.valid & (out.packets.dst == TURRET_RED) & (out.packets.src == 0))
        total += float(np.asarray(out.packets.raw)[m].sum())
        spawned |= bool(out.writes.write[5])
    assert total == pytest.approx(16.0 * 8)                 # melee, 3 stacks: 16 true per 0.5 s for 4 s
    assert spawned and int(w.a["team"][SLOT0 + 5]) == 0 and mtype(obj, 5) == O.T_ALLY_MITE
    assert int(out.desired[5]) == TURRET_RED


# ---- Elemental rift terrain ---------------------------------------------------------------------------

def test_rift_terrain_changes_walkable_mask_after_transformation():
    from lanerl_jax.modern.map import rift as R
    rt = R.load_rift_terrain()
    base = np.asarray(rt.walkable[R.variant_index(0, 0)])
    mtn = np.asarray(rt.walkable[R.variant_index(O.E_MOUNTAIN, 0)])
    ocean_b = np.asarray(rt.bush_ids[R.variant_index(O.E_OCEAN, 0)]) > 0
    # Mountain rock formation at cells x 159..165, z 100..110 (overlay rect): open before, wall after.
    assert base[0, 103, 161] and not mtn[0, 103, 161]
    assert (base != mtn).sum() > 500
    assert ocean_b.sum() > (np.asarray(rt.bush_ids[0]) > 0).sum() + 400        # brush grows
    assert (np.asarray(rt.walkable[R.variant_index(O.E_CLOUD, 0)]) == base).all()   # no Cloud overlay
    obj, _ = fresh()
    obj = obj._replace(rift_element=jnp.int32(O.E_MOUNTAIN), rift_at=jnp.float32(700.0), baron_form=jnp.int32(2))
    assert int(O.terrain_variant(obj, 699.0)) == R.variant_index(0, 2)
    v = O.terrain_variant(obj, 700.0)
    assert int(v) == R.variant_index(O.E_MOUNTAIN, 2)
    ter = R.terrain_pair(rt, v)
    from lanerl_jax.modern.map.terrain import is_walkable
    x = rt.min_x + 161.5 * rt.cell_size
    z = rt.min_z + 103.5 * rt.cell_size
    assert not bool(is_walkable(x, z, 0.0, ter[0]))
    assert bool(is_walkable(x, z, 0.0, R.terrain_pair(rt, R.variant_index(0, 0))[0]))


def test_all_entry_points_trace_under_jit():
    """Trace-only (eval_shape): no Python control flow on traced values."""
    obj, w = fresh()
    u = w.units()
    tb = table()
    lv = jnp.asarray([10, 10])
    f = lambda o, u, now: O.objectives_step(o, tb, u, now=now, dt=DT, levels=lv, damage_matrix=jnp.zeros((N, N), bool),
                                            champ=info(), last_damaged=jnp.zeros((C,), jnp.float32),
                                            key=jax.random.PRNGKey(0))
    jax.eval_shape(f, obj, u, jnp.float32(1200.0))
    launch = W.AttackLaunch(jnp.zeros((N,), bool), jnp.zeros((N,), jnp.int32), jnp.zeros((N,), bool),
                            jnp.zeros((N,), bool), jnp.zeros((N,), jnp.int32))
    jax.eval_shape(lambda o, u, now: O.objectives_attack(o, tb, u, launch, now=now), obj, u, jnp.float32(1.0))
    pk = D.packets(jnp.asarray([True]), 0, 1, 10.0, D.PHYSICAL)
    jax.eval_shape(lambda o, u, now: O.objectives_packet_mods(o, tb, pk, u, now=now), obj, u, jnp.float32(1.0))
    jax.eval_shape(lambda o, u, now: O.objectives_after_damage(
        o, tb, u, pk, jnp.asarray([10.0]), died=jnp.zeros((N,), bool), killer=jnp.full((N,), -1, jnp.int32),
        hp_after=u.hp, now=now, levels=lv, champ=info()), obj, u, jnp.float32(1.0))
    jax.eval_shape(lambda o, u, now: O.baron_minion_buffs(o, tb, u, now=now), obj, u, jnp.float32(1.0))
    jax.eval_shape(lambda o, now: O.team_buff_stats(o, tb, jnp.asarray([0, 1]), jnp.asarray([True, True]),
                                                   jnp.ones((2,)), jnp.ones((2,)), now=now,
                                                   out_of_combat=jnp.ones((2,))), obj, jnp.float32(1.0))
