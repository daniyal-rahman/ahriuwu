"""Jungle symptom tests (docs/modern/JUNGLE.md), eager over 2 champions + the 38 jungle slots."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.stats import MAGIC, PHYSICAL, TRUE
from lanerl_jax.modern.core.types import (JUNGLE_SLOTS, KIND_CHAMPION, KIND_MONSTER, KIND_NONE, NEUTRAL, UNIT_COLUMNS,
                                          AttackLaunch, CastOrder, WorldUnits, init_attack_state)
from lanerl_jax.modern.jungle import camps as J

M = J.Monster
C = 2
M0 = 2


@pytest.fixture(scope="module")
def table():
    return J.build_table(M0)


class World:
    def __init__(self, table, now=0.0):
        n = M0 + table.n_slots
        self.table, self.n, self.now = table, n, now
        f = lambda v: np.full(n, v, np.float32)
        self.a = dict(kind=np.full(n, KIND_NONE, np.int32), sub=np.zeros(n, np.int32), team=np.full(n, NEUTRAL, np.int32),
                      alive=np.zeros(n, bool), targetable=np.zeros(n, bool), x=f(0.0), y=f(0.0), hp=f(0.0),
                      max_hp=f(0.0), radius=f(50.0), armor=f(0.0), magic_resist=f(0.0), attack_damage=f(0.0),
                      attack_range=f(0.0), attack_speed=f(0.0), move_speed=f(0.0), windup=f(0.0),
                      missile_speed=f(0.0), spawn_time=f(0.0),
                      spawn_seq=np.arange(n, dtype=np.int32))
        for c, (x, y) in enumerate(((3000.0, 3000.0), (12000.0, 12000.0))):
            self.a["kind"][c], self.a["team"][c], self.a["alive"][c], self.a["targetable"][c] = KIND_CHAMPION, c, True, True
            self.a["x"][c], self.a["y"][c], self.a["hp"][c], self.a["max_hp"][c] = x, y, 1000.0, 1000.0
            self.a["radius"][c], self.a["attack_range"][c], self.a["move_speed"][c] = 65.0, 125.0, 345.0
        self.state = J.init_jungle(table, C, n)
        self.seq = n

    def units(self):
        a = {k: jnp.asarray(v) for k, v in self.a.items()}
        return WorldUnits(**{f: a[f] for f in WorldUnits._fields if f != "targetable"},
                          targetable=a["targetable"] & a["alive"])

    def spawn(self, level=(1, 1)):
        self.state, w = J.spawn_step(self.state, self.table, now=self.now, champion_level=jnp.asarray(level))
        u = jax.tree.map(np.asarray, J.unit_write(self.table, w, self.n))       # what world.units.write_units applies
        for k in UNIT_COLUMNS:
            self.a[k] = np.where(u.mask, getattr(u, k), self.a[k]).astype(self.a[k].dtype)
        self.a["alive"] |= u.mask
        self.a["targetable"] |= u.mask
        self.a["spawn_time"] = np.where(u.new, self.now, self.a["spawn_time"]).astype(np.float32)
        self.a["spawn_seq"] = np.where(u.new, self.seq + np.cumsum(u.new) - 1, self.a["spawn_seq"]).astype(np.int32)
        self.seq += int(u.new.sum())
        return w

    def ai(self, dmg=None, dt=1.0 / 30.0):
        dm = np.zeros((self.n, self.n), bool) if dmg is None else dmg
        self.state, out = J.monster_ai(self.state, self.table, self.units(), init_attack_state(self.n), now=self.now,
                                       dt=dt, damage_events=jnp.asarray(dm))
        s = slice(M0, M0 + self.table.n_slots)
        self.a["hp"][s] = np.minimum(self.a["hp"][s] + np.asarray(out.heal), self.a["max_hp"][s])
        self.a["alive"][s] &= ~np.asarray(out.despawn)
        self.a["targetable"][s] = np.asarray(out.targetable)
        return out

    def kill(self, slots, killer=0, level=(1.0, 1.0), **kw):
        died = np.zeros(self.n, bool)
        died[[M0 + s for s in slots]] = True
        kill = np.full(self.n, -1, np.int32)
        kill[died] = killer
        self.a["alive"][died] = False
        self.a["hp"][died] = 0.0
        hp = kw.pop("hp", self.a["hp"][:C])
        mx = kw.pop("max_hp", self.a["max_hp"][:C])
        self.state, rw = J.death_step(self.state, self.table, self.units(), now=self.now, died=jnp.asarray(died),
                                      killer=jnp.asarray(kill), avg_level=jnp.float32(np.mean(level)),
                                      champion_level=jnp.asarray(level, jnp.float32), hp=jnp.asarray(hp),
                                      max_hp=jnp.asarray(mx), mana=jnp.asarray([300.0, 300.0]),
                                      max_mana=jnp.asarray([300.0, 300.0]), **kw)
        return rw

    def hit(self, champ, slot):
        dm = np.zeros((self.n, self.n), bool)
        dm[champ, M0 + slot] = True
        return dm

    def put_champ(self, c, x, y):
        self.a["x"][c], self.a["y"][c] = x, y


def slots_of(table, camp_name):
    k = J.CAMP_NAMES.index(camp_name)
    return [int(s) for s in np.flatnonzero(np.asarray(table.slot_camp) == k)]


def slot_type(table, s):
    return int(np.asarray(table.slot_init_type)[s])


# ---- table --------------------------------------------------------------------------------------

def test_table_fits_the_regular_monster_budget(table):
    assert table.n_slots <= JUNGLE_SLOTS == 40
    blue = slots_of(table, "Order Blue")
    assert [slot_type(table, s) for s in blue] == [M.BLUE]
    np.testing.assert_allclose(np.asarray(table.slot_home)[blue[0]], (3821.5, 7901.1), atol=0.2)   # client placement
    wolves = [slot_type(table, s) for s in slots_of(table, "Chaos Wolves")]
    assert wolves == [M.WOLF, M.WOLF_MINI, M.WOLF_MINI]
    raptors = [slot_type(table, s) for s in slots_of(table, "Order Wraiths")]
    assert raptors == [M.RAPTOR] + [M.RAPTOR_MINI] * 5
    krugs = [slot_type(table, s) for s in slots_of(table, "Order Small Golems")]
    assert krugs[:2] == [M.KRUG, M.KRUG_MEDIUM] and krugs[2:] == [-1] * 4        # 6 minis: 2 reuse the parents
    st = J.monster_stats(table, M.BLUE, 3)
    assert float(st.hp) == pytest.approx(2760.0) and float(st.attack_damage) == pytest.approx(66 * 1.1)
    st = J.monster_stats(table, M.GROMP, 12)
    assert float(st.hp) == pytest.approx(2050 * 2.05)
    assert float(J.monster_stats(table, M.SCUTTLE, 1, True).hp) == pytest.approx(1550 * 0.65)


# ---- camps --------------------------------------------------------------------------------------

def test_camp_spawns_on_time_and_respawns_after_the_last_member(table):
    w = World(table, now=54.9)
    assert not np.any(np.asarray(w.spawn().mask))
    w.now = 55.0
    m = np.asarray(w.spawn().mask)
    wolves, gromp = slots_of(table, "Order Wolves"), slots_of(table, "Order OwlBear")
    assert m[wolves].all() and not m[gromp].any()                     # Gromp/Krugs at 1:07
    assert w.a["kind"][M0 + wolves[0]] == KIND_MONSTER and w.a["sub"][M0 + wolves[1]] == M.WOLF_MINI
    assert w.a["hp"][M0 + wolves[0]] == pytest.approx(1600.0)
    w.now = 67.0
    assert np.asarray(w.spawn().mask)[gromp].all()
    w.now = 100.0
    w.kill(wolves[:2])
    k = J.CAMP_NAMES.index("Order Wolves")
    assert not np.isfinite(float(w.state.camp_respawn_at[k]))         # one wolf still alive
    w.now = 110.0
    w.kill(wolves[2:])
    assert float(w.state.camp_respawn_at[k]) == pytest.approx(110.0 + 135.0)
    w.now = 244.9
    assert not np.asarray(w.spawn().mask)[wolves].any()
    w.now = 245.0
    assert np.asarray(w.spawn(level=(5, 5)).mask)[wolves].all()
    assert int(w.state.level[wolves[0]]) == 5
    assert w.a["hp"][M0 + wolves[0]] == pytest.approx(1600.0 * 1.4)


def test_krug_splits_and_marked_for_death(table):
    w = World(table, now=67.0)
    w.spawn(level=(3, 3))
    krugs = slots_of(table, "Order Small Golems")
    ancient, medium = krugs[0], krugs[1]
    w.now = 80.0
    w.kill([ancient])
    w.now = 80.5
    assert not np.asarray(w.spawn().mask).any()
    w.now = 81.0
    m = np.asarray(w.spawn().mask)
    minis = [s for s in krugs if m[s]]
    assert len(minis) == 4 and ancient in minis
    assert all(w.a["sub"][M0 + s] == M.KRUG_MINI for s in minis)
    assert all(int(w.state.level[s]) == 2 for s in minis)            # one level below the parent
    # Marked for death: 10 s without champion combat after the large monster died -> despawn.
    w.now = 85.0
    w.ai(dmg=w.hit(0, minis[0]))
    w.now = 94.9
    assert not np.asarray(w.ai().despawn).any()
    w.now = 95.1
    out = w.ai()
    assert np.asarray(out.despawn)[minis + [medium]].all()
    k = J.CAMP_NAMES.index("Order Small Golems")
    assert float(w.state.camp_respawn_at[k]) == pytest.approx(95.1 + 135.0)


def test_krug_medium_splits_in_two(table):
    w = World(table, now=67.0)
    w.spawn()
    krugs = slots_of(table, "Chaos Small Golems")
    w.now = 70.0
    rw = w.kill([krugs[1]])
    assert float(rw.gold[0]) == pytest.approx(10.0) and float(rw.xp[0]) == pytest.approx(10.0)
    w.now = 71.0
    m = np.asarray(w.spawn().mask)
    assert sorted(s for s in krugs if m[s]) == sorted([krugs[1], krugs[5]])


# ---- monster AI ---------------------------------------------------------------------------------

def test_attacked_monster_aggroes_its_camp_and_targets_the_attacker(table):
    w = World(table, now=55.0)
    w.spawn()
    wolves = slots_of(table, "Order Wolves")
    w.put_champ(0, 3900.0, 6300.0)
    w.now = 56.0
    out = w.ai(dmg=w.hit(0, wolves[2]))                               # hit a small wolf
    assert (np.asarray(out.desired)[wolves] == 0).all()               # the whole camp aggroes champion 0
    assert np.asarray(out.moving)[wolves].any()


def test_leash_reset_returns_home_and_heals(table):
    w = World(table, now=55.0)
    w.spawn()
    blue = slots_of(table, "Order Blue")[0]
    u = M0 + blue
    w.put_champ(0, 3900.0, 7700.0)
    w.now = 56.0
    out = w.ai(dmg=w.hit(0, blue))
    assert int(out.desired[blue]) == 0
    w.a["hp"][u] = 1000.0
    # Kite it far outside its 650 leash; the monster follows (world movement is faked here).
    w.put_champ(0, 3900.0, 6200.0)
    w.a["y"][u] = 6900.0
    mode = []
    for i in range(1, 200):
        w.now = 56.0 + i / 10.0
        out = w.ai(dt=0.1)
        mode.append(int(w.state.reset_mode[blue]))
        if mode[-1] != J.NO_RESET:
            break
    assert mode[-1] == J.SOFT and int(out.desired[blue]) == -1
    gx, gy = float(out.goal_x[blue]), float(out.goal_y[blue])
    np.testing.assert_allclose((gx, gy), np.asarray(table.slot_home)[blue], atol=1e-3)
    assert float(out.move_speed[blue]) == pytest.approx(275.0 * J.MS_IMPATIENT)
    hp0 = w.a["hp"][u]
    w.now += 1.0
    w.ai(dt=1.0)
    assert w.a["hp"][u] == pytest.approx(hp0 + 0.06 * 2300.0, rel=1e-4)   # 6% max HP per second
    w.now += 6.0
    w.ai(dt=0.1)
    assert int(w.state.reset_mode[blue]) == J.HARD                    # soft -> hard after 6 s
    w.a["x"][u], w.a["y"][u] = np.asarray(table.slot_home)[blue]
    w.now += 0.1
    out = w.ai(dt=0.1)
    assert int(w.state.reset_mode[blue]) == J.NO_RESET and not bool(w.state.aggro[blue])
    assert w.a["hp"][u] == pytest.approx(2300.0)                      # back at camp: full health
    w.now += 0.1
    assert int(w.ai(dt=0.1).desired[blue]) == -1                      # ignores the old attacker


def test_monster_attack_packets_and_holder_reduction(table):
    w = World(table, now=55.0)
    w.spawn()
    blue = slots_of(table, "Order Blue")[0]
    w.a["hp"][0] = 800.0
    launch = AttackLaunch(jnp.zeros(w.n, bool).at[M0 + blue].set(True), jnp.full(w.n, -1, jnp.int32).at[M0 + blue].set(0),
                          jnp.zeros(w.n, bool), jnp.zeros(w.n, bool), jnp.arange(w.n, dtype=jnp.int32) + 1)
    main, bonus = J.monster_attack_packets(w.state, table, w.units(), launch)
    v = np.asarray(main.valid)
    assert v.sum() == 1 and int(main.dtype[blue]) == PHYSICAL
    assert float(main.raw[blue]) == pytest.approx(66.0 + 0.05 * 800.0)
    assert not np.asarray(bonus.valid).any()
    held = w.state._replace(pet=w.state.pet._replace(ptype=jnp.asarray([J.PET_SCORCHCLAW, 0], jnp.int32)))
    main, _ = J.monster_attack_packets(held, table, w.units(), launch)
    assert float(main.raw[blue]) == pytest.approx((66.0 + 40.0) * 0.5)    # junglers take 50%
    gromp = slots_of(table, "Order OwlBear")[0]
    w.now = 67.0
    w.spawn()
    launch = launch._replace(launched=jnp.zeros(w.n, bool).at[M0 + gromp].set(True),
                             target=jnp.full(w.n, -1, jnp.int32).at[M0 + gromp].set(0))
    main, bonus = J.monster_attack_packets(w.state, table, w.units(), launch)
    assert int(bonus.dtype[gromp]) == MAGIC and float(bonus.raw[gromp]) == pytest.approx(0.05 * 800.0)


# ---- rewards ------------------------------------------------------------------------------------

def test_gold_and_xp_to_the_killer_match_the_table(table):
    w = World(table, now=67.0)
    w.spawn(level=(3, 3))
    gromp = slots_of(table, "Order OwlBear")[0]
    rw = w.kill([gromp], killer=1, level=(3.0, 3.0))
    assert float(rw.gold[1]) == pytest.approx(80.0) and float(rw.xp[1]) == pytest.approx(120.0 * 1.25)
    assert float(rw.gold[0]) == 0.0 and int(rw.large_kills[1]) == 1
    # Jungle item: +80 XP per large monster, +150 for the first one, comeback +50 per level behind.
    w2 = World(table, now=55.0)
    w2.state = J.latch_pets(w2.state, _own(table, {0: 1102}))
    w2.spawn()
    red = slots_of(table, "Order Red")[0]
    rw = w2.kill([red], killer=0, level=(1.0, 4.0))
    assert float(rw.xp[0]) == pytest.approx(95.0 + 80.0 + 150.0 + 50.0 * 2)   # avg 2.5 -> 1.5 behind -> 2
    assert int(w2.state.pet.treats[0]) == 1 and float(rw.heal[0]) == 0.0      # full HP: no kill heal


def test_first_scuttlers_and_cycle(table):
    w = World(table, now=174.0)
    crabs = slots_of(table, "Baron Crab") + slots_of(table, "Dragon Crab")
    assert not np.asarray(w.spawn().mask)[crabs].any()
    w.now = 175.0
    assert np.asarray(w.spawn().mask)[crabs].all()
    assert w.a["hp"][M0 + crabs[0]] == pytest.approx(1550 * 0.65)
    out = w.ai()
    assert not bool(out.targetable[crabs[0]])                         # 1.5 s untargetable spawn
    w.now = 177.0
    out = w.ai()
    assert bool(out.targetable[crabs[0]]) and int(out.desired[crabs[0]]) == -1   # never attacks
    assert float(out.slow_resist[crabs[0]]) == 1.0 and float(out.cc_duration_mult[crabs[0]]) == 2.0
    w.now = 200.0
    rw = w.kill([crabs[0]], killer=1)
    assert float(rw.xp[1]) == pytest.approx(100.0 * 0.2) and float(rw.gold[1]) == pytest.approx(55.0)
    assert int(rw.shrine_team[0]) == 1 and float(rw.shrine_until[0]) == pytest.approx(290.0)
    assert not np.isfinite(float(w.state.crab_respawn_at))            # the other first crab lives
    w.now = 210.0
    w.kill([crabs[1]], killer=0)
    assert float(w.state.crab_respawn_at) == pytest.approx(360.0)
    w.now = 360.0
    m = np.asarray(w.spawn(level=(6, 6)).mask)
    assert m[crabs].sum() == 1                                        # one Scuttler at a time
    s = crabs[int(np.flatnonzero(m[crabs])[0])]
    assert w.a["hp"][M0 + s] == pytest.approx(1550 * 1.5)
    bs = J.buff_stats(w.state, now=210.5, level=jnp.asarray([5, 5]), max_mana=jnp.asarray([300.0, 300.0]),
                      max_hp=jnp.asarray([1000.0, 1000.0]), x=jnp.asarray([4400.0, 4400.0]),
                      y=jnp.asarray([9600.0, 9600.0]), team=jnp.asarray([0, 1]),
                      champion_combat_recent=jnp.asarray([False, False]))
    assert float(bs.shrine_ms[1]) == pytest.approx(0.30) and float(bs.shrine_ms[0]) == 0.0


# ---- crests -------------------------------------------------------------------------------------

def _ctx(w, level=(1.0, 1.0), ranged=(False, False), in_combat=(False, False), **kw):
    from lanerl_jax.modern.items.effects.core import Ctx
    c = jnp.asarray
    z = c([0.0, 0.0])
    base = dict(now=jnp.float32(w.now), dt=jnp.float32(1.0 / 30.0), unit=c([0, 1]), team=c([0, 1]),
                alive=c([True, True]), level=c(level), is_ranged=c(ranged), x=c(w.a["x"][:2]), y=c(w.a["y"][:2]),
                facing_x=z, facing_y=z, moved=z, base_ad=c([60.0, 60.0]), bonus_ad=z, ap=z, base_hp=c([1000.0, 1000.0]),
                max_hp=c([1000.0, 1000.0]), hp=c(w.a["hp"][:2]), base_armor=c([30.0, 30.0]), bonus_armor=z,
                base_mr=c([30.0, 30.0]), bonus_mr=z, mana=c([300.0, 300.0]), max_mana=c([300.0, 300.0]),
                base_ms=c([345.0, 345.0]), move_speed=c([345.0, 345.0]), crit_chance=z, crit_damage=c([2.0, 2.0]),
                life_steal=z, bonus_attack_speed=z, ability_haste=z, lethality=z, heal_shield_power=z,
                attack_windup=c([0.2, 0.2]), in_combat=c(in_combat), in_shop=c([False, False]))
    base.update(kw)
    return Ctx(**base)


def _effects(w, hit=(False, False), target=(-1, -1), **kw):
    ctx = _ctx(w, **kw.pop("ctx", {}))
    w.state, fx = J.combat_effects(w.state, w.table, w.units(), ctx, attack_hit=jnp.asarray(hit),
                                   attack_target=jnp.asarray(target, jnp.int32), **kw)
    return fx


def _valid(pk):
    v = np.asarray(pk.valid)
    return [(int(s), int(d), round(float(r), 4), int(t)) for s, d, r, t, ok in
            zip(np.asarray(pk.src), np.asarray(pk.dst), np.asarray(pk.raw), np.asarray(pk.dtype), v) if ok]


def test_red_buff_on_kill_slows_and_burns(table):
    w = World(table, now=55.0)
    w.spawn()
    red = slots_of(table, "Order Red")[0]
    rw = w.kill([red], killer=0)
    assert bool(rw.red_granted[0]) and float(w.state.red_until[0]) == pytest.approx(175.0)
    w.put_champ(1, 3100.0, 3000.0)
    w.now = 60.0
    fx = _effects(w, hit=(True, False), target=(1, -1))
    assert float(fx.cc.slow[0, 1]) == pytest.approx(0.10) and float(fx.cc.slow_duration[0, 1]) == pytest.approx(3.0)
    assert _valid(fx.packets) == [(0, 1, 5.0, TRUE)]                   # 15 total, first third on hit
    w.now = 60.5
    assert _valid(_effects(w, hit=(True, False), target=(1, -1)).packets) == []   # refresh only
    ticks = []
    for t in (61.0, 62.0, 63.0, 64.0):
        w.now = t
        ticks += _valid(_effects(w).packets)
    # Instances at +1 s and +2 s; the 60.5 refresh extends the burn to 62.5, so no tick at 63.
    assert ticks == [(0, 1, 5.0, TRUE)] * 2
    assert float(J.red_slow(11, True)) == pytest.approx(0.125)
    assert float(J.red_burn_total(10)) == pytest.approx(30.0)


def test_crest_transfers_to_the_champion_killer(table):
    w = World(table, now=55.0)
    w.spawn()
    blue = slots_of(table, "Chaos Blue")[0]
    w.kill([blue], killer=1)
    assert float(w.state.blue_until[1]) == pytest.approx(175.0)
    w.now = 100.0
    w.kill([], champion_died=jnp.asarray([False, True]), champion_killer=jnp.asarray([-1, 0]))
    assert float(w.state.blue_until[0]) == pytest.approx(220.0) and float(w.state.blue_until[1]) < 100.0
    bs = J.buff_stats(w.state, now=101.0, level=jnp.asarray([6, 6]), max_mana=jnp.asarray([400.0, 400.0]),
                      max_hp=jnp.asarray([1000.0, 1000.0]), x=jnp.zeros(2), y=jnp.zeros(2), team=jnp.asarray([0, 1]),
                      champion_combat_recent=jnp.asarray([True, True]))
    assert float(bs.ability_haste[0]) == 15.0 and float(bs.mana_per_s[0]) == pytest.approx(9.0)


# ---- Smite --------------------------------------------------------------------------------------

def _own(table, held):
    from lanerl_jax.modern.items.catalog import catalog
    cat = catalog()
    own = np.zeros((C, len(cat.ids)), np.int32)
    for c, iid in held.items():
        own[c, cat.row(iid)] = 1
    return jnp.asarray(own)


def _smite(w, target, *, slot=0, x=0.0, y=0.0, haste=0.0):
    req = CastOrder(jnp.asarray([slot, -1], jnp.int32), jnp.asarray([target, -1], jnp.int32),
                    jnp.asarray([x, 0.0]), jnp.asarray([y, 0.0]))
    w.state, out = J.smite_step(w.state, w.table, w.units(), req, jnp.asarray([[J.SMITE, 4], [4, 12]]),
                                now=jnp.float32(w.now), summoner_haste=jnp.asarray([haste, 0.0]),
                                alive=jnp.asarray([True, True]))
    return out


def test_smite_kills_a_low_monster_and_the_kill_heals(table):
    w = World(table, now=55.0)
    w.state = J.latch_pets(w.state, _own(table, {0: 1101}))
    w.spawn()
    raptor = slots_of(table, "Order Wraiths")[0]
    w.a["hp"][M0 + raptor] = 500.0
    w.put_champ(0, *np.asarray(table.slot_home)[raptor] + np.asarray([300.0, 0.0]))
    w.now = 56.0
    out = _smite(w, M0 + raptor)
    assert bool(out.cast[0]) and _valid(out.packets) == [(0, M0 + raptor, 600.0, TRUE)]
    assert int(out.packets.flags[0]) & D.PROP_NO_OMNIVAMP and int(out.packets.flags[0]) & D.PROP_NO_DAMAGE_MOD
    assert float(out.charges[0]) == 0.0
    rw = w.kill([raptor], killer=0, level=(2.0, 2.0), hp=np.asarray([400.0, 1000.0]))
    assert float(rw.heal[0]) == pytest.approx(min(70 + 20 * 2, 250) * min(1.25 * 0.6, 1.0))
    assert float(rw.mana[0]) == pytest.approx(15.0 + 4.0 * 2.0)       # full mana: base restore, no bonus
    # Small monsters cannot be smitten; nothing to cast without a charge either.
    mini = slots_of(table, "Order Wraiths")[1]
    w.now = 75.0
    assert not bool(_smite(w, M0 + mini).cast[0])


def test_smite_charges_and_recharge(table):
    w = World(table, now=10.0)
    w.spawn()
    w.now = 10.0
    blue = slots_of(table, "Order Blue")[0]
    assert not bool(_smite(w, M0 + blue).cast[0])                     # start-of-game cooldown 15 s
    w.now = 55.0
    w.spawn()
    w.put_champ(0, *np.asarray(table.slot_home)[blue])
    w.now = 47.0
    _smite(w, -1)
    assert float(w.state.smite.max_charges[0]) == 1.0
    w.now = 48.0
    out = _smite(w, -1)
    assert float(w.state.smite.max_charges[0]) == 2.0 and float(out.recharge_left[0]) == pytest.approx(90.0)
    w.now = 138.0
    assert float(_smite(w, -1).charges[0]) == 2.0
    w.now = 140.0
    assert bool(_smite(w, M0 + blue).cast[0])
    w.now = 150.0
    assert not bool(_smite(w, M0 + blue).cast[0])                     # 15 s between casts, unhasted
    w.now = 155.0
    out = _smite(w, -1, x=float(w.a["x"][M0 + blue]) + 100.0, y=float(w.a["y"][M0 + blue]))
    assert bool(out.cast[0]) and int(out.target[0]) == M0 + blue      # cursor forgiveness (125)
    assert float(out.recharge_left[0]) == pytest.approx(140.0 + 90.0 - 155.0)


# ---- pets ---------------------------------------------------------------------------------------

def test_pet_evolution_changes_smite(table):
    w = World(table, now=55.0)
    w.state = J.latch_pets(w.state, _own(table, {0: 1103}))
    w.spawn()
    blue, red = slots_of(table, "Order Blue")[0], slots_of(table, "Order Red")[0]
    w.state = w.state._replace(pet=w.state.pet._replace(treats=jnp.asarray([14, 0], jnp.int32)))
    w.kill([red])
    assert int(J.pet_stage(w.state)[0]) == 1
    w.put_champ(0, *np.asarray(table.slot_home)[blue])
    w.now = 60.0
    out = _smite(w, M0 + blue)
    assert _valid(out.packets) == [(0, M0 + blue, 1000.0, TRUE)]       # Unleashed Smite
    # Champion Smite after the first evolution: 40 true + 20% slow 2 s.
    w.put_champ(1, float(w.a["x"][0]) + 200.0, float(w.a["y"][0]))
    w.now = 200.0
    out = _smite(w, 1)
    assert _valid(out.packets) == [(0, 1, 40.0, TRUE)]
    assert float(out.cc.slow[0, 1]) == pytest.approx(0.2) and float(out.cc.slow_duration[0, 1]) == 2.0
    # Final evolution: egg consumed, quest done, Primal Smite hits the camp around the target.
    w.state = w.state._replace(pet=w.state.pet._replace(treats=jnp.asarray([34, 0], jnp.int32)))
    w.now = 255.0
    w.spawn()
    wolves = slots_of(table, "Order Wolves")
    w.now = 256.0
    rw = w.kill([slots_of(table, "Order Wraiths")[0]])
    assert bool(rw.consume_pet[0]) and bool(rw.quest_completed[0]) and int(J.pet_stage(w.state)[0]) == 2
    w.put_champ(0, *np.asarray(table.slot_home)[wolves[0]])
    w.now = 300.0
    out = _smite(w, M0 + wolves[0])
    hits = sorted(_valid(out.packets))
    assert (0, M0 + wolves[0], 1400.0, TRUE) in hits and len(hits) == 3   # Primal: small wolves too
    # Quest rewards: +10 g / +10 XP per large monster afterwards; Mosstomper shield granted.
    fx = _effects(w, ctx=dict(level=(9.0, 1.0)))
    assert float(fx.shield[0]) == pytest.approx(200.0)
    rw = w.kill([wolves[0]])
    assert float(rw.gold[0]) == pytest.approx(55.0 + 10.0)


def test_pet_attacks_monsters_attacking_the_holder(table):
    w = World(table, now=55.0)
    w.state = J.latch_pets(w.state, _own(table, {0: 1101}))
    w.spawn()
    wolves = slots_of(table, "Order Wolves")
    w.put_champ(0, 3850.0, 6400.0)
    w.now = 56.0
    w.ai(dmg=w.hit(0, wolves[0]))
    fx = _effects(w, ctx=dict(level=(1.0, 1.0)))
    hits = _valid(fx.packets)
    assert sorted(d for _, d, _, _ in hits) == sorted(M0 + s for s in wolves)
    assert all(r == pytest.approx(20.0) and t == TRUE for _, _, r, t in hits)
    assert float(fx.heal[0]) == pytest.approx(6.0)
    w.now = 56.5
    assert _valid(_effects(w).packets) == []                          # once per second


def test_item_module_amp_and_coverage(table):
    from lanerl_jax.modern.items import effects as E
    from lanerl_jax.modern.items.effects import jungle as JI
    from lanerl_jax.modern.items.effects.core import Units
    rep = E.coverage_report()
    assert all(rep[i].startswith("jungle:") for i in (1101, 1102, 1103))
    w = World(table)
    n = w.n
    cls = jnp.asarray([0, 0] + [3] * (n - 2), jnp.int32)
    units = Units(jnp.zeros(n), jnp.zeros(n), jnp.zeros(n, jnp.int32), cls, jnp.ones(n, bool), jnp.ones(n), jnp.ones(n),
                  jnp.ones(n), jnp.ones(n, bool), jnp.zeros(n, bool), jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    st = JI.init(C, n)
    own = _own(table, {0: 1102})
    st, _ = JI.periodic(st, own, _ctx(w), units)
    pk = D.packets(jnp.asarray([True, True, True]), jnp.asarray([0, 0, 1]), jnp.asarray([5, 5, 5]),
                   jnp.asarray([100.0, 100.0, 100.0]), jnp.asarray([PHYSICAL, TRUE, PHYSICAL]))
    amp = np.asarray(JI.packet_amp(st, jnp.zeros_like(own), _ctx(w), units, pk))   # latched after consumption
    np.testing.assert_allclose(amp, [0.10, 0.0, 0.0])


def test_minion_penalties_for_holders(table):
    w = World(table)
    w.state = J.latch_pets(w.state, _own(table, {0: 1101}))
    g, x = J.minion_reward_mods(w.state, now=0.0, champion_level=jnp.asarray([1.0, 1.0]), avg_level=1.0)
    assert float(x[0]) == pytest.approx(0.30) and float(x[1]) == 1.0 and float(g[0]) == 0.0
    st = w.state._replace(pet=w.state.pet._replace(minion_gold=jnp.asarray([100.0, 0.0]),
                                                   monster_gold=jnp.asarray([200.0, 0.0])))
    g, x = J.minion_reward_mods(st, now=600.0, champion_level=jnp.asarray([5.0, 5.0]), avg_level=5.0)
    assert float(g[0]) == -13.0 and float(x[0]) == pytest.approx(0.5 * (1 - 0.7 * 0.5))


def test_gustwalker_brush_entry_is_readable_from_state(table):
    """30% on brush entry (latched by ``combat_effects``), decaying to 0 over 1.5 s."""
    w = World(table, now=100.0)
    w.state = J.latch_pets(w.state, _own(table, {0: 1102}))
    w.state = w.state._replace(pet=w.state.pet._replace(treats=jnp.asarray([35, 0], jnp.int32)))
    assert float(J.gust_bonus_ms(w.state, 100.0)[0]) == 0.0
    fx = _effects(w, in_brush=jnp.asarray([True, False]))
    assert float(fx.bonus_ms[0]) == pytest.approx(0.30)
    got = np.asarray(J.gust_bonus_ms(w.state, 100.0 + J.GUST_DECAY_S / 2))
    assert got[0] == pytest.approx(0.15) and got[1] == 0.0
    assert float(J.gust_bonus_ms(w.state, 100.0 + J.GUST_DECAY_S)[0]) == 0.0
