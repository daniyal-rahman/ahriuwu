"""Generic modern mechanics: attack machine, missiles, CC timers, blink."""
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import mechanics as M
from lanerl_jax.modern.core import types as W


def units(n=3, *, x=(0.0, 100.0, 1000.0), aspd=(1.0, 1.0, 1.0), rng=(125.0, 125.0, 125.0)):
    f = lambda v: jnp.asarray(v, jnp.float32)
    return W.WorldUnits(kind=jnp.asarray([1, 1, 2]), sub=jnp.zeros(n, jnp.int32), team=jnp.asarray([0, 1, 1]),
                        alive=jnp.ones(n, bool), targetable=jnp.ones(n, bool), x=f(x), y=f([0.0] * n),
                        radius=f([65.0] * n), hp=f([500.0] * n), max_hp=f([500.0] * n), armor=f([0.0] * n),
                        magic_resist=f([0.0] * n), attack_damage=f([60.0] * n), attack_range=f(rng),
                        attack_speed=f(aspd), move_speed=f([340.0] * n), spawn_seq=jnp.arange(n, dtype=jnp.int32),
                        spawn_time=f([0.0] * n))


def test_attack_cycle_windup_launch_period():
    u = units()
    att = W.init_attack_state(3)
    desired = jnp.asarray([1, -1, -1])
    launches = []
    for k in range(70):                                  # 30 Hz, 1 attack/s, 0.3 s windup
        att, launched = M.attack_step(att, u, desired, can_attack=jnp.ones(3, bool),
                                      windup=jnp.full((3,), 0.3), dt=1 / 30)
        launches.append(bool(launched[0]))
    hits = [i for i, v in enumerate(launches) if v]
    assert hits[:2] == [8, 38]                            # launch at ~0.3 s, then every 1.0 s (30 ticks)


def test_attack_cancel_resets_timer_and_range_is_edge_to_edge():
    u = units(x=(0.0, 125.0 + 65.0 + 65.0, 1000.0))     # exactly at edge-to-edge range
    att = W.init_attack_state(3)
    att, _ = M.attack_step(att, u, jnp.asarray([1, -1, -1]), can_attack=jnp.ones(3, bool),
                           windup=jnp.full((3,), 0.3), dt=1 / 30)
    assert float(att.windup_left[0]) > 0                  # started: in range edge-to-edge
    att, _ = M.attack_step(att, u, jnp.asarray([2, -1, -1]), can_attack=jnp.ones(3, bool),
                           windup=jnp.full((3,), 0.3), dt=1 / 30)
    assert float(att.windup_left[0]) == 0 and float(att.cooldown_left[0]) == 0   # retarget cancels, timer reset


def test_last_windup_tick_ignores_a_new_order_but_not_range_loss():
    """Grace tick: a move order on the final windup tick still launches; losing range does not."""
    u = units()
    ones = jnp.ones(3, bool)
    att = W.init_attack_state(3)
    for _ in range(8):                                    # 0.3 s windup: launches on tick 9 (index 8)
        att, launched = M.attack_step(att, u, jnp.asarray([1, -1, -1]), can_attack=ones,
                                      windup=jnp.full((3,), 0.3), dt=1 / 30)
        assert not bool(launched[0])
    moved, launched = M.attack_step(att, u, jnp.asarray([-1, -1, -1]), can_attack=ones,
                                    windup=jnp.full((3,), 0.3), dt=1 / 30)
    assert bool(launched[0])                              # the move order arrived in the grace tick
    far = units(x=(0.0, 900.0, 1000.0))
    _, launched = M.attack_step(att, far, jnp.asarray([1, -1, -1]), can_attack=ones,
                                windup=jnp.full((3,), 0.3), dt=1 / 30)
    assert not bool(launched[0])                          # out of range: cancelled
    early = W.init_attack_state(3)
    early, _ = M.attack_step(early, u, jnp.asarray([1, -1, -1]), can_attack=ones, windup=jnp.full((3,), 0.3),
                             dt=1 / 30)
    early, _ = M.attack_step(early, u, jnp.asarray([-1, -1, -1]), can_attack=ones, windup=jnp.full((3,), 0.3),
                             dt=1 / 30)
    assert float(early.windup_left[0]) == 0               # earlier in the windup a move cancels


def test_missile_homes_and_hits():
    u = units()
    ms = M.init_missiles(4)
    launch = jnp.asarray([True, False, False])
    ms, over = M.spawn_missiles(ms, launch, u, jnp.asarray([2, -1, -1]), jnp.full(3, 50.0), jnp.zeros(3, jnp.int32),
                                jnp.zeros(3, jnp.int32), jnp.full(3, 2000.0), jnp.ones(3, jnp.int32), jnp.zeros(3, bool))
    assert int(over) == 0 and int(ms.alive.sum()) == 1
    hit_at = None
    for k in range(30):
        ms, arrive = M.advance_missiles(ms, u, 1 / 30)
        if bool(arrive.any()):
            hit_at = k
            break
    assert hit_at == int(np.ceil((1000 - 65) / (2000 / 30))) - 1


def test_cc_tenacity_floor_and_strongest_slow():
    cc = M.init_cc(3)
    out = W.no_cc(1, 3)._replace(stun=jnp.zeros((1, 3)).at[0, 1].set(1.5),
                                 slow=jnp.zeros((1, 3)).at[0, 1].set(0.3),
                                 slow_duration=jnp.zeros((1, 3)).at[0, 1].set(2.0))
    ten = jnp.asarray([0.0, 1 - 0.7 * 0.8 * 0.85, 0.0])
    cc = M.apply_cc(cc, out, ten, jnp.zeros(3), 10.0, source_is_champion=jnp.asarray([True]))
    assert float(cc.stun_until[1]) == pytest.approx(10.0 + 0.714, abs=1e-3)       # DAMAGE F17
    caps = M.capabilities(cc, 10.5)
    assert not bool(caps["can_move"][1]) and bool(caps["can_move"][0])
    weaker = W.no_cc(1, 3)._replace(slow=jnp.zeros((1, 3)).at[0, 1].set(0.1),
                                    slow_duration=jnp.zeros((1, 3)).at[0, 1].set(5.0))
    cc2 = M.apply_cc(cc, weaker, ten, jnp.zeros(3), 10.1, source_is_champion=jnp.asarray([True]))
    assert float(cc2.slow[1]) == pytest.approx(0.3)                                  # strongest slow kept
