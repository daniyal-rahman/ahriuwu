"""Scripted players that act through the SAME interface the trained policy uses.

Why this exists
---------------
Most from-scratch runs never learn to farm (`RL-007`). Before blaming
learning, rule out the interface: if a rule-based player that sees ONLY the
`Observation` tensors and emits ONLY the factored 4-tuple
``(button, screen_x, screen_y, target_slot)`` cannot farm through
``orders_from``, no amount of training will.

Contract
--------
Every function here is pure JAX, per agent (unbatched: ``obs.entities`` is
``(32, 16)``), jit/vmap-able, and reads nothing but the observation:
``entities``, ``entity_pad_mask`` and ``self_vec``/``global_vec``. It never
reads ``obs.slot_unit`` (that is the DECODER's table, not a policy input) and
never reads ``LaneState``.

Static game knowledge it is allowed (a player knows these; the observation
deliberately does not carry them -- see ``obs/builder.py``'s "max health is
deliberately not fed"):

* minion max HP and armour by the subtype one-hot (melee/caster/cannon), from
  the profile table (`sim/profiles.py`, ``lane_params``). Blue and red rows are
  identical for max HP and armour (455/290/700 HP, 0/0/15 armour), so the
  policy does not need to know its side. Minion max HP does not grow within a
  10-minute episode in this sim (``SimConfig.minion_hp`` is None), so no clock
  term is needed.
* Garen's attack range (125) and the minion collision radius (40): the swing
  gate is ``attack_range + target.collision_radius`` centre-to-centre
  (`sim/autoattack.ideal_attack_range`), i.e. **165**, not the 175 the brief
  quoted. The obs positions are centres, so 165 is the number to compare with.
* a killable minion out of reach overrides the positioning target: the
  player walks to within reach of it (added after the first 600 s runs showed
  most uncredited nearby minion deaths happened out of reach, not in it).
* the top-lane polyline (`sim/init.TOP_LANE_PATH`) expressed in the canonical
  (blue) lane frame, to walk down the lane when no minion is visible. Both
  sides use this one canonical path: the frame reflection makes red's lane
  approximately the same curve, which is good enough for walking.

What the observation does NOT carry, and the consequence
--------------------------------------------------------
* **Attack readiness.** No self field reports the auto-attack cooldown or
  wind-up (the four attack-cycle fields were removed on 2026-09-12). The
  scripted player therefore cannot tell whether a swing is ready; it issues
  the attack and the engine swings when the cooldown allows. Failed attempts
  of that kind are classified by the harness, from the state, as
  "swing not ready".
* **Exact HP.** ``hp_frac`` is quantised to 1/60. The last-hit test uses the
  UPPER bound of the bar bucket, ``(hp_frac + 1/120) * max_hp``, so a "killable"
  call is never wrong because of quantisation (it can only be late).

Damage estimate: ``AD * 100 / (100 + armour)`` with AD = ``self_vec[S_AD] *
NORM_AD`` (the builder's AD is the exact level-scaled value the sim swings
with -- checked by ``tests/test_action_interface.py``).

Screen click selection
----------------------
The decoder maps a cell to a champion-centred offset ``(ds, dn)`` in the
agent's OWN lane frame (red mirrored), via ``actions._screen_to_centred_lane``.
The same function is evaluated once over the 96x54 grid here, and a desired
offset is turned into the grid cell whose decoded offset is nearest to it
(minimap cells excluded, desired offsets longer than ``CLICK_RADIUS``
shortened along their direction first so the pick stays on-screen). That is
"search the grid through the decoder's own projection"; test (b) in
``tests/test_action_interface.py`` checks the round trip in both frames.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_rl import constants as C

from ..obs.builder import NORM_AD, NORM_DIST, NORM_XY, HP_BAR_STEPS
from .actions import MINIMAP_X_MIN, MINIMAP_Y_MIN, _screen_to_centred_lane

__all__ = ["scripted_act", "scripted_act_any", "noop_act", "PLAYERS",
           "screen_grid", "cell_for_offset", "static_tables",
           "CHAMP_ATTACK_RANGE", "MINION_COLLISION_RADIUS", "STAND_BEHIND",
           "TURRET_KEEP_OUT"]

#: centre-to-centre swing reach vs a lane minion: 125 + 40 (see module doc)
CHAMP_ATTACK_RANGE = 125.0
MINION_COLLISION_RADIUS = 40.0
#: stay this far inside the reach when deciding "in range", so a minion that
#: steps during the two ticks before the order applies does not escape
RANGE_MARGIN = 10.0
#: stand this far behind (own side of) the front enemy minion
STAND_BEHIND = 140.0
#: never pick a stand point closer than this to a visible enemy turret
#: (turret range 775 + champion radius, rounded up)
TURRET_KEEP_OUT = 900.0
#: longest offset clicked; the screen reaches only ~780 units toward the
#: bottom edge, so anything longer is shortened along its direction
CLICK_RADIUS = 700.0

_AM = C.BUTTON_INDEX["attack_move"]
_MOVE = C.BUTTON_INDEX["move"]
_NOOP = C.BUTTON_INDEX["noop"]

_ENEMY_MINION = (13, 25)       # builder.SLOT_ENEMY_MINION
_TURRET = (25, 27)             # builder.SLOT_TURRET


def _eager(fn):
    """Cache a host-side constant table; build it eagerly even if the first
    call happens inside a jit trace."""
    @functools.lru_cache(maxsize=1)
    def wrapped():
        with jax.ensure_compile_time_eval():
            return fn()
    wrapped.__doc__ = fn.__doc__
    return wrapped


@_eager
def static_tables():
    """Per-subtype (melee, caster, cannon) max HP and armour, and the champion
    reach, read from the SAME profile table the sim steps with."""
    from ..sim.init import lane_params
    from ..sim.profiles import profile_id
    from ..sim.state import Kind, Team
    from ..sim.targeting import MinionType

    p = lane_params()
    rows = [profile_id(Kind.LANE_MINION, t, Team.BLUE)
            for t in (MinionType.MELEE, MinionType.CASTER, MinionType.CANNON)]
    rows_red = [profile_id(Kind.LANE_MINION, t, Team.RED)
                for t in (MinionType.MELEE, MinionType.CASTER, MinionType.CANNON)]
    max_hp = np.asarray(p["max_hp"])[rows]
    armor = np.asarray(p["armor"])[rows]
    radius = np.asarray(p["collision_radius"])[rows]
    # the policy is side-blind, so the tables it uses must not depend on side
    assert np.allclose(max_hp, np.asarray(p["max_hp"])[rows_red])
    assert np.allclose(armor, np.asarray(p["armor"])[rows_red])
    assert np.allclose(radius, MINION_COLLISION_RADIUS)
    champ = profile_id(Kind.CHAMPION, -1, Team.BLUE)
    assert float(p["attack_range"][champ]) == CHAMP_ATTACK_RANGE
    return {"max_hp": max_hp.astype(np.float32),
            "armor": armor.astype(np.float32)}


@_eager
def screen_grid():
    """``(ds, dn, ok)`` each ``(N_SCREEN_Y, N_SCREEN_X)``: the champion-centred
    own-frame offset the decoder sends for every cell, and whether a MOVE on
    that cell is executed (minimap cells decode to NOOP)."""
    sx = (np.arange(C.N_SCREEN_X, dtype=np.float32) + 0.5) / C.N_SCREEN_X
    sy = (np.arange(C.N_SCREEN_Y, dtype=np.float32) + 0.5) / C.N_SCREEN_Y
    SX, SY = np.meshgrid(sx, sy)
    ds, dn = _screen_to_centred_lane(jnp.asarray(SX), jnp.asarray(SY))
    ok = ~((SX >= MINIMAP_X_MIN) & (SY >= MINIMAP_Y_MIN))
    return np.asarray(ds), np.asarray(dn), ok


def cell_for_offset(ds, dn):
    """Nearest executable grid cell to the own-frame offset ``(ds, dn)``."""
    gds, gdn, ok = (jnp.asarray(a) for a in screen_grid())
    r = jnp.sqrt(ds * ds + dn * dn)
    k = jnp.where(r > CLICK_RADIUS, CLICK_RADIUS / jnp.maximum(r, 1e-6), 1.0)
    ds, dn = ds * k, dn * k
    err = (gds - ds) ** 2 + (gdn - dn) ** 2
    err = jnp.where(ok, err, jnp.inf)
    flat = jnp.argmin(err.reshape(-1))
    return (flat % C.N_SCREEN_X).astype(jnp.int32), (flat // C.N_SCREEN_X).astype(jnp.int32)


@_eager
def _canonical_lane_path():
    """``TOP_LANE_PATH`` in the blue lane frame, ``(W, 2)`` of ``(s, n)``."""
    from ..obs.frame import make_lane_frame, to_lane
    from ..sim.init import TOP_LANE_PATH, TOP_OUTER_TURRET
    from ..sim.state import Team
    from .trainer import BLUE_NEXUS
    f = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE], TOP_OUTER_TURRET[Team.RED],
                        BLUE_NEXUS)
    p = np.asarray(TOP_LANE_PATH, np.float32)
    s, n = to_lane(f, jnp.asarray(p[:, 0]), jnp.asarray(p[:, 1]))
    return np.stack([np.asarray(s), np.asarray(n)], -1), float(f.length)


def _minion_view(obs):
    """Per enemy-minion slot: valid, ds, dn, est. HP upper bound, max HP, armour."""
    tab = static_tables()
    e = obs.entities[_ENEMY_MINION[0]:_ENEMY_MINION[1]]
    valid = ((e[:, C.E_VALID] > 0.5) & (e[:, C.E_TYPE_ONEHOT][:, 1] > 0.5)
             & (e[:, C.E_TEAM_ONEHOT][:, 1] > 0.5))
    sub = e[:, C.E_MINION_SUBTYPE]
    known = sub.sum(-1) > 0.5
    max_hp = sub @ jnp.asarray(tab["max_hp"])
    armor = sub @ jnp.asarray(tab["armor"])
    hp_hi = (e[:, C.E_HP_FRAC] + 0.5 / HP_BAR_STEPS) * max_hp
    ds = e[:, C.E_DS] * NORM_DIST
    dn = e[:, C.E_DN] * NORM_DIST
    return valid & known, ds, dn, hp_hi, max_hp, armor


def _move_target(obs):
    """Own-frame offset to walk to: behind the wave front, or down the lane."""
    valid, ds, dn, *_ = _minion_view(obs)
    my_s = obs.self_vec[C.S_LANE_S] * NORM_XY
    my_n = obs.self_vec[C.S_LANE_N] * NORM_XY

    # wave front = the enemy minion nearest OUR side (smallest ds)
    front = jnp.argmin(jnp.where(valid, ds, jnp.inf))
    stand_ds = ds[front] - STAND_BEHIND
    stand_dn = dn[front]

    # no minion visible: next lane vertex ahead of me, capped short of the
    # enemy outer turret (s = L in the canonical frame)
    path, length = _canonical_lane_path()
    path = jnp.asarray(path)
    s_cap = length - TURRET_KEEP_OUT
    ahead = path[:, 0] > my_s + 200.0
    nxt = jnp.argmax(ahead)                 # first vertex ahead
    has_ahead = jnp.any(ahead)
    ws = jnp.where(has_ahead, path[nxt, 0], s_cap)
    wn = jnp.where(has_ahead, path[nxt, 1], 0.0)
    # clamp to the cap along the segment toward that vertex
    capped = ws > s_cap
    ws = jnp.where(capped, s_cap, ws)
    wn = jnp.where(capped, 0.0, wn)
    walk_ds, walk_dn = ws - my_s, wn - my_n

    any_minion = jnp.any(valid)
    tds = jnp.where(any_minion, stand_ds, walk_ds)
    tdn = jnp.where(any_minion, stand_dn, walk_dn)

    # keep the stand point out of visible enemy turret range
    t = obs.entities[_TURRET[0]:_TURRET[1]]
    t_enemy = (t[:, C.E_VALID] > 0.5) & (t[:, C.E_TEAM_ONEHOT][:, 1] > 0.5)
    t_ds = t[:, C.E_DS] * NORM_DIST
    t_dn = t[:, C.E_DN] * NORM_DIST
    for i in range(_TURRET[1] - _TURRET[0]):
        off_n = tdn - t_dn[i]
        d = jnp.sqrt((tds - t_ds[i]) ** 2 + off_n ** 2)
        lim = t_ds[i] - jnp.sqrt(jnp.maximum(TURRET_KEEP_OUT ** 2 - off_n ** 2, 0.0))
        tds = jnp.where(t_enemy[i] & (d < TURRET_KEEP_OUT), jnp.minimum(tds, lim), tds)
    return tds, tdn


def _act(obs, key, *, require_lasthit: bool):
    del key                                      # deterministic
    valid, ds, dn, hp_hi, max_hp, armor = _minion_view(obs)
    ad = obs.self_vec[C.S_AD] * NORM_AD
    dmg = ad * 100.0 / (100.0 + armor)
    dist = jnp.sqrt(ds * ds + dn * dn)
    in_range = valid & (dist <= CHAMP_ATTACK_RANGE + MINION_COLLISION_RADIUS
                        - RANGE_MARGIN)
    if require_lasthit:
        cand = in_range & (hp_hi <= dmg)
        # among killable, the one closest to dying relative to its bar
        score = jnp.where(cand, hp_hi, jnp.inf)
    else:
        cand = in_range
        score = jnp.where(cand, hp_hi, jnp.inf)
    pick = jnp.argmin(score)
    attack = jnp.any(cand)

    tds, tdn = _move_target(obs)
    # A minion that one auto would kill but that is out of reach: walk to it
    # (a point `CHAMP_ATTACK_RANGE` short of its centre) instead of the wave
    # front. Same threshold for both players; the brawler has no threshold
    # for attacking, but walking to a kill is still the useful move.
    killable_far = valid & ~in_range & (hp_hi <= dmg)
    far = jnp.argmin(jnp.where(killable_far, dist, jnp.inf))
    k = jnp.maximum(dist[far] - CHAMP_ATTACK_RANGE, 0.0) / jnp.maximum(dist[far], 1e-6)
    go = jnp.any(killable_far)
    tds = jnp.where(go, ds[far] * k, tds)
    tdn = jnp.where(go, dn[far] * k, tdn)
    sx, sy = cell_for_offset(tds, tdn)
    dead = obs.self_vec[C.S_IS_DEAD] > 0.5
    button = jnp.where(attack, _AM, _MOVE)
    button = jnp.where(dead, _NOOP, button).astype(jnp.int32)
    slot = jnp.where(attack, _ENEMY_MINION[0] + pick, 0).astype(jnp.int32)
    return button, sx, sy, slot


def scripted_act(obs, key):
    """Pure last-hitter: attack only a minion one auto kills, else position."""
    return _act(obs, key, require_lasthit=True)


def scripted_act_any(obs, key):
    """Brawler: attack the lowest-HP enemy minion in range, killable or not."""
    return _act(obs, key, require_lasthit=False)


def noop_act(obs, key):
    del key
    z = jnp.int32(0)
    return jnp.int32(_NOOP), z, z, z


PLAYERS = {"lasthit": scripted_act, "brawler": scripted_act_any,
           "noop": noop_act}
