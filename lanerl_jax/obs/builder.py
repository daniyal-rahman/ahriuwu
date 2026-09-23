"""The observation, in JAX.

Why this is on the critical path and not an afterthought
--------------------------------------------------------
In the production stack ``ObservationBuilder.build`` is **~55% of a decision**
and pure Python, so it holds the GIL for all of it -- which is why actors had to
become processes at all (``lanerl_train/procactor.py``: 4 CPU-bound Python
threads measure 1.00 cores, the same as 1, so 14 of 16 cores were unreachable).
Porting the simulator without porting this would move the bottleneck, not remove
it.

It also has a **better oracle than the simulator does**: the existing Python
builder is a pure function that can be diffed offline against this one on
recorded frames, in the same language, with no server in the loop.

The layout is `lanerl_rl.constants`, and it is smaller than it looks
--------------------------------------------------------------------
Rebuilt on 2026-09-12 from 64 self fields to 16 and 48 global fields to 6. The
removed ones are worth knowing because they are the shape of mistakes to not
re-make: 18 level one-hots beside a level scalar, gold in two encodings, region
one-hots that are thresholds of ``(s, n)``, velocity and hp-delta fields that a
GRU exists to compute, four attack-cycle fields that were identically zero
across all 99,654 BC rows, and 8 enemy-ability-cooldown estimates built on a
constant whose documented safety direction was backwards.

    entities (32, 16)  valid, ds, dn, hp_frac, 6 type, 3 team, 3 minion subtype
    self     (16,)     lane_s, lane_n, hp_frac, level, gold, cs, 4 cooldowns,
                       ad, ap, armor, mr, is_dead, recalling
    global   (6,)      clock, enemy_visible, 4x time-since-observed-cast

Three rules this port has to honour
-----------------------------------
**Only visible units are slotted.** ``valid = 0`` for an empty slot, and a
fogged entity is simply absent -- the older design kept it with a stale position
and the comment records why that was dangerous: writing ``ds = dn = 0,
valid = 1`` for a fogged enemy tells the policy *"the enemy is standing on top
of me"*, the single worst hallucination available.

**No slot-index feature.** Slots are grouped by type and ordered within a block,
but nothing encodes *which* index a row landed in, so the attention encoder
stays permutation-equivariant within a block.

**Minion subtype is categorical, and max health is deliberately not fed.** Melee
/ caster / cannon have max health 455 / 290 / 700, so the ``hp_frac`` at which
one auto-attack kills them is 0.172 / 0.263 / 0.112 -- a 2.3x spread the agent
was previously asked to learn as one threshold. The one-hot says *what the thing
is*; working out what that implies about damage is the network's job.

``LAST_HIT_SORT_K`` is 0 and staying there
------------------------------------------
Enemy minion slots are ordered by **distance**, not re-sorted by ascending HP.
The HP sort was removed deliberately: it does not add information (``hp_frac``
is already per-slot) and only removes the *comparison* from the network's job,
so the target head learns "click slot 13" instead of "find the minion about to
die". This repo has been burned by exactly that shape once already -- the
movement head was blind for weeks because a free crutch meant the signal was
never learned.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..sim.combat import growth_sum, stat_total
from ..sim.spells import (
    E_COOLDOWNS,
    Q_COOLDOWN,
    R_COOLDOWNS,
    Slot,
    W_COOLDOWNS,
    status_of,
    w_passive_modifiers,
)
from ..sim.state import Kind, LaneState, Team
from .fog import visible_to
from .frame import LaneFrame, delta_to_lane, to_lane

__all__ = [
    "N_SLOTS", "ENTITY_DIM", "SELF_DIM", "GLOBAL_DIM",
    "SLOT_ENEMY_CHAMP", "SLOT_ALLY_MINION", "SLOT_ENEMY_MINION",
    "SLOT_TURRET", "SLOT_SPARE",
    "NORM_DIST", "NORM_XY", "NORM_GOLD", "NORM_CS", "NORM_AD", "HP_BAR_STEPS",
    "Observation", "build_observation",
]

N_SLOTS = 32
ENTITY_DIM = 16
SELF_DIM = 16
GLOBAL_DIM = 6

SLOT_ENEMY_CHAMP = (0, 1)
SLOT_ALLY_MINION = (1, 13)
SLOT_ENEMY_MINION = (13, 25)
SLOT_TURRET = (25, 27)
SLOT_SPARE = (27, 32)

NORM_DIST = 3000.0
NORM_XY = 3000.0
NORM_GOLD = 3000.0
NORM_CS = 200.0
NORM_AD = 200.0
#: health bars have finite resolution, and a player reads a bar, not a float
HP_BAR_STEPS = 60.0

#: entity type one-hot order, from ``constants.ENTITY_TYPES``
_TYPE_CHAMPION, _TYPE_MINION, _TYPE_TURRET = 0, 1, 2
_N_TYPES = 6
_N_TEAMS = 3


class Observation(NamedTuple):
    entities: jax.Array        # (32, 16)
    entity_pad_mask: jax.Array  # (32,) True where the slot must be masked out
    self_vec: jax.Array        # (16,)
    global_vec: jax.Array      # (6,)
    slot_unit: jax.Array       # (32,) unit index per slot, -1 if empty


def _topk_slots(score: jax.Array, eligible: jax.Array, k: int):
    """Indices of the ``k`` smallest ``score`` among ``eligible``; -1 to pad.

    ``top_k`` on the negated score, which ties toward the lowest index -- the
    same tie-break the server's iteration order gives and the same one
    ``argmin`` gives elsewhere in this codebase.
    """
    big = jnp.asarray(jnp.inf, score.dtype)
    masked = jnp.where(eligible, score, big)
    neg, idx = jax.lax.top_k(-masked, k)
    return jnp.where(jnp.isfinite(-neg), idx, -1).astype(jnp.int32)


def build_observation(state: LaneState, me: int, frame: LaneFrame, *, params,
                      horizon_s: float = 600.0) -> Observation:
    """Build one agent's observation. ``me`` is the champion's unit index.

    ``params`` is the profile table used by ``step_decision``.  It is explicit
    because live combat stats are profile- and level-dependent; using a second
    hidden stat table here would let policy inputs drift from the simulator.
    """
    n = state.kind.shape[0]
    my_team = state.team[me]
    vis = visible_to(my_team, state.x, state.y, state.kind, state.team,
                     state.alive)

    dx = state.x - state.x[me]
    dy = state.y - state.y[me]
    d2 = dx * dx + dy * dy
    not_me = jnp.arange(n) != me

    is_champ = state.kind == Kind.CHAMPION
    is_minion = state.kind == Kind.LANE_MINION
    is_turret = state.kind == Kind.TURRET
    ally = state.team == my_team
    enemy = (state.team != my_team) & (state.kind != Kind.NONE)

    base = vis & not_me & state.alive
    enemy_champ = _topk_slots(d2, base & is_champ & enemy, 1)
    ally_minion = _topk_slots(d2, base & is_minion & ally, 12)
    enemy_minion = _topk_slots(d2, base & is_minion & enemy, 12)
    turret = _topk_slots(d2, base & is_turret, 2)

    # The spare block holds the nearest visible units NOT already slotted, so
    # anything the typed blocks overflowed is still visible to the policy
    # instead of vanishing.
    taken = jnp.zeros((n,), bool)
    for sel in (enemy_champ, ally_minion, enemy_minion, turret):
        hit = jnp.arange(n)[:, None] == jnp.where(sel >= 0, sel, -1)[None, :]
        taken = taken | jnp.any(hit, axis=1)
    spare = _topk_slots(d2, base & ~taken, 5)

    slot_unit = jnp.concatenate([enemy_champ, ally_minion, enemy_minion,
                                 turret, spare])
    valid = slot_unit >= 0
    u = jnp.clip(slot_unit, 0, n - 1)

    ds, dn = delta_to_lane(frame, dx[u], dy[u])
    hp_frac = jnp.where(state.max_hp[u] > 0, state.hp[u] / state.max_hp[u], 0.0)
    # quantised to health-bar resolution: a player reads a bar, not a float
    hp_frac = jnp.round(hp_frac * HP_BAR_STEPS) / HP_BAR_STEPS

    k = state.kind[u]
    type_1h = jnp.stack(
        [k == Kind.CHAMPION, k == Kind.LANE_MINION, k == Kind.TURRET]
        + [jnp.zeros_like(k, bool)] * (_N_TYPES - 3), axis=-1).astype(jnp.float32)
    team_1h = jnp.stack(
        [state.team[u] == my_team, state.team[u] != my_team,
         state.team[u] == Team.NEUTRAL], axis=-1).astype(jnp.float32)

    from ..sim.profiles import PROFILES
    mt_table = jnp.asarray([p[1] for p in PROFILES], jnp.int8)
    mt = mt_table[state.model[u]]
    # MELEE(0) -> 0, SUPER(3) -> 1, CANNON(2) -> 2, CASTER(1) -> ... see
    # constants.MINION_TYPE_INDEX {0:0, 3:1, 2:2}; CASTER is index 1 there via
    # the ENTITY_FIELD_NAMES order (melee, caster, cannon).
    sub_1h = jnp.stack([mt == 0, mt == 1, mt == 2], axis=-1).astype(jnp.float32)
    sub_1h = jnp.where(is_minion[u][:, None], sub_1h, 0.0)

    entities = jnp.concatenate([
        valid[:, None].astype(jnp.float32),
        (ds / NORM_DIST)[:, None], (dn / NORM_DIST)[:, None],
        hp_frac[:, None], type_1h, team_1h, sub_1h], axis=-1)
    entities = jnp.where(valid[:, None], entities, 0.0)

    # ---- self ------------------------------------------------------------
    s_, n_ = to_lane(frame, state.x[me], state.y[me])
    my_hp = jnp.where(state.max_hp[me] > 0, state.hp[me] / state.max_hp[me], 0.0)
    # The stat table is the simulator's source of truth. AP has no source in
    # this no-items state and remains explicitly 0.
    p = lambda key: params[key][state.model[me]]  # noqa: E731
    growth = growth_sum(state.level[me], jnp)
    ad = p("attack_damage") + p("ad_per_level") * growth
    armor_base = p("armor") + p("armor_per_level") * growth
    mr_base = p("magic_resist") + p("mr_per_level") * growth
    # The same `GarenWPassive` modifiers the sim mitigates with (`step.py`).
    wp = w_passive_modifiers(state.buffs, state.alive, state.x.dtype)
    armor = stat_total(
        armor_base - p("armor_flat_bonus"),
        flat_bonus=p("armor_flat_bonus"),
        percent_base_bonus=wp.armor_percent_base_bonus[me],
        percent_bonus=wp.armor_percent_bonus[me],
    )
    mr = stat_total(
        mr_base,
        percent_base_bonus=wp.mr_percent_base_bonus[me],
        percent_bonus=wp.mr_percent_bonus[me],
    )

    rank = state.spell_level[me].astype(jnp.int32)
    w_cd = jnp.asarray(W_COOLDOWNS, state.x.dtype)
    e_cd = jnp.asarray(E_COOLDOWNS, state.x.dtype)
    r_cd = jnp.asarray(R_COOLDOWNS, state.x.dtype)
    base_cd = jnp.stack([
        jnp.asarray(Q_COOLDOWN, state.x.dtype),
        w_cd[jnp.clip(rank[Slot.W], 1, len(W_COOLDOWNS)) - 1],
        e_cd[jnp.clip(rank[Slot.E], 1, len(E_COOLDOWNS)) - 1],
        r_cd[jnp.clip(rank[Slot.R], 1, len(R_COOLDOWNS)) - 1],
    ])
    # E's cooldown starts after its spin and Q's after the empowerment window,
    # so both read a zero countdown while their buff is live. Whether a press
    # does anything then is `spells.status`'s `cast_locked` -- the SAME rule
    # `apply_orders` gates the cast on (`STRUCT-001`): Q is locked for its
    # whole window; E for the first `E_CANCEL_MIN_S` of its spin and then
    # AVAILABLE, because a press from 1.0 s on CANCELS the spin (and starts
    # its cooldown). This used to report E locked for the whole spin while
    # the sim accepted the cancel -- the observation and the cast gate
    # disagreeing on the spell the policy farms with (`OBS-01`).
    cast_locked = status_of(state).cast_locked[me]
    cooldowns = jnp.where(
        (rank > 0) & ~cast_locked,
        jnp.clip(state.spell_cooldown[me] / base_cd, 0.0, 1.0),
        1.0,
    )

    self_vec = jnp.stack([
        s_ / NORM_XY, n_ / NORM_XY,
        jnp.round(my_hp * HP_BAR_STEPS) / HP_BAR_STEPS,
        state.level[me].astype(jnp.float32) / 18.0,
        state.gold[me] / NORM_GOLD,
        state.cs[me].astype(jnp.float32) / NORM_CS,
        cooldowns[Slot.Q], cooldowns[Slot.W], cooldowns[Slot.E], cooldowns[Slot.R],
        ad / NORM_AD,
        jnp.float32(0.0),          # no AP source exists in LaneState/profiles
        armor / NORM_AD,
        mr / NORM_AD,
        (~state.alive[me]).astype(jnp.float32),
        (state.recall_channel_ms[me] > 0).astype(jnp.float32),
    ])

    # ---- global ----------------------------------------------------------
    enemy_champ_visible = (enemy_champ[0] >= 0).astype(jnp.float32)
    # This is witnessed-event memory, not an estimate of the enemy's live
    # cooldown.  A never-seen event uses the saturated value deliberately:
    # both "unknown" and "long ago" mean an old cast carries no useful timing
    # information, while the state keeps ``-1`` as an auditable sentinel.
    # Normalise against rank-1 bases only, so we never leak the enemy's level
    # or spell ranks into the actor observation.
    observed_ms = state.observed_enemy_cast_ms[me]
    observed_base_ms = jnp.asarray([
        Q_COOLDOWN * 1000.0,
        W_COOLDOWNS[0] * 1000.0,
        E_COOLDOWNS[0] * 1000.0,
        R_COOLDOWNS[0] * 1000.0,
    ], state.x.dtype)
    since_observed_cast = jnp.where(
        observed_ms >= 0,
        jnp.clip(observed_ms / observed_base_ms, 0.0, 1.0),
        1.0,
    )
    global_vec = jnp.concatenate([
        jnp.stack([state.t_ms / (horizon_s * 1000.0), enemy_champ_visible]),
        since_observed_cast,
    ])

    return Observation(entities=entities, entity_pad_mask=~valid,
                       self_vec=self_vec, global_vec=global_vec,
                       slot_unit=slot_unit)
