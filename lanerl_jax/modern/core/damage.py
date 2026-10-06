"""Damage packets, typed shields and healing (DMG.*/HEAL.*/SHIELD.*, DAMAGE_AND_STATS §2.2, §5-7).

Per packet: DMG.10 prevent -> .15 crit -> .30 pre-mitigation flat -> .40 dealt (summed) -> .45 unit class
-> .50 resist -> .60 received (product) -> .70 post-mitigation flat -> Lifeline -> .80 shields -> Death's Dance
-> .85 health -> .92 vamp (caller). Packets on stateful units resolve sequentially in emission order; modifiers
arrive as precomputed ``Defense``/``Offense`` profiles.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .stats import MAGIC, PHYSICAL, TRUE, mitigation_multiplier

# Packet tags (client DamageSourceSettings) plus engine properties, one bitmask.
TAG_AOE = 1 << 0
TAG_PERIODIC = 1 << 1
TAG_INDIRECT = 1 << 2
TAG_BASIC_ATTACK = 1 << 3
TAG_ACTIVE_SPELL = 1 << 4
TAG_PROC = 1 << 5
TAG_PET = 1 << 6
TAG_ITEM = 1 << 8
TAG_DOES_NOT_AGGRO_JUNGLE = 1 << 9
TAG_ON_HIT = 1 << 10
TAG_BURN = 1 << 12
TAG_NON_AMPABLE = 1 << 13
PROP_LIFESTEAL = 1 << 16       # ApplyLifesteal
PROP_NO_OMNIVAMP = 1 << 17     # inverse of ApplyOmnivamp (Ignite/Smite/reactive)
PROP_NO_DAMAGE_MOD = 1 << 18   # lacks ApplyDamageModifier
PROP_REACTIVE = 1 << 19        # Thorns-style reflected damage
PROP_CRIT = 1 << 20            # the crit-capable portion critically struck
PROP_EXECUTE = 1 << 21         # destroys shields, deals current health
PROP_ULTIMATE = 1 << 22        # champion's R
PROP_SUMMONER = 1 << 23

BASIC_ATTACK = TAG_BASIC_ATTACK | PROP_LIFESTEAL
ON_HIT_ITEM = TAG_ON_HIT | TAG_PROC | TAG_ITEM

SHIELD_ALL = 0
SHIELD_PHYSICAL = 1
SHIELD_MAGIC = 2

# Unit classes for DMG.45 (client CLASSIC GameModeConstants dr_*).
CLASS_CHAMPION = 0
CLASS_MINION = 1
CLASS_STRUCTURE = 2
CLASS_MONSTER = 3
UNIT_CLASS_RATIO = (
    # dst:  champion minion structure monster
    (1.00, 1.00, 1.00, 1.00),   # champion source
    (0.55, 1.00, 0.60, 1.00),   # minion source (MINIONS §4)
    (1.00, 1.00, 1.00, 1.00),   # structure source
    (1.00, 1.00, 1.00, 1.00),   # monster source
)
OMNIVAMP_MODIFIED_RATIO = 0.333       # ov_OmnivampModifiedRatio
GRIEVOUS_WOUNDS = 0.40                # non-stacking
LIFELINE_THRESHOLD = 0.30
RESOLVE_CAPACITY = 64                 # sequential packets per pass; ``Resolved.overflow`` reports excess


class Packets(NamedTuple):
    """Fixed-capacity damage packets, shape (P,). ``valid`` masks padding."""
    valid: Any
    src: Any            # int32 unit index
    dst: Any            # int32 unit index
    raw: Any            # pre-mitigation, crit applied
    dtype: Any          # int32 PHYSICAL/MAGIC/TRUE
    flags: Any          # TAG_*/PROP_* bitmask
    amp: Any            # DMG.40 additive source-side sum
    item: Any           # provenance: item id > 0, rune perk as -id, 0 none
    cast_id: Any        # cast instance (> 0); 0 = its own instance
    block: Any          # flat reduction on every damage type (Bone Plating)


def packets(valid, src, dst, raw, dtype, flags=0, amp=0.0, item=0, cast_id=0, block=0.0) -> Packets:
    valid = jnp.asarray(valid, bool)
    shape = jnp.broadcast_shapes(valid.shape, jnp.shape(src), jnp.shape(dst),
                                 jnp.shape(raw), jnp.shape(dtype), jnp.shape(flags),
                                 jnp.shape(amp), jnp.shape(item), jnp.shape(cast_id), jnp.shape(block))
    b = lambda x, t: jnp.broadcast_to(jnp.asarray(x, t), shape).reshape(-1)
    return Packets(b(valid, bool), b(src, jnp.int32), b(dst, jnp.int32),
                   b(raw, jnp.float32), b(dtype, jnp.int32), b(flags, jnp.int32),
                   b(amp, jnp.float32), b(item, jnp.int32), b(cast_id, jnp.int32), b(block, jnp.float32))


def empty_packets(n: int = 0) -> Packets:
    return packets(jnp.zeros((n,), bool), 0, 0, 0.0, PHYSICAL)


def concat_packets(*batches: Packets) -> Packets:
    batches = [b for b in batches if b is not None]
    if not batches:
        return empty_packets(0)
    return Packets(*(jnp.concatenate([getattr(b, f) for b in batches])
                     for f in Packets._fields))


def compact_packets(p: Packets, capacity: int) -> tuple[Packets, Any]:
    """Valid packets first in emission order, padded to ``capacity``; ``(packets, dropped count)``."""
    n = p.valid.shape[0]
    (idx,) = jnp.nonzero(p.valid, size=capacity, fill_value=n)
    pad = lambda a: jnp.concatenate([a, jnp.zeros((1,), a.dtype)])[idx]
    out = Packets(*(pad(getattr(p, f)) for f in Packets._fields))
    return out, jnp.maximum(jnp.sum(p.valid) - capacity, 0)


def has(flags: Any, bit: int) -> Any:
    return (jnp.asarray(flags) & bit) != 0


def per_unit(val: Any, idx: Any, n_units: int, op: str = "any") -> Any:
    """(C, P) per-packet ``val`` scattered onto (C, N) at unit ``idx`` (P,), negative dropped:
    ``"any"`` (bool), ``"add"`` (packet order) or ``"max"`` (from 0)."""
    rows = jnp.arange(val.shape[0])[:, None]
    cols = jnp.where(idx >= 0, idx, n_units)[None, :]
    if op == "any":
        out = jnp.zeros((val.shape[0], n_units), bool)
        return out.at[rows, jnp.where(val, cols, n_units)].set(True, mode="drop")
    out = jnp.zeros((val.shape[0], n_units), val.dtype).at[rows, cols]
    return out.add(val, mode="drop") if op == "add" else out.max(val, mode="drop")


def first_per_key(sel: Any, *keys: Any) -> Any:
    """(C, P) ``sel`` keeping, per row, only the first selected packet of each distinct ``keys`` (P,) tuple."""
    P = sel.shape[-1]
    idx = jnp.arange(P)
    order = jnp.lexsort((idx,) + keys[::-1])
    ks = [k[order] for k in keys]
    new = ks[0][1:] != ks[0][:-1]
    for k in ks[1:]:
        new = new | (k[1:] != k[:-1])
    start = jnp.concatenate([jnp.ones((1,), bool), new])
    group = jnp.zeros((P,), jnp.int32).at[order].set(jnp.cumsum(start, dtype=jnp.int32) - 1)
    first = jax.vmap(lambda s: jax.ops.segment_min(jnp.where(s, idx, P), group, num_segments=P))(sel)
    return sel & (first[:, group] == idx[None, :])


class Shields(NamedTuple):
    """Per-unit shield slots, (N, K); times absolute."""
    amount: Any         # remaining absorb before the decay cap
    initial: Any        # amount at grant
    kind: Any           # SHIELD_*
    expires_at: Any
    decay_start: Any    # +inf: no decay
    order: Any          # grant counter


def init_shields(n_units: int, k: int = 6) -> Shields:
    z = jnp.zeros((n_units, k), jnp.float32)
    return Shields(z, z, jnp.zeros((n_units, k), jnp.int32), z,
                   jnp.full((n_units, k), jnp.inf, jnp.float32),
                   jnp.zeros((n_units, k), jnp.int32))


def shield_value(sh: Shields, now: Any) -> Any:
    """Current absorb per slot, decaying linearly to 0 at expiry."""
    span = jnp.maximum(sh.expires_at - sh.decay_start, 1e-6)
    frac = jnp.clip((sh.expires_at - now) / span, 0.0, 1.0)
    cap = jnp.where(now > sh.decay_start, sh.initial * frac, sh.initial)
    live = (sh.amount > 0.0) & (now < sh.expires_at)
    return jnp.where(live, jnp.minimum(sh.amount, cap), 0.0)


def grant_shield(sh: Shields, unit: Any, amount: Any, kind: Any, now: Any,
                 duration: Any, *, decay_hold: Any = jnp.inf, enabled: Any = True) -> Shields:
    """Insert one shield into the emptiest-soonest slot (SHIELD.40)."""
    value = shield_value(sh, now)
    slot_score = jnp.where(value[unit] > 0.0, sh.expires_at[unit], -jnp.inf)
    slot = jnp.argmin(slot_score)
    ok = jnp.asarray(enabled) & (jnp.asarray(amount) > 0.0)
    order = jnp.max(sh.order) + 1

    def put(arr, val):
        return arr.at[unit, slot].set(jnp.where(ok, jnp.asarray(val, arr.dtype), arr[unit, slot]))
    return Shields(put(sh.amount, amount), put(sh.initial, amount), put(sh.kind, kind),
                   put(sh.expires_at, now + duration), put(sh.decay_start, now + decay_hold),
                   put(sh.order, order))


def total_shield(sh: Shields, now: Any, kind: int | None = None) -> Any:
    v = shield_value(sh, now)
    if kind is not None:
        v = jnp.where((sh.kind == kind) | (sh.kind == SHIELD_ALL), v, 0.0)
    return jnp.sum(v, axis=-1)


class Defense(NamedTuple):
    """Per-unit target-side profile for one resolution pass, (N,)."""
    armor: Any
    magic_resist: Any
    flat_armor_reduction: Any
    percent_armor_reduction: Any      # combined 1 - prod(1 - p) (Carve, Garen E)
    flat_mr_reduction: Any
    percent_mr_reduction: Any
    received_mult: Any                # DMG.60 product of reductions (not on true damage)
    champion_received_mult: Any       # ... on damage from champions only
    received_amp: Any                 # DMG.60 additive vulnerability
    magic_received_amp: Any           # ... magic only (Abyssal Mask)
    basic_attack_mult: Any            # Plating 0.9 on non-turret basic attacks
    crit_taken_mult: Any              # Randuin's 0.7 on critical basic attacks
    champion_attack_block: Any        # Warden's Mail flat block (15), capped 20%
    postmit_flat: Any                 # DMG.70 flat (Bone Plating etc.)
    store_fraction: Any               # Death's Dance stored share
    invulnerable: Any
    spell_shield: Any                 # blocks enemy-champion ActiveSpell packets (Annul)
    unit_class: Any                   # CLASS_*
    lifeline_ready: Any
    lifeline_magic_only: Any
    lifeline_shield: Any
    lifeline_shield_kind: Any
    lifeline_duration: Any
    lifeline_decay_hold: Any
    lifeline_bonus_health: Any        # Protoplasm
    dodge_basic: Any = None           # non-turret basic attacks dodged (Jax E); None = never
    aoe_received_mult: Any = None     # on AoE packets (Jax E); None = 1
    received_mult_all: Any = None     # on every type incl. true (turret backdoor); None = 1


def default_defense(n: int, *, armor=0.0, magic_resist=0.0, unit_class=CLASS_MINION) -> Defense:
    f = lambda v, t=jnp.float32: jnp.broadcast_to(jnp.asarray(v, t), (n,))
    return Defense(
        armor=f(armor), magic_resist=f(magic_resist), flat_armor_reduction=f(0.),
        percent_armor_reduction=f(0.), flat_mr_reduction=f(0.), percent_mr_reduction=f(0.),
        received_mult=f(1.), champion_received_mult=f(1.), received_amp=f(0.), magic_received_amp=f(0.), basic_attack_mult=f(1.),
        crit_taken_mult=f(1.), champion_attack_block=f(0.), postmit_flat=f(0.), store_fraction=f(0.),
        invulnerable=f(False, bool), spell_shield=f(False, bool), unit_class=f(unit_class, jnp.int32), lifeline_ready=f(False, bool),
        lifeline_magic_only=f(False, bool), lifeline_shield=f(0.), lifeline_shield_kind=f(SHIELD_ALL, jnp.int32),
        lifeline_duration=f(0.), lifeline_decay_hold=f(jnp.inf), lifeline_bonus_health=f(0.))


class Offense(NamedTuple):
    """Per-unit attacker-side profile, (N,)."""
    lethality: Any
    percent_armor_pen: Any
    magic_pen: Any
    percent_magic_pen: Any
    dealt_reduction: Any              # Exhaust (not on true damage)
    unit_class: Any
    is_turret: Any                    # turret attacks bypass basic-attack reductions


def default_offense(n: int, *, unit_class=CLASS_MINION) -> Offense:
    f = lambda v, t=jnp.float32: jnp.broadcast_to(jnp.asarray(v, t), (n,))
    return Offense(lethality=f(0.), percent_armor_pen=f(0.), magic_pen=f(0.),
                   percent_magic_pen=f(0.), dealt_reduction=f(0.), unit_class=f(unit_class, jnp.int32),
                   is_turret=f(False, bool))


def effective_resist(resist, flat_red, pct_red, pct_pen, flat_pen):
    """DMG.50 order (§4.2); keeps negative resist."""
    r = resist - flat_red
    r = jnp.where(r > 0.0, r * (1.0 - pct_red), r)
    r = jnp.where(r > 0.0, r * (1.0 - pct_pen), r)
    return jnp.where(r > 0.0, jnp.maximum(0.0, r - flat_pen), r)


def premitigation_to_final(p: Packets, off: Offense, dfn: Defense) -> Any:
    """DMG.15-75 for every packet in parallel."""
    s, d = p.src, p.dst
    ratio = jnp.asarray(UNIT_CLASS_RATIO, jnp.float32)[off.unit_class[s], dfn.unit_class[d]]
    no_mod = has(p.flags, PROP_NO_DAMAGE_MOD)
    is_true = p.dtype == TRUE
    reduction = jnp.where(is_true, 0.0, off.dealt_reduction[s])
    dealt = jnp.where(no_mod, 1.0, jnp.maximum(1.0 + p.amp - reduction, 0.0))
    raw = p.raw * dealt * ratio
    armor = effective_resist(dfn.armor[d], dfn.flat_armor_reduction[d], dfn.percent_armor_reduction[d],
                             off.percent_armor_pen[s], off.lethality[s])
    mr = effective_resist(dfn.magic_resist[d], dfn.flat_mr_reduction[d], dfn.percent_mr_reduction[d],
                          off.percent_magic_pen[s], off.magic_pen[s])
    mult = jnp.where(p.dtype == PHYSICAL, mitigation_multiplier(armor, jnp),
                     jnp.where(p.dtype == MAGIC, mitigation_multiplier(mr, jnp), 1.0))
    post = raw * mult
    # DMG.60: true damage keeps only amplifiers.
    amp = 1.0 + dfn.received_amp[d] + jnp.where(p.dtype == MAGIC, dfn.magic_received_amp[d], 0.0)
    from_champion = off.unit_class[s] == CLASS_CHAMPION
    reduction_mult = dfn.received_mult[d] * jnp.where(from_champion, dfn.champion_received_mult[d], 1.0)
    received = jnp.where(is_true, 1.0, reduction_mult) * amp
    basic = has(p.flags, TAG_BASIC_ATTACK) & ~off.is_turret[s]
    received = received * jnp.where(basic & ~is_true, dfn.basic_attack_mult[d], 1.0)
    received = received * jnp.where(basic & has(p.flags, PROP_CRIT), dfn.crit_taken_mult[d], 1.0)
    post = jnp.where(no_mod, post, post * received)
    champ_basic = basic & (off.unit_class[s] == CLASS_CHAMPION)
    block = jnp.where(champ_basic, jnp.minimum(dfn.champion_attack_block[d], 0.2 * post), 0.0)
    post = jnp.where(is_true, post, jnp.maximum(post - block - dfn.postmit_flat[d], 0.0))
    post = jnp.maximum(post - p.block, 0.0)
    if dfn.aoe_received_mult is not None:
        post = post * jnp.where(has(p.flags, TAG_AOE), dfn.aoe_received_mult[d], 1.0)
    if dfn.received_mult_all is not None:
        post = post * dfn.received_mult_all[d]
    return jnp.where(p.valid & ~dfn.invulnerable[d] & ~spell_blocked(p, off, dfn) & ~dodged(p, off, dfn),
                     jnp.maximum(post, 0.0), 0.0)


def dodged(p: Packets, off: Offense, dfn: Defense) -> Any:
    """DMG.10: non-turret basic attacks and their on-hits on a dodging target deal nothing."""
    if dfn.dodge_basic is None:
        return jnp.zeros(p.valid.shape, bool)
    return p.valid & dfn.dodge_basic[p.dst] & has(p.flags, TAG_BASIC_ATTACK) & ~off.is_turret[p.src]


def spell_blocked(p: Packets, off: Offense, dfn: Defense) -> Any:
    """DMG.10: a spell shield blocks every enemy-champion ActiveSpell packet of the pass (§5.2)."""
    return p.valid & dfn.spell_shield[p.dst] & has(p.flags, TAG_ACTIVE_SPELL) \
        & (off.unit_class[p.src] == CLASS_CHAMPION) & (p.src != p.dst)


class Resolved(NamedTuple):
    final: Any          # (P,) post-mitigation, before shields (vamp reads this)
    absorbed: Any       # (P,)
    stored: Any         # (P,) to the Death's Dance pool
    health_loss: Any    # (P,) capped at remaining HP
    killed: Any         # (P,)
    hp: Any             # (N,)
    max_hp: Any         # (N,)
    shields: Shields
    lifeline_fired: Any  # (N,)
    dd_pool_add: Any    # (N,)
    spell_shield_popped: Any  # (N,)
    overflow: Any       # () sequential packets beyond RESOLVE_CAPACITY (dropped)


def sequential_units(dfn: Defense, shields: Shields, now: Any) -> Any:
    """(N,) units needing the sequential pass: champions and units with shields, Lifeline or storage."""
    return (dfn.unit_class == CLASS_CHAMPION) | (total_shield(shields, now) > 0.0) \
        | dfn.lifeline_ready | (dfn.store_fraction > 0.0)


def resolve(p: Packets, off: Offense, dfn: Defense, hp: Any, max_hp: Any,
            shields: Shields, now: Any) -> Resolved:
    """Lifeline -> shields -> Death's Dance -> HP. Other units carry none of that state, so their packets
    resolve in parallel with per-target running sums in emission order."""
    final = premitigation_to_final(p, off, dfn)
    execute = has(p.flags, PROP_EXECUTE)
    n_units = hp.shape[0]

    def typed_value(sh, d, dtype):
        value = shield_value(sh, now)[d]
        typed = (sh.kind[d] == SHIELD_ALL) | ((sh.kind[d] == SHIELD_PHYSICAL) & (dtype == PHYSICAL)) \
            | ((sh.kind[d] == SHIELD_MAGIC) & (dtype == MAGIC))
        return value, typed

    def body(carry, x):
        hp, max_hp, sh, fired = carry
        valid, d, dmg, dtype, exe = x
        alive = hp[d] > 0.0
        go = valid & alive
        # Lifeline: HP after existing shields would drop below 30%; its shield absorbs this packet (ITEMS §6.3).
        value, typed = typed_value(sh, d, dtype)
        pre_shield = jnp.sum(jnp.where(typed, value, 0.0))
        trigger = go & ~exe & dfn.lifeline_ready[d] & ~fired[d] & (dmg > 0.0) \
            & (~dfn.lifeline_magic_only[d] | (dtype == MAGIC)) \
            & (hp[d] - jnp.maximum(dmg - pre_shield, 0.0) < LIFELINE_THRESHOLD * max_hp[d])
        sh = grant_shield(sh, d, dfn.lifeline_shield[d], dfn.lifeline_shield_kind[d], now,
                          dfn.lifeline_duration[d], decay_hold=dfn.lifeline_decay_hold[d],
                          enabled=trigger)
        bonus = jnp.where(trigger, dfn.lifeline_bonus_health[d], 0.0).astype(hp.dtype)
        hp = hp.at[d].add(bonus.astype(hp.dtype))
        max_hp = max_hp.at[d].add(bonus.astype(max_hp.dtype))
        fired = fired.at[d].set(fired[d] | trigger)
        # DMG.80: execute destroys shields; else typed shields absorb soonest-expiry, earliest-grant first.
        value, typed = typed_value(sh, d, dtype)
        usable = jnp.where(typed & go & ~exe, value, 0.0)
        key = jnp.where(usable > 0.0, sh.expires_at[d] * 1e3 + sh.order[d] * 1e-6, jnp.inf)
        order = jnp.argsort(key)
        sorted_avail = usable[order]
        before = jnp.cumsum(sorted_avail) - sorted_avail
        take_sorted = jnp.clip(dmg - before, 0.0, sorted_avail)
        take = jnp.zeros_like(take_sorted).at[order].set(take_sorted)
        absorbed = jnp.sum(take)
        new_amount = jnp.where(take > 0.0, value - take, sh.amount[d]).astype(sh.amount.dtype)
        sh = sh._replace(amount=sh.amount.at[d].set(new_amount))
        sh = sh._replace(amount=jnp.where(exe & go, sh.amount.at[d].set(0.0), sh.amount))
        through = jnp.where(exe, hp[d], dmg - absorbed)
        # Death's Dance stores physical and magic only.
        stored = jnp.where(exe | (dtype == TRUE), 0.0, through * dfn.store_fraction[d])
        to_hp = jnp.where(go, through - stored, 0.0)
        loss = jnp.minimum(to_hp, jnp.maximum(hp[d], 0.0))
        new_hp = hp[d] - to_hp
        killed = go & (new_hp <= 0.0)
        hp = hp.at[d].set(jnp.where(go, new_hp, hp[d]).astype(hp.dtype))
        out = (jnp.where(go, absorbed, 0.0), jnp.where(go, stored, 0.0), loss, killed)
        return (hp, max_hp, sh, fired), out

    # Sequential packets go to a fixed buffer, independent of the emission grid size.
    n_packets = p.valid.shape[0]
    stateful = sequential_units(dfn, shields, now)
    on_champion = stateful[p.dst]
    seq = p.valid & on_champion
    par = p.valid & ~on_champion
    cap = min(n_packets, RESOLVE_CAPACITY)
    (idx,) = jnp.nonzero(seq, size=cap, fill_value=n_packets)
    take = lambda a, fill: jnp.concatenate([a, jnp.asarray([fill], a.dtype)])[idx]
    xs = (take(seq, False), take(p.dst, 0), take(final, 0.0), take(p.dtype, PHYSICAL), take(execute, False))
    carry0 = (hp, max_hp, shields, jnp.zeros((n_units,), bool))
    # Only real packets are stepped (padding is a no-op): a vmapped batch runs as long as its busiest env.
    count = jnp.minimum(jnp.sum(seq), cap)
    outs0 = (jnp.zeros((cap,), jnp.float32), jnp.zeros((cap,), jnp.float32), jnp.zeros((cap,), jnp.float32),
             jnp.zeros((cap,), bool))

    def step_one(i, c):
        carry, outs = c
        carry, y = body(carry, jax.tree.map(lambda a: a[i], xs))
        return carry, tuple(o.at[i].set(jnp.asarray(v, o.dtype)) for o, v in zip(outs, y))
    (hp_s, max_hp, shields, fired), (absorbed_c, stored_c, loss_c, killed_c) = jax.lax.fori_loop(
        0, count, step_one, (carry0, outs0))

    def back(vals, dtype):
        return jnp.zeros((n_packets + 1,), dtype).at[idx].set(vals.astype(dtype))[:n_packets]
    absorbed, stored = back(absorbed_c, jnp.float32), back(stored_c, jnp.float32)
    loss_s, killed_s = back(loss_c, jnp.float32), back(killed_c, bool)

    # Parallel pass: damage dealt to each target before this packet.
    f = jnp.where(par, jnp.where(execute, jnp.maximum(hp[p.dst], 0.0), final), 0.0).astype(jnp.float32)
    order = jnp.lexsort((jnp.arange(n_packets), p.dst))      # by target, then emission order
    fs, ds = f[order], p.dst[order]
    cs = jnp.cumsum(fs)
    first = jnp.concatenate([jnp.ones((1,), bool), ds[1:] != ds[:-1]])
    base = jax.lax.cummax(jnp.where(first, cs - fs, -jnp.inf))
    before = jnp.zeros_like(f).at[order].set(cs - fs - base)
    hp0 = hp[p.dst].astype(jnp.float32)
    left = hp0 - before
    loss_p = jnp.where(par, jnp.clip(left, 0.0, f), 0.0)
    killed_p = par & (left > 0.0) & (left - f <= 0.0)
    total = jnp.zeros((n_units,), jnp.float32).at[p.dst].add(jnp.where(par, f, 0.0))
    alive0 = hp > 0.0
    hp = jnp.where(stateful, hp_s, jnp.where(alive0, hp - total.astype(hp.dtype), hp))
    loss = loss_s + loss_p
    killed = killed_s | killed_p
    dd_add = jnp.zeros((n_units,), jnp.float32).at[p.dst].add(stored)
    popped = jnp.zeros((n_units,), bool).at[p.dst].max(spell_blocked(p, off, dfn))
    overflow = jnp.maximum(jnp.sum(seq) - cap, 0)
    return Resolved(final, absorbed, stored, loss, killed, hp, max_hp, shields, fired, dd_add, popped,
                    overflow)


class Vamp(NamedTuple):
    life_steal: Any     # (N,) per source unit
    omnivamp: Any       # (N,)


def vamp_heal(p: Packets, res: Resolved, vamp: Vamp, dst_class: Any,
              *, lifesteal_scale: Any = None) -> Any:
    ls, ov = vamp_heal_split(p, res, vamp, dst_class, lifesteal_scale=lifesteal_scale)
    return ls + ov


def vamp_heal_split(p: Packets, res: Resolved, vamp: Vamp, dst_class: Any,
                    *, lifesteal_scale: Any = None) -> tuple[Any, Any]:
    """DMG.92 ``(life steal, omnivamp)`` heal per source unit from ``final``, before HEAL.20/30.

    Never vs structures. Omnivamp skips PROP_NO_OMNIVAMP/PROP_REACTIVE; AoE, pet and periodic packets heal 33.3%
    vs minions and monsters. ``lifesteal_scale`` (P,): per-packet effectiveness."""
    n = vamp.life_steal.shape[0]
    cls = dst_class[p.dst]
    scale = jnp.ones_like(res.final) if lifesteal_scale is None else lifesteal_scale
    ls = jnp.where(has(p.flags, PROP_LIFESTEAL) & (cls != CLASS_STRUCTURE),
                   vamp.life_steal[p.src] * res.final * scale, 0.0)
    modified = ((cls == CLASS_MINION) | (cls == CLASS_MONSTER)) & \
        (has(p.flags, TAG_AOE) | has(p.flags, TAG_PET) | has(p.flags, TAG_PERIODIC))
    ov_ok = ~has(p.flags, PROP_NO_OMNIVAMP) & ~has(p.flags, PROP_REACTIVE) & (cls != CLASS_STRUCTURE)
    ov = jnp.where(ov_ok, vamp.omnivamp[p.src] * res.final
                   * jnp.where(modified, OMNIVAMP_MODIFIED_RATIO, 1.0), 0.0)
    z = jnp.zeros((n,), jnp.float32)
    return (z.at[p.src].add(jnp.where(p.valid, ls, 0.0).astype(jnp.float32)),
            z.at[p.src].add(jnp.where(p.valid, ov, 0.0).astype(jnp.float32)))


def heal_amount(base: Any, *, source_power: Any = 0.0, incoming: Any = 0.0,
                grievous: Any = False) -> Any:
    """HEAL.10-30 ``(1 + HSP) * (1 + sum incoming) * (1 - 0.4 GW)``."""
    gw = jnp.where(grievous, 1.0 - GRIEVOUS_WOUNDS, 1.0)
    return jnp.maximum(base, 0.0) * (1.0 + source_power) * (1.0 + incoming) * gw


def apply_heal(hp: Any, max_hp: Any, amount: Any, alive: Any = True) -> Any:
    """HEAL.40: no overheal, dead units are not healed."""
    return jnp.where(alive, jnp.minimum(max_hp, hp + jnp.maximum(amount, 0.0)), hp)
