"""26.19 damage-packet pipeline, typed shields and healing (DMG.*/HEAL.*/SHIELD.*).

Implements the ordered per-packet contract of docs/modern/DAMAGE_AND_STATS.md
§2.2 and §5–7 as pure, fixed-shape JAX:

    DMG.10 prevent -> DMG.15 crit -> DMG.30 pre-mitigation flat
    -> DMG.40 dealt modifiers (source-side, summed) -> DMG.45 unit class
    -> DMG.50 resist (reduction then penetration) -> DMG.60 received (product)
    -> DMG.70 post-mitigation flat -> DMG.75 final
    -> Lifeline check (items, before shields) -> DMG.80 typed shields
    -> Death's Dance storage -> DMG.85 health -> DMG.92 vamp (caller)

Packets resolve sequentially (``lax.scan``) so each one sees the shields, HP
and Lifeline state left by the previous packet; the caller fixes packet order
(TICK.70 deterministic emission order). Every value a modifier needs is an
explicit per-unit or per-packet input: items, runes and champions fill the
``Defense``/``Offense`` profiles before resolution instead of being called
from inside the scan.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .modern_stats import MAGIC, PHYSICAL, TRUE, mitigation_multiplier

# Packet tags (client DamageSourceSettings) plus engine properties, one bitmask.
TAG_AOE = 1 << 0
TAG_PERIODIC = 1 << 1
TAG_INDIRECT = 1 << 2
TAG_BASIC_ATTACK = 1 << 3
TAG_ACTIVE_SPELL = 1 << 4
TAG_PROC = 1 << 5
TAG_PET = 1 << 6
TAG_NON_REDIRECTABLE = 1 << 7
TAG_ITEM = 1 << 8
TAG_DOES_NOT_AGGRO_JUNGLE = 1 << 9
TAG_ON_HIT = 1 << 10
TAG_AUGMENT = 1 << 11
TAG_BURN = 1 << 12
TAG_NON_AMPABLE = 1 << 13
PROP_LIFESTEAL = 1 << 16       # ApplyLifesteal
PROP_NO_OMNIVAMP = 1 << 17     # inverse of ApplyOmnivamp (Ignite/Smite/reactive)
PROP_NO_DAMAGE_MOD = 1 << 18   # lacks ApplyDamageModifier
PROP_REACTIVE = 1 << 19        # Thorns-style reflected damage
PROP_CRIT = 1 << 20            # the crit-capable portion critically struck
PROP_EXECUTE = 1 << 21         # destroys shields, deals current health
PROP_ULTIMATE = 1 << 22        # emitted by the champion's R (Axiom Arcanist, Malignance)
PROP_SUMMONER = 1 << 23        # summoner-spell damage (Ignite)

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
    (0.55, 1.00, 0.60, 1.00),   # minion source (MINIONS.md §4, README X-2)
    (1.00, 1.00, 1.00, 1.00),   # structure source
    (1.00, 1.00, 1.00, 1.00),   # monster source
)
OMNIVAMP_MODIFIED_RATIO = 0.333       # ov_OmnivampModifiedRatio
GRIEVOUS_WOUNDS = 0.40                # every SR source, non-stacking
LIFELINE_THRESHOLD = 0.30
# Maximum valid packets resolved sequentially per pass (packets on champions or
# shielded units; two champions in lane receive far fewer). ``Resolved.overflow``
# reports any excess.
RESOLVE_CAPACITY = 64


class Packets(NamedTuple):
    """Fixed-capacity damage packets, shape (P,). ``valid`` masks padding."""
    valid: Any
    src: Any            # int32 unit index
    dst: Any            # int32 unit index
    raw: Any            # pre-mitigation amount (crit already applied by emitter)
    dtype: Any          # int32 PHYSICAL/MAGIC/TRUE
    flags: Any          # int32 TAG_*/PROP_* bitmask
    amp: Any            # additive source-side modifier sum (DMG.40), e.g. +0.08
    item: Any           # int32 provenance: item id > 0, rune perk id as -id, 0 for none
    cast_id: Any        # int32 cast instance (world-assigned, > 0); 0 = the packet is its own instance
    block: Any          # DMG.70 per-packet flat reduction on every damage type (Bone Plating)


def packets(valid, src, dst, raw, dtype, flags=0, amp=0.0, item=0, cast_id=0, block=0.0) -> Packets:
    """Broadcast any packet fields to one flat batch."""
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
    """Valid packets first, emission order kept, padded to ``capacity``.

    Returns ``(packets, overflow)``; overflow counts valid packets that did not
    fit (they are dropped, so callers must assert it stays 0).
    """
    n = p.valid.shape[0]
    (idx,) = jnp.nonzero(p.valid, size=capacity, fill_value=n)
    pad = lambda a: jnp.concatenate([a, jnp.zeros((1,), a.dtype)])[idx]
    out = Packets(*(pad(getattr(p, f)) for f in Packets._fields))
    return out, jnp.maximum(jnp.sum(p.valid) - capacity, 0)


def has(flags: Any, bit: int) -> Any:
    return (jnp.asarray(flags) & bit) != 0


class Shields(NamedTuple):
    """Per-unit shield slots, shape (N, K)."""
    amount: Any         # remaining absorb before decay cap
    initial: Any        # amount at grant (decay cap reference)
    kind: Any           # SHIELD_ALL / SHIELD_PHYSICAL / SHIELD_MAGIC
    expires_at: Any     # absolute seconds; slot empty when amount <= 0
    decay_start: Any    # absolute seconds; +inf for non-decaying shields
    order: Any          # monotone grant counter (ties: earliest first)


def init_shields(n_units: int, k: int = 6) -> Shields:
    z = jnp.zeros((n_units, k), jnp.float32)
    return Shields(z, z, jnp.zeros((n_units, k), jnp.int32), z,
                   jnp.full((n_units, k), jnp.inf, jnp.float32),
                   jnp.zeros((n_units, k), jnp.int32))


def shield_value(sh: Shields, now: Any) -> Any:
    """Current absorb of each slot, including linear decay to 0 at expiry."""
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
    """Per-unit target-side profile for one resolution pass, shape (N,)."""
    armor: Any
    magic_resist: Any
    flat_armor_reduction: Any
    percent_armor_reduction: Any      # combined 1 - prod(1 - p) (Carve, Garen E)
    flat_mr_reduction: Any
    percent_mr_reduction: Any
    received_mult: Any                # DMG.60 product of damage reductions (not on true damage)
    champion_received_mult: Any       # DMG.60 reduction on damage from champions only
    received_amp: Any                 # DMG.60 additive vulnerability, all damage types
    magic_received_amp: Any           # DMG.60 additive vulnerability, magic only (Abyssal Mask)
    basic_attack_mult: Any            # Plating 0.9 on non-turret basic attacks
    crit_taken_mult: Any              # Randuin's 0.7 on critical basic attacks
    champion_attack_block: Any        # Warden's Mail flat block (15), capped 20%
    postmit_flat: Any                 # DMG.70 flat (Bone Plating etc.)
    store_fraction: Any               # Death's Dance stored share (true over 3 s)
    invulnerable: Any
    spell_shield: Any                 # blocks enemy-champion ActiveSpell packets this pass (Annul)
    unit_class: Any                   # CLASS_*
    lifeline_ready: Any
    lifeline_magic_only: Any
    lifeline_shield: Any              # shield granted on trigger
    lifeline_shield_kind: Any
    lifeline_duration: Any
    lifeline_decay_hold: Any
    lifeline_bonus_health: Any        # Protoplasm max-HP grant on trigger
    dodge_basic: Any = None           # (N,) bool: non-turret basic attacks are dodged (Jax E); None = never
    aoe_received_mult: Any = None     # (N,) DMG.60 multiplier on AoE-tagged packets (Jax E 0.75); None = 1
    received_mult_all: Any = None     # (N,) DMG.60 multiplier on every damage type incl. true (turret backdoor 0.2)


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
    """Per-unit attacker-side profile, shape (N,)."""
    lethality: Any
    percent_armor_pen: Any
    magic_pen: Any
    percent_magic_pen: Any
    dealt_reduction: Any              # Exhaust 0.35 (not on true damage)
    unit_class: Any
    is_turret: Any                    # turret basic attacks bypass Plating


def default_offense(n: int, *, unit_class=CLASS_MINION) -> Offense:
    f = lambda v, t=jnp.float32: jnp.broadcast_to(jnp.asarray(v, t), (n,))
    return Offense(lethality=f(0.), percent_armor_pen=f(0.), magic_pen=f(0.),
                   percent_magic_pen=f(0.), dealt_reduction=f(0.), unit_class=f(unit_class, jnp.int32),
                   is_turret=f(False, bool))


def effective_resist(resist, flat_red, pct_red, pct_pen, flat_pen):
    """DMG.50 order (DAMAGE_AND_STATS §4.2); keeps negative resist."""
    r = resist - flat_red
    r = jnp.where(r > 0.0, r * (1.0 - pct_red), r)
    r = jnp.where(r > 0.0, r * (1.0 - pct_pen), r)
    return jnp.where(r > 0.0, jnp.maximum(0.0, r - flat_pen), r)


def premitigation_to_final(p: Packets, off: Offense, dfn: Defense) -> Any:
    """DMG.15–75 for every packet in parallel (no HP/shield state needed)."""
    s, d = p.src, p.dst
    ratio = jnp.asarray(UNIT_CLASS_RATIO, jnp.float32)[off.unit_class[s], dfn.unit_class[d]]
    no_mod = has(p.flags, PROP_NO_DAMAGE_MOD)
    is_true = p.dtype == TRUE
    # DMG.40: source-side amplifiers add; Exhaust reduction joins the same sum
    # but does not reduce true damage.
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
    # DMG.60 received modifiers multiply; true damage keeps only amplifiers.
    amp = 1.0 + dfn.received_amp[d] + jnp.where(p.dtype == MAGIC, dfn.magic_received_amp[d], 0.0)
    from_champion = off.unit_class[s] == CLASS_CHAMPION
    reduction_mult = dfn.received_mult[d] * jnp.where(from_champion, dfn.champion_received_mult[d], 1.0)
    received = jnp.where(is_true, 1.0, reduction_mult) * amp
    basic = has(p.flags, TAG_BASIC_ATTACK) & ~off.is_turret[s]
    received = received * jnp.where(basic & ~is_true, dfn.basic_attack_mult[d], 1.0)
    received = received * jnp.where(basic & has(p.flags, PROP_CRIT), dfn.crit_taken_mult[d], 1.0)
    post = jnp.where(no_mod, post, post * received)
    # DMG.70: flat post-mitigation reductions (not true damage).
    champ_basic = basic & (off.unit_class[s] == CLASS_CHAMPION)
    block = jnp.where(champ_basic, jnp.minimum(dfn.champion_attack_block[d], 0.2 * post), 0.0)
    post = jnp.where(is_true, post, jnp.maximum(post - block - dfn.postmit_flat[d], 0.0))
    # Per-packet flat reductions (Bone Plating) apply to every damage type.
    post = jnp.maximum(post - p.block, 0.0)
    if dfn.aoe_received_mult is not None:
        post = post * jnp.where(has(p.flags, TAG_AOE), dfn.aoe_received_mult[d], 1.0)
    if dfn.received_mult_all is not None:
        post = post * dfn.received_mult_all[d]
    return jnp.where(p.valid & ~dfn.invulnerable[d] & ~spell_blocked(p, off, dfn) & ~dodged(p, off, dfn),
                     jnp.maximum(post, 0.0), 0.0)


def dodged(p: Packets, off: Offense, dfn: Defense) -> Any:
    """DMG.10 dodge: a non-turret basic attack (and its on-hit packets) on a
    dodging target deals nothing (Jax E Counter Strike)."""
    if dfn.dodge_basic is None:
        return jnp.zeros(p.valid.shape, bool)
    return p.valid & dfn.dodge_basic[p.dst] & has(p.flags, TAG_BASIC_ATTACK) & ~off.is_turret[p.src]


def spell_blocked(p: Packets, off: Offense, dfn: Defense) -> Any:
    """DMG.10 spell shield: an enemy champion's ActiveSpell packets are blocked.

    The whole cast instance is blocked; without cast ids every ActiveSpell
    packet from enemy champions in this pass counts as one instance
    (DAMAGE_AND_STATS §5.2, INFERRED M).
    """
    return p.valid & dfn.spell_shield[p.dst] & has(p.flags, TAG_ACTIVE_SPELL) \
        & (off.unit_class[p.src] == CLASS_CHAMPION) & (p.src != p.dst)


class Resolved(NamedTuple):
    final: Any          # (P,) post-mitigation damage (vamp and "damage dealt" triggers read this)
    absorbed: Any       # (P,) taken by shields
    stored: Any         # (P,) moved to Death's Dance pool
    health_loss: Any    # (P,) actual HP removed (capped at remaining HP)
    killed: Any         # (P,) this packet brought dst to <= 0
    hp: Any             # (N,) after all packets
    max_hp: Any         # (N,)
    shields: Shields
    lifeline_fired: Any  # (N,) bool
    dd_pool_add: Any    # (N,) newly stored Death's Dance damage
    spell_shield_popped: Any  # (N,) bool: a spell shield blocked a packet
    overflow: Any       # () valid packets beyond RESOLVE_CAPACITY (dropped; must stay 0)


def sequential_units(dfn: Defense, shields: Shields, now: Any) -> Any:
    """(N,) units whose packets need the sequential pass: champions, and any
    unit with a live shield, a ready Lifeline or Death's Dance storage."""
    return (dfn.unit_class == CLASS_CHAMPION) | (total_shield(shields, now) > 0.0) \
        | dfn.lifeline_ready | (dfn.store_fraction > 0.0)


def resolve(p: Packets, off: Offense, dfn: Defense, hp: Any, max_hp: Any,
            shields: Shields, now: Any) -> Resolved:
    """Sequential resolution: Lifeline -> shields -> Death's Dance -> HP."""
    final = premitigation_to_final(p, off, dfn)
    execute = has(p.flags, PROP_EXECUTE)
    n_units = hp.shape[0]

    def body(carry, x):
        hp, max_hp, sh, fired = carry
        valid, d, dmg, dtype, exe = x
        alive = hp[d] > 0.0
        go = valid & alive
        # Lifeline: triggers on a packet that would leave HP below 30% after
        # existing shields absorb it; the granted shield then absorbs this
        # packet (ITEMS.md §6.3).
        value = shield_value(sh, now)[d]
        typed = (sh.kind[d] == SHIELD_ALL) | ((sh.kind[d] == SHIELD_PHYSICAL) & (dtype == PHYSICAL)) \
            | ((sh.kind[d] == SHIELD_MAGIC) & (dtype == MAGIC))
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
        # DMG.80: execute destroys shields; otherwise typed shields absorb in
        # soonest-expiry, earliest-grant order.
        value = shield_value(sh, now)[d]
        typed = (sh.kind[d] == SHIELD_ALL) | ((sh.kind[d] == SHIELD_PHYSICAL) & (dtype == PHYSICAL)) \
            | ((sh.kind[d] == SHIELD_MAGIC) & (dtype == MAGIC))
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
        # Death's Dance stores physical and magic damage only (its own bleed is true).
        stored = jnp.where(exe | (dtype == TRUE), 0.0, through * dfn.store_fraction[d])
        to_hp = jnp.where(go, through - stored, 0.0)
        loss = jnp.minimum(to_hp, jnp.maximum(hp[d], 0.0))
        new_hp = hp[d] - to_hp
        killed = go & (new_hp <= 0.0)
        hp = hp.at[d].set(jnp.where(go, new_hp, hp[d]).astype(hp.dtype))
        out = (jnp.where(go, absorbed, 0.0), jnp.where(go, stored, 0.0), loss, killed)
        return (hp, max_hp, sh, fired), out

    # Packets on champions resolve sequentially (shields, Lifeline, Death's
    # Dance, spell shields). Other units carry none of those, so their packets
    # resolve exactly in parallel with per-target running sums in emission
    # order. Sequential packets are compacted into a fixed buffer so the scan
    # length does not scale with the dense (holder x unit) emission grids.
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
    (hp_s, max_hp, shields, fired), (absorbed_c, stored_c, loss_c, killed_c) = jax.lax.scan(body, carry0, xs)

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
    """DMG.92: raw vamp healing per source unit (before HEAL.20/30).

    Life steal applies to packets with PROP_LIFESTEAL, never vs structures.
    Omnivamp applies to every packet without PROP_NO_OMNIVAMP/PROP_REACTIVE;
    AoE, pet and periodic packets heal 33.3% against minions and monsters.
    Vamp reads ``final`` (post-mitigation, before shields; README X-4).
    ``lifesteal_scale`` (P,) supports per-packet effectiveness (Ravenous VampAmp).
    """
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
    """HEAL.10–30: (1 + HSP_src) · (1 + Σ incoming) · (1 − 0.4·GW)."""
    gw = jnp.where(grievous, 1.0 - GRIEVOUS_WOUNDS, 1.0)
    return jnp.maximum(base, 0.0) * (1.0 + source_power) * (1.0 + incoming) * gw


def apply_heal(hp: Any, max_hp: Any, amount: Any, alive: Any = True) -> Any:
    """HEAL.40: no overheal; dead units are not healed."""
    return jnp.where(alive, jnp.minimum(max_hp, hp + jnp.maximum(amount, 0.0)), hp)
