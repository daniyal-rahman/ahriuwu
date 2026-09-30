"""26.19 Garen/Jax combat integrated with the existing lane/world step.

Rules are champion-specific; damage is attributed before the world's death
and reward pass. No training-only health corrections or balance multipliers.
The historical entry points remain usable by the existing parity fixtures.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..data.modern import IDS, champion, cooldowns, stat, values
from .combat import growth_sum, post_mitigation_damage
from .state import Kind, MoveOrder

GAREN, JAX = 86, 24


def ranked(name, slot, key, rank):
    return jnp.asarray(values(name, slot, key), jnp.float32)[jnp.clip(rank, 0, 6)]


def cooldown_table(ids, ranks):
    def table(name):
        return jnp.stack([jnp.asarray(cooldowns(name, slot),jnp.float32)[jnp.clip(ranks[:, i] - 1, 0, 2 if i == 3 else 4)]
                          for i, slot in enumerate("QWER")], -1)
    return jnp.where((ids == JAX)[:, None], table("Jax"), table("Garen"))


def mana_cost(s):
    r = s.spell_level
    costs = jnp.stack([jnp.full_like(s.hp, 50), jnp.full_like(s.hp, 30),
                       40 + 10 * r[:, 2], jnp.full_like(s.hp, 100)], -1)
    costs = jnp.where((s.champion.id == JAX)[:, None], costs, 0)
    return costs.at[:, 2].set(jnp.where(s.champion.jax_e_ms > 0, 0, costs[:, 2]))


def status(s):
    from .spells import Status
    c = s.champion
    free = s.alive & (c.stun_ms <= 0)
    may = free & (s.silenced_ms <= 0) & (s.r_cast_ms <= 0) & (s.recall_windup_ms <= 0) & (c.jax_r_cast_ms <= 0)
    ready = (s.spell_cooldown <= 0) & (s.spell_level > 0) & (c.mana[:, None] >= mana_cost(s))
    garen_e = (c.id == GAREN) & s.buffs.e.active
    jax_e = (c.id == JAX) & (c.jax_e_ms > 0)
    ready = ready.at[:, 2].set(jnp.where(garen_e, s.buffs.e.elapsed_s >= 1,
                                jnp.where(jax_e, c.jax_e_ms <= 1000, ready[:, 2])))
    ready = ready.at[:, 1].set(ready[:, 1] & ~((c.id == JAX) & (c.jax_w_ms > 0)))
    # Root/stun and dashes are represented separately from silence.
    can_attack = free & ~garen_e & (c.dash_ms <= 0) & (s.r_cast_ms <= 0) & (c.jax_r_cast_ms <= 0) & (s.recall_windup_ms <= 0) & (s.recall_channel_ms <= 0)
    return Status(garen_e | (c.dash_ms > 0), can_attack, may,
                  ~ready, ready & may[:, None])


def make_params(names=("Garen", "Jax"), patch=None, dtype=jnp.float32):
    from .profiles import build_profile_tables
    if len(names) != 2 or any(n not in IDS for n in names):
        raise ValueError("modern lane requires two champions from Garen/Jax")
    p = build_profile_tables(patch, dtype)
    mapping = {"max_hp": "baseHPModifiable", "hp_per_level": "hpPerLevelModifiable",
               "attack_damage": "baseDamageModifiable", "ad_per_level": "damagePerLevelModifiable",
               "armor": "baseArmorModifiable", "armor_per_level": "armorPerLevelModifiable",
               "magic_resist": "baseMR", "mr_per_level": "mrPerLevel",
               "move_speed": "baseMoveSpeedModifiable", "attack_range": "attackRangeModifiable",
               "attack_speed_per_level": "attackSpeedPerLevelModifiable",
               "hp_regen": "baseStaticHPRegenModifiable", "hp_regen_per_level": "hpRegenPerLevelModifiable"}
    for i, name in enumerate(names):
        for dest, src in mapping.items():
            p[dest] = p[dest].at[i].set(jnp.asarray(stat(name, src),dtype))
        record = champion(name)["character"]
        period = 1000 / stat(name, "attackSpeedModifiable")
        p["attack_period"] = p["attack_period"].at[i].set(jnp.asarray(period,dtype))
        p["attack_windup"] = p["attack_windup"].at[i].set(jnp.asarray(period * (0.3 + record["basicAttack"]["mAttackDelayCastOffsetPercent"]),dtype))
        p["armor_flat_bonus"] = p["armor_flat_bonus"].at[i].set(0)
        p["collision_radius"] = p["collision_radius"].at[i].set(65)
        p["pathfinding_radius"] = p["pathfinding_radius"].at[i].set(35)
    return p


def init_lane(names=("Garen", "Jax"), *, seed=0, patch=None, **kwargs):
    from .init import init_lane as legacy_init
    s = legacy_init(patch=patch, seed=seed, **kwargs)
    p = make_params(names, patch, s.x.dtype)
    c = s.champion.replace(id=s.champion.id.at[:2].set(jnp.asarray([IDS[n] for n in names], jnp.int16)))
    mana = jnp.where(c.id == JAX, jnp.asarray(339.,s.hp.dtype), 0.)
    s = s.replace(modern=True, champion=c.replace(mana=mana, max_mana=mana),
                  hp=s.hp.at[:2].set(p["max_hp"][:2]), max_hp=s.max_hp.at[:2].set(p["max_hp"][:2]))
    return s.replace(spell_level=skill_ranks(s))


def skill_ranks(s, level=None):
    from .spells import ranks_for_level
    # Garen E>Q>W; Jax W>E>Q, taking E,Q,W at levels1,2,3.
    order = (2, 0, 1, 1, 1, 3, 1, 2, 1, 2, 3, 2, 2, 0, 0, 3, 0, 0)
    rows = [[0, 0, 0, 0]]
    for slot in order:
        row = rows[-1].copy(); row[slot] += 1; rows.append(row)
    lv = jnp.clip(s.level if level is None else level, 0, 18)
    g = jnp.asarray([ranks_for_level(i) for i in range(19)], jnp.int8)[lv]
    j = jnp.asarray(rows, jnp.int8)[lv]
    return jnp.where((s.kind == Kind.CHAMPION)[:, None], jnp.where((s.champion.id == JAX)[:, None], j, g), s.spell_level)


def effective_stats(s, params, armor, mr):
    c = s.champion
    stacks = jnp.minimum(c.garen_w_stacks, 150) * .2
    armor = armor + jnp.where(c.id == GAREN, stacks, 0) + jnp.where(c.jax_r_ms > 0, c.jax_r_armor, 0)
    mr = mr + jnp.where(c.id == GAREN, stacks, 0) + jnp.where(c.jax_r_ms > 0, .6 * c.jax_r_armor, 0)
    return armor * jnp.where(c.garen_shred_ms > 0, jnp.asarray(.75,s.hp.dtype), 1), mr


def attack_speed_bonus(s, params):
    c = s.champion
    per_stack = .05 + .015 * jnp.floor((jnp.minimum(s.level, 18).astype(s.hp.dtype) - 1) / 3)
    return c.bonus_as + jnp.where(c.id == JAX, per_stack * c.jax_stacks, 0)


def attack_range_bonus(s):
    return jnp.where(((s.champion.id == GAREN) & s.buffs.q.active)
                     | ((s.champion.id == JAX) & (s.champion.jax_w_ms > 0)), jnp.asarray(50.,s.hp.dtype), 0.)


def apply_casts(s, orders, params, vision=None):
    from .orders import OrderKind, _record_observed_enemy_casts
    n = s.hp.shape[0]; m = orders.kind.shape[0]
    k = jnp.pad(orders.kind, (0, n-m)); target = jnp.pad(orders.target, (0, n-m), constant_values=-1)
    t = jnp.clip(target, 0, n-1); idx = jnp.arange(n)
    st = status(s); r = s.spell_level; c = s.champion; b = s.buffs
    g, j = c.id == GAREN, c.id == JAX
    cast = jnp.stack([k == v for v in (OrderKind.CAST_Q, OrderKind.CAST_W, OrderKind.CAST_E, OrderKind.CAST_R)], -1) & st.can_cast
    distance = jnp.sqrt((s.x-s.x[t])**2 + (s.y-s.y[t])**2)
    valid_q = (target >= 0) & (target != idx) & s.alive[t] & (s.kind[t] != Kind.TURRET) & (s.kind[t] != Kind.NONE) & (distance <= 700 + params['collision_radius'][s.model[t]])
    valid_r = (target >= 0) & s.alive[t] & (s.kind[t] == Kind.CHAMPION) & (s.team[t] != s.team) & (distance <= 400 + params['collision_radius'][s.model[t]])
    cast = cast.at[:, 0].set(cast[:, 0] & (~j | valid_q) & (c.dash_ms <= 0))
    cast = cast.at[:, 3].set(cast[:, 3] & (~g | valid_r))
    q, w, e, rr = [cast[:, z] for z in range(4)]
    ge_start, ge_end = e & g & ~b.e.active, e & g & b.e.active
    je_start, je_end = e & j & (c.jax_e_ms <= 0), e & j & (c.jax_e_ms > 0)
    cd = jnp.where(cast, cooldown_table(c.id, r), s.spell_cooldown)
    # Garen E and Jax E/W cooldown begins at expiration/consumption.
    cd = cd.at[:, 2].set(jnp.where(ge_start | je_start | je_end, 0, cd[:, 2]))
    cd = cd.at[:, 1].set(jnp.where(w & j, 0, cd[:, 1]))
    mana = c.mana - (jnp.where(cast, mana_cost(s), 0)).sum(-1)
    growth = growth_sum(s.level, jnp)
    ad = params['attack_damage'][s.model] + params['ad_per_level'][s.model]*growth + c.bonus_ad
    ticks = 7 + jnp.floor((params['attack_speed_per_level'][s.model]*growth/100 + c.bonus_as) / .25)
    b = b.replace(
        q=b.q.replace(active=jnp.where(q & g, True, b.q.active), elapsed_s=jnp.where(q & g, 0, b.q.elapsed_s), skip_next=jnp.zeros_like(b.q.skip_next)),
        q_haste=b.q_haste.replace(active=b.q_haste.active | (q & g), elapsed_s=jnp.where(q & g, 0, b.q_haste.elapsed_s), rank=jnp.where(q & g, r[:, 0], b.q_haste.rank)),
        w=b.w.replace(active=b.w.active | (w & g), elapsed_s=jnp.where(w & g, 0, b.w.elapsed_s), rank=jnp.where(w & g, r[:, 1], b.w.rank)),
        e=b.e.replace(active=(b.e.active | ge_start) & ~ge_end, elapsed_s=jnp.where(ge_start, 0, b.e.elapsed_s), power=jnp.where(ge_start, ranked('Garen','E','BaseDamagePerTick',r[:,2])+ranked('Garen','E','ADRatioPerTick',r[:,2])*ad,b.e.power)))
    c = c.replace(mana=mana,
        shield=jnp.where(w & g, ranked('Garen','W','BaseShield',r[:,1]) + .18*c.bonus_hp, c.shield),
        shield_ms=jnp.where(w & g, 750, c.shield_ms),
        slow_ms=jnp.where(q & g, 0, c.slow_ms),
        garen_tick_count=jnp.where(ge_start, ticks, c.garen_tick_count),
        garen_ticks=jnp.where(ge_start, 0, c.garen_ticks),
        garen_hits=jnp.where(ge_start[:,None], 0, c.garen_hits),
        jax_w_ms=jnp.where(w & j, 10000, c.jax_w_ms),
        jax_e_ms=jnp.where(je_start, 2000, jnp.where(je_end, 0, c.jax_e_ms)),
        jax_e_dodges=jnp.where(je_start, 0, c.jax_e_dodges),
        jax_e_release=jnp.where(je_end, 1, c.jax_e_release),
        jax_r_cast_ms=jnp.where(rr & j, 250, c.jax_r_cast_ms),
        dash_ms=jnp.where(q & j, jnp.maximum(distance / 1400 * 1000, 1), c.dash_ms),
        dash_target=jnp.where(q & j, target, c.dash_target),
        dash_target_seq=jnp.where(q & j, s.spawn_seq[t], c.dash_target_seq),
        r_target=jnp.where(rr & g, target, c.r_target),
        r_target_seq=jnp.where(rr & g, s.spawn_seq[t], c.r_target_seq))
    reset = (q & g) | (w & j)
    success = cast[:2]
    return s.replace(champion=c, buffs=b, spell_cooldown=cd,
        r_cast_ms=jnp.where(rr & g, 435, s.r_cast_ms),
        aa_cooldown=jnp.where(reset,0,s.aa_cooldown), aa_windup=jnp.where(reset,0,s.aa_windup),
        is_attacking=s.is_attacking & ~reset,
        recall_channel_ms=jnp.where(cast.any(-1),0,s.recall_channel_ms),
        observed_enemy_cast_ms=_record_observed_enemy_casts(s, success, vision))


def advance(s, params, armor, mr, dt):
    """Buff clocks, jumps and spell damage; returns state and BuffStep adapter."""
    from .spells import BuffStep
    c=s.champion; b=s.buffs; r=s.spell_level; n=s.hp.shape[0]; dtype=s.hp.dtype
    g,j=c.id==GAREN,c.id==JAX
    dec=lambda v:jnp.maximum(0,v-dt)
    rankcd=cooldown_table(c.id,r); cd=dec(s.spell_cooldown*1000)/1000
    # Regeneration is continuous in the modern champion layer. The world still
    # supplies its ordinary stat regen for every unit.
    lv=s.level.astype(dtype)
    rate=(1.5 + .2*jnp.minimum(lv-1,5) + .8*jnp.clip(lv-6,0,7) + .4*jnp.maximum(lv-13,0)) / 500
    hp=jnp.where(g & s.alive & (s.ms_since_damaged>=8000),jnp.minimum(s.max_hp,s.hp+s.max_hp*rate*dt/1000),s.hp)
    maxmana=jnp.where(j,339+52*growth_sum(s.level,jnp),0)
    mana=jnp.minimum(maxmana,c.mana+jnp.maximum(0,maxmana-c.max_mana)+jnp.where(j & s.alive,(1.64+.14*growth_sum(s.level,jnp))*dt/1000,0))
    # Fixed-duration targeted leap. Target identity prevents a recycled minion
    # slot becoming the recipient of a pending strike.
    t=jnp.clip(c.dash_target.astype(jnp.int32),0,n-1)
    dashing=(c.dash_ms>0)&s.alive&s.alive[t]&(s.spawn_seq[t]==c.dash_target_seq)
    fraction=jnp.minimum(1,dt/jnp.maximum(c.dash_ms,dt))
    x=jnp.where(dashing,s.x+(s.x[t]-s.x)*fraction,s.x)
    y=jnp.where(dashing,s.y+(s.y[t]-s.y)*fraction,s.y)
    arrived=dashing & (c.dash_ms<=dt)
    valid=arrived & s.alive[t] & (s.spawn_seq[t]==c.dash_target_seq) & (s.team[t]!=s.team)
    d2=(x[:,None]-x[None,:])**2+(y[:,None]-y[None,:])**2
    enemy=s.alive[None,:] & (s.team[:,None]!=s.team[None,:]) & (s.kind[None,:]!=Kind.NONE) & (s.kind[None,:]!=Kind.TURRET)
    reach=params['collision_radius'][s.model]
    within=lambda radius: enemy & (d2<=(radius+reach[None,:])**2)
    physical=jnp.zeros((n,n),dtype); magic=jnp.zeros_like(physical); true=jnp.zeros_like(physical)
    physical=physical.at[jnp.arange(n),t].add(jnp.where(valid,ranked('Jax','Q','Damage',r[:,0])+c.bonus_ad,0))
    w_on_q=valid & (c.jax_w_ms>0)
    magic=magic.at[jnp.arange(n),t].add(jnp.where(w_on_q,ranked('Jax','W','Damage',r[:,1])+.6*c.ap,0))
    # Spin ticks are on a fixed cast-time grid; no extra tick at expiry.
    elapsed=b.e.elapsed_s+dt/1000
    due=jnp.minimum(c.garen_tick_count,jnp.floor((elapsed-dt/1000)*c.garen_tick_count/3)+1)
    fires=b.e.active & g & s.alive & (c.garen_ticks<due) & (c.garen_ticks<c.garen_tick_count)
    hit=fires[:,None] & within(325)
    nearest=jnp.argmin(jnp.where(within(325),d2,jnp.inf),axis=1)
    bonus=jnp.where(jnp.arange(n)[None,:]==nearest[:,None],jnp.asarray(1.25,dtype),1.)
    # Crit RNG is explicit, deterministic under replay, independent per spin.
    key,sub=jax.random.split(s.key); crit=jax.random.uniform(sub,(n,))<c.crit
    physical+=jnp.where(hit,b.e.power[:,None]*bonus*jnp.where(crit[:,None],jnp.asarray(1.3,dtype),1),0)
    hits=c.garen_hits+hit.astype(jnp.int16)
    shred=(hit & (s.kind[None,:]==Kind.CHAMPION) & ((hits==6)|(hits==7)|((hits>7)&((hits-7)%6==0)))).any(0)
    e_end=b.e.active & ((elapsed>=3-1e-6)|~s.alive)
    cd=cd.at[:,2].set(jnp.where(e_end,rankcd[:,2],cd[:,2]))
    je_end=((c.jax_e_ms>0)&(c.jax_e_ms<=dt))|(c.jax_e_release>0)
    je_hit=je_end[:,None]&j[:,None]&s.alive[:,None]&within(375)
    je_raw=(ranked('Jax','E','BaseDamage',r[:,2])+.7*c.ap)[:,None]+.04*s.max_hp[None,:]
    magic+=jnp.where(je_hit,je_raw*(1+.2*jnp.minimum(c.jax_e_dodges,5))[:,None],0)
    stun=je_hit.any(0)*1000*jnp.where((c.shield_ms>0)&g,jnp.asarray(.4,dtype),1.)
    cd=cd.at[:,2].set(jnp.where(je_end,rankcd[:,2],cd[:,2]))
    jr_fire=(c.jax_r_cast_ms>0)&(c.jax_r_cast_ms<=dt)&s.alive
    jr_hit=jr_fire[:,None]&within(375)
    magic+=jnp.where(jr_hit,(ranked('Jax','R','SwingDamageBase',r[:,3])+c.ap)[:,None],0)
    champions_hit=(jr_hit & (s.kind[None,:]==Kind.CHAMPION)).sum(1)
    rarmor=ranked('Jax','R','BaseResists',r[:,3])+.4*c.bonus_ad+jnp.maximum(champions_hit-1,0)*(ranked('Jax','R','ResistsPerExtraTarget',r[:,3])+.1*c.bonus_ad)
    rt=jnp.clip(c.r_target.astype(jnp.int32),0,n-1)
    gr_fire=g & (s.r_cast_ms>0)&(s.r_cast_ms<=dt)&s.alive&s.alive[rt]&(s.spawn_seq[rt]==c.r_target_seq)
    rawr=ranked('Garen','R','BaseDamage',r[:,3])+ranked('Garen','R','ExecuteDamage',r[:,3])*jnp.maximum(s.max_hp[rt]-s.hp[rt],0)
    true=true.at[jnp.arange(n),rt].add(jnp.where(gr_fire,rawr,0))
    # E and R AoE are reduced by Counter Strike. Q leap is single-target.
    qmagic=jnp.zeros_like(magic).at[jnp.arange(n),t].add(jnp.where(w_on_q,ranked('Jax','W','Damage',r[:,1])+.6*c.ap,0))
    qphysical=jnp.zeros_like(physical).at[jnp.arange(n),t].add(jnp.where(valid,ranked('Jax','Q','Damage',r[:,0])+c.bonus_ad,0))
    aoe_reduction=jnp.where(c.jax_e_ms>0,jnp.asarray(.75,dtype),1)[None,:]
    dealt=post_mitigation_damage((physical-qphysical)*aoe_reduction+qphysical,armor[None,:],jnp)+post_mitigation_damage((magic-qmagic)*aoe_reduction+qmagic,mr[None,:],jnp)
    dr=jnp.where(b.w.active & g,1-ranked('Garen','W','DRPercent',b.w.rank),1)
    dealt=dealt*dr[None,:]+true
    damage=dealt.sum(0)
    source=jnp.where(damage>0,jnp.argmax(dealt,axis=0),-1).astype(jnp.int8)
    w_end=(c.jax_w_ms>0)&((c.jax_w_ms<=dt)|w_on_q|~s.alive)
    cd=cd.at[:,1].set(jnp.where(w_end,rankcd[:,1],cd[:,1]))
    stack_expired=(c.jax_stack_ms<=dt)&(c.jax_stacks>0)
    c=c.replace(mana=mana,max_mana=maxmana,
        shield=jnp.where(c.shield_ms>dt,c.shield,0),shield_ms=dec(c.shield_ms),
        stun_ms=jnp.maximum(dec(c.stun_ms),stun),slow_ms=dec(c.slow_ms),
        garen_ticks=c.garen_ticks+fires, garen_hits=hits,
        garen_shred_ms=jnp.where(shred,6000,dec(c.garen_shred_ms)),
        jax_e_ms=jnp.where(je_end,0,dec(c.jax_e_ms)),jax_e_release=jnp.zeros_like(c.jax_e_release),
        jax_w_ms=jnp.where(w_end,0,dec(c.jax_w_ms)),
        jax_r_ms=jnp.where(champions_hit>0,8000,dec(c.jax_r_ms)),
        jax_r_armor=jnp.where(champions_hit>0,rarmor,c.jax_r_armor),
        jax_r_cast_ms=dec(c.jax_r_cast_ms),
        jax_r_hits=jnp.where(c.jax_r_hit_ms<=dt,0,c.jax_r_hits),jax_r_hit_ms=dec(c.jax_r_hit_ms),
        jax_stacks=jnp.maximum(0,c.jax_stacks-stack_expired),
        jax_stack_ms=jnp.where(stack_expired,350,dec(c.jax_stack_ms)),
        dash_ms=dec(c.dash_ms))
    b=b.replace(e=b.e.replace(active=b.e.active&~e_end,elapsed_s=elapsed),
        q=b.q.replace(active=b.q.active&(b.q.elapsed_s+dt/1000<4.5)&s.alive,elapsed_s=b.q.elapsed_s+dt/1000),
        q_haste=b.q_haste.replace(active=b.q_haste.active&(b.q_haste.elapsed_s+dt/1000<ranked('Garen','Q','MovementSpeedDuration',b.q_haste.rank))&s.alive,elapsed_s=b.q_haste.elapsed_s+dt/1000),
        w=b.w.replace(active=b.w.active&(b.w.elapsed_s+dt/1000<4)&s.alive,elapsed_s=b.w.elapsed_s+dt/1000))
    s=s.replace(champion=c,buffs=b,spell_cooldown=cd,x=x,y=y,hp=hp,key=key)
    zero=jnp.zeros_like(hp)
    bs=BuffStep(b,cd,damage,source,jnp.ones_like(hp),zero,zero,zero,zero,b.q.active,jnp.zeros(n,bool))
    return s,bs


def auto_hits(s, params, aa, target, armor, mr, buffs):
    """Resolve dodge before on-hit effects, and keep physical/magic distinct."""
    c=s.champion; n=s.hp.shape[0]; j=c.id==JAX; g=c.id==GAREN
    dodge=aa.hit & (params["fires_missile"][s.model]<=0) & (c.jax_e_ms[target]>0) & (s.kind!=Kind.TURRET)
    landed=aa.hit & ~dodge
    q=landed & g & buffs.q.active
    w=landed & j & (c.jax_w_ms>0)
    rproc=landed & j & (s.spell_level[:,3]>0) & (c.jax_r_hits>=jnp.where(c.jax_r_ms>0,1,2))
    damage=jnp.where(dodge,0,aa.damage)
    magic=jnp.where(w,ranked('Jax','W','Damage',s.spell_level[:,1])+.6*c.ap,0)+jnp.where(rproc,ranked('Jax','R','PassiveBaseDamage',s.spell_level[:,3])+.6*c.ap,0)
    magic*=jnp.where(s.kind[target]==Kind.TURRET,jnp.asarray(.5,s.hp.dtype),1)
    damage+=post_mitigation_damage(magic,mr[target],jnp)
    started=aa.hit & j
    c=c.replace(jax_stacks=jnp.where(started,jnp.minimum(8,c.jax_stacks+1),c.jax_stacks),
        jax_stack_ms=jnp.where(started,2500,c.jax_stack_ms),
        jax_e_dodges=c.jax_e_dodges+jnp.zeros_like(s.hp).at[target].add(dodge.astype(s.hp.dtype)),
        jax_w_ms=jnp.where(w,0,c.jax_w_ms),
        jax_r_hits=jnp.where(landed & j & (s.spell_level[:,3]>0),jnp.where(rproc,0,c.jax_r_hits+1),c.jax_r_hits),
        jax_r_hit_ms=jnp.where(landed & j,2500,c.jax_r_hit_ms))
    cd=s.spell_cooldown.at[:,1].set(jnp.where(w,cooldown_table(c.id,s.spell_level)[:,1],s.spell_cooldown[:,1]))
    # A dodged Garen Q consumes the attack empowerment but applies no silence.
    buffs=buffs.replace(q=buffs.q.replace(active=buffs.q.active & ~(aa.hit & g)))
    return s.replace(champion=c,spell_cooldown=cd), damage, q, buffs


def finish(s, previous_alive, died, reborn, killer):
    c=s.champion; n=s.hp.shape[0]
    earned=jnp.zeros_like(s.hp).at[jnp.clip(killer,0,n-1)].add((died & (killer>=0)).astype(s.hp.dtype))
    clear=died|reborn
    c=c.replace(garen_w_stacks=jnp.minimum(150,c.garen_w_stacks+jnp.where(c.id==GAREN,earned,0)),
        stun_ms=jnp.where(clear,0,c.stun_ms),shield=jnp.where(clear,0,c.shield),shield_ms=jnp.where(clear,0,c.shield_ms),
        dash_ms=jnp.where(clear,0,c.dash_ms),jax_e_ms=jnp.where(clear,0,c.jax_e_ms),jax_w_ms=jnp.where(clear,0,c.jax_w_ms),
        jax_r_ms=jnp.where(clear,0,c.jax_r_ms),jax_r_cast_ms=jnp.where(clear,0,c.jax_r_cast_ms),
        jax_stacks=jnp.where(clear,0,c.jax_stacks),jax_r_hits=jnp.where(clear,0,c.jax_r_hits),
        mana=jnp.where(reborn,c.max_mana,c.mana))
    cd=s.spell_cooldown
    ranks=cooldown_table(c.id,s.spell_level)
    cd=cd.at[:,2].set(jnp.where(died & (s.buffs.e.active | (s.champion.jax_e_ms>0)),ranks[:,2],cd[:,2]))
    cd=cd.at[:,1].set(jnp.where(died & (s.champion.jax_w_ms>0),ranks[:,1],cd[:,1]))
    b=s.buffs
    b=b.replace(e=b.e.replace(active=b.e.active & ~clear),q=b.q.replace(active=b.q.active & ~clear),
                q_haste=b.q_haste.replace(active=b.q_haste.active & ~clear),w=b.w.replace(active=b.w.active & ~clear))
    return s.replace(champion=c,buffs=b,spell_cooldown=cd)
