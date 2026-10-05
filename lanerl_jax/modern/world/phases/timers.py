"""Phase 10, TIMERS: respawn and recall, cooldowns, mana/HP regen, fountain, inventory outputs, wards,
terrain ejection. Writes next-state columns to the scratch, not ``s``."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ... import economy as E
from ... import wards as WD
from ...core import types as W
from ...core.stat_pipeline import cooldown
from ...items import inventory as I
from ...items.catalog import catalog
from ...map import dynamic_terrain as DTR
from .. import views as V
from ..config import N_CHAMPIONS, WorldConfig
from ..config import layout as MW_layout
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """10. TIMERS: respawn, recall, cooldowns, mana/HP regen, fountain, inventory outputs, wards, ward units,
    terrain ejection. Leaves ``s`` untouched: the next-state columns go to the scratch for FOG/_commit."""
    c, dt = N_CHAMPIONS, cfg.dt
    cat = catalog()
    now, caps, champ, st, out, kit_all, extra = sc.now, sc.caps, sc.champ, sc.st, sc.out, sc.kit_all, sc.extra
    hp, max_hp, eco, econ, jrw, in_f, died = sc.hp, sc.max_hp, sc.eco, sc.econ, sc.jrw, sc.in_f, sc.died
    s0, s_out, in_stasis = sc.s0, sc.s_out, sc.in_stasis
    x, y = s.x, s.y                                                   # MOVE's positions
    alive = s.alive & ~died
    champ_dead = ~alive[:c]
    respawn = eco.respawned
    recall = eco.recalled
    fx_, fy_ = cfg.fountain[s.team[:c], 0], cfg.fountain[s.team[:c], 1]
    x = x.at[:c].set(jnp.where(respawn | recall, fx_, x[:c]))
    y = y.at[:c].set(jnp.where(respawn | recall, fy_, y[:c]))
    alive = alive.at[:c].set(jnp.where(respawn, True, alive[:c]))
    hp = hp.at[:c].set(jnp.where(respawn, max_hp[:c], hp[:c]))
    # Cooldowns: kit starts (hasted), refunds from runes, decrement.
    haste = jnp.stack([st.basic_ability_haste] * 3 + [st.ultimate_haste], -1)
    aw = sc.item_cleanse                                              # item-active world effects
    cd_rate = jnp.concatenate([jnp.broadcast_to(aw.basic_cd_rate[:, None], (c, 3)), jnp.ones((c, 1))], axis=1)
    cds = jnp.maximum(champ.cooldowns - dt * cd_rate, 0.0)
    cds = jnp.where(kit_all.cooldown_start, cooldown(kit_all.base_cooldown, haste), cds)
    ro = out.rune_outputs
    cds = cds.at[:, :3].multiply((1.0 - ro.basic_cd_refund)[:, None]).at[:, 3].multiply(1.0 - ro.ult_cd_refund)
    mana = jnp.clip(champ.mana - kit_all.mana_cost * aw.mana_cost_mult + out.effects.mana + extra.mana
                    + st.mana_regen * dt,
                    0.0, st.max_mana)
    mana = jnp.where(respawn, st.max_mana, mana)
    # HP regen (0.5 s ticks, §12) and the fountain (+ Homeguard heal).
    pulse = jnp.floor(now / 0.5 + 1e-6) - jnp.floor(s.t / 0.5 + 1e-6)
    regen = jnp.where(alive[:c] & ~respawn, st.hp_regen * 0.5 * pulse, 0.0)
    hp_c = jnp.minimum(hp[:c] + regen, max_hp[:c])
    hp_c, mana = E.fountain_regen(hp_c, max_hp[:c], mana, st.max_mana, in_f & alive[:c], s.t, now,
                                  homeguard=econ.homeguard.active)
    hp = hp.at[:c].set(hp_c)
    # Item outputs: Tear transforms, consumption, rune item grants.
    inv = champ.inventory
    frm, to, do = out.transforms
    inv = I.Inventory(*jax.vmap(lambda it, stk, f_, t_, d_: tuple(I.replace_item(I.Inventory(it, stk), f_, t_, d_)))(
        inv.item, inv.stack, frm, to, do))
    frm2, to2, do2 = aw.transform                                     # Seeker's -> Shattered Armguard
    inv = I.Inventory(*jax.vmap(lambda it, stk, f_, t_, d_: tuple(I.replace_item(I.Inventory(it, stk), f_, t_, d_)))(
        inv.item, inv.stack, frm2, to2, do2))
    consume = lambda inv, row, ok: I.Inventory(*jax.vmap(                                  # noqa: E731
        lambda it, stk, r, o: tuple(I.consume_one(I.Inventory(it, stk), jnp.argmax(it == r), o & jnp.any(it == r))))(
        inv.item, inv.stack, row, ok))
    inv = consume(inv, out.consume_row, out.consume_row >= 0)
    if jrw is not None:                                               # final pet evolution consumes the egg
        pet_rows = jnp.asarray([cat.row(i) for i in (1101, 1102, 1103)], jnp.int32)
        held_pet = jnp.max(jnp.where(jnp.isin(inv.item, pet_rows), inv.item, -1), axis=1)
        inv = consume(inv, held_pet, jrw.consume_pet & (held_pet >= 0))
    # Wards and trinkets (wards): placement, hits, expiry, rewards, Control Ward use.
    lay = MW_layout()
    w0, wn = lay["ward0"], 2 * W.MAX_WARDS_PER_TEAM
    ids = jnp.asarray(cat.arrays.item_id)
    trow = inv.item[:, 6]
    trinket_id = jnp.where(trow >= 0, ids[jnp.clip(trow, 0, ids.shape[0] - 1)], 0)
    cw_row = cat.row(2055)
    control_count = jnp.sum(jnp.where(inv.item == cw_row, inv.stack, 0), axis=1)
    wreq = WD.WardRequest(kind=-jnp.ones((c,), jnp.int32) if orders.ward_kind is None else orders.ward_kind,
                          x=jnp.zeros((c,)) if orders.ward_x is None else orders.ward_x,
                          y=jnp.zeros((c,)) if orders.ward_y is None else orders.ward_y)
    wards, wev = WD.ward_step(s.wards, now=now, dt=jnp.float32(dt), request=wreq, x=x[:c], y=y[:c], team=s.team[:c],
                              alive=alive[:c], level=econ.level, trinket_id=trinket_id, control_count=control_count,
                              grid=cfg.ward_grid, can_use=alive[:c] & ~caps["stunned"][:c] & ~in_stasis,
                              trinket_haste=st.item_haste + st.trinket_haste, hits=sc.ward_hits,
                              hitter=sc.ward_hitter, rune_pages=cfg.rune_pages, ward_visible=s0.visible[:, w0:w0 + wn])
    econ = econ._replace(gold=econ.gold + wev.gold, gold_total=econ.gold_total + wev.gold, xp=econ.xp + wev.xp)
    inv = consume(inv, jnp.full((c,), cw_row, jnp.int32), wev.consumed_control)
    grant_row = jnp.argmax(jnp.asarray(cat.arrays.item_id)[None, :] == ro.grant_item[:, None], axis=1)
    free = (inv.item[:, :6] < 0)
    can_grant = (ro.grant_item > 0) & jnp.any(free, axis=1)
    gslot = jnp.argmax(free, axis=1)
    put = can_grant[:, None] & (jnp.arange(7)[None, :] == gslot[:, None])
    inv = I.Inventory(jnp.where(put, grant_row[:, None], inv.item).astype(jnp.int32),
                      jnp.where(put, 1, inv.stack).astype(jnp.int32))
    lock = jnp.maximum(champ.cast_lock_until, jnp.where(kit_all.cast_started, now + kit_all.cast_lockout, 0.0))
    act = out.active                                                  # item actives: Hydra casts (cast time)
    lock = jnp.maximum(lock, jnp.where(act.used & ~act.can_move, now + act.cast_time, 0.0))
    item_lock = jnp.maximum(champ.item_cast_until, jnp.where(act.used & act.can_move, now + act.cast_time, 0.0))
    cast_now = kit_all.cast_started[:, None] & (jnp.arange(4)[None, :] == kit_all.cast_slot[:, None])
    last_dmg = jnp.where(sc.took_health[:c], now, champ.last_damaged)
    champ = champ._replace(
        cooldowns=cds, mana=mana, cast_lock_until=lock, item_cast_until=item_lock, last_damaged=last_dmg, inventory=inv,
        dyn=out.dynamic_stats, homeguard_ms=eco.homeguard_ms, blinked=s_out.blinked | sc.dstart,
        forbid=ro.forbid_purchase, reset_next=out.effects.attack_reset,
        bonus_points=champ.bonus_points + ro.skill_points, granted=jnp.where(can_grant, ro.grant_item, 0),
        cs=champ.cs + eco.kills.minion_kill.astype(jnp.int32),
        last_cast=jnp.where(cast_now, now, champ.last_cast),
        moving=jnp.where(respawn | recall | champ_dead, False, champ.moving),
        attack_order=jnp.where(respawn | recall | champ_dead, -1,
                               jnp.where(V.kit_attack_target(kit_all) >= 0, V.kit_attack_target(kit_all), champ.attack_order)))
    kills_next = eco.kills
    # Dead minions free their slot; dead champions keep theirs.
    kind = jnp.where(sc.minion_died, W.KIND_NONE, s.kind)
    # Ward units mirror the ward slots.
    view, _ = WD.ward_view(wards, now=now, x=x[:c], y=y[:c], team=s.team[:c], alive=alive[:c], level=econ.level)
    wsl = slice(w0, w0 + wn)
    kind = kind.at[wsl].set(jnp.where(view.alive, W.KIND_WARD, W.KIND_NONE))
    alive = alive.at[wsl].set(view.alive)
    x, y = x.at[wsl].set(view.x), y.at[wsl].set(view.y)
    hp, max_hp = hp.at[wsl].set(view.hp), max_hp.at[wsl].set(view.max_hp)
    w_sub = s.sub.at[wsl].set(view.sub)
    w_team = s.team.at[wsl].set(view.team)
    w_seq = s.spawn_seq.at[wsl].set((1 << 24) + wards.slots.seq)
    w_radius = s.radius.at[wsl].set(1.0)
    w_targ = s.targetable.at[wsl].set(view.alive)
    # Units left inside terrain that closed (inhibitor respawn, Rift transformation) step out. The test
    # disk is the movement clearance (route radius), the disk the MOVE clamp keeps walkable: the full
    # unit radius pulled champions walking along a structure pad back every tick.
    mobile = alive & ((kind == W.KIND_CHAMPION) | (kind == W.KIND_MINION))
    x, y = DTR.eject(x, y, jnp.clip(s.team, 0, 1), jnp.minimum(s.radius, cfg.routes.radius), sc.terrain,
                     active=mobile)
    return s, sc._replace(alive=alive, x=x, y=y, hp=hp, max_hp=max_hp, champ=champ, econ=econ, wards=wards,
                          kills=kills_next, kind=kind, sub=w_sub, team=w_team, spawn_seq=w_seq, radius=w_radius,
                          targetable=w_targ, cast_now=cast_now)
