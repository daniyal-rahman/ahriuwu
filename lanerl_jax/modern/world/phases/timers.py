"""Phase 10, TIMERS: respawn and recall, cooldowns, mana/HP regen, fountain, inventory outputs, wards and ward
units, terrain ejection. Writes next-state columns to the scratch, not ``s``."""
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
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    c, dt = N_CHAMPIONS, cfg.dt
    cat = catalog()
    now, champ, st, out, kit_all = sc.now, sc.champ, sc.st, sc.out, sc.kit_all
    hp, max_hp, eco, econ, jrw = sc.hp, sc.max_hp, sc.eco, sc.econ, sc.jrw
    x, y = s.x, s.y                                                   # MOVE's positions
    alive = s.alive & ~sc.died
    champ_dead = ~alive[:c]
    respawn = eco.respawned
    recall = eco.recalled
    fx_, fy_ = cfg.fountain[s.team[:c], 0], cfg.fountain[s.team[:c], 1]
    x = x.at[:c].set(jnp.where(respawn | recall, fx_, x[:c]))
    y = y.at[:c].set(jnp.where(respawn | recall, fy_, y[:c]))
    alive = alive.at[:c].set(jnp.where(respawn, True, alive[:c]))
    hp = hp.at[:c].set(jnp.where(respawn, max_hp[:c], hp[:c]))
    # Cooldowns: decrement, kit starts (hasted), rune refunds.
    haste = jnp.stack([st.basic_ability_haste] * 3 + [st.ultimate_haste], -1)
    aw = sc.item_cleanse
    cd_rate = jnp.concatenate([jnp.broadcast_to(aw.basic_cd_rate[:, None], (c, 3)), jnp.ones((c, 1))], axis=1)
    cds = jnp.maximum(champ.cooldowns - dt * cd_rate, 0.0)
    cds = jnp.where(kit_all.cooldown_start, cooldown(kit_all.base_cooldown, haste), cds)
    ro = out.rune_outputs
    cds = cds.at[:, :3].multiply((1.0 - ro.basic_cd_refund)[:, None]).at[:, 3].multiply(1.0 - ro.ult_cd_refund)
    mana = jnp.clip(champ.mana - kit_all.mana_cost * aw.mana_cost_mult + out.effects.mana + sc.extra.mana
                    + st.mana_regen * dt, 0.0, st.max_mana)
    mana = jnp.where(respawn, st.max_mana, mana)
    # HP regen in 0.5 s pulses (§12), then the fountain and Homeguard heal.
    pulse = jnp.floor(now / 0.5 + 1e-6) - jnp.floor(s.t / 0.5 + 1e-6)
    regen = jnp.where(alive[:c] & ~respawn, st.hp_regen * 0.5 * pulse, 0.0)
    hp_c = jnp.minimum(hp[:c] + regen, max_hp[:c])
    hp_c, mana = E.fountain_regen(hp_c, max_hp[:c], mana, st.max_mana, sc.in_f & alive[:c], s.t, now,
                                  homeguard=econ.homeguard.active)
    hp = hp.at[:c].set(hp_c)
    replace = lambda inv, frm, to, do: I.Inventory(*jax.vmap(                               # noqa: E731
        lambda it, stk, f_, t_, d_: tuple(I.replace_item(I.Inventory(it, stk), f_, t_, d_)))(
        inv.item, inv.stack, frm, to, do))
    inv = replace(champ.inventory, *out.transforms)                  # Tear
    inv = replace(inv, *aw.transform)                                 # Seeker's -> Shattered Armguard
    consume = lambda inv, row, ok: I.Inventory(*jax.vmap(                                  # noqa: E731
        lambda it, stk, r, o: tuple(I.consume_one(I.Inventory(it, stk), jnp.argmax(it == r), o & jnp.any(it == r))))(
        inv.item, inv.stack, row, ok))
    inv = consume(inv, out.consume_row, out.consume_row >= 0)
    if jrw is not None:                                               # final pet evolution consumes the egg
        pet_rows = jnp.asarray([cat.row(i) for i in (1101, 1102, 1103)], jnp.int32)
        held_pet = jnp.max(jnp.where(jnp.isin(inv.item, pet_rows), inv.item, -1), axis=1)
        inv = consume(inv, held_pet, jrw.consume_pet & (held_pet >= 0))
    lay = cfg.layout
    w0, wn = lay.ward0, 2 * W.MAX_WARDS_PER_TEAM
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
                              grid=cfg.ward_grid, can_use=alive[:c] & ~sc.caps["stunned"][:c] & ~sc.in_stasis,
                              trinket_haste=st.item_haste + st.trinket_haste, hits=sc.ward_hits,
                              hitter=sc.ward_hitter, rune_pages=cfg.rune_pages,
                              ward_visible=sc.s0.visible[:, w0:w0 + wn])
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
    act = out.active                                                  # item-active cast times (Hydra)
    lock = jnp.maximum(lock, jnp.where(act.used & ~act.can_move, now + act.cast_time, 0.0))
    item_lock = jnp.maximum(champ.item_cast_until, jnp.where(act.used & act.can_move, now + act.cast_time, 0.0))
    cast_now = kit_all.cast_started[:, None] & (jnp.arange(4)[None, :] == kit_all.cast_slot[:, None])
    last_dmg = jnp.where(sc.took_health[:c], now, champ.last_damaged)
    champ = champ._replace(
        cooldowns=cds, mana=mana, cast_lock_until=lock, item_cast_until=item_lock, last_damaged=last_dmg, inventory=inv,
        dyn=out.dynamic_stats, homeguard_ms=eco.homeguard_ms, blinked=sc.s_out.blinked | sc.dstart,
        forbid=ro.forbid_purchase, reset_next=out.effects.attack_reset,
        bonus_points=champ.bonus_points + ro.skill_points, granted=jnp.where(can_grant, ro.grant_item, 0),
        cs=champ.cs + eco.kills.minion_kill.astype(jnp.int32),
        last_cast=jnp.where(cast_now, now, champ.last_cast),
        moving=jnp.where(respawn | recall | champ_dead, False, champ.moving),
        attack_order=jnp.where(respawn | recall | champ_dead, -1,
                               jnp.where(V.kit_attack_target(kit_all) >= 0, V.kit_attack_target(kit_all),
                                         champ.attack_order)))
    kind = jnp.where(sc.minion_died, W.KIND_NONE, s.kind)            # dead minions free their slot
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
    # Units inside terrain that closed (inhibitor respawn, Rift transformation) step out; the test disk is the
    # movement clearance the MOVE clamp keeps walkable.
    mobile = alive & ((kind == W.KIND_CHAMPION) | (kind == W.KIND_MINION))
    x, y = DTR.eject(x, y, jnp.clip(s.team, 0, 1), jnp.minimum(s.radius, cfg.routes.radius), sc.terrain,
                     active=mobile)
    return s, sc._replace(alive=alive, x=x, y=y, hp=hp, max_hp=max_hp, champ=champ, econ=econ, wards=wards,
                          kills=eco.kills, kind=kind, sub=w_sub, team=w_team, spawn_seq=w_seq, radius=w_radius,
                          targetable=w_targ, cast_now=cast_now)
