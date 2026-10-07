// One combat tick of every item and rune effect around the world's packets (combat.py) and the item-side
// folding into the damage pipeline (items/effects/runtime.py).
#include "combat.hpp"

#include <algorithm>
#include <numeric>

#include "damage.hpp"
#include "dispatch.hpp"
#include "marshal.hpp"
#include "stats.hpp"

namespace lanesim::champ {

namespace {
constexpr int CARRY_CAPACITY = 64, EXTRA_ON_HIT_SLOTS = 2;
constexpr float CHAMPION_COMBAT_GAP = 10.f;
}  // namespace

// --- runtime.py --------------------------------------------------------------------------------------------------
Defense fold_defense(const Defense& base, const Ctx& ctx, const HolderDefense& holder, const Debuffs& debuffs,
                     float shield_power_unused, const Arr<float>& shield_power, const Arr<float>& incoming_heal) {
    Defense d = base;
    for (size_t c = 0; c < ctx.unit.size(); ++c) {
        int u = ctx.unit[c];
        d.received_mult[u] = d.received_mult[u] * holder.received_mult[c];
        d.basic_attack_mult[u] = d.basic_attack_mult[u] * holder.basic_attack_mult[c];
        d.crit_taken_mult[u] = d.crit_taken_mult[u] * holder.crit_taken_mult[c];
        d.champion_attack_block[u] = d.champion_attack_block[u] + holder.champion_attack_block[c];
        d.postmit_flat[u] = d.postmit_flat[u] + holder.postmit_flat[c];
        d.store_fraction[u] = d.store_fraction[u] + holder.store_fraction[c];
        d.lifeline_ready[u] = holder.lifeline_ready[c];
        d.lifeline_magic_only[u] = holder.lifeline_magic_only[c];
        d.lifeline_shield[u] = holder.lifeline_shield[c] * (1.f + shield_power[c]) * (1.f + incoming_heal[c]);
        d.champion_received_mult[u] = d.champion_received_mult[u] * holder.champion_received_mult[c];
        d.lifeline_shield_kind[u] = holder.lifeline_shield_kind[c];
        d.lifeline_duration[u] = holder.lifeline_duration[c];
        d.lifeline_decay_hold[u] = holder.lifeline_decay_hold[c];
        d.lifeline_bonus_health[u] = holder.lifeline_bonus_health[c];
        d.spell_shield[u] = d.spell_shield[u] | holder.spell_shield[c];
    }
    for (size_t j = 0; j < d.armor.size(); ++j) {
        d.percent_armor_reduction[j] = 1.f - (1.f - d.percent_armor_reduction[j]) * (1.f - debuffs.percent_armor_reduction[j]);
        d.flat_armor_reduction[j] = d.flat_armor_reduction[j] + debuffs.flat_armor_reduction[j];
        d.percent_mr_reduction[j] = 1.f - (1.f - d.percent_mr_reduction[j]) * (1.f - debuffs.percent_mr_reduction[j]);
        d.flat_mr_reduction[j] = d.flat_mr_reduction[j] + debuffs.flat_mr_reduction[j];
        d.received_amp[j] = d.received_amp[j] + debuffs.received_amp[j];
        d.magic_received_amp[j] = d.magic_received_amp[j] + debuffs.magic_received_amp[j];
    }
    return d;
}

Offense fold_offense(const Offense& base, const Ctx& ctx, const ItemStats& st) {
    Offense o = base;
    for (size_t c = 0; c < ctx.unit.size(); ++c) {
        int u = ctx.unit[c];
        o.lethality[u] = o.lethality[u] + st.lethality[c];
        o.percent_armor_pen[u] = 1.f - (1.f - o.percent_armor_pen[u]) * (1.f - st.percent_armor_pen[c]);
        o.magic_pen[u] = o.magic_pen[u] + st.magic_pen[c];
        o.percent_magic_pen[u] = 1.f - (1.f - o.percent_magic_pen[u]) * (1.f - st.percent_magic_pen[c]);
    }
    return o;
}

Report resolve_tick(const Packets& p, const Offense& off, const Defense& dfn, const Arr<float>& hp,
                    const Arr<float>& max_hp, const Shields& shields, float now, const Vamp& vamp) {
    Report r;
    r.packets = p;
    r.resolved = damage::resolve(p, off, dfn, hp, max_hp, shields, now);
    damage::vamp_heal_split(p, r.resolved, vamp, dfn.unit_class, nullptr, r.life_steal_heal, r.omnivamp_heal);
    return r;
}

void apply_effects(const Effects& eff, const Ctx& ctx, Arr<float>& hp, const Arr<float>& max_hp, Shields& shields,
                   UnitStatus& status, const Arr<float>& heal_power, const Arr<float>& incoming_heal,
                   const Arr<float>* vamp_heal, const Arr<float>* heal_mult) {
    const size_t c = ctx.unit.size(), n = hp.size();
    const float now = ctx.now;
    for (size_t h = 0; h < c; ++h) {
        int u = ctx.unit[h];
        bool gw = status.grievous_until[u] > now;
        bool alive = hp[u] > 0.f;
        float mult = heal_mult ? (*heal_mult)[h] : 1.f;
        float total = damage::heal_amount(eff.heal[h], heal_power[h], incoming_heal[h], gw) * mult
                      + damage::heal_amount(eff.heal_plain[h], 0.f, incoming_heal[h], gw);
        if (vamp_heal) total = total + damage::heal_amount((*vamp_heal)[u], 0.f, incoming_heal[h], gw);
        if (alive) hp[u] = std::min(max_hp[u], hp[u] + std::max(total, 0.f));
    }
    size_t S = c ? eff.shields.amount.size() / c : 0;
    for (size_t k = 0; k < S; ++k)
        for (size_t h = 0; h < c; ++h) {
            float mult = heal_mult ? (*heal_mult)[h] : 1.f;
            float amt = eff.shields.amount[h * S + k] * (1.f + heal_power[h]) * (1.f + incoming_heal[h]) * mult;
            damage::grant_shield(shields, ctx.unit[h], amt, eff.shields.kind[h * S + k], now,
                                 eff.shields.duration[h * S + k], eff.shields.decay_hold[h * S + k], amt > 0.f, (int)n);
        }
    for (size_t j = 0; j < n; ++j) {
        float active = status.slow_until[j] > now ? status.slow[j] : 0.f;
        bool take = eff.slow[j] > 0.f && eff.slow[j] >= active;
        float until = take ? std::max(now + eff.slow_duration[j], eff.slow[j] == active ? status.slow_until[j] : 0.f)
                           : status.slow_until[j];
        status.slow[j] = take ? eff.slow[j] : status.slow[j];
        status.slow_until[j] = until;
        status.grievous_until[j] = std::max(status.grievous_until[j], eff.grievous[j] > 0.f ? now + eff.grievous[j] : 0.f);
    }
}

// --- combat.py ---------------------------------------------------------------------------------------------------
CombatClocks update_clocks(const CombatClocks& clocks, const Report& report, const Ctx& ctx, const Units& units,
                           const CC* cc) {
    const Packets& p = report.packets;
    const int n = (int)units.x.size();
    CombatClocks out = clocks;
    auto combat_cls = [](int k) {
        return k == CLASS_CHAMPION || k == CLASS_MINION || k == CLASS_MONSTER || k == CLASS_STRUCTURE;
    };
    for (size_t h = 0; h < ctx.unit.size(); ++h) {
        int u = ctx.unit[h];
        bool any_combat = false, dealt_champ = false, took_champ = false, modern = false, hurt = false;
        for (size_t i = 0; i < size(p); ++i) {
            if (!p.valid[i]) continue;
            int s = clampi(p.src[i], 0, n - 1), d = clampi(p.dst[i], 0, n - 1);
            bool enemy = units.team[s] != units.team[d];
            bool o = p.src[i] == u && enemy, in = p.dst[i] == u && enemy;
            any_combat |= (o && combat_cls(units.cls[d])) || (in && combat_cls(units.cls[s]));
            dealt_champ |= o && units.cls[d] == CLASS_CHAMPION;
            took_champ |= in && units.cls[s] == CLASS_CHAMPION;
            modern |= o || in;
            hurt |= in && units.cls[s] == CLASS_CHAMPION && report.resolved.health_loss[i] > 0.f;
        }
        if (cc) {
            for (int j = 0; j < n; ++j) {
                bool any = cc->slowed[h * n + j] || cc->immobilized[h * n + j];
                bool champ_n = units.cls[j] == CLASS_CHAMPION && units.team[j] != ctx.team[h];
                dealt_champ |= any && champ_n;
                any_combat |= any;
            }
        }
        modern |= any_combat;
        bool champ = dealt_champ || took_champ;
        float now = ctx.now;
        bool new_episode = champ && (now - clocks.last_champion_combat[h] >= CHAMPION_COMBAT_GAP);
        if (any_combat) out.last_combat[h] = now;
        if (champ) out.last_champion_combat[h] = now;
        if (hurt) out.last_hit_by_champion[h] = now;
        if (new_episode) out.champion_combat_start[h] = now, out.struck_first[h] = dealt_champ && !took_champ;
        if (modern) out.last_combat_modern[h] = now;
    }
    return out;
}

namespace {
void shield_gained(const Effects& eff, const ItemStats& st, const Report& report, const Report& follow, const Ctx& ctx,
                   const Defense& dfn, const Arr<float>& mult, const RuneEvents& ev, Arr<float>& amount,
                   Arr<float>& duration) {
    size_t c = ctx.unit.size(), S = c ? eff.shields.amount.size() / c : 0;
    amount.assign(c, 0.f), duration.assign(c, 0.f);
    for (size_t h = 0; h < c; ++h) {
        float factor = (1.f + st.heal_shield_power[h]) * (1.f + st.incoming_heal[h]) * mult[h];
        float best = 0.f, dur = 0.f;
        size_t kbest = S;                                      // the zero pad column
        float best_v = -INF;
        for (size_t k = 0; k <= S; ++k) {
            float v = k < S ? eff.shields.amount[h * S + k] * factor : 0.f;
            if (v > best_v) best_v = v, kbest = k;
        }
        best = best_v;
        dur = kbest < S ? eff.shields.duration[h * S + kbest] : 0.f;
        int u = ctx.unit[h];
        bool fired = report.resolved.lifeline_fired[u] || follow.resolved.lifeline_fired[u];
        float life = fired ? dfn.lifeline_shield[u] : 0.f;
        if (life > best) dur = dfn.lifeline_duration[u];
        best = std::max(best, life);
        if (ev.shield_gained[h] > best) dur = ev.shield_gained_duration[h];
        amount[h] = std::max(best, ev.shield_gained[h]);
        duration[h] = dur;
    }
}

Packets concat(std::initializer_list<const Packets*> parts) {
    Packets out = empty_packets(0);
    for (const Packets* p : parts) append(out, *p);
    return out;
}
}  // namespace

CombatTickOut combat_tick(CombatState state, Owned own, Arr<int32_t> page, Ctx ctx, Units units, Attack attack,
                          Cast cast, Arr<int32_t> request, Packets base_packets, Offense base_offense,
                          Defense base_defense, Arr<float> hp, Arr<float> max_hp, Shields shields, UnitStatus status,
                          Kills kills, ItemStats holder_stats, CC cc, RuneEvents ev, int32_t main_capacity,
                          int32_t follow_up_capacity) {
    const int c = (int)ctx.unit.size(), n = (int)units.x.size();
    const bool have_cc = cc.slowed.size() > 0;
    ItemEffectState& items = state.items;
    RuneEffectState& runes = state.runes;
    ev.attack = attack, ev.cast = cast;
    if (have_cc) ev.cc = cc;
    else {
        ev.cc.slowed.assign((size_t)c * n, 0), ev.cc.immobilized.assign((size_t)c * n, 0);
    }
    ev.kills = kills, ev.own = own.counts, ev.clocks = state.clocks;
    ev.report = Report{};

    // 1. STAT.50; adaptive force split once from the pre-adaptive bonus AD/AP.
    ItemStats dyn = stats::combine2(items::dynamic_stats(items, own, ctx), runes::stats(runes, page, ctx, ev));
    for (int h = 0; h < c; ++h) {
        float af = dyn.adaptive_force[h] + holder_stats.adaptive_force[h];
        float af_ad, af_ap;
        stats::resolve_adaptive(af, ctx.bonus_ad[h] + dyn.attack_damage[h], ctx.ap[h] + dyn.ability_power[h],
                                ev.adaptive_physical[h], &af_ad, &af_ap);
        dyn.attack_damage[h] = dyn.attack_damage[h] + af_ad, dyn.ability_power[h] = dyn.ability_power[h] + af_ap;
        dyn.adaptive_force[h] = 0.f;
    }
    ItemStats holder0 = holder_stats;
    holder0.adaptive_force.assign(c, 0.f);
    ItemStats st = stats::combine2(holder0, dyn);
    for (int h = 0; h < c; ++h) {
        ev.bonus_ad[h] = ctx.bonus_ad[h] + dyn.attack_damage[h], ev.ap[h] = ctx.ap[h] + dyn.ability_power[h];
        ev.bonus_attack_speed[h] = ctx.bonus_attack_speed[h] + dyn.attack_speed[h];
        ev.summoner_haste[h] = st.summoner_haste[h];
    }

    // 2. STAT.70 max-HP sync: gains heal (except silent_health), losses clamp.
    Arr<float> target(c);
    for (int h = 0; h < c; ++h) {
        int u = ctx.unit[h];
        float static_max = max_hp[u] - state.dyn_health[h];
        target[h] = dyn.health[h] + dyn.percent_health[h] * (static_max + dyn.health[h]);
        float delta = target[h] - state.dyn_health[h];
        float heal_part = std::max(delta - std::max(dyn.silent_health[h] - state.dyn_silent[h], 0.f), 0.f);
        float new_max = std::max(max_hp[u] + delta, 1.f);
        hp[u] = hp[u] > 0.f ? std::min(std::max(hp[u] + heal_part, 0.f), new_max) : hp[u];
        max_hp[u] = new_max;
        ctx.max_hp[h] = new_max, ctx.hp[h] = hp[u];
    }
    units.hp = hp, units.max_hp = max_hp;
    Arr<float> dyn_silent = dyn.silent_health;

    // 3. Action phase: items first, then runes, in each hook.
    std::vector<Effects> parts;
    parts.push_back(items::on_cast(items, own, ctx, units, cast));
    parts.push_back(runes::on_cast(runes, page, ctx, units, ev));
    parts.push_back(items::on_attack(items, own, ctx, units, attack));
    parts.push_back(runes::on_attack(runes, page, ctx, units, ev));
    parts.push_back(items::on_hit(items, own, ctx, units, attack));
    Attack again = items::spellblade_extra_on_hit_attack(items.spellblade);
    Arr<uint8_t> phantom = items::marksman_phantom_hit_due(items.marksman, ctx);
    for (int h = 0; h < c; ++h) {
        bool ph = phantom[h] && attack.hit[h];
        again.hit[h] = again.hit[h] | ph;
        if (ph) again.target[h] = attack.target[h];
    }
    parts.push_back(items::on_hit(items, own, ctx, units, again));
    Arr<uint8_t> extra = items::marksman_extra_on_hit_targets(items.marksman, ctx);
    for (int k = 0; k < EXTRA_ON_HIT_SLOTS; ++k) {
        Attack a;
        a.launched.assign(c, 0), a.hit.assign(c, 0), a.target.assign(c, 0), a.raw.assign(c, 0.f), a.is_crit.assign(c, 0);
        for (int h = 0; h < c; ++h) {     // argsort(where(extra, 0, 1)): the extra units first, then the rest
            int rank = 0, pick = 0;
            bool found = false;
            for (int pass = 0; pass < 2 && !found; ++pass)
                for (int j = 0; j < n; ++j) {
                    bool e = extra[(size_t)h * n + j];
                    if ((pass == 0) != e) continue;
                    if (rank++ == k) { pick = j, found = true; break; }
                }
            a.target[h] = pick;
            a.hit[h] = extra[(size_t)h * n + pick];
        }
        parts.push_back(items::on_hit(items, own, ctx, units, a));
    }
    parts.push_back(runes::on_hit(runes, page, ctx, units, ev));
    ActiveOut act;
    parts.push_back(items::active(items, own, ctx, units, request, act));
    for (int h = 0; h < c; ++h) ev.potion_drunk[h] = std::max(ev.potion_drunk[h], items.consumables.drank[h]);
    parts.push_back(items::periodic(items, own, ctx, units));
    parts.push_back(runes::periodic(runes, page, ctx, units, ev));
    if (have_cc) parts.push_back(items::on_cc(items, own, ctx, units, cc));
    parts.push_back(runes::on_cc(runes, page, ctx, units, ev));
    Effects pre = no_effects(c, n);
    for (const Effects& e : parts) merge_into(pre, e);

    // 4. Defense / offense.
    HolderDefense holder = items::holder_defense(items, own, ctx);
    Debuffs id = items::target_debuffs(items, own, ctx, units), rd = runes::debuffs(runes, page, ctx, units, ev);
    Debuffs debuffs = combine_debuffs({&id, &rd}, n);
    Defense dfn = fold_defense(base_defense, ctx, holder, debuffs, 0.f, st.heal_shield_power, st.incoming_heal);
    for (int h = 0; h < c; ++h) {
        int u = ctx.unit[h];
        dfn.armor[u] = (dfn.armor[u] + dyn.armor[h]) * (1.f + dyn.percent_armor[h]);
        dfn.magic_resist[u] = (dfn.magic_resist[u] + dyn.magic_resist[h]) * (1.f + dyn.percent_magic_resist[h]);
    }
    Offense off = fold_offense(base_offense, ctx, st);
    Vamp vamp;
    vamp.life_steal.assign(n, 0.f), vamp.omnivamp.assign(n, 0.f);
    for (int h = 0; h < c; ++h) vamp.life_steal[ctx.unit[h]] = st.life_steal[h], vamp.omnivamp[ctx.unit[h]] = st.omnivamp[h];
    auto prepare = [&](Packets& p, const Units& live) {
        Arr<float> ia = items::packet_amp(items, own, ctx, live, p), ra = runes::packet_amp(runes, page, ctx, live, ev, p);
        for (size_t i = 0; i < size(p); ++i) p.amp[i] = p.amp[i] + ia[i] + ra[i];
        Arr<float> rb = runes::packet_block(runes, page, ctx, live, ev, p);
        for (size_t i = 0; i < size(p); ++i) p.block[i] = p.block[i] + rb[i];
    };

    // 5. Main resolution.
    CombatTickOut out;
    Packets packets;
    int overflow = compact(concat({&state.carry, &base_packets, &pre.packets}), (size_t)main_capacity, packets);
    prepare(packets, units);
    out.report = resolve_tick(packets, off, dfn, hp, max_hp, shields, ctx.now, vamp);
    hp = out.report.resolved.hp, max_hp = out.report.resolved.max_hp, shields = out.report.resolved.shields;
    Units live = units;
    live.hp = hp, live.max_hp = max_hp;
    for (int j = 0; j < n; ++j) live.alive[j] = units.alive[j] && hp[j] > 0.f;
    CombatClocks clocks = update_clocks(state.clocks, out.report, ctx, units, have_cc ? &cc : nullptr);
    Ctx ctx_d = ctx;
    for (int h = 0; h < c; ++h) ctx_d.hp[h] = hp[ctx.unit[h]], ctx_d.max_hp[h] = max_hp[ctx.unit[h]];

    // 6. On-damage triggers.
    RuneEvents ev_d = ev;
    ev_d.report = out.report, ev_d.clocks = clocks;
    Effects eff_i = items::on_damage(items, own, ctx_d, live, out.report);
    Effects eff_r = runes::on_damage(runes, page, ctx_d, live, ev_d);

    // 7. Follow-up pass; its own trigger packets carry into the next tick.
    Packets follow;
    int overflow2 = compact(concat({&eff_i.packets, &eff_r.packets}), (size_t)follow_up_capacity, follow);
    prepare(follow, live);
    out.follow_up = resolve_tick(follow, off, dfn, hp, max_hp, shields, ctx.now, vamp);
    hp = out.follow_up.resolved.hp, max_hp = out.follow_up.resolved.max_hp, shields = out.follow_up.resolved.shields;
    live.hp = hp, live.max_hp = max_hp;
    for (int j = 0; j < n; ++j) live.alive[j] = live.alive[j] && hp[j] > 0.f;
    clocks = update_clocks(clocks, out.follow_up, ctx, units, nullptr);
    for (int h = 0; h < c; ++h) ctx_d.hp[h] = hp[ctx.unit[h]], ctx_d.max_hp[h] = max_hp[ctx.unit[h]];
    RuneEvents ev_f = ev;
    ev_f.report = out.follow_up, ev_f.clocks = clocks;
    Effects eff_i2 = items::on_damage(items, own, ctx_d, live, out.follow_up);
    Effects eff_r2 = runes::on_damage(runes, page, ctx_d, live, ev_f);
    Packets carried;
    int overflow3 = compact(concat({&eff_i2.packets, &eff_r2.packets}), CARRY_CAPACITY, carried);

    // 8. Takedowns, heals/shields, end of tick.
    RuneEvents ev_t = ev_f;
    ev_t.report = Report{};
    Effects eff_t = items::on_takedown(items, own, ctx_d, live, kills);
    Effects eff_rt = runes::on_takedown(runes, page, ctx_d, live, ev_t);
    Effects total = no_effects(c, n);
    merge_into(total, pre);
    for (Effects* e : {&eff_i, &eff_r, &eff_i2, &eff_r2, &eff_t, &eff_rt}) {
        e->packets = empty_packets(0);
        merge_into(total, *e);
    }
    total.packets = empty_packets(0);
    Arr<float> vamp_heal(n);
    for (int j = 0; j < n; ++j)
        vamp_heal[j] = out.report.life_steal_heal[j] + out.report.omnivamp_heal[j] + out.follow_up.life_steal_heal[j]
                       + out.follow_up.omnivamp_heal[j];
    Arr<float> mult = runes::heal_mult(runes, page, ctx_d, ev_t);
    apply_effects(total, ctx, hp, max_hp, shields, status, st.heal_shield_power, st.incoming_heal, &vamp_heal, &mult);
    Arr<float> gained, gained_for;
    shield_gained(total, st, out.report, out.follow_up, ctx, dfn, mult, ev, gained, gained_for);
    RuneEvents ev_p = ev_t;
    ev_p.shield_gained = gained, ev_p.shield_gained_duration = gained_for;
    Ctx ctx_p = ctx;
    for (int h = 0; h < c; ++h) ctx_p.hp[h] = hp[ctx.unit[h]], ctx_p.max_hp[h] = max_hp[ctx.unit[h]];
    Units live_p = live;
    live_p.hp = hp;
    runes::post_tick(runes, page, ctx_p, live_p, ev_p);
    items::on_shop(items, own, ctx);
    out.rune_outputs = runes::outputs(runes, page, ctx, ev_p);
    state.clocks = clocks;
    state.dyn_health = target;
    state.dyn_silent = dyn_silent;
    state.carry = carried;
    out.state = state;
    out.hp = hp, out.max_hp = max_hp, out.shields = shields, out.status = status;
    out.packet_overflow = overflow + overflow2 + overflow3;
    out.effects = total;
    out.active = act;
    out.dynamic_stats = dyn;
    items::starters_pending_transforms(state.items.starters, own, out.transform_from, out.transform_to, out.transform_do);
    out.consume_row = state.items.consumables.consume_row;
    out.events = ev_p;
    return out;
}

LANESIM_TEST(combat_tick, "combat.combat_tick", combat_tick);

}  // namespace lanesim::champ
