#include "core.hpp"

#include <unordered_map>

namespace lanesim::champ {

int compact(const Packets& in, size_t capacity, Packets& out) {
    out = empty_packets(capacity);
    size_t k = 0;
    int dropped = 0;
    for (size_t i = 0; i < size(in); ++i) {
        if (!in.valid[i]) continue;
        if (k >= capacity) { ++dropped; continue; }
        out.valid[k] = 1, out.src[k] = in.src[i], out.dst[k] = in.dst[i], out.raw[k] = in.raw[i];
        out.dtype[k] = in.dtype[i], out.flags[k] = in.flags[i], out.amp[k] = in.amp[i], out.item[k] = in.item[i];
        out.cast_id[k] = in.cast_id[i], out.block[k] = in.block[i];
        ++k;
    }
    return dropped;
}

namespace {
const std::unordered_map<int, int>& rows(const char* key) {
    static thread_local std::unordered_map<std::string, std::unordered_map<int, int>> cache;
    auto& m = cache[key];
    if (m.empty()) {
        const auto& ids = data::table(key);
        for (size_t i = 0; i < ids.size(); ++i) m[(int)ids[i]] = (int)i;
    }
    return m;
}
}  // namespace

int item_row(int item_id) { return rows("catalog.ids").at(item_id); }
int n_items() { return (int)data::table("catalog.ids").size(); }
int rune_row(int perk_id) { return rows("runes.ids").at(perk_id); }

bool can_hold(const Owned& own, std::initializer_list<int> item_ids) {
    if (own.allowed.size() == 0) return true;
    int ni = n_items();
    size_t c = own.allowed.size() / ni;
    for (int id : item_ids) {
        int r = item_row(id);
        for (size_t h = 0; h < c; ++h)
            if (own.allowed[h * ni + r]) return true;
    }
    return false;
}

Effects no_effects(int c, int n) {
    Effects e;
    e.packets = empty_packets(0);
    e.heal.assign(c, 0.f), e.heal_plain.assign(c, 0.f), e.mana.assign(c, 0.f);
    e.shields.amount.assign(0, 0.f), e.shields.kind.assign(0, 0), e.shields.duration.assign(0, 0.f);
    e.shields.decay_hold.assign(0, 0.f);
    e.slow.assign(n, 0.f), e.slow_duration.assign(n, 0.f), e.grievous.assign(n, 0.f);
    e.gold.assign(c, 0.f), e.attack_reset.assign(c, 0), e.revive.assign(c, 0), e.revive_delay.assign(c, 0.f);
    e.revive_hp.assign(c, 0.f);
    return e;
}

namespace {
template <class T>
void concat_cols(Arr<T>& a, const Arr<T>& b, size_t c) {   // (C, Sa) ++ (C, Sb) along axis 1
    size_t sa = c ? a.size() / c : 0, sb = c ? b.size() / c : 0;
    if (sb == 0) return;
    Arr<T> out(c * (sa + sb));
    for (size_t h = 0; h < c; ++h) {
        for (size_t k = 0; k < sa; ++k) out[h * (sa + sb) + k] = a[h * sa + k];
        for (size_t k = 0; k < sb; ++k) out[h * (sa + sb) + sa + k] = b[h * sb + k];
    }
    a = out;
}
}  // namespace

void concat_shields(ShieldGrant& a, const ShieldGrant& b, size_t c) {
    concat_cols(a.amount, b.amount, c), concat_cols(a.kind, b.kind, c), concat_cols(a.duration, b.duration, c);
    concat_cols(a.decay_hold, b.decay_hold, c);
}

void merge_into(Effects& out, const Effects& p) {
    size_t c = out.heal.size(), n = out.slow.size();
    append(out.packets, p.packets);
    for (size_t h = 0; h < c; ++h) {
        out.heal[h] = out.heal[h] + p.heal[h], out.heal_plain[h] = out.heal_plain[h] + p.heal_plain[h];
        out.mana[h] = out.mana[h] + p.mana[h];
        out.gold[h] = out.gold[h] + p.gold[h];
        out.attack_reset[h] = out.attack_reset[h] | p.attack_reset[h];
        out.revive[h] = out.revive[h] | p.revive[h];
        if (p.revive[h]) out.revive_delay[h] = p.revive_delay[h], out.revive_hp[h] = p.revive_hp[h];
    }
    concat_shields(out.shields, p.shields, c);
    for (size_t j = 0; j < n; ++j) {
        bool stronger = p.slow[j] > out.slow[j];
        float dur = stronger ? p.slow_duration[j]
                  : (p.slow[j] == out.slow[j] ? std::max(out.slow_duration[j], p.slow_duration[j]) : out.slow_duration[j]);
        out.slow[j] = stronger ? p.slow[j] : out.slow[j];
        out.slow_duration[j] = dur;
        out.grievous[j] = std::max(out.grievous[j], p.grievous[j]);
    }
}

Effects merge(const std::vector<const Effects*>& parts, int c, int n) {
    Effects out = no_effects(c, n);
    for (const Effects* p : parts) merge_into(out, *p);
    return out;
}

ShieldGrant shield_grants(const Arr<float>& amount, int kind, float duration, float decay_hold) {
    ShieldGrant g;
    size_t c = amount.size();
    g.amount = amount;
    g.kind.assign(c, kind), g.duration.assign(c, duration), g.decay_hold.assign(c, decay_hold);
    return g;
}

HolderDefense neutral_defense(int c) {
    HolderDefense d;
    d.received_mult.assign(c, 1.f), d.basic_attack_mult.assign(c, 1.f), d.crit_taken_mult.assign(c, 1.f);
    d.champion_attack_block.assign(c, 0.f), d.postmit_flat.assign(c, 0.f), d.store_fraction.assign(c, 0.f);
    d.lifeline_ready.assign(c, 0), d.lifeline_magic_only.assign(c, 0), d.lifeline_shield.assign(c, 0.f);
    d.lifeline_shield_kind.assign(c, SHIELD_ALL), d.lifeline_duration.assign(c, 0.f);
    d.lifeline_decay_hold.assign(c, INF), d.lifeline_bonus_health.assign(c, 0.f), d.spell_shield.assign(c, 0);
    d.champion_received_mult.assign(c, 1.f);
    return d;
}

HolderDefense combine_defense(const std::vector<const HolderDefense*>& parts, int c) {
    HolderDefense o = neutral_defense(c);
    for (const HolderDefense* p : parts)
        for (int h = 0; h < c; ++h) {
            bool ready = p->lifeline_ready[h];
            o.received_mult[h] = o.received_mult[h] * p->received_mult[h];
            o.basic_attack_mult[h] = o.basic_attack_mult[h] * p->basic_attack_mult[h];
            o.crit_taken_mult[h] = o.crit_taken_mult[h] * p->crit_taken_mult[h];
            o.champion_attack_block[h] = o.champion_attack_block[h] + p->champion_attack_block[h];
            o.postmit_flat[h] = o.postmit_flat[h] + p->postmit_flat[h];
            o.store_fraction[h] = o.store_fraction[h] + p->store_fraction[h];
            o.lifeline_ready[h] = o.lifeline_ready[h] | ready;
            if (ready) {
                o.lifeline_magic_only[h] = p->lifeline_magic_only[h], o.lifeline_shield[h] = p->lifeline_shield[h];
                o.lifeline_shield_kind[h] = p->lifeline_shield_kind[h];
                o.lifeline_duration[h] = p->lifeline_duration[h], o.lifeline_decay_hold[h] = p->lifeline_decay_hold[h];
                o.lifeline_bonus_health[h] = p->lifeline_bonus_health[h];
            }
            o.spell_shield[h] = o.spell_shield[h] | p->spell_shield[h];
            o.champion_received_mult[h] = o.champion_received_mult[h] * p->champion_received_mult[h];
        }
    return o;
}

Debuffs neutral_debuffs(int n) {
    Debuffs d;
    d.visit([&](auto& m) { m.assign(n, 0.f); });
    return d;
}

Debuffs combine_debuffs(const std::vector<const Debuffs*>& parts, int n) {
    Debuffs o = neutral_debuffs(n);
    for (const Debuffs* p : parts)
        for (int j = 0; j < n; ++j) {
            o.percent_armor_reduction[j] = 1.f - (1.f - o.percent_armor_reduction[j]) * (1.f - p->percent_armor_reduction[j]);
            o.flat_armor_reduction[j] = o.flat_armor_reduction[j] + p->flat_armor_reduction[j];
            o.percent_mr_reduction[j] = 1.f - (1.f - o.percent_mr_reduction[j]) * (1.f - p->percent_mr_reduction[j]);
            o.flat_mr_reduction[j] = o.flat_mr_reduction[j] + p->flat_mr_reduction[j];
            o.received_amp[j] = o.received_amp[j] + p->received_amp[j];
            o.magic_received_amp[j] = o.magic_received_amp[j] + p->magic_received_amp[j];
            o.attack_speed_cripple[j] = std::max(o.attack_speed_cripple[j], p->attack_speed_cripple[j]);
        }
    return o;
}

RuneOutputs no_outputs(int c, int ni) {
    RuneOutputs o;
    o.grant_item.assign(c, 0), o.forbid_purchase.assign((size_t)c * ni, 0), o.skill_points.assign(c, 0);
    o.basic_cd_refund.assign(c, 0.f), o.ult_cd_refund.assign(c, 0.f), o.move_locked.assign(c, 0);
    o.blink.assign(c, 0), o.blink_range.assign(c, 0.f), o.spellbook_swap_ready.assign(c, 0);
    o.first_strike_gold.assign(c, 0.f), o.ghosted.assign(c, 0);
    return o;
}

void merge_outputs_into(RuneOutputs& o, const RuneOutputs& p) {
    size_t c = o.grant_item.size();
    for (size_t h = 0; h < c; ++h) {
        if (p.grant_item[h] != 0) o.grant_item[h] = p.grant_item[h];
        o.skill_points[h] = o.skill_points[h] + p.skill_points[h];
        o.basic_cd_refund[h] = 1.f - (1.f - o.basic_cd_refund[h]) * (1.f - p.basic_cd_refund[h]);
        o.ult_cd_refund[h] = 1.f - (1.f - o.ult_cd_refund[h]) * (1.f - p.ult_cd_refund[h]);
        o.move_locked[h] = o.move_locked[h] | p.move_locked[h];
        o.blink[h] = o.blink[h] | p.blink[h];
        o.blink_range[h] = std::max(o.blink_range[h], p.blink_range[h]);
        o.spellbook_swap_ready[h] = o.spellbook_swap_ready[h] | p.spellbook_swap_ready[h];
        o.first_strike_gold[h] = o.first_strike_gold[h] + p.first_strike_gold[h];
        o.ghosted[h] = o.ghosted[h] | p.ghosted[h];
    }
    for (size_t k = 0; k < o.forbid_purchase.size(); ++k) o.forbid_purchase[k] = o.forbid_purchase[k] | p.forbid_purchase[k];
}

CCOut no_cc(int c, int n) {
    CCOut o;
    size_t cn = (size_t)c * n;
    o.stun.assign(cn, 0.f), o.root.assign(cn, 0.f), o.silence.assign(cn, 0.f), o.knockup.assign(cn, 0.f);
    o.slow.assign(cn, 0.f), o.slow_duration.assign(cn, 0.f), o.cast_id.assign(cn, 0);
    return o;
}

void merge_cc_into(CCOut& a, const CCOut& b) {
    for (size_t k = 0; k < a.stun.size(); ++k) {
        bool stronger = b.slow[k] > a.slow[k];
        a.stun[k] = std::max(a.stun[k], b.stun[k]), a.root[k] = std::max(a.root[k], b.root[k]);
        a.silence[k] = std::max(a.silence[k], b.silence[k]), a.knockup[k] = std::max(a.knockup[k], b.knockup[k]);
        a.slow_duration[k] = stronger ? b.slow_duration[k] : std::max(a.slow_duration[k], b.slow_duration[k]);
        a.slow[k] = std::max(a.slow[k], b.slow[k]);
        if (b.cast_id[k] != 0) a.cast_id[k] = b.cast_id[k];
    }
}

Dash no_dash(int c) {
    Dash d;
    d.active.assign(c, 0), d.to_x.assign(c, 0.f), d.to_y.assign(c, 0.f), d.speed.assign(c, 0.f);
    d.target.assign(c, -1), d.blink.assign(c, 0);
    return d;
}

KitOut no_out(int c, int n) {
    KitOut o;
    o.packets = empty_packets(0);
    o.cc = no_cc(c, n);
    o.dash = no_dash(c);
    o.heal.assign(c, 0.f);
    o.shield.amount.assign(0, 0.f), o.shield.kind.assign(0, 0), o.shield.duration.assign(0, 0.f);
    o.shield.decay_hold.assign(0, 0.f);
    o.mana_cost.assign(c, 0.f), o.cooldown_start.assign((size_t)c * 4, 0), o.base_cooldown.assign((size_t)c * 4, 0.f);
    o.attack_reset.assign(c, 0), o.cast_started.assign(c, 0), o.cast_slot.assign(c, -1), o.cast_id.assign(c, 0);
    o.cast_lockout.assign(c, 0.f), o.cleanse_slow.assign(c, 0), o.attack_target.assign(c, -1);
    return o;
}

void merge_out_into(KitOut& acc, const KitOut& p, int c, int n) {
    append(acc.packets, p.packets);
    merge_cc_into(acc.cc, p.cc);
    for (int h = 0; h < c; ++h) {
        if (p.dash.active[h]) {
            acc.dash.active[h] = p.dash.active[h], acc.dash.to_x[h] = p.dash.to_x[h], acc.dash.to_y[h] = p.dash.to_y[h];
            acc.dash.speed[h] = p.dash.speed[h], acc.dash.target[h] = p.dash.target[h], acc.dash.blink[h] = p.dash.blink[h];
        }
        acc.heal[h] = acc.heal[h] + p.heal[h];
        acc.mana_cost[h] = acc.mana_cost[h] + p.mana_cost[h];
        for (int s = 0; s < 4; ++s) {
            acc.cooldown_start[h * 4 + s] = acc.cooldown_start[h * 4 + s] | p.cooldown_start[h * 4 + s];
            if (p.base_cooldown[h * 4 + s] > 0.f) acc.base_cooldown[h * 4 + s] = p.base_cooldown[h * 4 + s];
        }
        acc.attack_reset[h] = acc.attack_reset[h] | p.attack_reset[h];
        acc.cast_started[h] = acc.cast_started[h] | p.cast_started[h];
        if (p.cast_started[h]) acc.cast_slot[h] = p.cast_slot[h], acc.cast_id[h] = p.cast_id[h];
        acc.cast_lockout[h] = std::max(acc.cast_lockout[h], p.cast_lockout[h]);
        bool cs = p.cleanse_slow.size() ? p.cleanse_slow[h] : 0;
        acc.cleanse_slow[h] = acc.cleanse_slow[h] | cs;
        int at = p.attack_target.size() ? p.attack_target[h] : -1;
        if (at >= 0) acc.attack_target[h] = at;
    }
    concat_shields(acc.shield, p.shield, c);
}

KitDefense neutral_kit_defense(int c) {
    KitDefense d;
    d.received_mult.assign(c, 1.f), d.dodge_basic.assign(c, 0), d.aoe_received_mult.assign(c, 1.f);
    d.tenacity_bonus.assign(c, 0.f);
    return d;
}

KitAttackMods neutral_attack_mods(int c) {
    KitAttackMods m;
    m.extra_range.assign(c, 0.f), m.attack_reset.assign(c, 0), m.cannot_attack.assign(c, 0), m.cannot_crit.assign(c, 0);
    m.windup.assign(c, 0.f), m.period.assign(c, 0.f), m.uncancellable.assign(c, 0);
    return m;
}

}  // namespace lanesim::champ
