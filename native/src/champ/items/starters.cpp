// items/effects/starters.py. Reachable for Garen/Jax: Doran's Shield (Enduring Focus, Helping Hand). The Manaflow,
// Glory, Cull, Seraph's, Muramana and Fimbulwinter paths are not holdable; their always-run state resets and
// always-emitted padded packets / shield grants are kept.
#include "../marshal.hpp"
#include "items.hpp"

namespace lanesim::items::starters {

namespace {
constexpr int DORANS_SHIELD = 1054, MURAMANA = 3042;
constexpr float DS_FULL_MISSING = 0.75f;

float K(const char* name) { return data::f(std::string("items.starters.") + name); }
}  // namespace

// starters.stats: Enduring Focus regen (other stat lines need non-holdable items)
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx) {
    static const float max_regen_m = K("ds_max_regen"), max_regen_r = K("ds_max_range_regen"),
                       regen_dur = K("ds_regen_duration");
    size_t c = ctx.unit.size();
    ItemStats o = default_stats();
    o.health = o.attack_damage = o.ability_power = o.percent_move_speed = o.health_regen = o.mana = o.mana_regen =
        o.heal_shield_power = zeros_c(c);
    for (size_t h = 0; h < c; ++h) {
        bool ds = holds(own, DORANS_SHIELD, h) && ctx.now < s.ds_until[h] && ctx.alive[h];
        float max_regen = ctx.is_ranged[h] ? max_regen_r : max_regen_m;
        float missing = std::min(std::max(1.f - ctx.hp[h] / std::max(ctx.max_hp[h], 1.f), 0.f), 1.f);
        float ds_regen = ds ? max_regen / regen_dur * std::min(missing / DS_FULL_MISSING, 1.f) * s.ds_eff[h] : 0.f;
        o.health_regen[h] = ds_regen + 0.f;      // + Doran's Ring HP (not holdable)
    }
    return o;
}

// starters.defense: Seraph's Lifeline (not holdable: never ready)
HolderDefense defense(const State& s, const Owned& own, const Ctx& ctx) {
    static const float dur = K("seraph_shield_duration");
    int c = (int)ctx.unit.size();
    HolderDefense d = neutral_defense(c);
    for (int h = 0; h < c; ++h) d.lifeline_duration[h] = dur;
    return d;
}

// starters.on_hit: Helping Hand (Doran's Shield), Muramana Shock (padded), Manaflow charge (not holdable)
std::tuple<State, Effects> on_hit(State s, const Owned& own, const Ctx& ctx, const Units& u, const Attack& a) {
    static const float hh_ds = K("ds_minion_bonus"), mura = K("muramana_onhit");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    Effects e = no_effects(c, n);
    Packets shock = empty_packets(0);
    for (int h = 0; h < c; ++h) {
        bool hit = a.hit[h] && ctx.alive[h] && a.target[h] >= 0;
        int tcls = target_class(u, a.target[h]);
        int tgt = std::max(a.target[h], 0);
        bool ds = holds(own, DORANS_SHIELD, h);
        float hh_val = ds ? hh_ds : 0.f;
        int hh_item = ds ? DORANS_SHIELD : 0;
        bool hh = hit && hh_item != 0 && tcls == CLASS_MINION;
        push(e.packets, hh, ctx.unit[h], tgt, hh_val, PHYSICAL, ON_HIT_ITEM, 0.f, hh_item);
        float mx = ctx.max_mana[h] + 0.f;        // + Manaflow stacks (not holdable)
        push(shock, false, ctx.unit[h], tgt, mura * mx, PHYSICAL, ON_HIT_ITEM, 0.f, MURAMANA);
    }
    append(e.packets, shock);
    return {s, e};
}

// starters.on_cast: cast_start only for Manaflow/Muramana holders (not holdable)
std::tuple<State, Effects> on_cast(State s, const Owned& own, const Ctx& ctx, const Units& u, const Cast& cast) {
    return {s, no_effects((int)ctx.unit.size(), (int)u.x.size())};
}

// starters.on_damage: Enduring Focus trigger; Muramana ability Shock (padded)
std::tuple<State, Effects> on_damage(State s, const Owned& own, const Ctx& ctx, const Units& u, const Report& r) {
    static const float regen_dur = K("ds_regen_duration"), range_mult = K("ds_range_regen_mult"),
                       mura_m = K("muramana_ability_melee"), mura_r = K("muramana_ability_ranged");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    const Packets& p = r.packets;
    size_t np = size(p);
    Effects e = no_effects(c, n);
    for (int h = 0; h < c; ++h) {
        bool any_hit = false, strong = false;
        for (size_t k = 0; k < np; ++k) {
            int src = clampi(p.src[k], 0, n - 1);
            bool from_champ = p.valid[k] && r.resolved.final[k] > 0.f && u.cls[src] == CLASS_CHAMPION;
            bool to_me = p.dst[k] == ctx.unit[h] && from_champ && u.team[src] != ctx.team[h];
            bool weak = has(p.flags[k], TAG_AOE) || has(p.flags[k], TAG_PERIODIC);
            any_hit = any_hit || to_me;
            strong = strong || (to_me && !weak);
        }
        if (any_hit && holds(own, DORANS_SHIELD, h)) {
            s.ds_until[h] = ctx.now + regen_dur;
            s.ds_eff[h] = strong ? 1.f : range_mult;
        }
        float dmg = (ctx.is_ranged[h] ? mura_r : mura_m) * (ctx.max_mana[h] + 0.f);
        for (int j = 0; j < n; ++j)
            push_cn(e.packets, false, ctx.unit[h], j, dmg, PHYSICAL, TAG_PROC | TAG_ITEM, MURAMANA);
    }
    return {s, e};
}

// starters.periodic: Manaflow charges / Cull counters reset without their items; Consonance (not holdable)
std::tuple<State, Effects> periodic(State s, const Owned& own, const Ctx& ctx, const Units& u) {
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    for (int h = 0; h < c; ++h) {
        s.charges[h] = 0.f, s.next_charge[h] = BIG, s.tear_mana[h] = 0.f;
        s.cull_kills[h] = 0.f, s.cull_done[h] = 0;
    }
    return {s, no_effects(c, n)};
}

// starters.on_takedown: Glory / Cull (not holdable)
std::tuple<State, Effects> on_takedown(State s, const Owned& own, const Ctx& ctx, const Units& u, const Kills& k) {
    return {s, no_effects((int)ctx.unit.size(), (int)u.x.size())};
}

// starters.on_cc -> everlasting (Fimbulwinter not holdable: a zero grant per holder)
std::tuple<State, Effects> on_cc(State s, const Owned& own, const Ctx& ctx, const Units& u, const CC& cc) {
    static const float dur = K("fimbul_shield_duration");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    Effects e = no_effects(c, n);
    e.shields = shield_grants(zeros_c(c), SHIELD_ALL, dur);
    return {s, e};
}

LANESIM_TEST(items_starters_stats, "items.starters.stats", stats);
LANESIM_TEST(items_starters_defense, "items.starters.defense", defense);
LANESIM_TEST(items_starters_on_hit, "items.starters.on_hit", on_hit);
LANESIM_TEST(items_starters_on_cast, "items.starters.on_cast", on_cast);
LANESIM_TEST(items_starters_on_damage, "items.starters.on_damage", on_damage);
LANESIM_TEST(items_starters_periodic, "items.starters.periodic", periodic);
LANESIM_TEST(items_starters_on_takedown, "items.starters.on_takedown", on_takedown);
LANESIM_TEST(items_starters_on_cc, "items.starters.on_cc", on_cc);

}  // namespace lanesim::items::starters
