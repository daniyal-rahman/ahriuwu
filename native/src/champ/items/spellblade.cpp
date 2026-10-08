// items/effects/spellblade.py. Reachable for Garen/Jax: Sheen, Trinity Force, Dusk and Dawn, Phage. Iceborn, Lich
// Bane, Essence Reaver and Bloodsong are not holdable (their slow / Expose Weakness outputs stay zero).
#include "../marshal.hpp"
#include "items.hpp"

namespace lanesim::items::spellblade {

namespace {
constexpr int SHEEN = 3057, TRINITY = 3078, ICEBORN = 6662, LICH_BANE = 3100, ESSENCE_REAVER = 3508,
              DUSK_DAWN = 2510, BLOODSONG = 3877, PHAGE = 3044;

float K(const char* name) { return data::f(std::string("items.spellblade.") + name); }

bool any_spellblade(const Owned& own, int c) {
    return holds_any(own, {SHEEN, TRINITY, ICEBORN, LICH_BANE, ESSENCE_REAVER, DUSK_DAWN, BLOODSONG}, c);
}
bool armed(const State& s, const Owned& own, const Ctx& ctx, int c) {
    return any_spellblade(own, c) && ctx.now < s.armed_until[c];
}
// _which: the held Spellblade item, priority Trinity > ... > Sheen (reachable: Trinity, Dusk and Dawn, Sheen)
int which(const Owned& own, int c) {
    for (int iid : {TRINITY, DUSK_DAWN, SHEEN})
        if (holds(own, iid, c)) return iid;
    return 0;
}
}  // namespace

// spellblade.stats: Quicken / Rage move speed (Lich Bane AS not holdable)
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx) {
    static const float q_ms = K("quicken_ms"), r_ms = K("rage_ms"), r_ranged = K("rage_ranged");
    size_t c = ctx.unit.size();
    ItemStats o = default_stats();
    o.move_speed = o.attack_speed = zeros_c(c);
    for (size_t h = 0; h < c; ++h) {
        bool quicken = holds(own, TRINITY, h) && ctx.now < s.quicken_until[h];
        bool rage = holds(own, PHAGE, h) && ctx.now < s.rage_until[h];
        float rage_ms = r_ms * (ctx.is_ranged[h] ? r_ranged : 1.f);
        o.move_speed[h] = (quicken ? q_ms : 0.f) + (rage ? rage_ms : 0.f);
    }
    return o;
}

// spellblade.on_cast: arm the Spellblade
std::tuple<State, Effects> on_cast(State s, const Owned& own, const Ctx& ctx, const Units& u, const Cast& cast) {
    static const float window = K("sb_window");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    for (int h = 0; h < c; ++h)
        if (cast.started[h] && ctx.alive[h] && any_spellblade(own, h) && ctx.now >= s.cd_until[h])
            s.armed_until[h] = ctx.now + window;
    return {s, no_effects(c, n)};
}

// spellblade.on_hit (+ proc_damage)
std::tuple<State, Effects> on_hit(State s, const Owned& own, const Ctx& ctx, const Units& u, const Attack& a) {
    static const float sheen_ad = K("sheen_ad"), tri_ad = K("trinity_ad"), dd_ad = K("dd_ad"), dd_ap = K("dd_ap"),
                       dd_heal_ap = K("dd_heal_ap"), dd_heal_hp = K("dd_heal_bonus_hp"), sb_cd = K("sb_cooldown"),
                       dd_delay = K("dd_extra_delay"), q_dur = K("quicken_duration"), r_dur = K("rage_duration");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    Effects e = no_effects(c, n);
    for (int h = 0; h < c; ++h) {
        bool hit = a.hit[h] && ctx.alive[h] && a.target[h] >= 0;
        float now = ctx.now;
        int item = which(own, h);
        bool proc = hit && armed(s, own, ctx, h);
        int tgt = std::max(a.target[h], 0);
        float b = ctx.base_ad[h], ap = ctx.ap[h];
        float dmg = item == TRINITY ? tri_ad * b : item == DUSK_DAWN ? dd_ad * b + dd_ap * ap
                  : item == SHEEN ? sheen_ad * b : 0.f;
        int dtype = item == DUSK_DAWN ? MAGIC : PHYSICAL;
        push(e.packets, proc && dmg > 0.f, ctx.unit[h], tgt, dmg, dtype, ON_HIT_ITEM | PROP_LIFESTEAL, 0.f, item);
        e.heal[h] = proc && item == DUSK_DAWN ? dd_heal_ap * ctx.ap[h] + dd_heal_hp * (ctx.max_hp[h] - ctx.base_hp[h])
                                               : 0.f;
        if (proc && item == DUSK_DAWN) s.dd_extra_at[h] = now + dd_delay, s.dd_extra_target[h] = a.target[h];
        if (proc) s.armed_until[h] = -1e9f, s.cd_until[h] = now + sb_cd;
        if (hit && holds(own, TRINITY, h)) s.quicken_until[h] = now + q_dur;
        if (hit && holds(own, PHAGE, h)) s.rage_until[h] = now + r_dur;
    }
    return {s, e};
}

// spellblade.periodic: Dusk and Dawn re-application due (Iceborn field / Bloodsong gold not holdable)
std::tuple<State, Effects> periodic(State s, const Owned& own, const Ctx& ctx, const Units& u) {
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    for (int h = 0; h < c; ++h) {
        int t = clampi(s.dd_extra_target[h], 0, n - 1);
        bool fire = ctx.now >= s.dd_extra_at[h];
        bool due = fire && holds(own, DUSK_DAWN, h) && ctx.alive[h] && s.dd_extra_target[h] >= 0 && u.alive[t];
        s.dd_extra_due[h] = due;
        s.dd_due_target[h] = due ? s.dd_extra_target[h] : -1;
        if (fire) s.dd_extra_at[h] = INF;
    }
    return {s, no_effects(c, n)};
}

// spellblade.debuffs: Bloodsong Expose Weakness (not holdable)
Debuffs debuffs(const State& s, const Owned& own, const Ctx& ctx, const Units& u) {
    return neutral_debuffs((int)u.x.size());
}

LANESIM_TEST(items_spellblade_stats, "items.spellblade.stats", stats);
LANESIM_TEST(items_spellblade_on_cast, "items.spellblade.on_cast", on_cast);
LANESIM_TEST(items_spellblade_on_hit, "items.spellblade.on_hit", on_hit);
LANESIM_TEST(items_spellblade_periodic, "items.spellblade.periodic", periodic);
LANESIM_TEST(items_spellblade_debuffs, "items.spellblade.debuffs", debuffs);

}  // namespace lanesim::items::spellblade
