// items/effects/hydra.py. Reachable for Garen/Jax: Tiamat, Stridebreaker. Ravenous, Titanic and Profane are not
// holdable; their always-emitted padded packets keep the JAX values and the Titanic window bookkeeping (which also
// rewrites an infinite cooldown) is kept.
#include "../marshal.hpp"
#include "items.hpp"

namespace lanesim::items::hydra {

namespace {
constexpr int TIAMAT = 3077, RAVENOUS = 3074, TITANIC = 3748, PROFANE = 6698, STRIDEBREAKER = 6631;
constexpr int ACTIVES[4] = {TIAMAT, RAVENOUS, PROFANE, STRIDEBREAKER};      // _ACTIVES order
constexpr bool CD_AT_START[4] = {false, false, true, true};
constexpr float CLEAVE_RATIO_MELEE = 0.40f, CLEAVE_RATIO_RANGED = 0.20f, ACTIVE_OFFSET = 100.0f;

float K(const char* name) { return data::f(std::string("items.hydra.") + name); }
const std::vector<float>& T(const char* name) { return data::table(std::string("items.hydra.") + name); }
}  // namespace

// hydra.stats: Stridebreaker decaying move speed
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx) {
    static const float decay = K("stride_decay");
    size_t c = ctx.unit.size();
    ItemStats o = default_stats();
    o.percent_move_speed = zeros_c(c);
    for (size_t h = 0; h < c; ++h) {
        float left = std::min(std::max(1.f - (ctx.now - s.stride_ms_start[h]) / decay, 0.f), 1.f);
        o.percent_move_speed[h] = s.stride_ms[h] * left;
    }
    return o;
}

// hydra.on_hit: Cleave (Titanic on-hit and cone are padded)
std::tuple<State, Effects> on_hit(State s, const Owned& own, const Ctx& ctx, const Units& u, const Attack& a) {
    static const int max_splash = (int)K("max_splash");
    static const float radius = K("cleave_radius"), t_ranged = K("titanic_ranged"), t_primary = K("titanic_primary"),
                       t_splash = K("titanic_splash");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    Packets p_titanic = empty_packets(0), p_cleave = empty_packets(0), p_cone = empty_packets(0);
    Arr<float> dist(n);
    Arr<uint8_t> near(n), splash(n);
    for (int h = 0; h < c; ++h) {
        bool hit = a.hit[h] && ctx.alive[h];
        int tcls = target_class(u, a.target[h]);
        bool not_structure = tcls != CLASS_STRUCTURE;
        int ti = clampi(a.target[h], 0, n - 1);
        float tx = u.x[ti], ty = u.y[ti];
        bool tiamat = holds(own, TIAMAT, h), stride = holds(own, STRIDEBREAKER, h);
        bool do_cleave = hit && not_structure && (tiamat || stride);
        for (int j = 0; j < n; ++j) {
            bool primary = j == a.target[h] && a.target[h] >= 0;
            bool enemies = enemy(ctx, u, h, j) && u.cls[j] != CLASS_STRUCTURE && !primary;
            dist[j] = std::sqrt(sq(u.x[j] - tx) + sq(u.y[j] - ty));
            near[j] = in_circle(u, j, tx, ty, radius) && enemies;
        }
        nearest_k(dist.data(), near.data(), n, max_splash, splash.data());
        float ratio = ctx.is_ranged[h] ? CLEAVE_RATIO_RANGED : CLEAVE_RATIO_MELEE;
        float cleave_dmg = ratio * (ctx.base_ad[h] + ctx.bonus_ad[h]);
        int cleave_item = stride ? STRIDEBREAKER : TIAMAT;
        for (int j = 0; j < n; ++j)
            push_cn(p_cleave, splash[j] && do_cleave, ctx.unit[h], j, cleave_dmg, PHYSICAL,
                    TAG_AOE | TAG_PROC | TAG_ITEM, cleave_item);
        float rmult = ctx.is_ranged[h] ? t_ranged : 1.f;
        push(p_titanic, false, ctx.unit[h], std::max(a.target[h], 0), t_primary * rmult * ctx.max_hp[h], PHYSICAL,
             ON_HIT_ITEM | PROP_LIFESTEAL, 0.f, TITANIC);
        float cone_raw = t_splash * rmult * ctx.max_hp[h];
        for (int j = 0; j < n; ++j)
            push_cn(p_cone, false, ctx.unit[h], j, cone_raw, PHYSICAL, TAG_AOE | TAG_PROC | TAG_ITEM, TITANIC);
    }
    Effects e = no_effects(c, n);
    append(e.packets, p_titanic), append(e.packets, p_cleave), append(e.packets, p_cone);
    return {s, e};
}

// hydra.active: start a Tiamat/Stridebreaker cast; resolve casts ending this tick
std::tuple<State, Effects, ActiveOut> active(State s, const Owned& own, const Ctx& ctx, const Units& u,
                                             const Arr<int32_t>& request) {
    static const std::vector<float> ratio = T("active_ratio"), radius = T("active_radius"),
                                    cooldown = T("active_cooldown"), base_cast = T("active_base_cast");
    static const float t_cd = K("titanic_cooldown"), s_slow = K("stride_slow"), s_dur = K("stride_duration"),
                       s_ms = K("stride_active_ms");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    ActiveOut out;
    out.used.assign(c, 0), out.cast_time.assign(c, 0.f), out.can_move.assign(c, 1), out.attack_reset.assign(c, 0);
    Effects e = no_effects(c, n);
    Packets per_item[4];
    for (auto& p : per_item) p = empty_packets(0);
    for (int h = 0; h < c; ++h) {
        bool ready = ctx.now >= s.cd_until[h] && s.cast_item[h] == 0 && ctx.alive[h];
        // Titanic Crescent not holdable; an unused empowerment starts the cooldown at window end.
        if (std::isinf(s.cd_until[h]) && ctx.now >= s.titanic_until[h]) s.cd_until[h] = s.titanic_until[h] + t_cd;
        for (int k = 0; k < 4; ++k) {
            int item = ACTIVES[k];
            if (!(ready && request[h] == item && holds(own, item, h))) continue;
            float t = std::min(base_cast[k], ctx.attack_windup[h]);
            s.cast_item[h] = item;
            s.cast_end[h] = ctx.now + t;
            s.cd_until[h] = CD_AT_START[k] ? ctx.now + cooldown[k] : INF;
            out.used[h] = 1;
            out.cast_time[h] = t;
            out.can_move[h] = item == STRIDEBREAKER;
        }
    }
    for (int h = 0; h < c; ++h) {
        bool finishing = s.cast_item[h] != 0 && ctx.now >= s.cast_end[h] && ctx.alive[h];
        float cx = ctx.x[h] + ACTIVE_OFFSET * ctx.facing_x[h], cy = ctx.y[h] + ACTIVE_OFFSET * ctx.facing_y[h];
        float total_ad = ctx.base_ad[h] + ctx.bonus_ad[h];
        for (int k = 0; k < 4; ++k) {
            int item = ACTIVES[k];
            bool fire = finishing && s.cast_item[h] == item;
            int flags = TAG_AOE | TAG_ACTIVE_SPELL | TAG_ITEM | (item == RAVENOUS ? PROP_LIFESTEAL : 0);
            float champs = 0.f;
            for (int j = 0; j < n; ++j) {
                bool hit = fire && in_circle(u, j, cx, cy, radius[k]) && enemy(ctx, u, h, j)
                           && u.cls[j] != CLASS_STRUCTURE;
                push_cn(per_item[k], hit, ctx.unit[h], j, ratio[k] * total_ad, PHYSICAL, flags, item);
                if (item == STRIDEBREAKER && hit) {
                    e.slow[j] = s_slow, e.slow_duration[j] = s_dur;
                    if (u.cls[j] == CLASS_CHAMPION) champs = champs + 1.f;
                }
            }
            if (fire && !CD_AT_START[k]) s.cd_until[h] = ctx.now + cooldown[k];
            if (item == STRIDEBREAKER && fire && champs > 0.f) {
                s.stride_ms[h] = s_ms * champs;
                s.stride_ms_start[h] = ctx.now;
            }
        }
        if (finishing) s.cast_item[h] = 0;
    }
    for (auto& p : per_item) append(e.packets, p);
    return {s, e, out};
}

LANESIM_TEST(items_hydra_stats, "items.hydra.stats", stats);
LANESIM_TEST(items_hydra_on_hit, "items.hydra.on_hit", on_hit);
LANESIM_TEST(items_hydra_active, "items.hydra.active", active);

}  // namespace lanesim::items::hydra
