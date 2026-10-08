// items/effects/actives.py. Reachable for Garen/Jax: Zhonya's, Seeker's Armguard, Youmuu's, Hextech Gunblade. The
// other actives are not holdable; the state-driven parts that run regardless (buff timers, Actualizer amp,
// Rocketbelt dash end point, Redemption landing) and their padded packets / grants are kept.
#include "../marshal.hpp"
#include "items.hpp"

namespace lanesim::items::actives {

namespace {
constexpr int ZHONYAS = 3157, SEEKERS = 2420, QUICKSILVER = 3140, MERCURIAL = 3139, YOUMUU = 3142, RANDUINS = 3143,
              GUNBLADE = 3146, ROCKETBELT = 3152, SHURELYA = 2065, LOCKET = 3190, REDEMPTION = 3107,
              ACTUALIZER = 2522;
constexpr int ACTIVE_ITEMS[12] = {ZHONYAS, SEEKERS, QUICKSILVER, MERCURIAL, YOUMUU, RANDUINS, GUNBLADE, ROCKETBELT,
                                  SHURELYA, LOCKET, REDEMPTION, ACTUALIZER};
constexpr int NK = 12;
constexpr int K_ZHONYAS = 0, K_SEEKERS = 1, K_YOUMUU = 4, K_GUNBLADE = 6, K_ROCKETBELT = 7;
constexpr float ROCKET_DASH = 275.0f, GUNBLADE_RANGE = 700.0f;

float K(const char* name) { return data::f(std::string("items.actives.") + name); }

// actualizer_amp: Mana Made Real ability damage / heal-shield power
float actualizer_amp(const State& s, const Ctx& ctx, int c) {
    return ctx.now < s.actualizer_until[c] ? (15.0f + 0.005f * ctx.max_mana[c]) * 0.01f : 0.f;
}
}  // namespace

// actives.stats
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx) {
    static const float shurelya = K("shurelya_ms"), mercurial = K("mercurial_ms");
    size_t c = ctx.unit.size();
    ItemStats o = default_stats();
    o.percent_move_speed = o.heal_shield_power = zeros_c(c);
    for (size_t h = 0; h < c; ++h) {
        float now = ctx.now;
        o.percent_move_speed[h] = (now < s.shurelya_until[h] ? shurelya : 0.f)
                                  + (now < s.mercurial_until[h] ? mercurial : 0.f)
                                  + (now < s.youmuu_until[h] ? s.youmuu_ms[h] : 0.f);
        o.heal_shield_power[h] = actualizer_amp(s, ctx, h);
    }
    return o;
}

// actives.status: Youmuu's ghosting
StatusFlags status(const State& s, const Owned& own, const Ctx& ctx) {
    StatusFlags f;
    f.ghosted.assign(ctx.unit.size(), 0);
    for (size_t h = 0; h < ctx.unit.size(); ++h) f.ghosted[h] = ctx.now < s.youmuu_until[h];
    return f;
}

// actives.packet_amp: Actualizer amp on the holder's ability packets
Arr<float> packet_amp(const State& s, const Owned& own, const Ctx& ctx, const Units& u, const Packets& p) {
    Arr<float> out(size(p), 0.f);
    for (size_t k = 0; k < size(p); ++k) {
        bool ability = has(p.flags[k], TAG_ACTIVE_SPELL) && !has(p.flags[k], TAG_ITEM);
        float sum = 0.f;
        for (size_t h = 0; h < ctx.unit.size(); ++h)
            sum = sum + (p.src[k] == ctx.unit[h] && ability ? actualizer_amp(s, ctx, h) : 0.f);
        out[k] = sum;
    }
    return out;
}

// actives.active
std::tuple<State, Effects, ActiveOut> active(State s, const Owned& own, const Ctx& ctx, const Units& u,
                                             const Arr<int32_t>& request) {
    static const std::vector<float> cds = data::table("items.actives.cooldown");
    static const float stasis = K("stasis"), y_dur_m = K("youmuu_duration"), y_dur_r = K("youmuu_duration_ranged"),
                       y_ms_m = K("youmuu_ms_melee"), y_ms_r = K("youmuu_ms_ranged"), g_slow = K("gunblade_slow"),
                       g_slow_d = K("gunblade_slow_duration"), rb_base = K("rocket_raw_base"),
                       rb_ap = K("rocket_raw_ap"), locket_dur = K("locket_duration"), red_aoe = K("red_aoe"),
                       red_dmg = K("red_damage"), red_heal_min = K("red_heal_min"), red_heal_span = K("red_heal_span");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    Effects e = no_effects(c, n);
    ActiveOut out;
    out.used.assign(c, 0), out.cast_time.assign(c, 0.f), out.can_move.assign(c, 1), out.attack_reset.assign(c, 0);
    Packets p_g = empty_packets(0), p_rb = empty_packets(0), p_red = empty_packets(0);
    Arr<float> amount(c, 0.f);
    for (int h = 0; h < c; ++h) {
        int req = request[h];
        bool in_stasis = now < s.stasis_until[h];
        bool go[NK];
        for (int k = 0; k < NK; ++k) {
            int iid = ACTIVE_ITEMS[k];
            bool alive_ok = iid == REDEMPTION ? true : (bool)ctx.alive[h];
            go[k] = req == iid && holds(own, iid, h) && now >= s.cd_until[h * NK + k] && !in_stasis && alive_ok;
        }
        int hu = ctx.unit[h];
        float hx = u.x[hu], hy = u.y[hu], hr = u.radius[hu];
        // Gunblade: the aimed enemy champion, else the nearest in range.
        int aim = clampi(s.aim_unit[h], 0, n - 1);
        bool any_in = false, aim_in = false;
        int near = 0;
        float best = INF;
        for (int j = 0; j < n; ++j) {
            bool champ = enemy(ctx, u, h, j) && u.cls[j] != CLASS_STRUCTURE && u.cls[j] == CLASS_CHAMPION;
            float d = std::sqrt(sq(u.x[j] - hx) + sq(u.y[j] - hy));
            bool g_in = champ && d <= GUNBLADE_RANGE + hr + u.radius[j];
            float key = g_in ? d : INF;
            if (key < best) best = key, near = j;
            any_in = any_in || g_in;
            if (j == aim) aim_in = g_in;
        }
        bool aimed = s.aim_unit[h] >= 0 && aim_in;
        int g_tgt = aimed ? aim : near;
        go[K_GUNBLADE] = go[K_GUNBLADE] && any_in;
        bool used = false;
        for (int k = 0; k < NK; ++k) {
            used = used || go[k];
            if (go[k]) s.cd_until[h * NK + k] = now + cds[k];
        }
        if (go[K_ZHONYAS] || go[K_SEEKERS]) s.stasis_until[h] = now + stasis;
        bool ranged = ctx.is_ranged[h];
        if (go[K_YOUMUU]) {
            s.youmuu_until[h] = now + (ranged ? y_dur_r : y_dur_m);
            s.youmuu_ms[h] = (ranged ? y_ms_r : y_ms_m) * 0.01f;
        }
        // Gunblade bolt (Randuin's slow not holdable)
        float lv = std::min(std::max(ctx.level[h], 1.f), 18.f);
        float g_raw = 175.0f + (lv - 1.f) / 17.f * (253.0f - 175.0f) + 0.3f * ctx.ap[h];
        for (int j = 0; j < n; ++j) {
            bool g_hit = j == g_tgt && go[K_GUNBLADE];
            push_cn(p_g, g_hit, hu, j, g_raw, MAGIC, TAG_ACTIVE_SPELL | TAG_ITEM, GUNBLADE);
            if (g_hit && g_slow > e.slow[j]) e.slow[j] = g_slow, e.slow_duration[j] = g_slow_d;
        }
        // Rocketbelt (not holdable): dash end point along the aim / facing, padded rockets.
        float ax = s.aim_set[h] ? s.aim_x[h] - hx : ctx.facing_x[h];
        float ay = s.aim_set[h] ? s.aim_y[h] - hy : ctx.facing_y[h];
        float norm = std::sqrt(sq(ax) + sq(ay));
        float ux = norm > 1e-6f ? ax / std::max(norm, 1e-6f) : 1.f;
        float uy = norm > 1e-6f ? ay / std::max(norm, 1e-6f) : 0.f;
        float ex = hx + ROCKET_DASH * ux, ey = hy + ROCKET_DASH * uy;
        float rb_raw = rb_base + rb_ap * ctx.ap[h];
        for (int j = 0; j < n; ++j)
            push_cn(p_rb, false, hu, j, rb_raw, MAGIC, TAG_AOE | TAG_ACTIVE_SPELL | TAG_ITEM, ROCKETBELT);
        // Redemption (not holdable): a pending landing still resolves.
        bool land = now >= s.red_at[h];
        bool self_in = false;
        for (int j = 0; j < n; ++j) {
            bool area = in_circle(u, j, s.red_x[h], s.red_y[h], red_aoe);
            bool champ = enemy(ctx, u, h, j) && u.cls[j] != CLASS_STRUCTURE && u.cls[j] == CLASS_CHAMPION;
            push_cn(p_red, area && champ && land, hu, j, red_dmg * u.max_hp[j], TRUE_DMG,
                    TAG_AOE | TAG_ACTIVE_SPELL | TAG_ITEM, REDEMPTION);
            if (j == hu) self_in = area && ctx.alive[h] && land;
        }
        float rl = std::max(ctx.level[h], 1.f);
        e.heal[h] = self_in ? red_heal_min + (rl - 1.f) / 17.f * red_heal_span : 0.f;
        if (land) s.red_at[h] = INF;
        s.aim_set[h] = 0, s.cleanse_now[h] = 0, s.dash_now[h] = go[K_ROCKETBELT];
        s.dash_x[h] = ex, s.dash_y[h] = ey, s.shatter_now[h] = go[K_SEEKERS];
        e.attack_reset[h] = go[K_ROCKETBELT];
        out.used[h] = used, out.attack_reset[h] = go[K_ROCKETBELT];
    }
    append(e.packets, p_g), append(e.packets, p_rb), append(e.packets, p_red);
    e.shields = shield_grants(amount, SHIELD_ALL, locket_dur, 0.f);    // Locket not holdable
    return {s, e, out};
}

// actives.with_aim: store this tick's order target for targeted actives
State with_aim(State s, const Arr<int32_t>& unit, const Arr<float>& x, const Arr<float>& y) {
    size_t c = s.aim_unit.size();
    for (size_t h = 0; h < c; ++h) {
        s.aim_unit[h] = unit.size() ? unit[unit.size() == 1 ? 0 : h] : -1;
        if (x.size()) {
            s.aim_x[h] = x[x.size() == 1 ? 0 : h], s.aim_y[h] = y[y.size() == 1 ? 0 : h];
        }
        s.aim_set[h] = x.size() != 0;
    }
    return s;
}

// actives.request_allowed: only cleanse items while disabled, nothing in stasis
Arr<int32_t> request_allowed(const Arr<int32_t>& request, const Arr<uint8_t>& disabled, const Arr<uint8_t>& in_stasis) {
    Arr<int32_t> out(request.size(), 0);
    for (size_t h = 0; h < request.size(); ++h) {
        int req = request[h];
        bool qss = req == QUICKSILVER || req == MERCURIAL;
        bool ok = !disabled[h] || qss;
        if (in_stasis.size()) ok = ok && !in_stasis[h];
        out[h] = ok ? req : 0;
    }
    return out;
}

// actives.world: world effects of item actives
ActiveWorld world(const State& s, float now) {
    static const float mana_mult = K("actualizer_mana_cost_mult"), cd_rate = K("actualizer_cd_rate");
    constexpr float ROCKET_DASH_SPEED = 1500.0f;
    int c = (int)s.stasis_until.size();
    ActiveWorld w;
    w.stasis.assign(c, 0), w.stasis_until = s.stasis_until, w.cleanse = s.cleanse_now;
    w.dash = no_dash(c);
    w.mana_cost_mult.assign(c, 1.f), w.basic_cd_rate.assign(c, 1.f);
    w.transform_from.assign(c, item_row(SEEKERS)), w.transform_to.assign(c, item_row(2421 /*Shattered*/));
    w.transform_do = s.shatter_now;
    for (int h = 0; h < c; ++h) {
        bool on = now < s.actualizer_until[h];
        w.stasis[h] = now < s.stasis_until[h];
        w.dash.active[h] = s.dash_now[h], w.dash.to_x[h] = s.dash_x[h], w.dash.to_y[h] = s.dash_y[h];
        w.dash.speed[h] = ROCKET_DASH_SPEED;
        if (on) w.mana_cost_mult[h] = mana_mult, w.basic_cd_rate[h] = cd_rate;
    }
    return w;
}

LANESIM_TEST(items_actives_with_aim, "items.actives.with_aim", with_aim);
LANESIM_TEST(items_actives_request_allowed, "items.actives.request_allowed", request_allowed);
LANESIM_TEST(items_actives_world, "items.actives.world", world);
LANESIM_TEST(items_actives_stats,"items.actives.stats", stats);
LANESIM_TEST(items_actives_status, "items.actives.status", status);
LANESIM_TEST(items_actives_packet_amp, "items.actives.packet_amp", packet_amp);
LANESIM_TEST(items_actives_active, "items.actives.active", active);

}  // namespace lanesim::items::actives
