// items.effects.marksman: marksman / lethality item passives (lanerl_jax/modern/items/effects/marksman.py).
// Ported in full (hooks only; the integrator helpers at the bottom of marksman.py are not hooks). Reachable for
// Garen/Jax: Recurve Bow (Sting), Phantom Dancer (ghosted status) and Youmuu's Ghostblade (Haunt; Wraith Step lives
// in actives).
#include <cmath>

#include "../marshal.hpp"
#include "items_b.hpp"

namespace lanesim::items::marksman {

using namespace champ;
using itemsb::by_range;
using itemsb::holds_any;

namespace {

constexpr int RECURVE = 1043, FIENDHUNTER = 2512, HEXOPTICS = 2523, YUNTAL = 3032, LDR = 3036, PHANTOM = 3046;
constexpr int BLOODTHIRSTER = 3072, RUNAANS = 3085, STATIKK = 3087, RFC = 3094, STORMRAZOR = 3095, GUINSOO = 3124;
constexpr int SLINGSHOT = 3144, KRAKEN = 6672, NAVORI = 6675, COLLECTOR = 6676, YOUMUU = 3142, HUBRIS = 6697;
constexpr int AXIOM = 6696, SERPENT = 6695, VOLTAIC = 6699;
constexpr std::initializer_list<int> ENERGIZED_ITEMS = {RFC, STATIKK, STORMRAZOR, VOLTAIC};

#define KK(name) ([] { static const float v_ = itemsb::k("marksman", name); return v_; }())

bool enemy_champ(const Ctx& ctx, const Units& u, int c, int j) { return u.cls[j] == CLASS_CHAMPION && u.team[j] != ctx.team[c]; }
float level_bp(float start, float per_level, float from_level, float level) {
    return start + per_level * std::max(0.0f, level - from_level + 1.0f);
}
float statikk_bounces(float lv) {
    float s = 0.0f;
    for (float k : {6.0f, 10.0f, 14.0f, 20.0f}) s = s + (lv >= k ? 1.0f : 0.0f);
    return 4.0f + s;
}
int clip_unit(const Units& u, int idx) { return clampi(idx, 0, itemsb::n_units(u) - 1); }
float dist(const Units& u, int j, float px, float py) { return std::sqrt(sq(u.x[j] - px) + sq(u.y[j] - py)); }
// _energized: fire every owned Energized effect at ``tgt`` for holders ``fire``; appends the packets.
void energized(State& state, const Owned& own, const Ctx& ctx, const Units& units, const bool* fire, const int* tgt_in,
               Packets& out) {
    int n = itemsb::n_units(units);
    Packets p_rfc = empty_packets(), p_storm = empty_packets(), p_volt = empty_packets(), p_st = empty_packets();
    for (int c = 0; c < C; ++c) {
        int tgt = std::max(tgt_in[c], 0);
        int tcls = target_class(units, tgt);
        bool champ_ = tcls == CLASS_CHAMPION;
        bool rfc = fire[c] && holds(own, RFC, c);
        push(p_rfc, rfc, ctx.unit[c], tgt, KK("RFC_DMG"), MAGIC, ON_HIT_ITEM, 0.f, RFC);
        bool storm = fire[c] && holds(own, STORMRAZOR, c);
        push(p_storm, storm, ctx.unit[c], tgt, KK("STORM_DMG"), MAGIC, ON_HIT_ITEM, 0.f, STORMRAZOR);
        // Voltaic: % current HP before this attack, capped vs non-champions.
        bool volt = fire[c] && holds(own, VOLTAIC, c);
        float vdmg = (ctx.is_ranged[c] ? KK("VOLT_PCT_R") : KK("VOLT_PCT_M")) * units.hp[tgt];
        vdmg = champ_ ? vdmg : std::min(vdmg, KK("VOLT_CAP"));
        push(p_volt, volt, ctx.unit[c], tgt, vdmg, PHYSICAL, ON_HIT_ITEM, 0.f, VOLTAIC);
        // Statikk: chain to the nearest unhit enemy in range.
        bool st = fire[c] && holds(own, STATIKK, c) && tcls != CLASS_STRUCTURE;
        float count = statikk_bounces(ctx.level[c]);
        std::vector<uint8_t> hit(n, 0);
        for (int j = 0; j < n; ++j) hit[j] = itemsb::onehot(tgt, j) && st;
        int ti = clip_unit(units, tgt);
        float cx = units.x[ti], cy = units.y[ti];
        for (int k = 1; k < (int)KK("STATIKK_MAX_BOUNCES"); ++k) {
            std::vector<uint8_t> cand(n);
            std::vector<float> d(n);
            for (int j = 0; j < n; ++j) {
                cand[j] = enemy(ctx, units, c, j) && units.cls[j] != CLASS_STRUCTURE && !hit[j]
                          && in_circle(units, j, cx, cy, KK("STATIKK_RANGE"));
                d[j] = dist(units, j, cx, cy);
            }
            std::vector<uint8_t> pick = itemsb::nearest_k(d, cand, 1);
            bool on = st && (float)k < count;
            int nxt = -1;
            for (int j = 0; j < n; ++j) {
                pick[j] = pick[j] && on;
                hit[j] = hit[j] || pick[j];
                if (pick[j] && nxt < 0) nxt = j;
            }
            if (nxt >= 0) cx = units.x[nxt], cy = units.y[nxt];
        }
        bool any_secondary = false;
        size_t row = (size_t)c * n;
        bool same_tick = ctx.now == state.extra_at[c];
        for (int j = 0; j < n; ++j) {
            bool primary = itemsb::onehot(tgt, j);
            float chain = units.cls[j] == CLASS_CHAMPION ? KK("STATIKK_CHAMP") : KK("STATIKK_OTHER");
            push(p_st, hit[j], ctx.unit[c], j, chain, MAGIC, primary ? ON_HIT_ITEM : TAG_PROC | TAG_ITEM | TAG_AOE, 0.f, STATIKK);
            bool secondary = hit[j] && !primary;
            state.extra_hits[row + j] = same_tick ? (state.extra_hits[row + j] || secondary) : secondary;
            any_secondary = any_secondary || secondary;
        }
        if (fire[c]) state.energy[c] = 0.0f, state.en_pending[c] = 0;
        if (storm) state.storm_until[c] = ctx.now + KK("STORM_DUR");
        if (volt) state.volt_until[c] = ctx.now + KK("VOLT_DUR");
        if (any_secondary) state.extra_at[c] = ctx.now;
    }
    append(out, p_rfc), append(out, p_storm), append(out, p_volt), append(out, p_st);
}

}  // namespace

// marksman.stats
ItemStats stats(State state, Owned own, Ctx ctx) {
    float now = ctx.now;
    ItemStats o = itemsb::stats_out({&ItemStats::attack_speed, &ItemStats::crit_chance, &ItemStats::percent_move_speed, &ItemStats::move_speed, &ItemStats::lethality, &ItemStats::attack_damage, &ItemStats::ultimate_haste});
    for (int c = 0; c < C; ++c) {
        float guinsoo = holds(own, GUINSOO, c) && now < state.guinsoo_until[c] ? KK("GUINSOO_AS") * state.guinsoo_stacks[c] : 0.0f;
        float flurry = holds(own, YUNTAL, c) && now < state.yt_as_until[c] ? KK("YT_AS") : 0.0f;
        float barrage = holds(own, FIENDHUNTER, c) && state.fh_charges[c] > 0 && now < state.fh_until[c] ? KK("FH_AS") : 0.0f;
        bool ooc = holds(own, YOUMUU, c) && (now - state.champ_combat[c] >= KK("YOUMUU_TIMER"));
        o.attack_speed[c] = guinsoo + flurry + barrage;
        o.crit_chance[c] = holds(own, YUNTAL, c) ? state.yt_crit[c] : 0.0f;
        o.percent_move_speed[c] = holds(own, STORMRAZOR, c) && now < state.storm_until[c] ? KK("STORM_MS") : 0.0f;
        o.move_speed[c] = ooc ? KK("YOUMUU_OOC_MS") * by_range(ctx, c, KK("YOUMUU_RANGED")) : 0.0f;
        o.lethality[c] = holds(own, VOLTAIC, c) && now < state.volt_until[c]
                             ? (ctx.is_ranged[c] ? KK("VOLT_LETH_R") : KK("VOLT_LETH_M")) : 0.0f;
        o.attack_damage[c] = holds(own, HUBRIS, c) && now < state.hubris_until[c]
                                 ? KK("HUBRIS_BASE") + KK("HUBRIS_PER") * state.hubris_stacks[c] : 0.0f;
        o.ultimate_haste[c] = holds(own, FIENDHUNTER, c) ? KK("FH_HASTE") : 0.0f;
    }
    return o;
}

// marksman.status: Phantom Dancer Spectral Waltz.
StatusFlags status(State state, Owned own, Ctx ctx) {
    StatusFlags s;
    s.ghosted.assign(C, 0);
    for (int c = 0; c < C; ++c) s.ghosted[c] = holds(own, PHANTOM, c) && ctx.alive[c];
    return s;
}

// marksman.dealt_amp: Lord Dominik's Giant Slayer. (C, N)
Arr<float> dealt_amp(State state, Owned own, Ctx ctx, Units units) {
    int n = itemsb::n_units(units);
    Arr<float> out((size_t)C * n, 0.f);
    for (int c = 0; c < C; ++c)
        for (int j = 0; j < n; ++j) {
            float frac = std::min(std::max(units.bonus_hp[j] / KK("LDR_HP"), 0.0f), 1.0f);
            out[(size_t)c * n + j] = holds(own, LDR, c) && enemy_champ(ctx, units, c, j) ? KK("LDR_MAX") * frac : 0.0f;
        }
    return out;
}

// marksman.attack_mods: Fiendhunter Opening Barrage forced crit.
AttackMods attack_mods(State state, Owned own, Ctx ctx, Units units, Arr<int32_t> target) {
    AttackMods m;
    m.force_crit.assign(C, 0), m.crit_scale.assign(C, KK("FH_CRIT"));
    for (int c = 0; c < C; ++c)
        m.force_crit[c] = holds(own, FIENDHUNTER, c) && state.fh_charges[c] > 0 && ctx.now < state.fh_until[c] && ctx.alive[c];
    return m;
}

// marksman.packet_amp: Hexoptics Magnification on basic attacks.
Arr<float> packet_amp(State state, Owned own, Ctx ctx, Units units, Packets p) {
    int n = itemsb::n_units(units);
    std::vector<float> amp((size_t)C * n);   // basic_attack_amp
    for (int c = 0; c < C; ++c) {
        int h = clip_unit(units, ctx.unit[c]);
        float hx = units.x[h], hy = units.y[h], hr = units.radius[h];
        for (int j = 0; j < n; ++j) {
            float edge = std::max(dist(units, j, hx, hy) - hr - units.radius[j], 0.0f);
            amp[(size_t)c * n + j] = holds(own, HEXOPTICS, c) ? KK("HEX_AMP") * std::min(std::max(edge / KK("HEX_RANGE"), 0.0f), 1.0f) : 0.0f;
        }
    }
    Arr<float> out(size(p), 0.f);
    for (size_t i = 0; i < size(p); ++i) {
        int d = clampi(p.dst[i], 0, n - 1);
        float s = (p.src[i] == ctx.unit[0] ? amp[d] : 0.0f) + (p.src[i] == ctx.unit[1] ? amp[(size_t)n + d] : 0.0f);
        out[i] = has(p.flags[i], TAG_BASIC_ATTACK) ? s : 0.0f;
    }
    return out;
}

// marksman.on_cast: Fiendhunter Opening Barrage after R.
std::tuple<State, Effects> on_cast(State state, Owned own, Ctx ctx, Units units, Cast cast) {
    for (int c = 0; c < C; ++c) {
        bool go = cast.started[c] && cast.slot[c] == 3 && holds(own, FIENDHUNTER, c) && ctx.now >= state.fh_cd_until[c] && ctx.alive[c];
        if (go) state.fh_charges[c] = KK("FH_N"), state.fh_until[c] = ctx.now + KK("FH_DUR"), state.fh_cd_until[c] = ctx.now + KK("FH_CD");
    }
    return {state, no_effects(C, itemsb::n_units(units))};
}

// marksman.on_attack
std::tuple<State, Effects> on_attack(State state, Owned own, Ctx ctx, Units units, Attack attack) {
    int n = itemsb::n_units(units);
    float now = ctx.now;
    Effects e = no_effects(C, n);
    for (int c = 0; c < C; ++c) {
        size_t row = (size_t)c * n;
        bool la = attack.launched[c] && ctx.alive[c];
        int tgt = std::max(attack.target[c], 0);
        bool vs_champ = attack.target[c] >= 0 && target_class(units, tgt) == CLASS_CHAMPION;

        bool has_en = holds_any(own, ENERGIZED_ITEMS, c);
        bool en_now = la && has_en && state.energy[c] >= KK("ENERGY_MAX");
        float gain = KK("ENERGY_PER_ATTACK") + (holds(own, STATIKK, c) ? KK("STATIKK_BONUS") : 0.0f);
        if (la && has_en && !en_now) state.energy[c] = std::min(KK("ENERGY_MAX"), state.energy[c] + gain);
        if (la) state.en_pending[c] = en_now;

        // Guinsoo's: the attack reaching max stacks grants the first Phantom stack.
        bool g = la && holds(own, GUINSOO, c);
        bool alive_g = now < state.guinsoo_until[c];
        float gst = alive_g ? state.guinsoo_stacks[c] : 0.0f;
        float ph = alive_g ? state.phantom[c] : 0.0f;
        bool fire_ph = g && ph >= KK("GUINSOO_PHANTOM_MAX");
        float gst_new = std::min(KK("GUINSOO_MAX"), gst + 1.0f);
        float ph_new = fire_ph ? 0.0f : (gst_new >= KK("GUINSOO_MAX") ? std::min(KK("GUINSOO_PHANTOM_MAX"), ph + 1.0f) : ph);
        if (g) state.guinsoo_stacks[c] = gst_new, state.guinsoo_until[c] = now + KK("GUINSOO_DUR"), state.phantom[c] = ph_new;
        if (la) state.phantom_pending[c] = fire_ph;

        // Kraken (ranged): stacks at launch; the 3rd consumes them.
        bool kr = la && holds(own, KRAKEN, c) && ctx.is_ranged[c];
        float kst = now < state.kraken_until[c] ? state.kraken_stacks[c] : 0.0f;
        bool k_fire = kr && kst >= KK("KRAKEN_COUNT_M1");
        if (kr) state.kraken_stacks[c] = k_fire ? 0.0f : kst + 1.0f, state.kraken_until[c] = now + KK("KRAKEN_DUR");
        if (la && ctx.is_ranged[c]) state.kraken_pending[c] = k_fire;

        bool y = la && holds(own, YUNTAL, c);
        if (y) state.yt_crit[c] = std::min(KK("YT_CRIT_MAX"), state.yt_crit[c] + KK("YT_CRIT_PER") * by_range(ctx, c, KK("YT_RANGED")));
        bool flurry = y && vs_champ && now >= state.yt_cd_until[c];
        if (flurry) state.yt_as_until[c] = now + KK("YT_DUR"), state.yt_cd_until[c] = now + KK("YT_CD");

        bool fh = la && state.fh_charges[c] > 0 && now < state.fh_until[c] && holds(own, FIENDHUNTER, c);
        if (la) state.fh_attack[c] = fh;
        if (fh) state.fh_charges[c] = state.fh_charges[c] - 1.0f;
        if (la && holds(own, SLINGSHOT, c)) state.sling_cd_until[c] = state.sling_cd_until[c] - KK("SLING_ATTACK_CDR");
        if (la && holds(own, NAVORI, c)) state.navori_at[c] = now;

        // Runaan's bolts at the nearest enemies in front.
        bool run = la && holds(own, RUNAANS, c);
        int h = clip_unit(units, ctx.unit[c]);
        float hx = units.x[h], hy = units.y[h];
        std::vector<uint8_t> cand(n);
        std::vector<float> d(n);
        for (int j = 0; j < n; ++j) {
            float rx = units.x[j] - hx, ry = units.y[j] - hy;
            bool front = rx * ctx.facing_x[c] + ry * ctx.facing_y[c] >= 0.0f;
            bool reach = in_circle(units, j, hx, hy, KK("RUNAAN_REACH"));
            bool enemies = enemy(ctx, units, c, j) && units.cls[j] != CLASS_STRUCTURE && !itemsb::onehot(attack.target[c], j);
            d[j] = std::sqrt(sq(rx) + sq(ry));
            cand[j] = enemies && front && reach;
        }
        int bolts_k = (int)(ctx.is_ranged[c] ? KK("RUNAAN_BOLTS_RANGED") : KK("RUNAAN_BOLTS_MELEE"));
        std::vector<uint8_t> bolts = itemsb::nearest_k(d, cand, bolts_k);
        float crit_mult = attack.is_crit[c] ? ctx.crit_damage[c] : 1.0f;
        int bolt_flags = TAG_PROC | TAG_ITEM | PROP_LIFESTEAL | (attack.is_crit[c] ? PROP_CRIT : 0);
        float raw = KK("RUNAAN_RATIO") * (ctx.base_ad[c] + ctx.bonus_ad[c]) * crit_mult;
        bool any_bolt = false;
        for (int j = 0; j < n; ++j) {
            bool b = bolts[j] && run;
            push(e.packets, b, ctx.unit[c], j, raw, PHYSICAL, bolt_flags, 0.f, RUNAANS);
            any_bolt = any_bolt || b;
        }
        if (any_bolt) {
            for (int j = 0; j < n; ++j) state.extra_hits[row + j] = bolts[j] && run;
            state.extra_at[c] = now;
        }
    }
    return {state, e};
}

// marksman.on_hit
std::tuple<State, Effects> on_hit(State state, Owned own, Ctx ctx, Units units, Attack attack) {
    int n = itemsb::n_units(units);
    float now = ctx.now;
    Effects e = no_effects(C, n);
    Packets p_rec = empty_packets(), p_gui = empty_packets(), p_kr = empty_packets(), p_fh = empty_packets();
    const int lsf = ON_HIT_ITEM | PROP_LIFESTEAL;
    bool en[C];
    int tgts[C];
    for (int c = 0; c < C; ++c) {
        bool h = attack.hit[c] && ctx.alive[c] && attack.target[c] >= 0;
        int tgt = std::max(attack.target[c], 0);
        tgts[c] = tgt;
        push(p_rec, h && holds(own, RECURVE, c), ctx.unit[c], tgt, KK("RECURVE_DMG"), PHYSICAL, lsf, 0.f, RECURVE);
        push(p_gui, h && holds(own, GUINSOO, c), ctx.unit[c], tgt, KK("GUINSOO_DMG"), MAGIC, lsf, 0.f, GUINSOO);

        // Kraken: melee stacks on-hit; ranged consumes what launch reserved.
        bool km = h && holds(own, KRAKEN, c) && !ctx.is_ranged[c];
        float kst = now < state.kraken_until[c] ? state.kraken_stacks[c] : 0.0f;
        bool km_fire = km && kst >= KK("KRAKEN_COUNT_M1");
        bool kr_fire = h && holds(own, KRAKEN, c) && ctx.is_ranged[c] && state.kraken_pending[c];
        float missing = 1.0f - units.hp[tgt] / std::max(units.max_hp[tgt], 1.0f);
        float kdmg = level_bp(150.0f, 5.0f, 9.0f, ctx.level[c]) * by_range(ctx, c, KK("KRAKEN_RANGED"))
                     * (1.0f + KK("KRAKEN_AMP_M1") * std::min(std::max(missing, 0.0f), 1.0f));
        push(p_kr, km_fire || kr_fire, ctx.unit[c], tgt, kdmg, PHYSICAL, lsf, 0.f, KRAKEN);

        // Fiendhunter: a natural crit on an empowered attack adds true damage.
        float forced_raw = (ctx.base_ad[c] + ctx.bonus_ad[c]) * (1.0f + KK("FH_CRIT") * (ctx.crit_damage[c] - 1.0f));
        bool natural = attack.natural_crit.size() ? (bool)attack.natural_crit[c]
                                                  : attack.is_crit[c] && attack.raw[c] > forced_raw * 1.001f;
        bool fh_true = h && state.fh_attack[c] && natural;
        push(p_fh, fh_true, ctx.unit[c], tgt, KK("FH_TRUE") * attack.raw[c], TRUE_DMG, TAG_PROC | TAG_ITEM, 0.f, FIENDHUNTER);

        bool yt = h && holds(own, YUNTAL, c);
        bool phantom_due = h && state.phantom_pending[c];
        if (km) state.kraken_stacks[c] = km_fire ? 0.0f : kst + 1.0f, state.kraken_until[c] = now + KK("KRAKEN_DUR");
        if (kr_fire) state.kraken_pending[c] = 0;
        if (h) state.fh_attack[c] = 0;
        if (yt) state.yt_cd_until[c] = state.yt_cd_until[c] - (attack.is_crit[c] ? KK("YT_CRIT_CDR") : KK("YT_AA_CDR"));
        if (h) state.phantom_pending[c] = 0;
        if (phantom_due) state.phantom_at[c] = now;
        en[c] = h && state.en_pending[c] && state.energy[c] >= KK("ENERGY_MAX") && holds_any(own, ENERGIZED_ITEMS, c);
    }
    append(e.packets, p_rec), append(e.packets, p_gui), append(e.packets, p_kr), append(e.packets, p_fh);
    energized(state, own, ctx, units, en, tgts, e.packets);
    return {state, e};
}

// marksman.on_damage
std::tuple<State, Effects> on_damage(State state, Owned own, Ctx ctx, Units units, Report report) {
    int n = itemsb::n_units(units);
    float now = ctx.now;
    const Packets& p = report.packets;
    const Resolved& r = report.resolved;
    int np = (int)size(p);
    const State s0 = state;
    Effects e = no_effects(C, n);
    auto src_is = [&](int c, int i) { return p.src[i] == ctx.unit[c]; };
    auto landed = [&](int i) { return p.valid[i] && r.final[i] > 0.0f; };
    // dealt_by_holder / hit_by_holder
    Arr<float> dealt = itemsb::per_unit_add(p, n, [&](int c, int i) { return src_is(c, i) && landed(i) ? r.final[i] : 0.0f; });
    Arr<uint8_t> sling_hit = itemsb::per_unit_any(p, n, [&](int c, int i) { return src_is(c, i) && landed(i) && p.item[i] != SLINGSHOT; });
    Arr<uint8_t> gal_hit = itemsb::per_unit_any(p, n, [&](int c, int i) {
        return src_is(c, i) && landed(i) && has(p.flags[i], TAG_ACTIVE_SPELL) && !has(p.flags[i], TAG_ITEM);
    });
    std::vector<uint8_t> already(n, 0);
    for (int j = 0; j < n; ++j)
        for (int c = 0; c < C; ++c) already[j] = already[j] || now < s0.venom_until[(size_t)c * n + j];
    Packets p_sling = empty_packets(), p_exec = empty_packets(), p_gal = empty_packets();
    Arr<float> grant(C, 0.f);
    bool gal[C];
    int gal_tgt[C];
    int nk = itemsb::shield_slots(r.shields, n);
    for (int c = 0; c < C; ++c) {
        size_t row = (size_t)c * n;
        bool any_hit = false, any_fresh = false, sling_any = false, gal_any = false;
        int sling_tgt = 0;
        gal_tgt[c] = 0;
        for (int j = 0; j < n; ++j) {
            bool hit_champ = dealt[row + j] > 0.0f && enemy_champ(ctx, units, c, j);
            any_hit = any_hit || hit_champ;
            if (hit_champ) state.last_dmg[row + j] = now;
            bool sf = holds(own, SERPENT, c) && hit_champ;
            any_fresh = any_fresh || (sf && !already[j]);
            if (sf) state.venom_until[row + j] = now + KK("SERPENT_DUR");
            bool sm = sling_hit[row + j] && enemy_champ(ctx, units, c, j);
            if (sm && !sling_any) sling_any = true, sling_tgt = j;
            bool gm = gal_hit[row + j] && enemy_champ(ctx, units, c, j);
            if (gm && !gal_any) gal_any = true, gal_tgt[c] = j;
            bool execute = holds(own, COLLECTOR, c) && hit_champ && r.hp[j] > 0.0f && r.hp[j] < KK("COLLECTOR_THRESHOLD") * r.max_hp[j];
            push(p_exec, execute, ctx.unit[c], j, r.hp[j], TRUE_DMG, PROP_EXECUTE | TAG_ITEM, 0.f, COLLECTOR);
        }
        bool took = false;
        for (int i = 0; i < np; ++i) {
            int sc = units.cls[clampi(p.src[i], 0, n - 1)];
            bool from_champ = landed(i) && sc == CLASS_CHAMPION && p.src[i] != p.dst[i];
            took = took || (p.dst[i] == ctx.unit[c] && from_champ);
        }
        if (any_hit || took) state.champ_combat[c] = now;
        if (any_fresh) {
            for (int j = 0; j < n; ++j) {
                bool hit_champ = dealt[row + j] > 0.0f && enemy_champ(ctx, units, c, j);
                state.venom_fresh[row + j] = holds(own, SERPENT, c) && hit_champ && !already[j];
            }
            state.venom_fresh_at[c] = now;
        }
        // Slingshot: first enemy champion damaged by another packet.
        bool sling = holds(own, SLINGSHOT, c) && ctx.alive[c] && now >= s0.sling_cd_until[c] && sling_any;
        push(p_sling, sling, ctx.unit[c], sling_tgt, KK("SLING_DMG"), MAGIC, TAG_PROC | TAG_ITEM, 0.f, SLINGSHOT);
        if (sling) state.sling_cd_until[c] = now + KK("SLING_CD");

        // Bloodthirster Ichorshield.
        int u = ctx.unit[c];
        float remaining = 0.0f;
        for (int s = 0; s < nk; ++s) remaining = remaining + itemsb::shield_value(r.shields, (size_t)u * nk + s, now);
        float ichor = std::min(s0.ichor[c], remaining);
        float heal = report.life_steal_heal[u];
        float missing = std::max(r.max_hp[u] - r.hp[u], 0.0f);
        float cap = level_bp(165.0f, 15.0f, 9.0f, ctx.level[c]);
        bool bt = holds(own, BLOODTHIRSTER, c) && ctx.alive[c] && r.hp[u] > 0.0f;
        grant[c] = bt ? std::max(std::min(heal - missing, cap - ichor), 0.0f) : 0.0f;
        state.ichor[c] = ichor + grant[c];

        // Voltaic Galvanize (uses the energy before this hook's Energized resets).
        gal[c] = holds(own, VOLTAIC, c) && ctx.alive[c] && state.energy[c] >= KK("ENERGY_MAX") && gal_any;
    }
    energized(state, own, ctx, units, gal, gal_tgt, p_gal);
    append(e.packets, p_sling), append(e.packets, p_exec), append(e.packets, p_gal);
    e.shields = shield_grants(grant, SHIELD_ALL, KK("ICHOR_DURATION"));
    return {state, e};
}

// marksman.periodic: Energize charge from movement.
std::tuple<State, Effects> periodic(State state, Owned own, Ctx ctx, Units units) {
    for (int c = 0; c < C; ++c) {
        float gain = holds_any(own, ENERGIZED_ITEMS, c) && ctx.alive[c] ? ctx.moved[c] / KK("ENERGY_UNITS_PER_STACK") : 0.0f;
        state.energy[c] = std::min(KK("ENERGY_MAX"), state.energy[c] + gain);
    }
    return {state, no_effects(C, itemsb::n_units(units))};
}

// marksman.on_takedown: Hubris, Hexoptics Arcane Aim, Axiom Arc, Collector gold.
std::tuple<State, Effects> on_takedown(State state, Owned own, Ctx ctx, Units units, Kills kills) {
    int n = itemsb::n_units(units);
    float now = ctx.now;
    Effects e = no_effects(C, n);
    for (int c = 0; c < C; ++c) {
        float count = 0.0f;
        for (int j = 0; j < n; ++j) {
            size_t cj = (size_t)c * n + j;
            count += kills.killed_units[cj] && enemy_champ(ctx, units, c, j) && (now - state.last_dmg[cj] <= KK("TAKEDOWN_WINDOW")) ? 1.0f : 0.0f;
        }
        bool any_td = count > 0;
        bool hub = any_td && holds(own, HUBRIS, c), ax = any_td && holds(own, AXIOM, c);
        float refund = count * (KK("AXIOM_BASE") + KK("AXIOM_PER_LETHALITY") * ctx.lethality[c]);
        e.gold[c] = holds(own, COLLECTOR, c) ? KK("COLLECTOR_GOLD") * kills.champion_kill[c] : 0.0f;
        if (hub) state.hubris_stacks[c] = state.hubris_stacks[c] + count, state.hubris_until[c] = now + KK("HUBRIS_DUR");
        if (any_td && holds(own, HEXOPTICS, c)) state.hex_until[c] = now + KK("HEX_DUR");
        if (ax) state.axiom_refund[c] = refund, state.axiom_at[c] = now;
    }
    return {state, e};
}

LANESIM_TEST(items_marksman_stats, "items.marksman.stats", marksman::stats);
LANESIM_TEST(items_marksman_status, "items.marksman.status", marksman::status);
LANESIM_TEST(items_marksman_dealt_amp, "items.marksman.dealt_amp", marksman::dealt_amp);
LANESIM_TEST(items_marksman_attack_mods, "items.marksman.attack_mods", marksman::attack_mods);
LANESIM_TEST(items_marksman_packet_amp, "items.marksman.packet_amp", marksman::packet_amp);
LANESIM_TEST(items_marksman_on_cast, "items.marksman.on_cast", marksman::on_cast);
LANESIM_TEST(items_marksman_on_attack, "items.marksman.on_attack", marksman::on_attack);
LANESIM_TEST(items_marksman_on_hit, "items.marksman.on_hit", marksman::on_hit);
LANESIM_TEST(items_marksman_on_damage, "items.marksman.on_damage", marksman::on_damage);
LANESIM_TEST(items_marksman_periodic, "items.marksman.periodic", marksman::periodic);
LANESIM_TEST(items_marksman_on_takedown, "items.marksman.on_takedown", marksman::on_takedown);

}  // namespace lanesim::items::marksman
