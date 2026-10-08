// items.effects.mage: burns, ability-damage passives, AP scaling and on-damage procs
// (lanerl_jax/modern/items/effects/mage.py). Ported in full; reachable for Garen/Jax: Hextech Alternator (Revved)
// and Hextech Gunblade (stats only, its active lives in actives).
#include <cmath>
#include <limits>

#include "../marshal.hpp"
#include "items_b.hpp"

namespace lanesim::items::mage {

using namespace champ;

namespace {

constexpr int ASHES = 2508, BLACKFIRE = 2503, RABADON = 3089;
constexpr int NASHOR = 3115, RYLAI = 3116, MALIGNANCE = 3118, CRYPTBLOOM = 3137;
constexpr int ALTERNATOR = 3145, GUISE = 3147;
constexpr int MORELLO = 3165, CHAPTER = 3802, CATALYST = 3803, ORB = 3916;
constexpr int HORIZON = 4628, COSMIC = 4629, RIFTMAKER = 4633, SHADOWFLAME = 4645;
constexpr int STORMSURGE = 4646, LIANDRY = 6653, LUDEN = 6655, ROA = 6657, BLOODLETTER = 8010;
constexpr float BIG = 1e9f;

#define DV(id, name) ([] { static const float v_ = itemsb::dv("mage", id, name); return v_; }())
#define KK(name) ([] { static const float v_ = itemsb::k("mage", name); return v_; }())

// _ticks: (ticks due by now, next tick time) for a timer ticking at nxt, nxt + period, ... <= until.
void ticks(float until, float nxt, float now, float period, float& k, float& next) {
    const float EPS_ = KK("EPS");
    float end = std::min(now, until);
    k = nxt <= end + EPS_ ? std::floor((end - nxt) / period + EPS_) + 1.0f : 0.0f;
    next = nxt + k * period;
}
// _apply_timer: (re)apply a ticking timer; refresh keeps phase, a finished timer restarts.
void apply_timer(float& until, float& nxt, bool trig, float now, float duration, float period) {
    bool active = nxt <= until + KK("EPS");
    float u = trig ? now + duration : until;
    nxt = trig && !active ? now + period : nxt;
    until = u;
}
// _stacks: whole seconds of champion combat since ``start``, capped.
float stacks_c(const State& s, const Ctx& ctx, int c, float start, float linger, float n_max) {
    bool in_combat = (ctx.now - s.combat_last[c]) <= linger;
    return in_combat ? std::min(std::max(std::floor(ctx.now - start + KK("EPS")), 0.0f), n_max) : 0.0f;
}
float guise_amp(const State& s, const Owned& own, const Ctx& ctx, int c) {
    float out = 0.0f;
    float sg = stacks_c(s, ctx, c, s.combat_start3[c], DV(GUISE, "BuffCounterDuration"), KK("GUISE_NMAX"));
    out = out + (holds(own, GUISE, c) ? DV(GUISE, "DamageIncreasePerSecond") * sg : 0.0f);
    float sl = stacks_c(s, ctx, c, s.combat_start3[c], DV(LIANDRY, "BuffCounterDuration"), KK("LIANDRY_NMAX"));
    out = out + (holds(own, LIANDRY, c) ? DV(LIANDRY, "DamageIncreasePerSecond") * sl : 0.0f);
    return out;
}
float rift_stacks(const State& s, const Ctx& ctx, int c) {
    return stacks_c(s, ctx, c, s.combat_start4[c], DV(RIFTMAKER, "BuffCounterDuration"), KK("RIFT_NMAX"));
}
float roa_stacks(const State& s, const Owned& own, int c) {
    float st = std::min(std::max(std::floor(s.roa_elapsed[c] / DV(ROA, "SecondsPerStack") + KK("EPS")), 0.0f), DV(ROA, "MaxStacks"));
    return holds(own, ROA, c) ? st : 0.0f;
}
float dist(const Units& u, int j, float px, float py) { return std::sqrt(sq(u.x[j] - px) + sq(u.y[j] - py)); }
int clip_unit(const Units& u, int idx) { return clampi(idx, 0, itemsb::n_units(u) - 1); }
// (C, Z, N) enemies inside Hatefog zone z of holder c (Z = N); ``inclusive`` keeps a zone on its final tick.
bool in_zone(const State& s, const Owned& own, const Ctx& ctx, const Units& u, int c, int z, int j, bool inclusive) {
    int n = itemsb::n_units(u);
    size_t cz = (size_t)c * n + z;
    bool live = inclusive ? (s.mal_until[cz] + KK("EPS") >= ctx.now) : (s.mal_until[cz] > ctx.now);
    if (!(holds(own, MALIGNANCE, c) && live)) return false;
    bool inside = dist(u, j, s.mal_x[cz], s.mal_y[cz]) <= s.mal_r[cz] + u.radius[j];
    return inside && enemy(ctx, u, c, j) && u.cls[j] != CLASS_STRUCTURE;
}
}  // namespace

// mage.stats
ItemStats stats(State state, const Owned& own, const Ctx& ctx) {
    int n = (int)(state.bf_until.size() / C);
    float now = ctx.now;
    ItemStats o = itemsb::stats_out({&ItemStats::health, &ItemStats::mana, &ItemStats::ability_power, &ItemStats::omnivamp, &ItemStats::ultimate_haste, &ItemStats::move_speed});
    for (int c = 0; c < C; ++c) {
        float roa = roa_stacks(state, own, c);
        float roa_hp = DV(ROA, "HealthPerStack") * roa;
        float rift_ap = holds(own, RIFTMAKER, c) ? DV(RIFTMAKER, "HealthToAPConversionPercent") * ((ctx.max_hp[c] - ctx.base_hp[c]) + roa_hp)
                                                 : 0.0f;
        float flat_ap = rift_ap + DV(ROA, "APPerStack") * roa;
        float burning = 0.0f;
        for (int j = 0; j < n; ++j) burning += (state.bf_until[(size_t)c * n + j] > now && state.bf_qual[(size_t)c * n + j]) ? 1.0f : 0.0f;
        float pct = (holds(own, RABADON, c) ? DV(RABADON, "APAmp") : 0.0f)
                    + (holds(own, BLACKFIRE, c) ? DV(BLACKFIRE, "APPerStack") * burning : 0.0f);
        bool rift_max = rift_stacks(state, ctx, c) >= KK("RIFT_NMAX");
        float vamp = ctx.is_ranged[c] ? DV(RIFTMAKER, "VampAmountRanged") : DV(RIFTMAKER, "VampAmountMelee");
        o.health[c] = roa_hp;
        o.mana[c] = DV(ROA, "ManaPerStack") * roa;
        o.ability_power[c] = flat_ap + pct * (ctx.ap[c] + flat_ap);
        o.omnivamp[c] = holds(own, RIFTMAKER, c) && rift_max ? vamp : 0.0f;
        o.ultimate_haste[c] = holds(own, MALIGNANCE, c) ? DV(MALIGNANCE, "UltimateHaste") : 0.0f;
        o.move_speed[c] = holds(own, COSMIC, c) && now < state.cosmic_until[c] ? 20.0f : 0.0f;   // calc MoveSpeedAmount
    }
    return o;
}

// mage.dealt_amp: Madness / Suffering / Void Corruption ramps, Hypershot mark. (C, N)
Arr<float> dealt_amp(State state, const Owned& own, const Ctx& ctx, const Units& units) {
    int n = itemsb::n_units(units);
    Arr<float> out((size_t)C * n, 0.f);
    for (int c = 0; c < C; ++c) {
        float rift = holds(own, RIFTMAKER, c) ? DV(RIFTMAKER, "EternityDamageIncreasePerSecond") * rift_stacks(state, ctx, c) : 0.0f;
        float base = guise_amp(state, own, ctx, c) + rift;
        for (int j = 0; j < n; ++j) {
            bool marked = holds(own, HORIZON, c) && state.hz_until[(size_t)c * n + j] > ctx.now;
            out[(size_t)c * n + j] = base + (marked ? DV(HORIZON, "DamageAmp") : 0.0f);
        }
    }
    return out;
}

// mage.debuffs: Hatefog flat MR shred, Bloodletter's %MR shred.
Debuffs debuffs(State state, const Owned& own, const Ctx& ctx, const Units& units) {
    int n = itemsb::n_units(units);
    Debuffs d = neutral_debuffs(n);
    // Any live zone (c, z) covering j curses it: the zone checks hoisted out of the per-unit loop (in_zone).
    std::vector<uint8_t> cursed(n, 0);
    for (int c = 0; c < C; ++c) {
        if (!holds(own, MALIGNANCE, c)) continue;
        for (int z = 0; z < n; ++z) {
            size_t cz = (size_t)c * n + z;
            if (!(state.mal_until[cz] > ctx.now)) continue;
            for (int j = 0; j < n; ++j)
                if (!cursed[j]) cursed[j] = in_zone(state, own, ctx, units, c, z, j, false);
        }
    }
    for (int j = 0; j < n; ++j) {
        float mx = 0.0f;
        for (int c = 0; c < C; ++c) {
            size_t cj = (size_t)c * n + j;
            mx = std::max(mx, holds(own, BLOODLETTER, c) && state.bl_until[cj] > ctx.now ? state.bl_stacks[cj] : 0.0f);
        }
        d.flat_mr_reduction[j] = cursed[j] ? 10.0f : 0.0f;   // calc MagicResistanceShred
        d.percent_mr_reduction[j] = mx * DV(BLOODLETTER, "ShredPerStack");
    }
    return d;
}

// mage.on_hit: Nashor's Tooth.
std::tuple<State, Effects> on_hit(State state, const Owned& own, const Ctx& ctx, const Units& units, const Attack& attack) {
    int n = itemsb::n_units(units);
    Effects e = no_effects(C, n);
    for (int c = 0; c < C; ++c) {
        bool go = attack.hit[c] && holds(own, NASHOR, c) && ctx.alive[c] && attack.target[c] >= 0;
        float dmg = DV(NASHOR, "NashorsBaseValue") + DV(NASHOR, "NashorsAPValue") * ctx.ap[c];
        push(e.packets, go, ctx.unit[c], std::max(attack.target[c], 0), dmg, MAGIC, ON_HIT_ITEM | PROP_LIFESTEAL, 0.f, NASHOR);
    }
    return {state, e};
}

// mage.on_cast: Malignance ult attribution window.
std::tuple<State, Effects> on_cast(State state, const Owned& own, const Ctx& ctx, const Units& units, const Cast& cast) {
    for (int c = 0; c < C; ++c)
        if (cast.started[c] && cast.slot[c] == 3) state.ult_until[c] = ctx.now + KK("ULT_ATTRIBUTION_WINDOW");
    return {state, no_effects(C, itemsb::n_units(units))};
}

// mage.on_damage
std::tuple<State, Effects> on_damage(State state, const Owned& own, const Ctx& ctx, const Units& units, const Report& report) {
    int n = itemsb::n_units(units);
    float now = ctx.now;
    const Packets& p = report.packets;
    const Resolved& r = report.resolved;
    int np = (int)size(p);
    const State s0 = state;
    const float EPS_ = KK("EPS");
    std::vector<int> dst(np), srcc(np), dcls(np), scls(np);
    std::vector<uint8_t> landed(np), champ_(np), struct_(np), ability(np), pet(np), magic(np), magic_true(np);
    for (int i = 0; i < np; ++i) {
        dst[i] = clampi(p.dst[i], 0, n - 1), srcc[i] = clampi(p.src[i], 0, n - 1);
        dcls[i] = units.cls[dst[i]], scls[i] = units.cls[srcc[i]];
        landed[i] = p.valid[i] && r.final[i] > 0.0f;
        champ_[i] = dcls[i] == CLASS_CHAMPION, struct_[i] = dcls[i] == CLASS_STRUCTURE;
        ability[i] = has(p.flags[i], TAG_ACTIVE_SPELL) && !has(p.flags[i], TAG_ITEM);
        pet[i] = has(p.flags[i], TAG_PET);
        magic[i] = p.dtype[i] == MAGIC, magic_true[i] = magic[i] || p.dtype[i] == TRUE_DMG;
    }
    auto src_is = [&](int c, int i) { return p.src[i] == ctx.unit[c]; };
    auto dst_is = [&](int c, int i) { return p.dst[i] == ctx.unit[c]; };
    auto dealt = [&](int c, int i) { return src_is(c, i) && landed[i] && units.team[dst[i]] != ctx.team[c]; };
    auto taken_champ = [&](int c, int i) {
        return dst_is(c, i) && p.valid[i] && p.raw[i] > 0.0f && scls[i] == CLASS_CHAMPION && units.team[srcc[i]] != ctx.team[c];
    };
    auto enemies = [&](int c, int j) { return enemy(ctx, units, c, j) && units.cls[j] != CLASS_STRUCTURE; };
    // first_dst: destination of the first selected packet, -1 if none.
    auto first_dst = [&](int c, auto sel) {
        for (int i = 0; i < np; ++i)
            if (sel(c, i)) return p.dst[i];
        return -1;
    };
    Effects e = no_effects(C, n);
    Packets l1 = empty_packets(), l2 = empty_packets(), l3 = empty_packets(), p_alt = empty_packets();

    Arr<uint8_t> ab_hit = itemsb::per_unit_any(p, n, [&](int c, int i) { return dealt(c, i) && ability[i] && !struct_[i]; });
    Arr<uint8_t> ab_pet_hit = itemsb::per_unit_any(p, n, [&](int c, int i) { return dealt(c, i) && (ability[i] || pet[i]) && !struct_[i]; });
    Arr<uint8_t> gw_hit = itemsb::per_unit_any(p, n, [&](int c, int i) { return dealt(c, i) && magic[i] && champ_[i]; });
    Arr<uint8_t> hyper_hit = itemsb::per_unit_any(p, n, [&](int c, int i) { return dealt(c, i) && ability[i] && !pet[i] && champ_[i]; });
    auto ult_sel = [&](int c, int i) {
        return dealt(c, i) && (ability[i] || pet[i]) && !has(p.flags[i], TAG_PROC) && champ_[i] && now <= s0.ult_until[c]
               && holds(own, MALIGNANCE, c);
    };
    Arr<float> inst = itemsb::per_unit_max(p, n, [&](int c, int i) { return ult_sel(c, i) ? r.final[i] : 0.0f; });
    Arr<uint8_t> zone_hit = itemsb::per_unit_any(p, n, ult_sel);
    Arr<uint8_t> dealt_any = itemsb::per_unit_any(p, n, dealt);
    Arr<uint8_t> bl_any = itemsb::per_unit_any(p, n, [&](int c, int i) { return dealt(c, i) && magic[i] && ability[i] && champ_[i]; });
    Arr<float> amount = itemsb::per_unit_add(p, n, [&](int c, int i) { return dealt(c, i) && champ_[i] ? r.final[i] : 0.0f; });

    int slots = (int)KK("STORM_SLOTS");
    int epoch = (int)std::floor(now / KK("STORM_BUCKET") + EPS_);
    int slot = ((epoch % slots) + slots) % slots;
    int n_extra = (int)KK("LUDEN_N_EXTRA");

    for (int c = 0; c < C; ++c) {
        size_t row = (size_t)c * n;
        bool event = false, any_luden = false, any_alt = false, cos_any = false, any_sf = false;
        float taken_raw = 0.0f;
        for (int i = 0; i < np; ++i) {
            event = event || (dealt(c, i) && champ_[i]) || taken_champ(c, i);
            any_luden = any_luden || (dealt(c, i) && ability[i] && !struct_[i]);
            any_alt = any_alt || (dealt(c, i) && champ_[i] && p.item[i] != ALTERNATOR);
            cos_any = cos_any || (dealt(c, i) && magic_true[i] && champ_[i]);
            taken_raw = taken_raw + (taken_champ(c, i) && p.valid[i] && scls[i] == CLASS_CHAMPION ? p.raw[i] : 0.0f);
        }
        (void)any_sf;
        float gap = now - s0.combat_last[c];
        if (event && gap > DV(GUISE, "BuffCounterDuration")) state.combat_start3[c] = now;
        if (event && gap > DV(RIFTMAKER, "BuffCounterDuration")) state.combat_start4[c] = now;
        if (event) state.combat_last[c] = now;

        for (int j = 0; j < n; ++j) {
            size_t cj = row + j;
            apply_timer(state.ashes_until[cj], state.ashes_next[cj], ab_hit[cj] && holds(own, ASHES, c), now,
                        DV(ASHES, "BurnDuration"), DV(ASHES, "TickFrequency"));
            bool bf_trig = ab_hit[cj] && holds(own, BLACKFIRE, c);
            apply_timer(state.bf_until[cj], state.bf_next[cj], bf_trig, now, DV(BLACKFIRE, "BurnDuration"), DV(BLACKFIRE, "TickFrequency"));
            if (bf_trig) state.bf_qual[cj] = units.cls[j] == CLASS_CHAMPION || units.cls[j] == CLASS_MONSTER;
            apply_timer(state.lia_until[cj], state.lia_next[cj], ab_pet_hit[cj] && holds(own, LIANDRY, c), now,
                        DV(LIANDRY, "BurnDuration"), DV(LIANDRY, "TickFrequency"));
        }

        // Luden's Echo.
        bool l_go = holds(own, LUDEN, c) && now >= s0.luden_cd[c] && any_luden;
        int prim = l_go ? first_dst(c, [&](int cc, int i) { return dealt(cc, i) && ability[i] && !struct_[i]; }) : -1;
        int pi = clip_unit(units, prim);
        float px = units.x[pi], py = units.y[pi];
        // nearest_k(dist, near, n_extra): k rounds of argmin (ties: lower index).
        std::vector<float> key(n);
        std::vector<uint8_t> near(n), pick(n, 0);
        for (int j = 0; j < n; ++j) {
            near[j] = in_circle(units, j, px, py, DV(LUDEN, "MissileRange")) && enemies(c, j) && !itemsb::onehot(prim, j);
            key[j] = near[j] ? dist(units, j, px, py) : std::numeric_limits<float>::infinity();
        }
        for (int round = 0; round < std::min(n_extra, n); ++round) {
            int am = 0;
            for (int j = 1; j < n; ++j)
                if (key[j] < key[am]) am = j;
            pick[am] = 1, key[am] = std::numeric_limits<float>::infinity();
        }
        float l_dmg = DV(LUDEN, "BaseDamage") + DV(LUDEN, "APRatio") * ctx.ap[c];
        int n_second = 0;
        for (int j = 0; j < n; ++j) n_second += near[j] && pick[j] && l_go;
        float left = (float)n_extra - (float)n_second;
        for (int j = 0; j < n; ++j) {
            bool pm = itemsb::onehot(prim, j);
            push(l1, pm && l_go, ctx.unit[c], j, l_dmg, MAGIC, TAG_ITEM | TAG_PROC, 0.f, LUDEN);
            push(l2, near[j] && pick[j] && l_go, ctx.unit[c], j, l_dmg, MAGIC, TAG_ITEM | TAG_PROC | TAG_AOE, 0.f, LUDEN);
            push(l3, pm && l_go && left > 0, ctx.unit[c], j, DV(LUDEN, "RepeatDamageReduction") * left * l_dmg, MAGIC,
                 TAG_ITEM | TAG_PROC, 0.f, LUDEN);
        }
        if (l_go) state.luden_cd[c] = now + DV(LUDEN, "Cooldown");

        // Hextech Alternator Revved.
        bool a_go = holds(own, ALTERNATOR, c) && now >= s0.alt_cd[c] && any_alt && ctx.alive[c];
        int a_tgt = first_dst(c, [&](int cc, int i) { return dealt(cc, i) && champ_[i] && p.item[i] != ALTERNATOR; });
        push(p_alt, a_go, ctx.unit[c], std::max(a_tgt, 0), 65.0f, MAGIC, TAG_ITEM | TAG_PROC, 0.f, ALTERNATOR);   // calc DamageAmount
        if (a_go) state.alt_cd[c] = now + DV(ALTERNATOR, "Cooldown");

        if (holds(own, COSMIC, c) && cos_any) state.cosmic_until[c] = now + DV(COSMIC, "StackDuration");

        // Horizon Focus Hypershot.
        bool focus_any = false;
        int f_src = 0;
        std::vector<uint8_t> hyper(n);
        for (int j = 0; j < n; ++j) {
            float hd = std::sqrt(sq(units.x[j] - ctx.x[c]) + sq(units.y[j] - ctx.y[c]));
            hyper[j] = hyper_hit[row + j] && hd >= DV(HORIZON, "SnipeRange") && holds(own, HORIZON, c);
            if (hyper[j]) state.hz_until[row + j] = std::max(s0.hz_until[row + j], now + DV(HORIZON, "BuffDuration"));
            if (hyper[j] && !focus_any) focus_any = true, f_src = j;
        }
        bool focus = focus_any && now >= s0.hz_cd[c];
        float fx = units.x[f_src], fy = units.y[f_src];
        for (int j = 0; j < n; ++j) {
            bool f_area = in_circle(units, j, fx, fy, KK("HORIZON_FOCUS_RADIUS"), false) && enemies(c, j)
                          && units.cls[j] == CLASS_CHAMPION && !hyper[j] && focus;
            if (f_area) state.hz_until[row + j] = std::max(state.hz_until[row + j], now + DV(HORIZON, "SecondaryBuffDuration"));
        }
        if (focus) state.hz_cd[c] = now + DV(HORIZON, "Cooldown");

        // Malignance Hatefog zones.
        for (int j = 0; j < n; ++j) {
            size_t cj = row + j;
            bool zone = zone_hit[cj] && now >= s0.mal_until[cj];
            if (!zone) continue;
            float radius = std::min(DV(MALIGNANCE, "AOESize") + std::pow(2.0f, std::min(inst[cj], 2000.0f) / 100.0f),
                                    DV(MALIGNANCE, "MaxRadius"));
            state.mal_x[cj] = units.x[j], state.mal_y[cj] = units.y[j], state.mal_r[cj] = radius;
            state.mal_until[cj] = now + DV(MALIGNANCE, "GroundDuration"), state.mal_next[cj] = now + KK("MALIGNANCE_TICK");
        }

        for (int j = 0; j < n; ++j)
            if (dealt_any[row + j]) state.crypt_last[row + j] = now;

        // Stormsurge sliding window.
        for (int b = 0; b < slots; ++b) {
            bool hot = b == slot;
            bool stale = hot && s0.storm_epoch[(size_t)c * slots + b] != epoch;
            for (int j = 0; j < n; ++j) {
                size_t h = (row + j) * slots + b;
                float v = stale ? 0.0f : s0.storm_hist[h];
                state.storm_hist[h] = v + (hot ? amount[row + j] : 0.0f);
            }
            if (hot) state.storm_epoch[(size_t)c * slots + b] = epoch;
        }
        bool s_any = false;
        int s_tgt = 0;
        for (int j = 0; j < n; ++j) {
            float window = 0.0f;
            for (int b = 0; b < slots; ++b) {
                bool recent = state.storm_epoch[(size_t)c * slots + b] > epoch - slots;
                window = window + (recent ? state.storm_hist[(row + j) * slots + b] : 0.0f);
            }
            bool over = window >= DV(STORMSURGE, "DamageThreshold") * units.max_hp[j] && units.cls[j] == CLASS_CHAMPION && enemies(c, j);
            if (over && !s_any) s_any = true, s_tgt = j;
        }
        bool s_go = holds(own, STORMSURGE, c) && now >= s0.storm_cd[c] && s0.storm_target[c] < 0 && s_any;
        if (s_go) {
            int st = clip_unit(units, s_tgt);
            state.storm_target[c] = s_tgt, state.storm_at[c] = now + DV(STORMSURGE, "DelayDuration");
            state.storm_cd[c] = now + DV(STORMSURGE, "Cooldown");
            state.storm_x[c] = units.x[st], state.storm_y[c] = units.y[st];
        }

        // Eternity: mana from pre-mitigation champion damage taken.
        bool eternity = holds(own, CATALYST, c) || holds(own, ROA, c);
        e.mana[c] = eternity && ctx.alive[c] ? DV(CATALYST, "EternityManaRestore") * taken_raw : 0.0f;

        // Bloodletter's Curse.
        for (int j = 0; j < n; ++j) {
            size_t cj = row + j;
            bool bl_hit = bl_any[cj] && holds(own, BLOODLETTER, c) && now >= s0.bl_icd[cj];
            if (!bl_hit) continue;
            bool live = s0.bl_until[cj] > now;
            state.bl_stacks[cj] = std::min((live ? s0.bl_stacks[cj] : 0.0f) + 1.0f, DV(BLOODLETTER, "MaxStacks"));
            state.bl_until[cj] = now + DV(BLOODLETTER, "DebuffDuration"), state.bl_icd[cj] = now + DV(BLOODLETTER, "InternalCD");
        }
    }

    // Rylai's slow and Grievous Wounds (any holder).
    for (int j = 0; j < n; ++j) {
        bool rylai = false, gw = false;
        for (int c = 0; c < C; ++c) {
            rylai = rylai || (ab_hit[(size_t)c * n + j] && holds(own, RYLAI, c));
            gw = gw || (gw_hit[(size_t)c * n + j] && (holds(own, MORELLO, c) || holds(own, ORB, c)));
        }
        e.slow[j] = rylai ? DV(RYLAI, "SlowAmount") : 0.0f;
        e.slow_duration[j] = rylai ? DV(RYLAI, "SlowDuration") : 0.0f;
        e.grievous[j] = gw ? DV(MORELLO, "GrievousDuration") : 0.0f;
    }

    // Shadowflame Cinderbloom: follow-up packets (P,).
    Packets p_sf = empty_packets();
    for (int i = 0; i < np; ++i) {
        float hp_frac = units.hp[dst[i]] / std::max(units.max_hp[dst[i]], 1e-6f);
        bool sf_src = false;
        for (int c = 0; c < C; ++c) sf_src = sf_src || (src_is(c, i) && holds(own, SHADOWFLAME, c) && units.team[dst[i]] != ctx.team[c]);
        bool sf = sf_src && landed[i] && magic_true[i] && !struct_[i] && p.item[i] != SHADOWFLAME
                  && hp_frac < DV(SHADOWFLAME, "HealthThreshold");
        int flags = TAG_ITEM | TAG_PROC | (p.flags[i] & (TAG_PERIODIC | TAG_AOE));
        push(p_sf, sf, p.src[i], p.dst[i], DV(SHADOWFLAME, "SpellItemDamageAmp") * p.raw[i], p.dtype[i], flags, p.amp[i], SHADOWFLAME);
    }
    append(e.packets, l1), append(e.packets, l2), append(e.packets, l3), append(e.packets, p_alt), append(e.packets, p_sf);
    return {state, e};
}

// mage.periodic: burns, Hatefog ticks, Stormsurge Squall, Enlighten mana, Rod of Ages clock.
std::tuple<State, Effects> periodic(State state, const Owned& own, const Ctx& ctx, const Units& units) {
    int n = itemsb::n_units(units);
    float now = ctx.now, dt = ctx.dt;
    const State s0 = state;
    Effects e = no_effects(C, n);
    Packets p_ashes = empty_packets(), p_bf = empty_packets(), p_lia = empty_packets(), p_mal = empty_packets();
    Packets p_strike = empty_packets(), p_field = empty_packets();
    const int burn = TAG_PERIODIC | TAG_ITEM;
    for (int c = 0; c < C; ++c) {
        size_t row = (size_t)c * n;
        float ap = ctx.ap[c];
        for (int j = 0; j < n; ++j) {
            size_t cj = row + j;
            bool monster = units.cls[j] == CLASS_MONSTER;
            bool minion_like = !monster && units.cls[j] != CLASS_CHAMPION;
            // Burns end with the target.
            if (!units.alive[j]) state.ashes_until[cj] = -BIG, state.bf_until[cj] = -BIG, state.lia_until[cj] = -BIG;
            float k;
            ticks(state.ashes_until[cj], s0.ashes_next[cj], now, DV(ASHES, "TickFrequency"), k, state.ashes_next[cj]);
            float per = KK("ASHES_PER") + (monster ? KK("ASHES_MONSTER") : 0.0f);
            push(p_ashes, k > 0 && holds(own, ASHES, c), ctx.unit[c], j, k * per, MAGIC, burn, 0.f, ASHES);

            ticks(state.bf_until[cj], s0.bf_next[cj], now, DV(BLACKFIRE, "TickFrequency"), k, state.bf_next[cj]);
            float rate = monster ? DV(BLACKFIRE, "MonsterDPS") + DV(BLACKFIRE, "MonsterAP") * ap
                       : (minion_like ? DV(BLACKFIRE, "MinionDPS") + DV(BLACKFIRE, "MinionAP") * ap
                                      : DV(BLACKFIRE, "BurnFlatDamagePerSecond") + DV(BLACKFIRE, "APRatio") * ap);
            push(p_bf, k > 0 && holds(own, BLACKFIRE, c), ctx.unit[c], j, k * rate * DV(BLACKFIRE, "TickFrequency"), MAGIC, burn, 0.f,
                 BLACKFIRE);

            ticks(state.lia_until[cj], s0.lia_next[cj], now, DV(LIANDRY, "TickFrequency"), k, state.lia_next[cj]);
            float lrate = DV(LIANDRY, "BurnPercentHealthDamage") * units.max_hp[j];
            lrate = monster ? std::min(lrate, DV(LIANDRY, "MonsterDamageCap")) : lrate;
            push(p_lia, k > 0 && holds(own, LIANDRY, c), ctx.unit[c], j, k * lrate * DV(LIANDRY, "TickFrequency"), MAGIC, burn, 0.f,
                 LIANDRY);
        }
        // Hatefog: one tick per unit per zone tick (max over overlapping zones).
        std::vector<float> kz(n);
        for (int z = 0; z < n; ++z) ticks(s0.mal_until[row + z], s0.mal_next[row + z], now, KK("MALIGNANCE_TICK"), kz[z], state.mal_next[row + z]);
        float zdmg = (DV(MALIGNANCE, "BaseDamage") + DV(MALIGNANCE, "APRatio") * ap) * KK("MALIGNANCE_TICK");
        for (int j = 0; j < n; ++j) {
            float kn = 0.0f;
            if (holds(own, MALIGNANCE, c))
                for (int z = 0; z < n; ++z) kn = std::max(kn, in_zone(s0, own, ctx, units, c, z, j, true) ? kz[z] : 0.0f);
            push(p_mal, kn > 0, ctx.unit[c], j, kn * zdmg, MAGIC, burn | TAG_AOE, 0.f, MALIGNANCE);
        }

        // Stormsurge Squall.
        bool pending = s0.storm_target[c] >= 0;
        int t = std::max(s0.storm_target[c], 0);
        bool t_alive = units.alive[t];
        float sx = pending && t_alive ? units.x[t] : s0.storm_x[c];
        float sy = pending && t_alive ? units.y[t] : s0.storm_y[c];
        float sq_dmg = DV(STORMSURGE, "BaseDamage") + DV(STORMSURGE, "APRatio") * ap;
        bool held = holds(own, STORMSURGE, c);
        bool strike = pending && t_alive && now >= s0.storm_at[c] && held;
        bool burst = pending && !t_alive && held;
        push(p_strike, strike, ctx.unit[c], t, sq_dmg, MAGIC, TAG_ITEM | TAG_PROC, 0.f, STORMSURGE);
        for (int j = 0; j < n; ++j) {
            bool field = in_circle(units, j, sx, sy, KK("STORM_AOE")) && enemy(ctx, units, c, j) && units.cls[j] == CLASS_CHAMPION && burst;
            push(p_field, field, ctx.unit[c], j, sq_dmg, MAGIC, TAG_ITEM | TAG_PROC | TAG_AOE, 0.f, STORMSURGE);
        }
        if (strike || burst || (pending && !held)) state.storm_target[c] = -1;
        state.storm_x[c] = sx, state.storm_y[c] = sy;

        // Fimbulwinter's Enlighten (Chapter): level up restores mana over time.
        bool chapter = holds(own, CHAPTER, c);
        float give = s0.ch_rem[c] * std::min(std::max(dt / std::max(s0.ch_until[c] - now + dt, 1e-6f), 0.0f), 1.0f);
        give = chapter ? give : 0.0f;
        float rem = chapter ? s0.ch_rem[c] - give : 0.0f;
        float gained = s0.ch_level[c] >= 0.0f ? std::max(ctx.level[c] - s0.ch_level[c], 0.0f) : 0.0f;
        bool up = chapter && gained > 0;
        rem = rem + (up ? DV(CHAPTER, "ManaRestorePercent") * ctx.max_mana[c] * gained : 0.0f);
        state.ch_rem[c] = rem;
        if (up) state.ch_until[c] = now + DV(CHAPTER, "RestorationDuration");
        state.ch_level[c] = ctx.level[c];
        e.mana[c] = give;

        // Rod of Ages clock resets when the item leaves the inventory.
        state.roa_elapsed[c] = holds(own, ROA, c) ? s0.roa_elapsed[c] + dt : 0.0f;
    }
    for (const Packets* q : {&p_ashes, &p_bf, &p_lia, &p_mal, &p_strike, &p_field}) append(e.packets, *q);
    return {state, e};
}

// mage.on_takedown: Cryptbloom Life From Death heal.
std::tuple<State, Effects> on_takedown(State state, const Owned& own, const Ctx& ctx, const Units& units, const Kills& kills) {
    int n = itemsb::n_units(units);
    Effects e = no_effects(C, n);
    for (int c = 0; c < C; ++c) {
        bool any = false;
        for (int j = 0; j < n; ++j) {
            size_t cj = (size_t)c * n + j;
            any = any || (kills.killed_units[cj] && units.cls[j] == CLASS_CHAMPION
                          && (ctx.now - state.crypt_last[cj]) <= DV(CRYPTBLOOM, "TakedownWindow"));
        }
        bool go = holds(own, CRYPTBLOOM, c) && ctx.alive[c] && ctx.now >= state.crypt_cd[c] && any;
        e.heal[c] = go ? DV(CRYPTBLOOM, "BaseHeal") + DV(CRYPTBLOOM, "HealAPRatio") * ctx.ap[c] : 0.0f;
        if (go) state.crypt_cd[c] = ctx.now + DV(CRYPTBLOOM, "Cooldown");
    }
    return {state, e};
}

LANESIM_TEST(items_mage_stats, "items.mage.stats", mage::stats);
LANESIM_TEST(items_mage_dealt_amp, "items.mage.dealt_amp", mage::dealt_amp);
LANESIM_TEST(items_mage_debuffs, "items.mage.debuffs", mage::debuffs);
LANESIM_TEST(items_mage_on_hit, "items.mage.on_hit", mage::on_hit);
LANESIM_TEST(items_mage_on_cast, "items.mage.on_cast", mage::on_cast);
LANESIM_TEST(items_mage_on_damage, "items.mage.on_damage", mage::on_damage);
LANESIM_TEST(items_mage_periodic, "items.mage.periodic", mage::periodic);
LANESIM_TEST(items_mage_on_takedown, "items.mage.on_takedown", mage::on_takedown);

}  // namespace lanesim::items::mage
