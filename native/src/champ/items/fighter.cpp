// items.effects.fighter: Fighter / AD-bruiser item passives (lanerl_jax/modern/items/effects/fighter.py).
// Ported in full (every item of the module), so the hooks are also correct beyond the Garen/Jax allow-lists.
#include <cmath>

#include "../marshal.hpp"
#include "items_b.hpp"

namespace lanesim::items::fighter {

using namespace champ;
using itemsb::by_range;

namespace {

constexpr int OVERLORD = 2501, HUNGER = 2517, BASTION = 2520, CLEAVER = 3071, HEXPLATE = 3073;
constexpr int BOTRK = 3153, SHOJIN = 3161, HULL = 3181, DMP = 3742, DEATHS_DANCE = 6333;
constexpr int CHEMPUNK = 6609, SUNDERED = 6610, ECLIPSE = 6692, EXECUTIONER = 3123, MORTAL = 3033;
constexpr int SERYLDA = 6694, WITS = 3091, TERMINUS = 3302;
constexpr float NEG = -1e9f;

// Data value, read once per call site.
#define DV(id, name) ([] { static const float v_ = itemsb::dv("fighter", id, name); return v_; }())

// Module constants (fighter.py top level).
struct K {
    float tyranny = itemsb::k("fighter", "TYRANNY"), retribution = itemsb::k("fighter", "RETRIBUTION");
    float retribution_full = itemsb::k("fighter", "RETRIBUTION_FULL");
    float famine_base = itemsb::k("fighter", "FAMINE_BASE"), famine_melee = itemsb::k("fighter", "FAMINE_MELEE");
    float famine_ranged = itemsb::k("fighter", "FAMINE_RANGED");
    float takedown_window = itemsb::k("fighter", "TAKEDOWN_WINDOW");
    float shaped_base = itemsb::k("fighter", "SHAPED_BASE"), shaped_leth = itemsb::k("fighter", "SHAPED_LETH");
    float sabotage_base = itemsb::k("fighter", "SABOTAGE_BASE");
    float sabotage_leth = itemsb::k("fighter", "SABOTAGE_LETH");
    float carve_per_stack = itemsb::k("fighter", "CARVE_PER_STACK"), carve_max = itemsb::k("fighter", "CARVE_MAX");
    float carve_duration = itemsb::k("fighter", "CARVE_DURATION");
    float carve_icd_eps = itemsb::k("fighter", "CARVE_ICD_EPS");
    float dd_melee = itemsb::k("fighter", "DD_MELEE"), dd_ranged = itemsb::k("fighter", "DD_RANGED");
    float dd_bleed = itemsb::k("fighter", "DD_BLEED"), dd_bucket = itemsb::k("fighter", "DD_BUCKET");
    int dd_slots = (int)itemsb::k("fighter", "DD_SLOTS");
    float hull_stacks = itemsb::k("fighter", "HULL_STACKS"), hull_stacks_m1 = itemsb::k("fighter", "HULL_STACKS_M1");
    float hull_ranged = itemsb::k("fighter", "HULL_RANGED");
    float dmp_max = itemsb::k("fighter", "DMP_MAX"), dmp_rate = itemsb::k("fighter", "DMP_RATE");
    float dmp_flat_full = itemsb::k("fighter", "DMP_FLAT_FULL");
    float wits_damage = itemsb::k("fighter", "WITS_DAMAGE");
    float term_base = itemsb::k("fighter", "TERM_BASE"), term_bad = itemsb::k("fighter", "TERM_BAD");
    float term_ap = itemsb::k("fighter", "TERM_AP"), term_res_l1 = itemsb::k("fighter", "TERM_RES_L1");
    std::vector<float> term_res_steps = data::table("items.fighter.TERM_RES_STEPS");   // (level, add) pairs
    float term_light_max = itemsb::k("fighter", "TERM_LIGHT_MAX");
    float term_dark_per = itemsb::k("fighter", "TERM_DARK_PER"), term_dark_max = itemsb::k("fighter", "TERM_DARK_MAX");
};
const K& k() {
    static const K v;
    return v;
}

// overlord_bonus_ad: Tyranny + Retribution AD.
float overlord_bonus_ad(const Ctx& ctx, int c) {
    const K& q = k();
    float tyranny = q.tyranny * (ctx.max_hp[c] - ctx.base_hp[c]);
    float missing = std::min(std::max(1.0f - ctx.hp[c] / std::max(ctx.max_hp[c], 1.0f), 0.0f), 1.0f);
    float frac = q.retribution * std::min(std::max(missing / q.retribution_full, 0.0f), 1.0f);
    return tyranny + frac * ((ctx.base_ad[c] + ctx.bonus_ad[c]) + tyranny);
}

float terminus_resist(float level) {
    const K& q = k();
    float out = q.term_res_l1 + 0.0f;
    for (size_t s = 0; s + 1 < q.term_res_steps.size(); s += 2)
        out = out + (level >= q.term_res_steps[s] ? q.term_res_steps[s + 1] : 0.0f);
    return out;
}

}  // namespace

// fighter.stats
ItemStats stats(State state, Owned own, Ctx ctx) {
    const K& q = k();
    float now = ctx.now;
    ItemStats o = itemsb::stats_out({&ItemStats::attack_damage, &ItemStats::ability_haste, &ItemStats::omnivamp, &ItemStats::move_speed, &ItemStats::attack_speed, &ItemStats::percent_move_speed, &ItemStats::ultimate_haste, &ItemStats::basic_ability_haste, &ItemStats::health, &ItemStats::armor, &ItemStats::magic_resist, &ItemStats::percent_armor_pen, &ItemStats::percent_magic_pen});
    for (int c = 0; c < C; ++c) {
        o.attack_damage[c] = holds(own, OVERLORD, c) ? overlord_bonus_ad(ctx, c) : 0.0f;
        float famine = q.famine_base + (ctx.is_ranged[c] ? q.famine_ranged : q.famine_melee) * ctx.bonus_ad[c];
        o.ability_haste[c] = holds(own, HUNGER, c) ? famine : 0.0f;
        o.omnivamp[c] = holds(own, HUNGER, c) && now < state.feast_until[c] ? DV(HUNGER, "OmnivampOnTakedown") : 0.0f;
        float fervor = holds(own, CLEAVER, c) && now < state.fervor_until[c]
                           ? DV(CLEAVER, "MoveSpeedBonus") * by_range(ctx, c, DV(CLEAVER, "RangedMod")) : 0.0f;
        float dmp_ms = holds(own, DMP, c) ? DV(DMP, "MaxMovementSpeed") * state.dmp[c] / q.dmp_max : 0.0f;
        bool hex_held = holds(own, HEXPLATE, c);
        bool hex_on = hex_held && now < state.hex_until[c];
        o.attack_speed[c] = hex_on ? (ctx.is_ranged[c] ? DV(HEXPLATE, "BonusASRanged") : DV(HEXPLATE, "BonusASMelee")) / 100.0f
                                   : 0.0f;
        o.percent_move_speed[c] =
            hex_on ? (ctx.is_ranged[c] ? DV(HEXPLATE, "BonusMSRanged") : DV(HEXPLATE, "BonusMSMelee")) / 100.0f : 0.0f;
        o.ultimate_haste[c] = hex_held ? DV(HEXPLATE, "UltimateHaste") : 0.0f;
        o.basic_ability_haste[c] = holds(own, SHOJIN, c) ? DV(SHOJIN, "AHBase") : 0.0f;
        o.health[c] = holds(own, SUNDERED, c) && now < state.sky_bonus_until[c] ? state.sky_bonus[c] : 0.0f;
        bool term = holds(own, TERMINUS, c);
        float light = (term && now < state.term_light_until[c] ? state.term_light[c] : 0.0f) * terminus_resist(ctx.level[c]);
        float dark = (term && now < state.term_dark_until[c] ? state.term_dark[c] : 0.0f) * q.term_dark_per;
        o.move_speed[c] = fervor + dmp_ms;
        o.armor[c] = light, o.magic_resist[c] = light;
        o.percent_armor_pen[c] = dark, o.percent_magic_pen[c] = dark;
    }
    return o;
}

// fighter.defense: Death's Dance store fraction.
HolderDefense defense(State state, Owned own, Ctx ctx) {
    const K& q = k();
    HolderDefense d = neutral_defense(C);
    for (int c = 0; c < C; ++c)
        d.store_fraction[c] = holds(own, DEATHS_DANCE, c) ? (ctx.is_ranged[c] ? q.dd_ranged : q.dd_melee) : 0.0f;
    return d;
}

// fighter.debuffs: Black Cleaver Carve shred.
Debuffs debuffs(State state, Owned own, Ctx ctx, Units units) {
    const K& q = k();
    int n = itemsb::n_units(units);
    Debuffs d = neutral_debuffs(n);
    for (int j = 0; j < n; ++j) {
        float keep = 1.0f;
        for (int c = 0; c < C; ++c) {
            size_t cj = (size_t)c * n + j;
            float stacks = holds(own, CLEAVER, c) && ctx.now < state.carve_until[cj] ? state.carve[cj] : 0.0f;
            float f = 1.0f - q.carve_per_stack * stacks;
            keep = c == 0 ? f : keep * f;
        }
        d.percent_armor_reduction[j] = 1.0f - keep;
    }
    return d;
}

// fighter.attack_mods: Sundered Sky forced crit per target.
AttackMods attack_mods(State state, Owned own, Ctx ctx, Units units, Arr<int32_t> target) {
    int n = itemsb::n_units(units);
    AttackMods m;
    m.force_crit.assign(C, 0), m.crit_scale.assign(C, 1.f);
    for (int c = 0; c < C; ++c) {
        int t = clampi(target[c], 0, n - 1);
        bool ready = ctx.now >= state.sky_cd[(size_t)c * n + t];
        bool force = holds(own, SUNDERED, c) && target[c] >= 0 && target_class(units, target[c]) == CLASS_CHAMPION && ready;
        m.force_crit[c] = force;
        m.crit_scale[c] = force ? DV(SUNDERED, "CritModifier") : 1.0f;
    }
    return m;
}

// fighter.packet_amp: Shojin Focused Will on ability packets.
Arr<float> packet_amp(State state, Owned own, Ctx ctx, Units units, Packets p) {
    float amp[C];
    for (int c = 0; c < C; ++c) {   // shojin_ability_amp
        float stacks = holds(own, SHOJIN, c) && ctx.now < state.shojin_until[c] ? state.shojin[c] : 0.0f;
        amp[c] = stacks * DV(SHOJIN, "SpellDamageIncrease") * by_range(ctx, c, DV(SHOJIN, "RangedMod"));
    }
    Arr<float> out(size(p), 0.f);
    for (size_t i = 0; i < size(p); ++i) {
        bool ability = has(p.flags[i], TAG_ACTIVE_SPELL) && !has(p.flags[i], TAG_ITEM);
        float s = (p.src[i] == ctx.unit[0] ? amp[0] : 0.0f) + (p.src[i] == ctx.unit[1] ? amp[1] : 0.0f);
        out[i] = ability ? s : 0.0f;
    }
    return out;
}

// fighter.on_cast: Hexplate Overdrive timers, Shojin fresh cast.
std::tuple<State, Effects> on_cast(State state, Owned own, Ctx ctx, Units units, Cast cast) {
    int n = itemsb::n_units(units);
    for (int c = 0; c < C; ++c) {
        bool go = cast.started[c] && ctx.alive[c];
        bool ult = go && cast.slot[c] == 3 && holds(own, HEXPLATE, c) && ctx.now >= state.hex_cd[c];
        if (ult) state.hex_until[c] = ctx.now + DV(HEXPLATE, "HasteDuration");
        if (ult) state.hex_cd[c] = ctx.now + DV(HEXPLATE, "Cooldown");
        state.shojin_fresh[c] = state.shojin_fresh[c] | (go && holds(own, SHOJIN, c));
    }
    return {state, no_effects(C, n)};
}

// fighter.on_hit
std::tuple<State, Effects> on_hit(State state, Owned own, Ctx ctx, Units units, Attack attack) {
    const K& q = k();
    int n = itemsb::n_units(units);
    float now = ctx.now;
    Effects e0 = no_effects(C, n), e1 = no_effects(C, n);
    Packets p_wits = empty_packets(), p_term = empty_packets(), p_hull = empty_packets(), p_dmp = empty_packets();
    State s0 = state;   // reads below use the state before this hook where JAX does
    for (int c = 0; c < C; ++c) {
        bool hit = attack.hit[c] && ctx.alive[c] && attack.target[c] >= 0;
        int t = clampi(attack.target[c], 0, n - 1);
        int tcls = target_class(units, t);
        bool champ_ = tcls == CLASS_CHAMPION, struct_ = tcls == CLASS_STRUCTURE;
        int oh = hit ? attack.target[c] : -1;
        size_t row = (size_t)c * n;

        // BotRK Mist's Edge / Clawing Shadows.
        bool b = hit && holds(own, BOTRK, c) && !struct_;
        float mist = (ctx.is_ranged[c] ? DV(BOTRK, "RangedValue") : DV(BOTRK, "MeleeValue")) * units.hp[t];
        bool capped = tcls == CLASS_MINION || tcls == CLASS_MONSTER;
        mist = capped ? std::min(mist, DV(BOTRK, "MonsterDamageCap")) : mist;
        push(e0.packets, b, ctx.unit[c], t, mist, PHYSICAL, ON_HIT_ITEM | PROP_LIFESTEAL, 0.f, BOTRK);
        bool claw = b && champ_ && now >= s0.botrk_cd[c];
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            bool claw_oh = itemsb::onehot(oh, j) && claw;
            float cnt = (now < s0.botrk_until[row + j] ? s0.botrk_count[row + j] : 0.0f) + (claw_oh ? 1.0f : 0.0f);
            state.botrk_count[row + j] = cnt;
            sum = sum + (claw_oh ? cnt : 0.0f);
        }
        bool proc = claw && sum >= 3;
        float strength = -(ctx.is_ranged[c] ? DV(BOTRK, "RangedMoveSpeedMod") : DV(BOTRK, "MoveSpeedMod"));
        for (int j = 0; j < n; ++j) {
            bool claw_oh = itemsb::onehot(oh, j) && claw;
            if (proc && claw_oh) state.botrk_count[row + j] = 0.0f;
            if (claw_oh) state.botrk_until[row + j] = now + DV(BOTRK, "AttackCounterDuration");
            e0.slow[j] = std::max(e0.slow[j], claw_oh && proc ? strength : 0.0f);   // max over holders from 0
        }
        if (proc) state.botrk_cd[c] = now + DV(BOTRK, "Cooldown");

        // Wit's End, Terminus.
        bool w = hit && holds(own, WITS, c) && !struct_;
        push(p_wits, w, ctx.unit[c], t, q.wits_damage, MAGIC, ON_HIT_ITEM | PROP_LIFESTEAL, 0.f, WITS);
        bool tm = hit && holds(own, TERMINUS, c) && !struct_;
        push(p_term, tm, ctx.unit[c], t, q.term_base + q.term_bad * ctx.bonus_ad[c] + q.term_ap * ctx.ap[c], MAGIC,
             ON_HIT_ITEM | PROP_LIFESTEAL, 0.f, TERMINUS);
        bool jux = tm && champ_;
        bool dark_hit = jux && s0.term_next_dark[c], light_hit = jux && !s0.term_next_dark[c];
        float dur = DV(TERMINUS, "BuffDuration");
        float light = now < s0.term_light_until[c] ? s0.term_light[c] : 0.0f;
        float dark = now < s0.term_dark_until[c] ? s0.term_dark[c] : 0.0f;
        if (light_hit) state.term_light[c] = std::min(light + 1, q.term_light_max), state.term_light_until[c] = now + dur;
        if (dark_hit) state.term_dark[c] = std::min(dark + 1, q.term_dark_max), state.term_dark_until[c] = now + dur;
        if (jux) state.term_next_dark[c] = !s0.term_next_dark[c];

        // Hullbreaker Skipper.
        bool hb = hit && holds(own, HULL, c);
        float stacks = now < s0.hull_until[c] ? s0.hull[c] : 0.0f;
        bool hproc = hb && (champ_ || struct_) && stacks >= q.hull_stacks_m1;
        float vs_champ = DV(HULL, "SkipperADRatio") * ctx.base_ad[c] + DV(HULL, "MaxStackDamageHPRatio") * ctx.max_hp[c];
        float vs_struct = DV(HULL, "SkipperADRatioVSStructures") * ctx.base_ad[c]
                          + DV(HULL, "MaxStackDamageVSStructuresHPRatio") * ctx.max_hp[c];
        float hdmg = (struct_ ? vs_struct : vs_champ) * by_range(ctx, c, q.hull_ranged);
        push(p_hull, hproc, ctx.unit[c], t, hdmg, PHYSICAL, ON_HIT_ITEM | PROP_LIFESTEAL, 0.f, HULL);
        state.hull[c] = hproc ? 0.0f : (hb ? std::min(stacks + 1, q.hull_stacks) : s0.hull[c]);
        if (hb && !hproc) state.hull_until[c] = now + DV(HULL, "SkipperStackDuration");

        // Dead Man's Plate (no life steal).
        bool dm = hit && holds(own, DMP, c) && !struct_ && s0.dmp[c] > 0;
        float sdmp = s0.dmp[c] / q.dmp_max;
        float ddmg = sdmp * (DV(DMP, "MaxStacksADRatio") * ctx.base_ad[c] + q.dmp_flat_full);
        push(p_dmp, dm, ctx.unit[c], t, ddmg, PHYSICAL, ON_HIT_ITEM, 0.f, DMP);
        if (dm) state.dmp[c] = 0.0f;

        // Sundered Sky heal.
        bool sky_ready = now >= s0.sky_cd[row + t];
        bool sk = hit && holds(own, SUNDERED, c) && champ_ && sky_ready;
        float missing = std::max(ctx.max_hp[c] - ctx.hp[c], 0.0f);
        float heal = DV(SUNDERED, "HealBaseADRatio") * ctx.base_ad[c] * by_range(ctx, c, DV(SUNDERED, "RangedHealMod"))
                     + DV(SUNDERED, "MissingHealthHeal") * missing;
        heal = sk ? heal : 0.0f;
        float excess = std::max(heal * (1.0f + ctx.heal_shield_power[c]) - missing, 0.0f);
        bool gain = sk && excess > 0;
        if (sk && oh >= 0) state.sky_cd[row + oh] = now + DV(SUNDERED, "Cooldown");
        if (gain) state.sky_bonus[c] = excess, state.sky_bonus_until[c] = now + 8.0f;
        e1.heal[c] = heal;

        // Bastionbreaker Sabotage.
        bool sab = hit && holds(own, BASTION, c) && struct_ && now < s0.sabotage_until[c];
        float total = (q.sabotage_base + q.sabotage_leth * ctx.lethality[c]) * by_range(ctx, c, DV(BASTION, "RangeModifier"));
        float dot_dur = DV(BASTION, "DoTDuration");
        if (sab) {
            state.sabotage_until[c] = NEG, state.bb_dot_target[c] = t;
            state.bb_dot_rate[c] = total / dot_dur, state.bb_dot_until[c] = now + dot_dur;
        }
    }
    for (int j = 0; j < n; ++j) e0.slow_duration[j] = e0.slow[j] > 0 ? DV(BOTRK, "MoveSpeedDuration") : 0.0f;
    append(e1.packets, p_wits), append(e1.packets, p_term), append(e1.packets, p_hull), append(e1.packets, p_dmp);
    return {state, merge({&e0, &e1}, C, n)};
}

// fighter.on_damage
std::tuple<State, Effects> on_damage(State state, Owned own, Ctx ctx, Units units, Report report) {
    const K& q = k();
    int n = itemsb::n_units(units);
    float now = ctx.now;
    const Packets& p = report.packets;
    const Resolved& r = report.resolved;
    int np = (int)size(p);
    std::vector<int> dcls(np);
    std::vector<uint8_t> landed(np), is_phys(np), basic(np), ability(np), champ_p(np);
    for (int i = 0; i < np; ++i) {
        dcls[i] = units.cls[clampi(p.dst[i], 0, n - 1)];
        landed[i] = p.valid[i] && r.final[i] > 0.0f;
        is_phys[i] = p.dtype[i] == PHYSICAL;
        basic[i] = has(p.flags[i], TAG_BASIC_ATTACK);
        ability[i] = has(p.flags[i], TAG_ACTIVE_SPELL) && !has(p.flags[i], TAG_ITEM);
        champ_p[i] = dcls[i] == CLASS_CHAMPION;
    }
    auto src_is = [&](int c, int i) { return p.src[i] == ctx.unit[c]; };
    // hit(mask): (C, N) holder's selected packets reached enemy unit n.
    auto hit = [&](auto mask) {
        Arr<uint8_t> h = itemsb::per_unit_any(p, n, [&](int c, int i) { return src_is(c, i) && mask(i); });
        for (int c = 0; c < C; ++c)
            for (int j = 0; j < n; ++j) h[(size_t)c * n + j] = h[(size_t)c * n + j] && units.team[j] != ctx.team[c];
        return h;
    };
    State s0 = state;
    Arr<uint8_t> any_champ = hit([&](int i) { return landed[i] && champ_p[i]; });
    for (size_t k2 = 0; k2 < any_champ.size(); ++k2)
        if (any_champ[k2]) state.last_dmg[k2] = now;

    // Black Cleaver.
    Arr<uint8_t> add_basic = hit([&](int i) { return landed[i] && is_phys[i] && champ_p[i] && basic[i]; });
    Arr<uint8_t> nonbasic_hit = hit([&](int i) { return landed[i] && is_phys[i] && champ_p[i] && !basic[i]; });
    Arr<uint8_t> phys_hit = hit([&](int i) { return landed[i] && is_phys[i]; });
    for (int c = 0; c < C; ++c) {
        bool bc = holds(own, CLEAVER, c) && ctx.alive[c];
        bool any_phys = false;
        for (int j = 0; j < n; ++j) {
            size_t cj = (size_t)c * n + j;
            bool nonbasic = nonbasic_hit[cj] && (now - s0.carve_last[cj] >= q.carve_icd_eps);
            float add = bc ? (float)add_basic[cj] + (float)nonbasic : 0.0f;
            float cur = now < s0.carve_until[cj] ? s0.carve[cj] : 0.0f;
            state.carve[cj] = add > 0 ? std::min(cur + add, q.carve_max) : cur;
            if (add > 0) state.carve_until[cj] = now + q.carve_duration;
            if (bc && nonbasic) state.carve_last[cj] = now;
            any_phys = any_phys || phys_hit[cj];
        }
        if (bc && any_phys) state.fervor_until[c] = now + DV(CLEAVER, "MoveSpeedDuration");
    }

    // Grievous Wounds items; Serylda's slow (target HP after this tick's damage).
    Effects e0 = no_effects(C, n);
    Arr<uint8_t> phys_champ = hit([&](int i) { return landed[i] && is_phys[i] && champ_p[i]; });
    Arr<uint8_t> ab_hit = hit([&](int i) { return landed[i] && ability[i]; });
    for (int j = 0; j < n; ++j) {
        bool gw = false, sl_any = false;
        bool low = r.hp[j] <= DV(SERYLDA, "SlowThreshold") * r.max_hp[j] && r.hp[j] > 0.0f;
        for (int c = 0; c < C; ++c) {
            size_t cj = (size_t)c * n + j;
            gw = gw || (phys_champ[cj] && itemsb::holds_any(own, {EXECUTIONER, MORTAL, CHEMPUNK}, c));
            sl_any = sl_any || (ab_hit[cj] && holds(own, SERYLDA, c) && low);
        }
        e0.grievous[j] = gw ? DV(EXECUTIONER, "GrievousDuration") : 0.0f;
        e0.slow[j] = sl_any ? DV(SERYLDA, "SlowAmount") : 0.0f;
        e0.slow_duration[j] = sl_any ? DV(SERYLDA, "SlowDuration") : 0.0f;
    }

    // Shojin Focused Will stacks.
    for (int c = 0; c < C; ++c) {
        bool any_ab = false;
        for (int j = 0; j < n; ++j) any_ab = any_ab || ab_hit[(size_t)c * n + j];
        bool sj = holds(own, SHOJIN, c) && any_ab;
        bool grant = sj && (s0.shojin_fresh[c] || (now - s0.shojin_last[c] >= DV(SHOJIN, "CastIDLockout")));
        float sj_cur = now < s0.shojin_until[c] ? s0.shojin[c] : 0.0f;
        if (grant) {
            state.shojin[c] = std::min(sj_cur + 1, DV(SHOJIN, "StackCount"));
            state.shojin_until[c] = now + DV(SHOJIN, "StackDuration"), state.shojin_last[c] = now;
        }
        state.shojin_fresh[c] = s0.shojin_fresh[c] && !grant;
    }

    // Eclipse Ever Rising Moon.
    Effects e1 = no_effects(C, n);
    Packets p_bb = empty_packets();
    Arr<uint8_t> ecl_hit = hit([&](int i) { return landed[i] && champ_p[i] && p.item[i] != ECLIPSE; });
    Arr<float> shield_amt(C, 0.f);
    float window = DV(ECLIPSE, "WindowDuration");
    for (int c = 0; c < C; ++c) {
        bool ecl = holds(own, ECLIPSE, c) && ctx.alive[c] && now >= s0.ecl_cd[c];
        int first_idx = 0;
        bool fire = false;
        for (int j = 0; j < n; ++j) {
            size_t cj = (size_t)c * n + j;
            bool stack = ecl_hit[cj] && ecl;
            bool open_ = (now - s0.ecl_first[cj] <= window) && (s0.ecl_first[cj] < now);
            if (stack && open_ && !fire) fire = true, first_idx = j;
        }
        for (int j = 0; j < n; ++j) {
            size_t cj = (size_t)c * n + j;
            bool stack = ecl_hit[cj] && ecl;
            bool open_ = (now - s0.ecl_first[cj] <= window) && (s0.ecl_first[cj] < now);
            state.ecl_first[cj] = (fire && j == first_idx) ? NEG : (stack && !open_ ? now : s0.ecl_first[cj]);
        }
        float pct = DV(ECLIPSE, "MeleePercMaxHP") * by_range(ctx, c, DV(ECLIPSE, "RangedPercMaxHPMult"));
        push(e1.packets, fire, ctx.unit[c], first_idx, pct * units.max_hp[first_idx], PHYSICAL, TAG_PROC | TAG_ITEM, 0.f,
             ECLIPSE);
        float shield = (DV(ECLIPSE, "MeleeBaseShield") + DV(ECLIPSE, "MeleeBonusADShieldRatio") * ctx.bonus_ad[c])
                       * by_range(ctx, c, DV(ECLIPSE, "RangedShieldMult"));
        shield_amt[c] = fire ? shield : 0.0f;
        if (fire) state.ecl_cd[c] = now + DV(ECLIPSE, "Cooldown");

        // Bastionbreaker Shaped Charge: first ability packet on an enemy champion.
        bool bb = holds(own, BASTION, c) && ctx.alive[c] && now >= s0.bb_cd[c];
        int first = -1;
        for (int i = 0; i < np && first < 0; ++i) {
            bool team_ok = units.team[clampi(p.dst[i], 0, n - 1)] != ctx.team[c];
            if (src_is(c, i) && landed[i] && ability[i] && champ_p[i] && team_ok && bb) first = i;
        }
        bool bb_go = first >= 0;
        int bb_dst = np ? p.dst[bb_go ? first : 0] : 0;
        float shaped = (q.shaped_base + q.shaped_leth * ctx.lethality[c]) * by_range(ctx, c, DV(BASTION, "AbilityDamageRangeMod"));
        push(p_bb, bb_go, ctx.unit[c], bb_dst, shaped, TRUE_DMG, TAG_PROC | TAG_ITEM, 0.f, BASTION);
        if (bb_go) state.bb_cd[c] = now + DV(BASTION, "Cooldown");
    }
    append(e1.packets, p_bb);
    e1.shields = shield_grants(shield_amt, SHIELD_ALL, DV(ECLIPSE, "ShieldDuration"));

    // Death's Dance: stored damage joins the current bucket.
    int bucket = (int)std::floor(now / q.dd_bucket);
    int ks = q.dd_slots;
    int slot = ((bucket % ks) + ks) % ks;
    for (int c = 0; c < C; ++c) {
        float pool = 0.0f;
        for (int s = 0; s < ks; ++s) pool = pool + s0.dd_amt[(size_t)c * ks + s];
        float stored = holds(own, DEATHS_DANCE, c) || pool > 0 ? r.dd_pool_add[ctx.unit[c]] : 0.0f;
        if (!(stored > 0)) continue;
        size_t cs = (size_t)c * ks + slot;
        bool same = s0.dd_bucket[cs] == bucket;
        float new_amt = s0.dd_amt[cs] + stored;
        state.dd_amt[cs] = new_amt;
        state.dd_rate[cs] = same ? s0.dd_rate[cs] + stored / q.dd_bleed : new_amt / q.dd_bleed;
        state.dd_bucket[cs] = bucket;
    }
    return {state, merge({&e0, &e1}, C, n)};
}

// fighter.periodic: DMP momentum, Death's Dance bleed and heal, Sabotage burn.
std::tuple<State, Effects> periodic(State state, Owned own, Ctx ctx, Units units) {
    const K& q = k();
    int n = itemsb::n_units(units);
    float now = ctx.now, dt = ctx.dt;
    int ks = q.dd_slots;
    Effects e = no_effects(C, n);
    Packets p_bb = empty_packets();
    for (int c = 0; c < C; ++c) {
        bool moving = holds(own, DMP, c) && ctx.alive[c] && ctx.moved[c] > 0.0f;
        float dmp = moving ? std::min(state.dmp[c] + q.dmp_rate * dt, q.dmp_max) : state.dmp[c];
        state.dmp[c] = holds(own, DMP, c) && ctx.alive[c] ? dmp : 0.0f;
        float bleed = 0.0f;
        for (int s = 0; s < ks; ++s) {
            size_t cs = (size_t)c * ks + s;
            float bk = std::min(state.dd_amt[cs], state.dd_rate[cs] * dt);
            bleed = bleed + bk;
            float amt = ctx.alive[c] ? state.dd_amt[cs] - bk : 0.0f;
            state.dd_rate[cs] = amt > 1e-6f ? state.dd_rate[cs] : 0.0f;
            state.dd_amt[cs] = amt > 1e-6f ? amt : 0.0f;
        }
        bleed = ctx.alive[c] ? bleed : 0.0f;
        push(e.packets, bleed > 0, ctx.unit[c], ctx.unit[c], bleed, TRUE_DMG,
             PROP_NO_OMNIVAMP | PROP_NO_DAMAGE_MOD | TAG_PERIODIC, 0.f, DEATHS_DANCE);
        float heal = state.dd_heal_rate[c] * std::min(std::max(state.dd_heal_until[c] - now, 0.0f), dt);
        e.heal[c] = ctx.alive[c] ? heal : 0.0f;
        float burn = state.bb_dot_rate[c] * std::min(std::max(state.bb_dot_until[c] - now, 0.0f), dt);
        push(p_bb, burn > 0, ctx.unit[c], state.bb_dot_target[c], burn, TRUE_DMG, TAG_PERIODIC | TAG_ITEM | PROP_NO_OMNIVAMP,
             0.f, BASTION);
    }
    append(e.packets, p_bb);
    return {state, e};
}

// fighter.on_takedown: Feast, Defy, Sabotage.
std::tuple<State, Effects> on_takedown(State state, Owned own, Ctx ctx, Units units, Kills kills) {
    const K& q = k();
    int n = itemsb::n_units(units);
    float now = ctx.now;
    for (int c = 0; c < C; ++c) {
        bool td = false;
        for (int j = 0; j < n; ++j) {
            size_t cj = (size_t)c * n + j;
            td = td || (kills.killed_units[cj] && units.cls[j] == CLASS_CHAMPION && (now - state.last_dmg[cj]) <= q.takedown_window);
        }
        td = td && ctx.alive[c];
        bool feast = td && holds(own, HUNGER, c), defy = td && holds(own, DEATHS_DANCE, c);
        bool sab = td && holds(own, BASTION, c);
        float heal_total = DV(DEATHS_DANCE, "BonusADRatio") * ctx.bonus_ad[c];
        float hd = DV(DEATHS_DANCE, "HealDuration");
        if (feast) state.feast_until[c] = now + DV(HUNGER, "OmnivampDuration");
        if (defy) {
            for (int s = 0; s < q.dd_slots; ++s) state.dd_amt[(size_t)c * q.dd_slots + s] = 0.0f, state.dd_rate[(size_t)c * q.dd_slots + s] = 0.0f;
            state.dd_heal_rate[c] = heal_total / hd, state.dd_heal_until[c] = now + hd;
        }
        if (sab) state.sabotage_until[c] = now + DV(BASTION, "BuffDuration");
    }
    return {state, no_effects(C, n)};
}

LANESIM_TEST(items_fighter_stats, "items.fighter.stats", fighter::stats);
LANESIM_TEST(items_fighter_defense, "items.fighter.defense", fighter::defense);
LANESIM_TEST(items_fighter_debuffs, "items.fighter.debuffs", fighter::debuffs);
LANESIM_TEST(items_fighter_attack_mods, "items.fighter.attack_mods", fighter::attack_mods);
LANESIM_TEST(items_fighter_packet_amp, "items.fighter.packet_amp", fighter::packet_amp);
LANESIM_TEST(items_fighter_on_cast, "items.fighter.on_cast", fighter::on_cast);
LANESIM_TEST(items_fighter_on_hit, "items.fighter.on_hit", fighter::on_hit);
LANESIM_TEST(items_fighter_on_damage, "items.fighter.on_damage", fighter::on_damage);
LANESIM_TEST(items_fighter_periodic, "items.fighter.periodic", fighter::periodic);
LANESIM_TEST(items_fighter_on_takedown, "items.fighter.on_takedown", fighter::on_takedown);

}  // namespace lanesim::items::fighter
