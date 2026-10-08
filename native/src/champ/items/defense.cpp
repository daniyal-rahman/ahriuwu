// items.effects.defense: tank, Lifeline, Annul, Thorns and Immolate items (lanerl_jax/modern/items/effects/defense.py).
// Reachable for Garen/Jax: Seeker's Armguard and Zhonya's Hourglass (no passive; actives live in actives) and Force
// of Nature. Every state update that runs regardless of items is ported, and every padded packet carries the JAX
// raw value. Omitted (holds() is false outside the allow-lists): the client calculations of Sterak's (BonusAD,
// ShieldSize), Hexdrinker / Maw / Shieldbow shields, Protoplasm (MaxHealthGain, TotalHealthRegen), Immolate
// DamagePerTick, Kaenic ShieldCalc, Warmog's Vitality / TotalHealing and Hollow Radiance bursts; those values are
// 0 here (marked "not ported" below).
#include <cmath>

#include "../marshal.hpp"
#include "items_b.hpp"

namespace lanesim::items::defense {

using namespace champ;
using itemsb::holds_any;

namespace {

constexpr int UNENDING = 2502, KAENIC = 2504, PROTOPLASM = 2525, GA = 3026, STERAKS = 3053;
constexpr int SPIRIT = 3065, SUNFIRE = 3068, THORNMAIL = 3075, BRAMBLE = 3076, WARDENS = 3082;
constexpr int WARMOGS = 3083, HEARTSTEEL = 3084, BANSHEES = 3102, FROZEN_HEART = 3110, RANDUINS = 3143;
constexpr int HEXDRINKER = 3155, MAW = 3156, FORCE_OF_NATURE = 4401, VERDANT = 4632;
constexpr int EDGE_OF_NIGHT = 3814, BAMIS = 6660, HOLLOW = 6664, JAKSHO = 6665, SHIELDBOW = 6673;
constexpr int ABYSSAL = 8020;
constexpr std::initializer_list<int> LIFELINE = {STERAKS, HEXDRINKER, MAW, SHIELDBOW, PROTOPLASM};
constexpr std::initializer_list<int> ANNUL = {BANSHEES, VERDANT, EDGE_OF_NIGHT};
constexpr std::initializer_list<int> THORNS = {BRAMBLE, THORNMAIL};
constexpr std::initializer_list<int> IMMOLATE = {BAMIS, SUNFIRE, HOLLOW};
constexpr float BIG = 1e9f;

// Data value / module constant, read once per call site.
#define DV(id, name) ([] { static const float v_ = itemsb::dv("defense", id, name); return v_; }())
#define KK(name) ([] { static const float v_ = itemsb::k("defense", name); return v_; }())

float level0(const Ctx& ctx, int c) { return 0.0f * ctx.level[c]; }   // calc: v = 0.0 * ctx.level

bool proto_active(const State& s, const Ctx& ctx, int c) {
    return ctx.now >= s.proto_start[c] && ctx.now < s.proto_start[c] + DV(PROTOPLASM, "Duration");
}
// _vitality: Warmog's % item HP (not ported: 0 unless Warmog's is held).
float vitality(const Owned& own, int c) { return 0.0f; }
// _max_hp: max HP including this module's dynamic health.
float max_hp(const State& s, const Owned& own, const Ctx& ctx, int c) {
    return ctx.max_hp[c] + vitality(own, c) + s.hs_hp[c] + (proto_active(s, ctx, c) ? s.proto_hp[c] : 0.0f);
}
bool in_champ_combat(const State& s, const Ctx& ctx, int c) { return ctx.now - s.champ_last[c] <= KK("CHAMP_COMBAT_WINDOW"); }
bool lifeline_ready(const State& s, const Owned& own, const Ctx& ctx, int c) {
    return holds_any(own, LIFELINE, c) && ctx.now >= s.lifeline_cd[c] && ctx.alive[c];
}
bool annul_ready(const State& s, const Owned& own, const Ctx& ctx, int c) {
    return holds_any(own, ANNUL, c) && ctx.now >= s.annul_ready[c] && ctx.alive[c];
}
// _pick: value of the held item of the table (later entries win).
float pick(const Owned& own, int c, std::initializer_list<std::pair<int, float>> table, float dflt = 0.0f) {
    float out = dflt;
    for (auto& [id, v] : table) out = holds(own, id, c) ? v : out;
    return out;
}
float lifeline_cooldown(const Owned& own, int c) {
    return pick(own, c, {{STERAKS, DV(STERAKS, "Cooldown")}, {HEXDRINKER, DV(HEXDRINKER, "Cooldown")},
                         {MAW, DV(MAW, "Cooldown")}, {SHIELDBOW, DV(SHIELDBOW, "Cooldown")},
                         {PROTOPLASM, DV(PROTOPLASM, "Cooldown")}});
}
float annul_cooldown(const Owned& own, int c) {
    return pick(own, c, {{BANSHEES, DV(BANSHEES, "Cooldown")}, {VERDANT, DV(VERDANT, "Cooldown")},
                         {EDGE_OF_NIGHT, DV(EDGE_OF_NIGHT, "Cooldown")}});
}
// calc(THORNMAIL / BRAMBLE, "TotalDamage")
float thornmail_damage(const Ctx& ctx, int c) {
    return level0(ctx, c) + DV(THORNMAIL, "BaseDamage") + DV(THORNMAIL, "BonusArmorDamageRatio") * ctx.bonus_armor[c];
}
float bramble_damage(const Ctx& ctx, int c) { return level0(ctx, c) + DV(BRAMBLE, "BaseDamage"); }
// calc(HEARTSTEEL, "DamageProcCalc", max_hp=mhp)
float heartsteel_damage(const Ctx& ctx, int c, float mhp) {
    return level0(ctx, c) + DV(HEARTSTEEL, "BaseDamage") + DV(HEARTSTEEL, "HPRatio") * (ctx.base_hp[c] + (mhp - ctx.base_hp[c]));
}
// calc(UNENDING, "DrainCalc", max_hp=mhp)
float drain_damage(const Ctx& ctx, int c, float mhp) {
    return level0(ctx, c) + DV(UNENDING, "BonusHealthDrainPercentage") * (mhp - ctx.base_hp[c]);
}
bool enemy_alive(const Ctx& ctx, const Units& u, int c, int j) { return enemy(ctx, u, c, j) && ctx.alive[c]; }

}  // namespace

// defense.stats
ItemStats stats(State state, Owned own, Ctx ctx) {
    float now = ctx.now;
    ItemStats o = itemsb::stats_out({&ItemStats::attack_damage, &ItemStats::incoming_heal, &ItemStats::health, &ItemStats::percent_move_speed, &ItemStats::tenacity, &ItemStats::omnivamp, &ItemStats::magic_resist, &ItemStats::armor});
    for (int c = 0; c < C; ++c) {
        bool proto = holds(own, PROTOPLASM, c) && proto_active(state, ctx, c);
        bool maw = holds(own, MAW, c) && now < state.maw_until[c];
        bool fon = holds(own, FORCE_OF_NATURE, c) && state.fon_stacks[c] >= DV(FORCE_OF_NATURE, "MaxStacks")
                   && now < state.fon_expire[c];
        bool jak = holds(own, JAKSHO, c) && in_champ_combat(state, ctx, c) && now - state.champ_start[c] >= KK("JAK_EPS");
        float jak_r = DV(JAKSHO, "BonusResistPercentage");
        o.attack_damage[c] = 0.0f;   // Sterak's BonusAD: not ported
        o.incoming_heal[c] = holds(own, SPIRIT, c) ? DV(SPIRIT, "HealingIncrease") : 0.0f;
        o.health[c] = vitality(own, c) + state.hs_hp[c] + (proto ? state.proto_hp[c] : 0.0f);
        o.percent_move_speed[c] = (proto ? DV(PROTOPLASM, "MSAmount") : 0.0f) + (fon ? DV(FORCE_OF_NATURE, "MoveSpeed") : 0.0f);
        o.tenacity[c] = proto ? DV(PROTOPLASM, "TenacityAmount") : 0.0f;
        o.omnivamp[c] = maw ? DV(MAW, "BuffVamp") : 0.0f;
        o.magic_resist[c] = (fon ? DV(FORCE_OF_NATURE, "BonusMagicResist") : 0.0f) + (jak ? jak_r * ctx.bonus_mr[c] : 0.0f);
        o.armor[c] = jak ? jak_r * ctx.bonus_armor[c] : 0.0f;
    }
    return o;
}

// defense.defense: Lifeline, Annul, Randuin's, Warden's.
HolderDefense defense(State state, Owned own, Ctx ctx) {
    HolderDefense d = neutral_defense(C);
    for (int c = 0; c < C; ++c) {
        d.crit_taken_mult[c] = holds(own, RANDUINS, c) ? KK("RANDUIN_CRIT_MULT") : 1.0f;
        d.champion_attack_block[c] = holds(own, WARDENS, c) ? DV(WARDENS, "BlockBase") : 0.0f;
        d.lifeline_ready[c] = lifeline_ready(state, own, ctx, c);
        d.lifeline_magic_only[c] = holds(own, HEXDRINKER, c) || holds(own, MAW, c);
        d.lifeline_shield[c] = 0.0f;   // Sterak's / Hexdrinker / Maw / Shieldbow shield calcs: not ported
        d.lifeline_shield_kind[c] = (int)pick(own, c, {{HEXDRINKER, SHIELD_MAGIC}, {MAW, SHIELD_MAGIC}}, SHIELD_ALL);
        d.lifeline_duration[c] = pick(own, c, {{STERAKS, DV(STERAKS, "ShieldDuration")}, {HEXDRINKER, DV(HEXDRINKER, "ShieldLifetime")},
                                               {MAW, DV(MAW, "ShieldDuration")}, {SHIELDBOW, DV(SHIELDBOW, "ShieldDuration")}});
        d.lifeline_decay_hold[c] = pick(own, c, {{STERAKS, DV(STERAKS, "TimeBeforeDecay")}}, INF);
        d.lifeline_bonus_health[c] = 0.0f;   // Protoplasm MaxHealthGain: not ported
        d.spell_shield[c] = annul_ready(state, own, ctx, c);
    }
    return d;
}

// defense.debuffs: Frozen Heart cripple, Abyssal Mask Unmake.
Debuffs debuffs(State state, Owned own, Ctx ctx, Units units) {
    int n = itemsb::n_units(units);
    Debuffs d = neutral_debuffs(n);
    for (int j = 0; j < n; ++j) {
        float cripple = 0.0f, unmake = 0.0f;
        for (int c = 0; c < C; ++c) {
            bool champs = enemy(ctx, units, c, j) && units.cls[j] == CLASS_CHAMPION && ctx.alive[c];
            bool fh = champs && holds(own, FROZEN_HEART, c) && in_circle(units, j, ctx.x[c], ctx.y[c], DV(FROZEN_HEART, "AuraRadius"));
            bool am = champs && holds(own, ABYSSAL, c) && in_circle(units, j, ctx.x[c], ctx.y[c], DV(ABYSSAL, "Radius"));
            cripple = std::max(cripple, fh ? KK("FH_SLOW") : 0.0f);
            unmake = std::max(unmake, am ? DV(ABYSSAL, "DamageAmp") : 0.0f);
        }
        d.magic_received_amp[j] = unmake, d.attack_speed_cripple[j] = cripple;
    }
    return d;
}

// defense.on_hit: Heartsteel Colossal Consumption.
std::tuple<State, Effects> on_hit(State state, Owned own, Ctx ctx, Units units, Attack attack) {
    int n = itemsb::n_units(units);
    Effects e = no_effects(C, n);
    for (int c = 0; c < C; ++c) {
        int tg = attack.target[c];
        size_t row = (size_t)c * n;
        float charged = 0.0f, cd = 0.0f;
        for (int j = 0; j < n; ++j) {
            charged = charged + (itemsb::onehot(tg, j) ? state.hs_charge[row + j] : 0.0f);
            cd = cd + (itemsb::onehot(tg, j) ? state.hs_cd[row + j] : 0.0f);
        }
        bool go = attack.hit[c] && ctx.alive[c] && holds(own, HEARTSTEEL, c) && target_class(units, tg) == CLASS_CHAMPION
                  && tg >= 0 && charged >= KK("HS_DEMOLISH_EPS") && ctx.now >= cd;
        float mhp = max_hp(state, own, ctx, c);
        float dmg = heartsteel_damage(ctx, c, mhp);
        float gain = heartsteel_damage(ctx, c, mhp) * DV(HEARTSTEEL, "DamageToMaxHealthRatio");
        push(e.packets, go, ctx.unit[c], std::max(tg, 0), dmg, PHYSICAL, ON_HIT_ITEM | PROP_LIFESTEAL, 0.f, HEARTSTEEL);
        state.hs_hp[c] = state.hs_hp[c] + (go ? gain : 0.0f);
        if (go && tg >= 0 && tg < n) state.hs_charge[row + tg] = 0.0f, state.hs_cd[row + tg] = ctx.now + DV(HEARTSTEEL, "PerTargetCooldown");
    }
    return {state, e};
}

// defense.on_damage
std::tuple<State, Effects> on_damage(State state, Owned own, Ctx ctx, Units units, Report report) {
    int n = itemsb::n_units(units);
    float now = ctx.now;
    const Packets& p = report.packets;
    const Resolved& r = report.resolved;
    int np = (int)size(p);
    const State s0 = state;
    Effects e = no_effects(C, n);
    auto src_i = [&](int i) { return clampi(p.src[i], 0, n - 1); };
    auto dst_i = [&](int i) { return clampi(p.dst[i], 0, n - 1); };
    auto to_h = [&](int c, int i) { return p.valid[i] && p.dst[i] == ctx.unit[c]; };
    auto from_h = [&](int c, int i) { return p.valid[i] && p.src[i] == ctx.unit[c]; };
    auto enemy_src = [&](int c, int i) { return units.team[src_i(i)] != ctx.team[c]; };
    auto enemy_dst = [&](int c, int i) { return units.team[dst_i(i)] != ctx.team[c]; };
    auto dmg = [&](int i) { return r.final[i] > 0.0f; };
    auto champ_src = [&](int i) { return units.cls[src_i(i)] == CLASS_CHAMPION; };
    auto champ_dst = [&](int i) { return units.cls[dst_i(i)] == CLASS_CHAMPION; };
    auto magic = [&](int i) { return p.dtype[i] == MAGIC; };
    auto taken = [&](int c, int i) { return to_h(c, i) && enemy_src(c, i) && dmg(i); };
    auto dealt = [&](int c, int i) { return from_h(c, i) && enemy_dst(c, i) && dmg(i); };

    Arr<uint8_t> dealt_n = itemsb::per_unit_any(p, n, dealt);
    Arr<uint8_t> fon_hits = [&] {   // per_unit(taken & magic & champ_src, p.src, n)
        Arr<uint8_t> h((size_t)C * n, 0);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < np; ++i)
                if (p.src[i] >= 0 && p.src[i] < n && taken(c, i) && magic(i) && champ_src(i)) h[(size_t)c * n + p.src[i]] = 1;
        return h;
    }();
    Packets p_thorns = empty_packets();
    std::vector<uint8_t> gw_src(np, 0);
    int nk = itemsb::shield_slots(r.shields, n);
    for (int c = 0; c < C; ++c) {
        bool took_champ = false, took_other = false, ev = false, lethal = false, trig_any = false, magic_taken = false;
        bool dealt_champ = false;
        float ud_dmg = 0.0f;
        for (int i = 0; i < np; ++i) {
            bool tk = taken(c, i), dl = dealt(c, i);
            took_champ = took_champ || (tk && champ_src(i));
            took_other = took_other || (tk && !champ_src(i));
            ev = ev || (to_h(c, i) && enemy_src(c, i) && champ_src(i)) || (from_h(c, i) && enemy_dst(c, i) && champ_dst(i));
            lethal = lethal || (to_h(c, i) && r.killed[i]);
            bool own_immo = from_h(c, i) && (p.item[i] == BAMIS || p.item[i] == SUNFIRE || p.item[i] == HOLLOW);
            trig_any = trig_any || ((tk || dl) && !own_immo);
            ud_dmg = ud_dmg + (from_h(c, i) && p.item[i] == UNENDING ? r.final[i] : 0.0f);
            magic_taken = magic_taken || (to_h(c, i) && magic(i) && dmg(i));
            dealt_champ = dealt_champ || (dl && champ_dst(i));
        }
        // Champion combat (Jak'Sho, Unending Despair, Maw).
        bool fresh = ev && (now - s0.champ_last[c] > KK("CHAMP_COMBAT_WINDOW"));
        if (fresh) state.champ_start[c] = now;
        if (ev) state.champ_last[c] = now;
        for (int j = 0; j < n; ++j)
            if (dealt_n[(size_t)c * n + j]) state.dealt_t[(size_t)c * n + j] = now;

        bool ll = r.lifeline_fired[ctx.unit[c]] && lifeline_ready(s0, own, ctx, c);
        if (ll) state.lifeline_cd[c] = now + lifeline_cooldown(own, c);
        bool proto = ll && holds(own, PROTOPLASM, c);
        if (proto) state.proto_start[c] = now;   // proto_hp / proto_rate calcs: not ported
        float maw_until = ll && holds(own, MAW, c) ? now + DV(MAW, "BuffDuration") : s0.maw_until[c];
        state.maw_until[c] = ev && now < maw_until ? std::max(maw_until, now + DV(MAW, "BuffExtension")) : maw_until;

        // Annul: pop starts the cooldown; champion damage restarts it while cooling.
        bool popped = r.spell_shield_popped[ctx.unit[c]] && annul_ready(s0, own, ctx, c);
        bool cooling = holds_any(own, ANNUL, c) && now < s0.annul_ready[c];
        if (popped || (cooling && took_champ)) state.annul_ready[c] = now + annul_cooldown(own, c);

        bool revive = lethal && holds(own, GA, c) && now >= s0.ga_cd[c] && ctx.alive[c];
        if (revive) state.ga_cd[c] = now + KK("GA_DELAY") + KK("GA_COOLDOWN"), state.ga_revive_at[c] = now + KK("GA_DELAY");
        e.revive[c] = revive;
        e.revive_delay[c] = revive ? KK("GA_DELAY") : 0.0f;
        e.revive_hp[c] = revive ? KK("GA_HP") * ctx.base_hp[c] : 0.0f;

        // Immolate activation (its own damage does not refresh it).
        bool trig = trig_any && holds_any(own, IMMOLATE, c) && ctx.alive[c];
        bool active = now <= s0.immo_until[c];
        if (trig && !active) state.immo_next[c] = now + KK("IMMO_PERIOD");
        if (trig) state.immo_until[c] = now + DV(SUNFIRE, "AuraDuration");

        e.heal[c] = holds(own, UNENDING, c) ? DV(UNENDING, "HealMultiplier") * ud_dmg : 0.0f;

        // Kaenic: magic damage resets the timer; track the remaining Kaenic shield.
        if (magic_taken) state.kaenic_last_magic[c] = now;
        state.kaenic_granted[c] = s0.kaenic_granted[c] && !magic_taken;
        float left = 0.0f;
        for (int s = 0; s < nk; ++s) {
            size_t idx = (size_t)ctx.unit[c] * nk + s;
            bool kslot = r.shields.kind[idx] == SHIELD_MAGIC && (r.shields.expires_at[idx] - now > KK("KAENIC_TAG"));
            left = left + (kslot ? itemsb::shield_value(r.shields, idx, now) : 0.0f);
        }
        state.kaenic_left[c] = left;

        // Force of Nature Steadfast.
        float gained = 0.0f;
        for (int j = 0; j < n; ++j) {
            size_t cj = (size_t)c * n + j;
            bool eligible = fon_hits[cj] && now >= s0.fon_src_ready[cj] && holds(own, FORCE_OF_NATURE, c);
            gained += eligible ? 1.0f : 0.0f;
            if (eligible) state.fon_src_ready[cj] = now + DV(FORCE_OF_NATURE, "StackRefreshTimer");
        }
        bool expired = now >= s0.fon_expire[c];
        float stacks = std::min((expired ? 0.0f : s0.fon_stacks[c]) + gained, DV(FORCE_OF_NATURE, "MaxStacks"));
        state.fon_stacks[c] = stacks;
        if (gained > 0 || (dealt_champ && stacks > 0)) state.fon_expire[c] = now + DV(FORCE_OF_NATURE, "BuffDuration");

        state.warmog_block[c] = std::max(s0.warmog_block[c], std::max(took_champ ? now + DV(WARMOGS, "OOCTimerChampion") : -BIG,
                                                                      took_other ? now + DV(WARMOGS, "OOCTimer") : -BIG));

        // Thorns: reactive magic on enemy basic attacks; Grievous Wounds on champion attackers.
        bool thorns_on = holds_any(own, THORNS, c) && ctx.alive[c];
        bool tm = holds(own, THORNMAIL, c);
        float thorn_dmg = tm ? thornmail_damage(ctx, c) : bramble_damage(ctx, c);
        int thorn_item = tm ? THORNMAIL : BRAMBLE;
        for (int i = 0; i < np; ++i) {
            bool struck = to_h(c, i) && enemy_src(c, i) && has(p.flags[i], TAG_BASIC_ATTACK) && !has(p.flags[i], PROP_REACTIVE)
                          && thorns_on;
            push(p_thorns, struck, ctx.unit[c], p.src[i], thorn_dmg, MAGIC, PROP_REACTIVE | TAG_ITEM | TAG_PROC, 0.f, thorn_item);
            gw_src[i] = gw_src[i] || (struck && champ_src(i));
        }
    }
    float gw_dur = DV(THORNMAIL, "GrievousDuration");
    for (int i = 0; i < np; ++i) {
        int s = src_i(i);
        e.grievous[s] = std::max(e.grievous[s], gw_src[i] ? gw_dur : 0.0f);
    }
    e.packets = p_thorns;
    return {state, e};
}

// defense.periodic
std::tuple<State, Effects> periodic(State state, Owned own, Ctx ctx, Units units) {
    int n = itemsb::n_units(units);
    float now = ctx.now, dt = ctx.dt;
    const float EPS_ = KK("EPS");
    Effects e = no_effects(C, n);
    Packets p_ud = empty_packets();
    Arr<float> kamount(C, 0.f);
    for (int c = 0; c < C; ++c) {
        bool alive = ctx.alive[c];
        float mhp = max_hp(state, own, ctx, c);
        size_t row = (size_t)c * n;

        float pdur = DV(PROTOPLASM, "Duration");
        float end = state.proto_start[c] + pdur;   // _overlap(now, dt, start, start + pdur)
        float overlap = std::max(std::min(now, end) - std::max(now - dt, state.proto_start[c]), 0.0f);
        e.heal[c] = holds(own, PROTOPLASM, c) && alive ? state.proto_rate[c] * overlap : 0.0f;

        // Guardian Angel revive completion restores mana.
        bool done = now >= state.ga_revive_at[c];
        e.mana[c] = done ? KK("GA_MANA") * ctx.max_mana[c] : 0.0f;
        if (done) state.ga_revive_at[c] = BIG;

        // Immolate ticks.
        float period = KK("IMMO_PERIOD");
        float last = std::min(now, state.immo_until[c]);
        bool due = state.immo_next[c] <= last + EPS_ && holds_any(own, IMMOLATE, c) && alive;
        float n_due = due ? std::floor((last - state.immo_next[c]) / period + EPS_) + 1.0f : 0.0f;
        state.immo_next[c] = state.immo_next[c] + n_due * period;
        float dpt = 0.0f;   // Immolate DamagePerTick calcs: not ported (0 unless an Immolate item is held)
        static const std::vector<float>& mults = data::table("items.defense.IMMO_MULT");
        float minion_m = pick(own, c, {{BAMIS, mults[0]}, {SUNFIRE, mults[2]}, {HOLLOW, mults[4]}}, 1.0f);
        float monster_m = pick(own, c, {{BAMIS, mults[1]}, {SUNFIRE, mults[3]}, {HOLLOW, mults[5]}}, 1.0f);
        int immo_item = (int)pick(own, c, {{BAMIS, (float)BAMIS}, {SUNFIRE, (float)SUNFIRE}, {HOLLOW, (float)HOLLOW}}, 0.0f);
        for (int j = 0; j < n; ++j) {
            int cls = units.cls[j];
            float mult = cls == CLASS_MINION ? minion_m : (cls == CLASS_MONSTER ? monster_m : 1.0f);
            bool near = enemy_alive(ctx, units, c, j) && cls != CLASS_STRUCTURE
                        && in_circle(units, j, ctx.x[c], ctx.y[c], DV(SUNFIRE, "Range")) && n_due > 0;
            push(e.packets, near, ctx.unit[c], j, dpt * n_due * mult, MAGIC, TAG_AOE | TAG_PERIODIC | TAG_ITEM, 0.f, immo_item);
        }
        if (!alive) state.immo_until[c] = -BIG;

        // Unending Despair pulse.
        bool pulse = holds(own, UNENDING, c) && alive && in_champ_combat(state, ctx, c) && now >= state.ud_next[c];
        float drain = drain_damage(ctx, c, mhp);
        for (int j = 0; j < n; ++j) {
            bool ud_t = enemy_alive(ctx, units, c, j) && units.cls[j] == CLASS_CHAMPION
                        && in_circle(units, j, ctx.x[c], ctx.y[c], DV(UNENDING, "DrainRange")) && pulse;
            push(p_ud, ud_t, ctx.unit[c], j, drain, MAGIC, TAG_AOE | TAG_ITEM, 0.f, UNENDING);
        }
        if (pulse) state.ud_next[c] = now + DV(UNENDING, "Cooldown");

        // Kaenic Rookern (ShieldCalc not ported: kgo needs the item).
        bool kgo = holds(own, KAENIC, c) && alive && !state.kaenic_granted[c]
                   && (now - state.kaenic_last_magic[c] >= KK("KAENIC_OOC_EPS"));
        float target = 0.0f;
        kamount[c] = kgo ? std::max(target - state.kaenic_left[c], 0.0f) : 0.0f;
        state.kaenic_left[c] = kgo ? std::max(target, state.kaenic_left[c]) : (alive ? state.kaenic_left[c] : 0.0f);
        state.kaenic_granted[c] = (state.kaenic_granted[c] || kgo) && alive && holds(own, KAENIC, c);

        // Warmog's Heart (TotalHealing not ported: wgo needs the item).
        e.heal_plain[c] = 0.0f;

        // Heartsteel charge.
        for (int j = 0; j < n; ++j) {
            bool hs_in = enemy_alive(ctx, units, c, j) && units.cls[j] == CLASS_CHAMPION && holds(own, HEARTSTEEL, c)
                         && in_circle(units, j, ctx.x[c], ctx.y[c], DV(HEARTSTEEL, "DistanceToChampion"));
            bool stale = now - state.hs_last_in[row + j] > DV(HEARTSTEEL, "RangeTrackingBuffDuration");
            state.hs_charge[row + j] = hs_in ? std::min(state.hs_charge[row + j] + dt, KK("HS_DEMOLISH"))
                                             : (stale ? 0.0f : state.hs_charge[row + j]);
            if (hs_in) state.hs_last_in[row + j] = now;
        }
        // Death clears Force of Nature stacks.
        state.fon_stacks[c] = alive && now < state.fon_expire[c] ? state.fon_stacks[c] : 0.0f;
    }
    append(e.packets, p_ud);
    e.shields = shield_grants(kamount, SHIELD_MAGIC, KK("KAENIC_DURATION"));
    return {state, e};
}

// defense.on_takedown: Hollow Radiance Desolate (bursts not ported: they need the item; raw 0 otherwise).
std::tuple<State, Effects> on_takedown(State state, Owned own, Ctx ctx, Units units, Kills kills) {
    int n = itemsb::n_units(units);
    Effects e = no_effects(C, n);
    for (int c = 0; c < C; ++c)
        for (int j = 0; j < n; ++j) push(e.packets, false, ctx.unit[c], j, 0.0f, MAGIC, TAG_AOE | TAG_PROC | TAG_ITEM, 0.f, HOLLOW);
    return {state, e};
}

LANESIM_TEST(items_defense_stats, "items.defense.stats", defense::stats);
LANESIM_TEST(items_defense_defense, "items.defense.defense", defense::defense);
LANESIM_TEST(items_defense_debuffs, "items.defense.debuffs", defense::debuffs);
LANESIM_TEST(items_defense_on_hit, "items.defense.on_hit", defense::on_hit);
LANESIM_TEST(items_defense_on_damage, "items.defense.on_damage", defense::on_damage);
LANESIM_TEST(items_defense_periodic, "items.defense.periodic", defense::periodic);
LANESIM_TEST(items_defense_on_takedown, "items.defense.on_takedown", defense::on_takedown);

}  // namespace lanesim::items::defense
