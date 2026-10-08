// items/effects/consumables.py: potions, elixirs and rune-granted consumables.
#include "../marshal.hpp"
#include "items.hpp"

namespace lanesim::items::consumables {

namespace {
constexpr int HEALTH_POTION = 2003, REFILLABLE = 2031, IRON = 2138, SORCERY = 2139, WRATH = 2140;
constexpr int BISCUIT = 2010, SKILL = 2150, AVARICE = 2151, FORCE = 2152;
constexpr int ELIXIRS[3] = {IRON, SORCERY, WRATH};
constexpr int HOT_SLOTS = 4;
constexpr float HOT_PERIOD = 0.5f, POTION_COOLDOWN = 1.0f, WRATH_AOE_MULT = 0.33f;
constexpr float BISCUIT_FLAT = 20.0f, BISCUIT_PCT = 0.015f, BISCUIT_MISSING_FULL = 0.70f, BISCUIT_PERMANENT_HP = 30.0f;
constexpr float BISCUIT_QUEUE = 3.f;

float K(const char* name) { return data::f(std::string("items.consumables.") + name); }

bool elixir_on(const State& s, const Ctx& ctx, int item, int c) {
    return s.elixir[c] == item && ctx.now < s.elixir_until[c];
}
}  // namespace

// consumables.stats
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx) {
    static const float iron_hp = K("iron_hp"), iron_ten = K("iron_tenacity"), sorc_ap = K("sorcery_ap"),
                       sorc_mr = K("sorcery_mana_regen"), wrath_ad = K("wrath_ad"), force_af = K("force_adaptive");
    size_t c = ctx.unit.size();
    ItemStats o = default_stats();
    o.health = o.silent_health = o.adaptive_force = o.tenacity = o.ability_power = o.mana_regen = o.attack_damage =
        zeros_c(c);
    for (size_t h = 0; h < c; ++h) {
        bool iron = elixir_on(s, ctx, IRON, h), sorc = elixir_on(s, ctx, SORCERY, h),
             wrath = elixir_on(s, ctx, WRATH, h);
        float biscuit_hp = BISCUIT_PERMANENT_HP * s.biscuits_eaten[h];
        o.health[h] = (iron ? iron_hp : 0.f) + biscuit_hp;
        o.silent_health[h] = biscuit_hp;
        o.adaptive_force[h] = ctx.now < s.force_until[h] ? force_af : 0.f;
        o.tenacity[h] = iron ? iron_ten : 0.f;
        o.ability_power[h] = sorc ? sorc_ap : 0.f;
        o.mana_regen[h] = sorc ? sorc_mr : 0.f;
        o.attack_damage[h] = wrath ? wrath_ad : 0.f;
    }
    return o;
}

// consumables.active (+ _add_hot)
std::tuple<State, Effects, ActiveOut> active(State s, const Owned& own, const Ctx& ctx, const Units& u,
                                             const Arr<int32_t>& request) {
    static const float ticks_hp = K("ticks_hp"), ticks_rf = K("ticks_rf"), per_hp = K("per_hp"),
                       per_rf = K("per_rf"), avarice_dur = K("avarice_duration"), force_dur = K("force_duration");
    static const std::vector<float> elixir_dur = data::table("items.consumables.elixir_duration");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    ActiveOut out;
    out.used.assign(c, 0), out.cast_time.assign(c, 0.f), out.can_move.assign(c, 1), out.attack_reset.assign(c, 0);
    for (int h = 0; h < c; ++h) {
        int req = request[h];
        bool potion_ok = ctx.alive[h] && ctx.now >= s.potion_cd[h];
        bool hp_go = req == HEALTH_POTION && holds(own, HEALTH_POTION, h) && potion_ok;
        bool rf_go = req == REFILLABLE && holds(own, REFILLABLE, h) && potion_ok && s.refill_charges[h] >= 1.f;
        float per = hp_go ? per_hp : per_rf, ticks = hp_go ? ticks_hp : ticks_rf;
        bool pot = hp_go || rf_go;
        if (pot) {   // _add_hot: first free slot, else the one with fewest ticks left
            int slot = -1;
            for (int k = 0; k < HOT_SLOTS; ++k)
                if (s.hot_left[h * HOT_SLOTS + k] <= 0.f) { slot = k; break; }
            if (slot < 0) {
                slot = 0;
                for (int k = 1; k < HOT_SLOTS; ++k)
                    if (s.hot_left[h * HOT_SLOTS + k] < s.hot_left[h * HOT_SLOTS + slot]) slot = k;
            }
            s.hot_per_tick[h * HOT_SLOTS + slot] = per;
            s.hot_next[h * HOT_SLOTS + slot] = ctx.now + HOT_PERIOD;
            s.hot_left[h * HOT_SLOTS + slot] = ticks;
        }
        int consume = hp_go ? item_row(HEALTH_POTION) : -1;
        for (int e = 0; e < 3; ++e) {
            int item = ELIXIRS[e];
            if (req == item && holds(own, item, h)) {      // usable while dead
                s.elixir[h] = item;
                s.elixir_until[h] = ctx.now + elixir_dur[e];
                consume = item_row(item);
            }
        }
        bool bis = req == BISCUIT && holds(own, BISCUIT, h) && ctx.alive[h];
        bool skill = req == SKILL && holds(own, SKILL, h);
        bool avarice = req == AVARICE && holds(own, AVARICE, h);
        bool force = req == FORCE && holds(own, FORCE, h);
        if (bis) consume = item_row(BISCUIT);
        if (skill) consume = item_row(SKILL);
        if (avarice) consume = item_row(AVARICE);
        if (force) consume = item_row(FORCE);
        if (bis) s.biscuit_queue[h] = std::min(s.biscuit_queue[h] + 1.f, BISCUIT_QUEUE);
        s.biscuits_eaten[h] = s.biscuits_eaten[h] + (bis ? 1.f : 0.f);
        s.skill_points[h] = s.skill_points[h] + (skill ? 1 : 0);
        if (avarice) s.avarice_until[h] = ctx.now + avarice_dur;
        if (force) s.force_until[h] = ctx.now + force_dur;
        out.used[h] = pot || consume >= 0;
        if (pot) s.potion_cd[h] = ctx.now + POTION_COOLDOWN;
        if (rf_go) s.refill_charges[h] = s.refill_charges[h] - 1.f;
        s.consume_row[h] = consume;
        s.drank[h] = hp_go ? HEALTH_POTION : (rf_go ? REFILLABLE : 0);
    }
    return {s, no_effects(c, n), out};
}

// consumables.periodic
std::tuple<State, Effects> periodic(State s, const Owned& own, const Ctx& ctx, const Units& u) {
    static const float avarice_gold = K("avarice_gold"), refill_max = K("refill_max"), b_ticks = K("biscuit_ticks");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    Effects e = no_effects(c, n);
    for (int h = 0; h < c; ++h) {
        bool alive = ctx.alive[h];
        float heal = 0.f;
        float k[HOT_SLOTS];
        for (int q = 0; q < HOT_SLOTS; ++q) {
            int i = h * HOT_SLOTS + q;
            bool due = s.hot_left[i] > 0.f && ctx.now >= s.hot_next[i];
            k[q] = due ? std::min(std::floor((ctx.now - s.hot_next[i]) / HOT_PERIOD) + 1.f, s.hot_left[i]) : 0.f;
            heal = heal + (alive ? k[q] * s.hot_per_tick[i] : 0.f);
        }
        bool expired = s.elixir_until[h] <= ctx.now;
        bool b_due = s.biscuit_left[h] > 0.f && ctx.now >= s.biscuit_next[h];
        float b_k = b_due ? std::min(std::floor((ctx.now - s.biscuit_next[h]) / HOT_PERIOD) + 1.f, s.biscuit_left[h])
                          : 0.f;
        heal = heal + (alive ? b_k * s.biscuit_per_tick[h] : 0.f);
        float b_left = alive ? s.biscuit_left[h] - b_k : 0.f;
        bool start = b_left <= 0.f && s.biscuit_queue[h] > 0.f && alive;
        float missing = std::min(std::max((1.f - ctx.hp[h] / std::max(ctx.max_hp[h], 1.f)) / BISCUIT_MISSING_FULL,
                                          0.f), 1.f);
        float b_total = (BISCUIT_FLAT + BISCUIT_PCT * ctx.max_hp[h]) * (1.f + missing);
        e.gold[h] = (s.avarice_until[h] > ctx.now - ctx.dt && s.avarice_until[h] <= ctx.now) ? avarice_gold : 0.f;
        s.biscuit_queue[h] = s.biscuit_queue[h] - (start ? 1.f : 0.f);
        s.biscuit_left[h] = start ? b_ticks : b_left;
        s.biscuit_next[h] = start ? ctx.now + HOT_PERIOD : s.biscuit_next[h] + b_k * HOT_PERIOD;
        if (start) s.biscuit_per_tick[h] = b_total / b_ticks;
        for (int q = 0; q < HOT_SLOTS; ++q) {
            int i = h * HOT_SLOTS + q;
            s.hot_next[i] = s.hot_next[i] + k[q] * HOT_PERIOD;
            s.hot_left[i] = alive ? s.hot_left[i] - k[q] : 0.f;      // HoTs end on death
        }
        if (!holds(own, REFILLABLE, h)) s.refill_charges[h] = refill_max;   // a re-bought Refillable starts full
        if (expired) s.elixir[h] = 0;
        e.heal_plain[h] = heal;
    }
    return {s, e};
}

// consumables.on_hit: Elixir of Avarice true damage on minions
std::tuple<State, Effects> on_hit(State s, const Owned& own, const Ctx& ctx, const Units& u, const Attack& a) {
    static const float dmg = K("avarice_on_hit");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    Effects e = no_effects(c, n);
    for (int h = 0; h < c; ++h) {
        bool go = a.hit[h] && ctx.now < s.avarice_until[h] && target_class(u, a.target[h]) == CLASS_MINION;
        push(e.packets, go, ctx.unit[h], std::max(a.target[h], 0), dmg, TRUE_DMG, ON_HIT_ITEM, 0.f, AVARICE);
    }
    return {s, e};
}

// consumables.on_shop: Refillable refills in the shop
State on_shop(State s, const Owned& own, const Ctx& ctx) {
    static const float refill_max = K("refill_max");
    for (size_t h = 0; h < ctx.unit.size(); ++h)
        if (holds(own, REFILLABLE, h) && ctx.in_shop[h]) s.refill_charges[h] = refill_max;
    return s;
}

// consumables.on_damage: Wrath drain heal, Sorcery true damage
std::tuple<State, Effects> on_damage(State s, const Owned& own, const Ctx& ctx, const Units& u, const Report& r) {
    static const float drain = K("wrath_drain"), sorc_true = K("sorcery_true"), sorc_icd = K("sorcery_icd");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    const Packets& p = r.packets;
    size_t np = size(p);
    Arr<uint8_t> single_sel(np), aoe_sel(np), dealt_sel(np);
    for (size_t k = 0; k < np; ++k) {
        int dcls = u.cls[clampi(p.dst[k], 0, n - 1)];
        bool phys = p.valid[k] && p.dtype[k] == PHYSICAL && dcls == CLASS_CHAMPION;
        single_sel[k] = phys && !has(p.flags[k], TAG_AOE);
        aoe_sel[k] = phys && has(p.flags[k], TAG_AOE);
        dealt_sel[k] = p.valid[k] && r.resolved.final[k] > 0.f && p.item[k] != SORCERY;
    }
    Arr<float> single = dealt_by_holder(r, ctx, n, single_sel), aoe = dealt_by_holder(r, ctx, n, aoe_sel);
    Arr<float> dealt = dealt_by_holder(r, ctx, n, dealt_sel);
    Effects e = no_effects(c, n);
    for (int h = 0; h < c; ++h) {
        float drained = 0.f;
        bool sorc_on = elixir_on(s, ctx, SORCERY, h);
        for (int j = 0; j < n; ++j) {
            bool enemy_n = u.team[j] != ctx.team[h];
            bool champ_n = u.cls[j] == CLASS_CHAMPION && enemy_n;
            drained = drained + (champ_n ? single[h * n + j] + WRATH_AOE_MULT * aoe[h * n + j] : 0.f);
            bool is_struct = u.cls[j] == CLASS_STRUCTURE && enemy_n;
            bool sorc = sorc_on && dealt[h * n + j] > 0.f && u.alive[j]
                        && ((champ_n && ctx.now >= s.sorcery_cd[h * n + j]) || is_struct);
            push_cn(e.packets, sorc, ctx.unit[h], j, sorc_true, TRUE_DMG, TAG_PROC | TAG_ITEM, SORCERY);
            if (sorc && champ_n) s.sorcery_cd[h * n + j] = ctx.now + sorc_icd;
        }
        e.heal[h] = elixir_on(s, ctx, WRATH, h) ? drain * drained : 0.f;
    }
    return {s, e};
}

LANESIM_TEST(items_consumables_stats, "items.consumables.stats", stats);
LANESIM_TEST(items_consumables_active, "items.consumables.active", active);
LANESIM_TEST(items_consumables_periodic, "items.consumables.periodic", periodic);
LANESIM_TEST(items_consumables_on_hit, "items.consumables.on_hit", on_hit);
LANESIM_TEST(items_consumables_on_shop, "items.consumables.on_shop", on_shop);
LANESIM_TEST(items_consumables_on_damage, "items.consumables.on_damage", on_damage);

}  // namespace lanesim::items::consumables
