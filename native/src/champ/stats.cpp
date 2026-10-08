// Champion stat composition (core/stat_pipeline.py, core/stats.py) and ItemStats combination
// (items/catalog.combine_stats). Elementwise over the (C,) champions, float32, JAX operation order.
#include "stats.hpp"

#include <algorithm>
#include <cmath>

#include "marshal.hpp"

namespace lanesim::stats {

namespace {
// items.catalog.MULTIPLICATIVE_FIELDS: stack as 1 - prod(1 - x).
bool multiplicative(int field) {
    static const int F[] = {13 /*tenacity*/, 14 /*slow_resist*/, 21 /*percent_armor_pen*/, 23 /*percent_magic_pen*/};
    return std::find(std::begin(F), std::end(F), field) != std::end(F);
}
}  // namespace

std::vector<Arr<float>*> fields(ItemStats& s) {
    std::vector<Arr<float>*> out;
    s.visit([&](auto& m) {
        if constexpr (std::is_same_v<std::decay_t<decltype(m)>, Arr<float>>) out.push_back(&m);
    });
    return out;
}

ItemStats zero(size_t c) {
    ItemStats s;
    for (Arr<float>* f : fields(s)) f->assign(c, 0.f);
    return s;
}

// Fields broadcast like JAX: a size-1 field (a hook's scalar 0.0 default) applies to every holder, an empty one
// is 0; the result has the widest size among the parts.
ItemStats combine(const std::vector<const ItemStats*>& parts) {
    ItemStats out;
    auto o = fields(out);
    std::vector<std::vector<Arr<float>*>> ps;
    for (const ItemStats* p : parts) ps.push_back(fields(const_cast<ItemStats&>(*p)));
    auto at = [](const Arr<float>& a, size_t i) { return a.size() == 0 ? 0.f : a[a.size() == 1 ? 0 : i]; };
    for (size_t k = 0; k < o.size(); ++k) {
        size_t n = 0;
        for (auto& p : ps) n = std::max(n, p[k]->size());
        o[k]->resize(n);
        for (size_t i = 0; i < n; ++i) {
            if (multiplicative((int)k)) {
                float keep = 1.f;
                for (auto& p : ps) keep = keep * (1.f - at(*p[k], i));
                (*o[k])[i] = 1.f - keep;
            } else {
                float total = at(*ps[0][k], i);
                for (size_t q = 1; q < ps.size(); ++q) total = total + at(*ps[q][k], i);
                (*o[k])[i] = total;
            }
        }
    }
    return out;
}

ItemStats combine2(const ItemStats& a, const ItemStats& b) { return combine({&a, &b}); }

float level_growth_sum(float level) {
    float n = std::max(level - 1.f, 0.f);
    return n * (0.7025f + 0.0175f * n);
}

void resolve_adaptive(float af, float bonus_ad, float ap, bool adaptive_physical, float* ad_out, float* ap_out) {
    bool to_ad = bonus_ad > ap ? true : (ap > bonus_ad ? false : adaptive_physical);
    *ad_out = to_ad ? af * ADAPTIVE_AD_RATIO : 0.f;
    *ap_out = to_ad ? 0.f : af;
}

float soft_cap_move_speed(float raw) {
    return raw > 490.f ? .5f * raw + 230.f
         : raw > 415.f ? .8f * raw + 83.f
         : raw >= 220.f ? raw
         : raw >= 0.f ? .5f * raw + 110.f : .01f * raw + 110.f;
}

float move_speed(float base_ms, float flat, float additive_pct, float multiplicative_pct, float slow, float slow_resist,
                 float bonus_ms_amp, float celerity_flat_pct) {
    float amp = 1.f + bonus_ms_amp;
    float raw = (base_ms + flat * amp) * (1.f + additive_pct * amp + celerity_flat_pct) * (1.f + multiplicative_pct * amp)
                * (1.f - slow * (1.f - slow_resist));
    return soft_cap_move_speed(raw);
}

float attack_speed(float base_as, float ratio, float bonus_as, float mult, float cripple, float cap_lift) {
    float a = (base_as + ratio * bonus_as) * (1.f + mult) * (1.f - cripple);
    float hi = cap_lift > 0.f ? AS_UNCAPPED : AS_MAX;
    return std::min(std::max(a, AS_MIN), hi);
}

float windup(float base_as, float as_now, float pct, float modifier) {
    float base = pct / base_as;
    return base + modifier * (pct / as_now - base);
}

float cooldown(float base_cd, float haste) { return base_cd * 100.f / (100.f + std::min(haste, HASTE_CAP)); }

float cc_duration(float duration, float tenacity, bool affected) {
    float t = affected ? tenacity : 0.f;
    float reduced = std::max(std::min(duration, TENACITY_FLOOR_SECONDS), duration * (1.f - t));
    return t <= 0.f ? duration * (1.f - t) : reduced;
}

ChampionStats compose(const ChampionBase& base, const Arr<int32_t>& level, const ItemStats& bonus,
                      const Arr<uint8_t>& adaptive_physical, const Arr<float>& slow, const Arr<float>& cripple,
                      const Arr<float>& extra_tenacity_b, const Arr<float>& extra_tenacity_c) {
    size_t c = base.base_hp.size();
    ChampionStats s;
    s.visit([&](auto& m) { m.assign(c, 0.f); });
    auto at = [](const Arr<float>& a, size_t i) { return a.size() == 0 ? 0.f : a[a.size() == 1 ? 0 : i]; };
    for (size_t i = 0; i < c; ++i) {
        float g = level_growth_sum((float)level[level.size() == 1 ? 0 : i]);
        float base_hp = base.base_hp[i] + base.hp_per_level[i] * g;
        float base_ad = base.base_ad[i] + base.ad_per_level[i] * g;
        float base_armor = base.base_armor[i] + base.armor_per_level[i] * g;
        float base_mr = base.base_mr[i] + base.mr_per_level[i] * g;
        bool phys = adaptive_physical.size() == 0 ? true : adaptive_physical[adaptive_physical.size() == 1 ? 0 : i];
        float af_ad, af_ap;
        resolve_adaptive(bonus.adaptive_force[i], bonus.attack_damage[i], bonus.ability_power[i], phys, &af_ad, &af_ap);
        float bonus_ad = bonus.attack_damage[i] + af_ad;
        float ap = bonus.ability_power[i] + af_ap;
        float max_hp = std::max((base_hp + bonus.health[i]) * (1.f + bonus.percent_health[i]), 1.f);
        float armor = (base_armor + bonus.armor[i]) * (1.f + bonus.percent_armor[i]);
        float mr = (base_mr + bonus.magic_resist[i]) * (1.f + bonus.percent_magic_resist[i]);
        float bonus_as = base.attack_speed_per_level[i] / 100.f * g + bonus.attack_speed[i];
        float aspd = attack_speed(base.attack_speed[i], base.attack_speed_ratio[i], bonus_as,
                                  bonus.multiplicative_attack_speed[i], at(cripple, i), bonus.attack_speed_cap_lift[i]);
        float slow_resist = std::min(bonus.slow_resist[i], 1.f);
        float ms = move_speed(base.base_ms[i], bonus.move_speed[i], bonus.percent_move_speed[i], 0.f, at(slow, i),
                              slow_resist, bonus.bonus_ms_amp[i], 0.f);
        float ah = bonus.ability_haste[i];
        float g_hp = base.hp_regen[i] + base.hp_regen_per_level[i] * g;
        float g_mana = base.mana_regen[i] + base.mana_regen_per_level[i] * g;
        s.base_ad[i] = base_ad, s.bonus_ad[i] = bonus_ad, s.ap[i] = ap, s.base_hp[i] = base_hp, s.max_hp[i] = max_hp;
        s.base_armor[i] = base_armor, s.bonus_armor[i] = armor - base_armor;
        s.base_mr[i] = base_mr, s.bonus_mr[i] = mr - base_mr;
        s.attack_speed[i] = aspd, s.bonus_attack_speed[i] = bonus_as, s.attack_period[i] = 1.f / aspd;
        s.attack_windup[i] = windup(base.attack_speed[i], aspd, base.windup_percent[i], base.windup_modifier[i]);
        s.move_speed[i] = ms;
        s.basic_ability_haste[i] = std::min(ah + bonus.basic_ability_haste[i], HASTE_CAP);
        s.ultimate_haste[i] = std::min(ah + bonus.ultimate_haste[i], HASTE_CAP);
        s.item_haste[i] = bonus.item_haste[i], s.summoner_haste[i] = bonus.summoner_haste[i];
        s.trinket_haste[i] = bonus.trinket_haste[i];
        s.tenacity[i] = std::min(bonus.tenacity[i] + at(extra_tenacity_b, i) + at(extra_tenacity_c, i), 1.f);
        s.slow_resist[i] = slow_resist;
        s.crit_chance[i] = std::min(std::max(bonus.crit_chance[i], 0.f), 1.f);
        s.crit_damage[i] = 2.f + bonus.crit_damage[i];
        s.life_steal[i] = bonus.life_steal[i], s.omnivamp[i] = bonus.omnivamp[i];
        s.heal_shield_power[i] = bonus.heal_shield_power[i], s.lethality[i] = bonus.lethality[i];
        s.percent_armor_pen[i] = bonus.percent_armor_pen[i], s.magic_pen[i] = bonus.magic_pen[i];
        s.percent_magic_pen[i] = bonus.percent_magic_pen[i];
        s.hp_regen[i] = g_hp * (1.f + bonus.percent_base_health_regen[i]) + bonus.health_regen[i];
        s.max_mana[i] = base.base_mana[i] + base.mana_per_level[i] * g + bonus.mana[i];
        s.mana_regen[i] = g_mana * (1.f + bonus.percent_base_mana_regen[i]) + bonus.mana_regen[i];
        s.attack_range[i] = base.attack_range[i];
    }
    return s;
}

namespace {
ChampionStats compose_test(ChampionBase base, Arr<int32_t> level, ItemStats bonus, Arr<uint8_t> phys, Arr<float> slow) {
    return compose(base, level, bonus, phys, slow, {}, {}, {});
}
ItemStats combine_test(ItemStats a, ItemStats b) { return combine2(a, b); }
}  // namespace
LANESIM_TEST(stats_compose, "stats.compose", compose_test);
LANESIM_TEST(stats_combine, "stats.combine", combine_test);

}  // namespace lanesim::stats
