// Champion stat composition (core/stat_pipeline.py, core/stats.py) and ItemStats combination.
#pragma once
#include <vector>

#include "../gen/types.hpp"

namespace lanesim::stats {

constexpr float AS_MIN = .2f;
const float AS_MAX = (float)(1.0 / 0.333);
constexpr float AS_UNCAPPED = 1e3f, HASTE_CAP = 500.f, TENACITY_FLOOR_SECONDS = .3f;
constexpr float ADAPTIVE_AD_RATIO = .6f;

std::vector<Arr<float>*> fields(ItemStats& s);       // the 39 fields in ItemStats order
ItemStats zero(size_t c);
ItemStats combine(const std::vector<const ItemStats*>& parts);
ItemStats combine2(const ItemStats& a, const ItemStats& b);

float level_growth_sum(float level);
void resolve_adaptive(float af, float bonus_ad, float ap, bool adaptive_physical, float* ad, float* ap_out);
float soft_cap_move_speed(float raw);
float move_speed(float base_ms, float flat, float additive_pct, float multiplicative_pct, float slow, float slow_resist,
                 float bonus_ms_amp, float celerity_flat_pct);
float attack_speed(float base_as, float ratio, float bonus_as, float mult, float cripple, float cap_lift);
float windup(float base_as, float as_now, float pct, float modifier);
float cooldown(float base_cd, float haste);
float cc_duration(float duration, float tenacity, bool affected = true);

// stat_pipeline.compose; optional per-champion inputs may be empty (0) or one value (broadcast).
ChampionStats compose(const ChampionBase& base, const Arr<int32_t>& level, const ItemStats& bonus,
                      const Arr<uint8_t>& adaptive_physical, const Arr<float>& slow, const Arr<float>& cripple,
                      const Arr<float>& extra_tenacity_b, const Arr<float>& extra_tenacity_c);

}  // namespace lanesim::stats
