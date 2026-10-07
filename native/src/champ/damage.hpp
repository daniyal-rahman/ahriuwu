// Damage resolution for the champion layer (core/damage.py).
#pragma once
#include "core.hpp"

namespace lanesim::champ::damage {

float shield_value(const Shields& sh, int unit, int k, int K, float now);
float total_shield(const Shields& sh, int unit, float now, int n);
void grant_shield(Shields& sh, int unit, float amount, int kind, float now, float duration, float decay_hold,
                  bool enabled, int n);
float effective_resist(float resist, float flat_red, float pct_red, float pct_pen, float flat_pen);
Arr<float> premitigation_to_final(const Packets& p, const Offense& off, const Defense& dfn);
Resolved resolve(const Packets& p, const Offense& off, const Defense& dfn, const Arr<float>& hp,
                 const Arr<float>& max_hp, const Shields& shields, float now);
void vamp_heal_split(const Packets& p, const Resolved& res, const Vamp& vamp, const Arr<int32_t>& dst_class,
                     const Arr<float>* lifesteal_scale, Arr<float>& ls, Arr<float>& ov);
float heal_amount(float base, float source_power = 0.f, float incoming = 0.f, bool grievous = false);

}  // namespace lanesim::champ::damage
