// Lane-minion and structure rules: ports of lanerl_jax/modern/lane/{minions,towers}.py (scalar form).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lanesim {

namespace minion {
enum Type { MELEE, CASTER, CANNON, SUPER, NONE = -1 };
constexpr float WAVE_FIRST_S = 30.f, WAVE_UNIT_GAP_S = .8f;
constexpr float ATTACK_SPEED[4] = {1.25f, .667f, 1.f, .85f};
constexpr float ATTACK_RANGE[4] = {110.f, 550.f, 300.f, 170.f};
constexpr float GAMEPLAY_RADIUS[4] = {48.f, 48.f, 65.f, 65.f};
const float WINDUP_S[4] = {.393f, .47f, .3f, (float)(.5 / 1.44 / .85)};
constexpr float MISSILE_SPEED[4] = {0.f, 650.f, 1200.f, 0.f};
constexpr float ACQUISITION_RANGE[4] = {750.f, 700.f, 750.f, 600.f};
constexpr float FIRST_ACQUISITION_RANGE[4] = {1000.f, 900.f, 750.f, 600.f};
constexpr float WAKE_UP_RANGE[4] = {450.f, 635.f, 750.f, 600.f};
constexpr float XP_BASE[4] = {62.f, 31.f, 75.f, 75.f};
constexpr float SLAYER[4] = {.02f, .035f, .05f, 0.f};
constexpr float CFH_GENERIC_RADIUS = 500.f, CFH_CHAMPION_RADIUS = 1000.f;
constexpr float BASE_HP[4] = {430.f, 275.f, 750.f, 1500.f}, HP_UP[4] = {35.f, 9.f, 85.f, 100.f},
                HP_MAX_BONUS[4] = {1120.f, 325.f, 5100.f, 6000.f};
constexpr float BASE_AD[4] = {11.f, 19.5f, 36.f, 180.f}, AD_UP[4] = {0.f, 1.5f, 1.5f, 5.f},
                AD_UP_LATE[4] = {3.f, 2.5f, 2.5f, 0.f}, AD_MAX_BONUS[4] = {69.f, 105.5f, 90.f, 300.f};
constexpr float BASE_ARMOR[4] = {0.f, 0.f, 0.f, 100.f}, BASE_MR[4] = {0.f, 0.f, 0.f, -30.f};

inline float wave_spawn_time(int i) {
    return i < 27 ? 30.f + 30.f * (float)i : (i < 66 ? 840.f + 25.f * (float)(i - 27) : 1810.f + 20.f * (float)(i - 66));
}
inline float wave_interval_s(float t) { return t < 840.f ? 30.f : (t < 1790.f ? 25.f : 20.f); }
inline int wave_index_at(float t) {
    if (t < WAVE_FIRST_S) return -1;
    if (t >= 1800.f && t < 1810.f) return 65;
    if (t < 840.f) return (int)std::floor((t - 30.f) / 30.f);
    if (t < 1810.f) return 27 + (int)std::floor((t - 840.f) / 25.f);
    return 66 + (int)std::floor((t - 1810.f) / 20.f);
}
inline bool cannon_wave(int i) {
    return i < 27 ? (i >= 2 && i % 3 == 2) : (i < 54 ? (i >= 28 && (i - 28) % 2 == 0) : i >= 54);
}
inline int upgrade_index_at(float t) { return t < 30.f ? 0 : 1 + (int)std::floor((t - 30.f) / 90.f); }
inline float melee_armor(float u) { return u >= 6.f ? std::min(.085f * (u - 6.f) * (u - 5.f) / 2.f, 20.f) : 0.f; }
inline float gold_bounty(int k, int u, int team) {
    if (k == MELEE) return 20.f;
    if (k == CASTER) return 14.f;
    if (k == SUPER && team == 1) return 49.f;
    if (k == CANNON || k == SUPER) return std::min(49.f + (float)std::max(u, 0), 90.f);
    return 0.f;
}
inline float base_move_speed(float t) {
    float steps = std::floor((t - 630.f) / 300.f) + 1.f;
    steps = steps < 0.f ? 0.f : (steps > 4.f ? 4.f : steps);
    return 350.f + 25.f * steps;
}
inline float sidelane_bonus(int wave_number, int lane, float wave_time, float tau) {
    float n = (float)wave_number;
    float b = std::max(0.f, 120.f - 4.5f * n);
    float step = std::floor(std::max(tau, 0.f) / 7.f);
    float bonus = std::max(0.f, b - 15.f * step);
    bool ok = lane != 1 && n >= 2.f && wave_time < 840.f && tau >= 0.f && tau < 25.f;
    return ok ? bonus : 0.f;
}
inline float soft_cap(float raw) { return raw > 490.f ? .5f * raw + 230.f : (raw > 415.f ? .8f * raw + 83.f : raw); }
inline void pushing(float level_adv, float tower_adv, float time_s, float* bonus, float* div) {
    float level = std::min(std::max(level_adv, 0.f), 3.f), towers = std::max(tower_adv, 0.f);
    bool on = time_s >= 210.f;
    *bonus = on ? (.05f + .05f * towers) * level : 0.f;
    *div = on ? 1.f + towers * level : 1.f;
}
inline int super_count(bool down, bool all_down, float respawn_at, float t) {
    int s = all_down ? 2 : (down ? 1 : 0);
    bool soon = (respawn_at - t) < 2.f * wave_interval_s(t);
    return (down && soon) ? 0 : s;
}
// (type, valid) of unit ``u`` of wave ``i`` with ``supers`` supers (minions.wave_unit_type).
inline int wave_unit_type(int i, int u, int supers) {
    int ns = std::max(supers, 0);
    float t = wave_spawn_time(i);
    int nm = t < 840.f ? 3 : (t < 1500.f ? (i % 2 == 0 ? 2 : 3) : 2);
    int nc = (cannon_wave(i) && ns == 0) ? 1 : 0;
    int nr = 3 - (t >= 1800.f ? 1 : 0);
    int a = u - ns, b = a - nm, c = b - nc;
    int kind = u < ns ? SUPER : (a < nm ? MELEE : (b < nc ? CANNON : (c < nr ? CASTER : NONE)));
    bool valid = i >= 0 && u >= 0 && kind != NONE;
    return valid ? kind : NONE;
}
inline float first_wave_ghost_s(int lane) { return lane == 1 ? 18.f : 28.f; }
}  // namespace minion

namespace tower {
enum Tier { OUTER, INNER, INHIB_TURRET, NEXUS_TURRET, INHIBITOR_BUILDING, NEXUS_BUILDING };
constexpr float ATTACK_RANGE = 750.f, GAMEPLAY_RADIUS = 88.4f, BACKDOOR_RADIUS = 1000.f, BULWARK_RADIUS = 850.f;
constexpr float PROTECTION_RADIUS = 1400.f, ARMOR_PEN = .3f, BUILDING_ARMOR = 20.f, BUILDING_MR = 0.f;
constexpr float OG_PROC_COOLDOWN = 90.f;
enum Priority { PET, CANNON_SUPER, MIST_WALKER, MELEE_P, CASTER_P, LOW_PET, CHAMPION_P };
constexpr float MINION_SHOT[4] = {.45f, .70f, .14f, .07f};

inline float decay_steps(float now) {
    float s = std::floor(now / 60.f) - 10.f;
    return s < 0.f ? 0.f : (s > 4.f ? 4.f : s);
}
inline float outer_ad(float now) {
    float s = std::floor((now - 30.f) / 60.f) + 1.f;
    return 182.f + 12.f * (s < 0.f ? 0.f : (s > 14.f ? 14.f : s));
}
inline float attack_damage(int tier, float now) {
    float g = std::floor((now - 180.f) / 60.f) + 1.f;
    float inner_growth = 16.f * (g < 0.f ? 0.f : (g > 15.f ? 15.f : g));
    return tier == OUTER ? outer_ad(now) : (tier == NEXUS_TURRET ? 165.f : 187.f) + inner_growth;
}
inline float warming(int stacks) { return 1.f + .5f * (float)std::min(std::max(stacks, 0), 3); }
inline float shot_fraction(int kind, int tier) {
    float siege = tier == OUTER ? .14f : (tier == INNER ? .11f : .08f);
    return kind == 2 ? siege : MINION_SHOT[std::min(std::max(kind, 0), 3)];
}
inline float respawn_delay(int tier) {
    return tier == NEXUS_TURRET ? 180.f : (tier == INHIBITOR_BUILDING ? 300.f : INFINITY);
}
}  // namespace tower

inline float mitigation(float r) { return r < 0.f ? 2.f - 100.f / (100.f - r) : 100.f / (100.f + r); }

}  // namespace lanesim
