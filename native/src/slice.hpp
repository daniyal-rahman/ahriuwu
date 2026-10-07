// The lane-slice building blocks (tick.cpp), shared by the slice step and the full champion tick
// (champ/world_tick.cpp).
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "rules.hpp"
#include "world.hpp"

namespace lanesim::slice {

constexpr float SWEEP_INTERVAL_S = .25f, GIVE_UP_S = 4.f, IGNORE_S = .5f, ATTACK_MEMORY_S = 2.f;
constexpr float WAYPOINT_MARGIN = 25.f, FIRST_WAVE_END_S = 59.f;
constexpr float SIEGE_TURRET_BONUS = 1.4f;
const float SUPER_BUILDING_SCALE = (float)(.125 / .60);
constexpr int ROUTE_REPLANS_PER_TICK = 16;
// collision.py
constexpr float AVOID_MAX_STEP = 60.f, CONTACT_MIN_FRAC = .25f, STATIONARY_MOBILITY = .25f, MAX_PUSH = 20.f;
constexpr int SEPARATION_ITERS = 3;
const float MINION_PATHING_RADIUS[4] = {35.7437f, 35.7437f, 55.7437f, 55.5208f};
constexpr float CHAMPION_PATHING_RADIUS = 35.f;
// core.damage
enum DType { PHYSICAL, MAGIC, TRUE_DMG };
constexpr int TAG_BASIC_ATTACK = 1 << 3, PROP_LIFESTEAL = 1 << 16, PROP_CRIT = 1 << 20;
constexpr int BASIC_ATTACK = TAG_BASIC_ATTACK | PROP_LIFESTEAL;
enum Class { CLASS_CHAMPION, CLASS_MINION, CLASS_STRUCTURE, CLASS_MONSTER };
constexpr float UNIT_CLASS_RATIO[4][4] = {
    {1.f, 1.f, 1.f, 1.f}, {.55f, 1.f, .60f, 1.f}, {1.f, 1.f, 1.f, 1.f}, {1.f, 1.f, 1.f, 1.f}};
constexpr int CAST_ID_STRIDE = 256;
// vision.py
constexpr float CHAMPION_SIGHT = 1350.f, MINION_SIGHT = 1200.f, SUPER_MINION_SIGHT = 1350.f, TURRET_SIGHT = 1350.f,
                NEXUS_SIGHT = 1350.f, WARD_SIGHT = 900.f, FARSIGHT_SIGHT = 500.f, REVEAL_RADIUS = 300.f,
                REVEAL_DURATION = 2.f;

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline int clip_team(int t) { return clampi(t, 0, 1); }
inline float sq(float v) { return v * v; }
inline float dist(const Env& e, int i, int j) { return std::sqrt(sq(e.x[i] - e.x[j]) + sq(e.y[i] - e.y[j])); }
// Pre-check before an exact ``dist(i, j) <= reach`` compare: true only when the pair is certainly farther (the
// squared distance clears (reach + 1)^2), so skipping it cannot change a result.
inline bool beyond(const Env& e, int i, int j, float reach) {
    return sq(e.x[i] - e.x[j]) + sq(e.y[i] - e.y[j]) > sq(reach + 1.f);
}
inline int damage_class(int k) {
    return k == CHAMPION ? CLASS_CHAMPION : (is_structure(k) ? CLASS_STRUCTURE : (k == MONSTER ? CLASS_MONSTER : CLASS_MINION));
}

struct Packet { int src, dst; float raw; int dtype, flags; float amp; };

// Per-thread working arrays, sized on first use.
// One live lane-AI column, copied out of the unit columns for the pair scans.
struct Live { int c, u; float x, y, r; int kind, sub, team; bool targetable; };

struct Scratch {
    std::vector<Live> live, foes[2];
    std::vector<int32_t> desired, live_cols, ids;
    std::vector<float> gx, gy, ms, nx, ny, x0, y0, prad, start_x, start_y;
    std::vector<uint8_t> stop, active, can_move, launched, collide, ghost, minion_rows, turret_rows, newu;
    std::vector<float> push_bonus, push_div, raw_all, hp_after, armor, t_mult;
    std::vector<uint8_t> invuln, dmg, died, attacking;
    std::vector<int32_t> dtype_all;
    std::vector<Packet> packets;
    std::vector<int32_t> victims_off, victims;
    void size(const World& w) {
        size_t n = w.n;
        if (desired.size() == n) return;
        for (auto* v : {&desired, &live_cols, &ids, &dtype_all}) v->assign(n, 0);
        for (auto* v : {&gx, &gy, &ms, &nx, &ny, &x0, &y0, &prad, &start_x, &start_y, &push_bonus, &push_div,
                        &raw_all, &hp_after, &armor, &t_mult})
            v->assign(n, 0.f);
        for (auto* v : {&stop, &active, &can_move, &launched, &collide, &ghost, &minion_rows, &turret_rows, &newu,
                        &invuln, &died})
            v->assign(n, 0);
        dmg.assign(n * n, 0);
        attacking.assign(w.cols.size() * w.cols.size(), 0);
        victims_off.assign(w.cols.size() + 1, 0);
        packets.reserve(n + w.missiles);
    }
};
Scratch& scratch();

// Per-phase wall time (ns), summed over the calling thread's ticks (ls_profile).
enum Phase { P_SPAWN, P_TURRET, P_SELECT, P_MOVE_PREP, P_ROUTE, P_COLLIDE, P_ATTACK, P_DAMAGE, P_DEATH, P_TIMERS, P_FOG,
             S_RESET, S_VICTIMS, S_PASS_A, S_PASS_B, S_TURRETS, P_CHAMP, N_PHASES };
extern thread_local double prof_ns[N_PHASES];
struct Clock {
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    void lap(int phase) {
        auto now = std::chrono::steady_clock::now();
        prof_ns[phase] += std::chrono::duration<double, std::nano>(now - t).count();
        t = now;
    }
};

void vulnerable(const World& w, Env& e);
void turret_tick(const World& w, Env& e, float now);
void structure_damage_events(const World& w, Env& e, const float* before, const float* hp_after, float now);
void spawn(const World& w, Env& e, float now);
void select_targets(const World& w, Env& e, float now, Scratch& sc);
void move_step(const World& w, Env& e, const float* gx, const float* gy, const float* ms, const uint8_t* active,
               float* nx, float* ny);
float pathing_radius(const Env& e, int i);
void collide(const World& w, Env& e, const float* x1, const float* y1, const float* gx, const float* gy,
             const uint8_t* moving, const uint8_t* solid, float* ox, float* oy);
bool in_range(const Env& e, int i, int target);
// mechanics.attack_step; ``windup`` (null: the unit column), ``period`` (> 0 overrides), ``uncancellable`` and
// ``reset`` are the champion kit inputs (null: none).
void attack_step(const World& w, Env& e, const int32_t* desired, const uint8_t* can_attack, uint8_t* launched,
                 const float* windup = nullptr, const float* period = nullptr, const uint8_t* uncancellable = nullptr,
                 const uint8_t* reset = nullptr);
bool hostile_ok(const Env& e, int i, int t);

}  // namespace lanesim::slice
