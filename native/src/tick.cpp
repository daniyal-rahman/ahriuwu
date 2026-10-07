// One 30 Hz tick of the lane slice of the 26.19 world (lanerl_jax/modern/world/tick.py), champions idle:
// wave spawning, structures, minion and turret AI, route movement, unit collision, the attack machine, missiles,
// damage, deaths and fog. Each function names the JAX function it ports; the JAX version evaluates every pair
// and masks, this one loops only over the units that can matter, with the same float32 operation order.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "rules.hpp"
#include "world.hpp"

namespace lanesim {
namespace {

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
thread_local Scratch scratch;
}  // namespace
thread_local float* debug_route = nullptr;           // (ward0, 16) route inputs/outputs of the next step, or null
namespace {

// Per-phase wall time (ns), summed over the calling thread's ticks (ls_profile).
enum Phase { P_SPAWN, P_TURRET, P_SELECT, P_MOVE_PREP, P_ROUTE, P_COLLIDE, P_ATTACK, P_DAMAGE, P_DEATH, P_TIMERS, P_FOG,
             S_RESET, S_VICTIMS, S_PASS_A, S_PASS_B, S_TURRETS, N_PHASES };
thread_local double prof_ns[N_PHASES] = {};
struct Clock {
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    void lap(int phase) {
        auto now = std::chrono::steady_clock::now();
        prof_ns[phase] += std::chrono::duration<double, std::nano>(now - t).count();
        t = now;
    }
};

// --- structures (lane.ai: _vulnerable, turret_tick, structure_unit_view, structure_damage_events) --------------
void vulnerable(const World& w, Env& e) {
    int n = w.n;
    bool inhib_dead[2] = {false, false}, nexus_t_alive[2] = {false, false};
    for (int i = 0; i < n; ++i) {
        bool alive = e.towers_is_structure[i] && e.towers_turret_hp[i] > 0.f;
        int t = clip_team(e.towers_team[i]);
        if (e.towers_is_structure[i] && e.towers_turret_tier[i] == tower::INHIBITOR_BUILDING && !alive) inhib_dead[t] = true;
        if (e.towers_is_structure[i] && e.towers_turret_tier[i] == tower::NEXUS_TURRET && alive) nexus_t_alive[t] = true;
    }
    for (int i = 0; i < n; ++i) {
        bool alive = e.towers_is_structure[i] && e.towers_turret_hp[i] > 0.f;
        int pre = clampi(e.towers_prereq[i], 0, n - 1);
        bool pre_dead = e.towers_prereq[i] < 0 || !(e.towers_is_structure[pre] && e.towers_turret_hp[pre] > 0.f);
        int t = clip_team(e.towers_team[i]);
        bool ok = e.towers_turret_tier[i] == tower::NEXUS_TURRET ? inhib_dead[t]
                : (e.towers_turret_tier[i] == tower::NEXUS_BUILDING ? inhib_dead[t] && !nexus_t_alive[t] : pre_dead);
        e.towers_targetable[i] = alive && ok;
    }
}

void turret_tick(const World& w, Env& e, float now) {
    int n = w.n;
    float dt = w.dt;
    static const float RATE[6] = {0.f, 0.f, 3.f, 6.f, 15.f, 20.f};
    for (int i = 0; i < n; ++i) {     // towers.regenerate_and_respawn on every row (vmapped in JAX)
        int tier = e.towers_turret_tier[i];
        float hp = e.towers_turret_hp[i], mx = e.towers_turret_max_hp[i];
        float frac = hp / mx;
        float low = tier == tower::NEXUS_TURRET ? .4f : .3f, high = tier == tower::NEXUS_TURRET ? .7f : .75f;
        float cap = (frac <= low ? low : (frac <= high ? high : 1.f)) * mx;
        if (tier >= tower::INHIBITOR_BUILDING) cap = mx;
        float rate = RATE[clampi(tier, 0, 5)];
        float nh = hp > 0.f ? std::min(cap, hp + rate * std::max(dt, 0.f)) : 0.f;
        bool respawn = (tier == tower::NEXUS_TURRET || tier == tower::INHIBITOR_BUILDING) && hp <= 0.f
                       && now >= e.towers_turret_respawn_at[i];
        float back = tier == tower::NEXUS_TURRET ? .4f : 1.f;
        e.towers_turret_hp[i] = respawn ? mx * back : nh;
        if (respawn) {
            e.towers_turret_respawn_at[i] = INF;
            e.towers_turret_warm_stacks[i] = 0;
            e.towers_turret_warm_until[i] = 0.f;
        }
    }
    vulnerable(w, e);
    for (int i = 0; i < n; ++i) {     // towers.unlock
        bool lane_turret = e.towers_is_structure[i] && e.towers_turret_tier[i] < tower::NEXUS_TURRET;
        float arg = (e.towers_targetable[i] && lane_turret) ? now : INF;
        if (std::isinf(e.towers_turret_growth_since[i])) e.towers_turret_growth_since[i] = arg;
    }
    // live minions and champions among the columns: the only units that refresh backdoor or suppress a crystal
    static thread_local std::vector<int32_t> walkers;
    walkers.clear();
    for (int c : w.cols)
        if (e.alive[c] && (e.kind[c] == MINION || e.kind[c] == CHAMPION)) walkers.push_back(c);
    float box[3][4];                  // per walker team: min x, max x, min y, max y
    for (auto& b : box) b[0] = b[2] = INF, b[1] = b[3] = -INF;
    for (int c : walkers) {
        auto& b = box[clampi(e.team[c], 0, 2)];
        b[0] = std::min(b[0], e.x[c]), b[1] = std::max(b[1], e.x[c]);
        b[2] = std::min(b[2], e.y[c]), b[3] = std::max(b[3], e.y[c]);
    }
    const float reach_max = tower::ATTACK_RANGE + tower::GAMEPLAY_RADIUS + 400.f;   // > every walker's reach
    auto far_from = [&](int i, int team) {
        const auto& b = box[team];
        return e.x[i] < b[0] - reach_max || e.x[i] > b[1] + reach_max || e.y[i] < b[2] - reach_max
               || e.y[i] > b[3] + reach_max;
    };
    for (int i = 0; i < n; ++i) {     // towers.advance; enemy pairs only over the structure rows
        bool minion_near = false, unit_near = false;
        bool any_foe = false;
        for (int tm = 0; tm < 3; ++tm) any_foe |= tm != e.towers_team[i] && !far_from(i, tm);
        if (w.col_of[i] >= 0 && i >= w.struct0 && any_foe) {
            for (int c : walkers) {
                if (e.towers_team[i] == e.team[c]) continue;
                bool minion = e.kind[c] == MINION;
                if (beyond(e, i, c, std::max(tower::BACKDOOR_RADIUS,
                                             tower::ATTACK_RANGE + tower::GAMEPLAY_RADIUS + e.radius[c])))
                    continue;
                float d = dist(e, i, c);
                if (minion && d <= tower::BACKDOOR_RADIUS) minion_near = true;
                if (d <= tower::ATTACK_RANGE + tower::GAMEPLAY_RADIUS + e.radius[c]) unit_near = true;
            }
            minion_near = minion_near && e.towers_is_structure[i];
        }
        bool alive = e.towers_turret_hp[i] > 0.f;
        if (minion_near && alive) e.towers_turret_backdoor_until[i] = now + 3.f;
        bool lane_turret = e.towers_is_structure[i] && e.towers_turret_tier[i] < tower::NEXUS_TURRET;
        e.towers_turret_growth_active[i] = alive && e.towers_turret_tier[i] < tower::NEXUS_TURRET
                                && (e.towers_turret_growth_active[i] || (now >= e.towers_turret_growth_since[i] + tower::OG_PROC_COOLDOWN
                                                              && !unit_near))
                                && lane_turret;
        if (now >= e.towers_turret_warm_until[i]) e.towers_turret_warm_stacks[i] = 0;
    }
    for (int i = 0; i < n; ++i) {     // structure_unit_view over the units view (targetable implies alive)
        if (!e.towers_is_structure[i]) {
            e.targetable[i] = e.targetable[i] && e.alive[i];
            continue;
        }
        e.hp[i] = e.towers_turret_hp[i];
        e.alive[i] = e.towers_turret_hp[i] > 0.f;
        e.targetable[i] = e.towers_targetable[i];
    }
}

void structure_damage_events(const World& w, Env& e, const float* before, const float* hp_after, float now) {
    static const float THR[5] = {.9f, .75f, .55f, .3f, 0.f};
    int n = w.n;
    bool any_kill = false, first_done = false;
    for (int i = 0; i < n; ++i) {
        bool s = e.towers_is_structure[i];
        int tier = e.towers_turret_tier[i];
        float after = s ? std::max(hp_after[i], 0.f) : e.towers_turret_hp[i];
        int plates = e.towers_turret_plates[i];
        if (s && tier < tower::NEXUS_TURRET) {
            int cnt = 0;
            for (float th : THR) cnt += after <= e.towers_turret_max_hp[i] * th;
            plates = std::max(plates, cnt);
        }
        for (int k = 0; k < 4; ++k)
            if (k >= e.towers_turret_plates[i] && k < plates) e.towers_turret_bulwark_until[i * 4 + k] = now + 20.f;
        bool destroyed = s && before[i] > 0.f && after <= 0.f;
        bool turret_kill = destroyed && tier <= tower::NEXUS_TURRET;
        if (turret_kill) {
            if (!first_done && !*e.towers_first_turret_taken) { /* first turret gold: an econ event */ }
            first_done = true;
            any_kill = true;
        }
        if (destroyed) {
            e.towers_turret_respawn_at[i] = now + tower::respawn_delay(tier);
            e.towers_turret_warm_stacks[i] = 0;
        }
        e.towers_turret_plates[i] = plates;
        e.towers_turret_hp[i] = after;
        e.towers_turret_growth_active[i] = e.towers_turret_growth_active[i] && after > 0.f;
    }
    if (any_kill) *e.towers_first_turret_taken = 1;
    vulnerable(w, e);
}

// --- wave spawning (lane.ai.spawn_lane_minions, world.units.write_units) -----------------------------------------
void spawn(const World& w, Env& e, float now) {
    int n = w.n;
    bool down[2][3] = {}, all_down[2];
    float resp[2][3];
    for (auto& r : resp) for (float& v : r) v = INF;
    for (int i = 0; i < n; ++i) {
        if (!(e.towers_is_structure[i] && e.towers_turret_tier[i] == tower::INHIBITOR_BUILDING && e.towers_turret_hp[i] <= 0.f)) continue;
        int o = e.towers_team[i], l = e.towers_lane[i];
        if (o < 0 || o > 1 || l < 0 || l > 2) continue;
        down[o][l] = true;
        resp[o][l] = std::min(resp[o][l], e.towers_turret_respawn_at[i]);
    }
    for (int o = 0; o < 2; ++o) all_down[o] = down[o][0] && down[o][1] && down[o][2];
    int due_type[2][3];
    for (int t = 0; t < 2; ++t)
        for (int l = 0; l < 3; ++l) {           // minions.lane_spawn_step; inputs describe the enemy's inhibitors
            int k = t * 3 + l, o = 1 - t;
            int wave = e.spawn_wave[k], unit = e.spawn_unit[k], latched = e.spawn_supers[k];
            float t_wave = minion::wave_spawn_time(wave);
            int supers = latched >= 0 ? latched : minion::super_count(down[o][l], all_down[o], resp[o][l], t_wave);
            int kind = minion::wave_unit_type(wave, unit, supers);
            bool due = kind != minion::NONE && now >= t_wave + minion::WAVE_UNIT_GAP_S * (float)unit;
            bool more = minion::wave_unit_type(wave, unit + 1, supers) != minion::NONE;
            bool close = due && !more;
            e.spawn_wave[k] = close ? wave + 1 : wave;
            e.spawn_unit[k] = close ? 0 : (due ? unit + 1 : unit);
            e.spawn_supers[k] = close ? -1 : (due ? supers : latched);
            due_type[t][l] = due ? kind : -1;
        }
    int level = 0;
    for (int c = 0; c < N_CHAMPIONS; ++c) level = std::max(level, e.econ_level[c]);
    int u = minion::upgrade_index_at(now);
    float uf = (float)std::max(u, 0), late = std::max(uf - 5.f, 0.f);
    int count = 0;
    int picked[6], picked_team[6], picked_type[6], picked_lane[6], n_picked = 0;
    for (int b = 0; b < w.n_lanes; ++b) {
        int l = w.lanes[b], lo = w.minion0 + 40 * b, hi = lo + 40;
        int from = lo;
        for (int t = 0; t < 2; ++t) {
            if (due_type[t][l] < 0) continue;
            int slot = -1;
            for (int i = from; i < hi; ++i)
                if (e.kind[i] == NONE || (e.kind[i] == MINION && !e.alive[i])) { slot = i; break; }
            if (slot < 0) continue;              // overflow: the unit is lost (counted in JAX)
            picked[n_picked] = slot, picked_team[n_picked] = t, picked_type[n_picked] = due_type[t][l];
            picked_lane[n_picked++] = l;
            from = slot + 1;
        }
    }
    // spawn_seq follows slot order.
    for (int a = 0; a < n_picked; ++a)
        for (int b = a + 1; b < n_picked; ++b)
            if (picked[b] < picked[a]) {
                std::swap(picked[a], picked[b]), std::swap(picked_team[a], picked_team[b]);
                std::swap(picked_type[a], picked_type[b]), std::swap(picked_lane[a], picked_lane[b]);
            }
    for (int a = 0; a < n_picked; ++a) {
        int i = picked[a], t = picked_team[a], k = picked_type[a], l = picked_lane[a];
        float hp = minion::BASE_HP[k] + std::min(minion::HP_UP[k] * uf, minion::HP_MAX_BONUS[k]);
        e.kind[i] = MINION, e.sub[i] = k, e.team[i] = t;
        e.x[i] = w.barracks[t][l][0], e.y[i] = w.barracks[t][l][1];
        e.hp[i] = hp, e.max_hp[i] = hp, e.radius[i] = minion::GAMEPLAY_RADIUS[k];
        e.armor[i] = minion::BASE_ARMOR[k] + (k == minion::MELEE ? minion::melee_armor(uf) : 0.f);
        e.magic_resist[i] = minion::BASE_MR[k];
        e.attack_damage[i] = minion::BASE_AD[k] + std::min(minion::AD_UP[k] * uf + minion::AD_UP_LATE[k] * late, minion::AD_MAX_BONUS[k]);
        e.attack_range[i] = minion::ATTACK_RANGE[k], e.attack_speed[i] = minion::ATTACK_SPEED[k];
        e.move_speed[i] = minion::base_move_speed(now), e.windup[i] = minion::WINDUP_S[k];
        e.missile_speed[i] = minion::MISSILE_SPEED[k];
        e.bounty_gold[i] = minion::gold_bounty(k, u, t), e.bounty_xp[i] = minion::XP_BASE[k];
        e.bounty_level[i] = level;
        e.alive[i] = 1, e.targetable[i] = 1;
        e.spawn_seq[i] = *e.next_seq + count++;
        e.spawn_time[i] = now;
        e.att_target[i] = -1, e.att_windup_left[i] = 0.f, e.att_cooldown_left[i] = 0.f;
        e.memo_anchor[i] = -3;
        e.cc_stun_until[i] = e.cc_root_until[i] = e.cc_silence_until[i] = e.cc_knockup_until[i] = 0.f;
        e.cc_slow[i] = e.cc_slow_until[i] = e.cc_champion_cc_until[i] = 0.f;
    }
    *e.next_seq += count;
}

// --- minion and turret targeting (lane.ai.select_targets) ---------------------------------------------------------
void reset_new_units(const World& w, Env& e, uint8_t* newu) {
    int n = w.n;
    size_t k = w.cols.size();
    for (int i = 0; i < n; ++i) {
        newu[i] = e.spawn_seq[i] != e.lane_ai_seq[i];
        if (newu[i]) {
            int t = clip_team(e.team[i]);
            int lane = 0;
            float best = INF;
            for (int l = 0; l < 3; ++l) {
                float d = sq(w.barracks[t][l][0] - e.x[i]) + sq(w.barracks[t][l][1] - e.y[i]);
                if (d < best) best = d, lane = l;
            }
            bool is_minion = e.kind[i] == MINION;
            e.lane_ai_seq[i] = e.spawn_seq[i];
            e.lane_ai_target[i] = -1, e.lane_ai_target_seq[i] = 0, e.lane_ai_target_priority[i] = NO_PRIORITY;
            e.lane_ai_sweep_timer[i] = SWEEP_INTERVAL_S, e.lane_ai_since_attack[i] = 0.f;
            e.lane_ai_lane[i] = is_minion ? lane : -1, e.lane_ai_waypoint[i] = 0;
            e.lane_ai_first_wave[i] = is_minion && e.spawn_time[i] < FIRST_WAVE_END_S;
            e.lane_ai_engaged[i] = 0, e.lane_ai_champion_aggro[i] = 0;
            int r = w.row_m_of[i];
            if (r >= 0) std::fill(e.lane_ai_ignore_until + r * k, e.lane_ai_ignore_until + (r + 1) * k, -INF);
            int c = w.col_of[i];
            if (c >= 0) {
                std::fill(e.lane_ai_last_attack + c * k, e.lane_ai_last_attack + (c + 1) * k, -INF);
                for (size_t a = 0; a < k; ++a) e.lane_ai_last_attack[a * k + c] = -INF;
            }
        }
        if (newu[i] || !e.alive[i]) e.lane_ai_warm_stacks[i] = 0, e.lane_ai_warm_until[i] = 0.f;
    }
}

// lane.ai._lane_goal for one minion row.
void lane_goal(const World& w, const Env& e, int i, int* waypoint, float* gx, float* gy) {
    int team = clip_team(e.team[i]), lane = clampi(e.lane_ai_lane[i], 0, 2);
    const float* path = w.path(team, lane);
    int length = w.lane_len[lane], cap = w.path_cap;
    float px = e.x[i], py = e.y[i];
    int k = e.lane_ai_waypoint[i];
    for (int it = 0; it < 3; ++it) {
        const float* wp = path + 2 * clampi(k, 0, cap - 1);
        const float* nx = path + 2 * clampi(k + 1, 0, cap - 1);
        bool near = std::sqrt(sq(px - wp[0]) + sq(py - wp[1])) < WAYPOINT_MARGIN;
        bool passed = (k + 1 < length)
                      && std::sqrt(sq(px - nx[0]) + sq(py - nx[1])) < std::sqrt(sq(wp[0] - nx[0]) + sq(wp[1] - nx[1]));
        if (k < length && (near || passed)) ++k;
    }
    *waypoint = k;
    if (k < length) {
        const float* p = path + 2 * clampi(k, 0, cap - 1);
        *gx = p[0], *gy = p[1];
        return;
    }
    for (int c : w.cols)
        if (e.kind[c] == NEXUS && e.team[c] != e.team[i]) { *gx = e.x[c], *gy = e.y[c]; return; }
    const float* last = path + 2 * clampi(length - 1, 0, cap - 1);
    *gx = last[0], *gy = last[1];
}


// lane.ai._lex_argmin over one row: the lowest (primary, secondary), first column on ties; -1 if none.
struct Best {
    int c = -1, p = NO_PRIORITY;
    float s = INF;
    void offer(int col, int prim, float sec) {
        if (c < 0 || prim < p || (prim == p && sec < s)) c = col, p = prim, s = sec;
    }
};

void select_targets(const World& w, Env& e, float now, Scratch& sc) {
    const int n = w.n;
    const float dt = w.dt;
    const size_t K = w.cols.size(), R = w.rows_m.size();
    const int32_t* cols = w.cols.data();
    Clock sub;
    reset_new_units(w, e, sc.newu.data());
    sub.lap(S_RESET);
    const uint8_t* dmg = e.prev_damage_matrix;

    // Live columns as compact records (column order), and per team the others' records: rows of team t scan
    // ``foes[t]`` (raw teams differ), still in column order so argmin ties keep the first column.
    auto& live = sc.live;
    live.clear();
    for (size_t c = 0; c < K; ++c) {
        int u = cols[c];
        if (e.alive[u])
            live.push_back({(int)c, u, e.x[u], e.y[u], e.radius[u], e.kind[u], e.sub[u], e.team[u], (bool)e.targetable[u]});
    }
    for (int t = 0; t < 2; ++t) {
        sc.foes[t].clear();
        for (const Live& L : live)
            if (L.team != t) sc.foes[t].push_back(L);
    }

    // Aggression memory: damage events (champions only champion-on-champion), then the recent-attack list (pairs
    // whose last event is within ATTACK_MEMORY_S; a native cache of last_attack's recent entries), then each live
    // attacker's live victims (attacking[c, v]; a dead victim is never an ally near anyone).
    const float before = *e.t;                                  // last tick's ``now``: the list was pruned with it
    for (int q = 0; q < *e.ev_n; ++q) {
        int s = e.ev_src[q], d = e.ev_dst[q], a = w.col_of[s], b = w.col_of[d];
        if (!(a >= 0 && b >= 0 && (e.kind[s] != CHAMPION || e.kind[d] == CHAMPION))) continue;
        float& la = e.lane_ai_last_attack[a * K + b];
        bool listed = (before - la) <= ATTACK_MEMORY_S;
        la = now;
        if (!listed) {
            if ((size_t)*e.rec_n < K * K) e.rec[(*e.rec_n)++] = (int)(a * K + b);
            else *e.rec_n = -1;                                 // overflow: rebuilt from the table below
        }
    }
    if (*e.rec_n < 0) {
        *e.rec_n = 0;
        for (size_t p = 0; p < K * K; ++p)
            if ((now - e.lane_ai_last_attack[p]) <= ATTACK_MEMORY_S) e.rec[(*e.rec_n)++] = (int)p;
    }
    int kept = 0;
    for (int q = 0; q < *e.rec_n; ++q)
        if ((now - e.lane_ai_last_attack[e.rec[q]]) <= ATTACK_MEMORY_S) e.rec[kept++] = e.rec[q];
    *e.rec_n = kept;
    int32_t* off = sc.victims_off.data();
    std::fill(off, off + K + 1, 0);
    auto each_pair = [&](auto&& f) {
        for (int q = 0; q < *e.rec_n; ++q) {
            int a = e.rec[q] / (int)K, b = e.rec[q] % (int)K;
            if (e.alive[cols[a]] && e.alive[cols[b]]) f(a, b);
        }
        for (const Live& L : live) {
            int v = L.u == L.u ? w.col_of[clampi(e.att_target[L.u], 0, n - 1)] : -1;
            if (e.att_windup_left[L.u] > 0.f && e.att_target[L.u] >= 0 && v >= 0 && e.alive[cols[v]]) f(L.c, v);
        }
    };
    each_pair([&](int a, int) { ++off[a + 1]; });
    for (size_t c = 0; c < K; ++c) off[c + 1] += off[c];
    sc.victims.resize(off[K]);
    {
        static thread_local std::vector<int32_t> fill;
        fill.assign(off, off + K);
        each_pair([&](int a, int b) { sc.victims[fill[a]++] = cols[b]; });
    }
    // Some ally of row unit i within ``radius`` of i that column c attacks, of the victim kind ``vkind``.
    auto attacks_ally_near = [&](int i, int c, int vkind, float radius) {
        for (int q = off[c]; q < off[c + 1]; ++q) {
            int v = sc.victims[q];
            if (e.kind[v] == vkind && e.team[v] == e.team[i] && dist(e, i, v) < radius) return true;
        }
        return false;
    };

    sub.lap(S_VICTIMS);
    int32_t* desired = sc.desired.data();
    float *gx = sc.gx.data(), *gy = sc.gy.data();
    uint8_t* stop = sc.stop.data();
    for (int i = 0; i < n; ++i) {
        desired[i] = -1;
        gx[i] = e.x[i], gy[i] = e.y[i];
        stop[i] = e.kind[i] != CHAMPION && e.kind[i] != MINION;
    }

    // --- minions (rows R): pass A holds, calls for help and gives up; pass B acquires with the team's attacker
    // counts. Each row keeps only its valid candidates (base_valid), in column order.
    struct Cell { int c; float d; int prio; };
    static thread_local std::vector<Cell> cells;
    static thread_local std::vector<int32_t> row_off, tgt, tprio_v;
    static thread_local std::vector<float> since_v, timer_v;
    static thread_local std::vector<uint8_t> acquire_v;
    cells.clear();
    row_off.resize(R + 1);
    tgt.resize(R), tprio_v.resize(R), since_v.resize(R), timer_v.resize(R), acquire_v.resize(R);
    auto unit_k = [&](int u) { return (int)std::nearbyint((e.spawn_time[u] - minion::WAVE_FIRST_S) / minion::WAVE_UNIT_GAP_S); };
    for (size_t r = 0; r < R; ++r) {
        int i = w.rows_m[r];
        row_off[r] = (int)cells.size();
        acquire_v[r] = 0;
        if (!(e.kind[i] == MINION && e.alive[i])) continue;
        int sub = clampi(e.sub[i], 0, 3);
        bool unengaged = e.lane_ai_first_wave[i] && !e.lane_ai_engaged[i];
        float acq = minion::ACQUISITION_RANGE[sub], first_acq = minion::FIRST_ACQUISITION_RANGE[sub];
        float wake = minion::WAKE_UP_RANGE[sub];
        const uint8_t* vis = e.visible + (size_t)clip_team(e.team[i]) * n;
        const float reach = std::max(std::max(acq, first_acq), wake) + 320.f;   // + the largest structure radius
        const float xi = e.x[i], yi = e.y[i], lim = sq(reach + 1.f);
        const auto& foes = sc.foes[e.team[i] == 1 ? 1 : 0];
        for (const Live& L : foes) {
            if (L.team == e.team[i]) continue;                   // raw teams (a team-2 row matches neither list)
            float d2 = sq(xi - L.x) + sq(yi - L.y);
            if (d2 > lim || !L.targetable || !vis[L.u]) continue;
            int k = L.kind;
            if (!(k == CHAMPION || k == MINION || is_structure(k) || (k == MONSTER && L.team < 2))) continue;
            float d = std::sqrt(d2);
            float scan = k == CHAMPION ? (unengaged ? wake : acq)
                       : (k == MINION ? (unengaged ? first_acq : acq) : acq + L.r);
            if (!(d < scan)) continue;
            int prio;
            if (k == CHAMPION)
                prio = attacks_ally_near(i, L.c, CHAMPION, minion::CFH_CHAMPION_RADIUS) ? 1 : 6;
            else if (k == MINION)
                prio = attacks_ally_near(i, L.c, CHAMPION, minion::CFH_GENERIC_RADIUS) ? 2
                     : (attacks_ally_near(i, L.c, MINION, minion::CFH_GENERIC_RADIUS) ? 3 : 5);
            else if (k == TURRET)
                prio = attacks_ally_near(i, L.c, MINION, minion::CFH_GENERIC_RADIUS) ? 4 : 7;
            else
                prio = 7;
            cells.push_back({L.c, d, prio});
        }
        const Cell* row = cells.data() + row_off[r];
        const int nrow = (int)cells.size() - row_off[r];
        float* ign = e.lane_ai_ignore_until + r * K;
        int held = e.lane_ai_target[i];
        int hs = clampi(held, 0, n - 1);
        int held_c = w.col_of[hs];
        bool held_valid = false;
        for (int q = 0; q < nrow && !held_valid; ++q) held_valid = row[q].c == held_c;
        bool held_ok = held >= 0 && e.spawn_seq[hs] == e.lane_ai_target_seq[i] && held_c >= 0 && held_valid;
        bool just_lost = held >= 0 && !held_ok;
        int target = held_ok ? held : -1;
        int tprio = held_ok ? e.lane_ai_target_priority[i] : NO_PRIORITY;
        int safe = clampi(target, 0, n - 1);
        bool in_windup = e.att_windup_left[i] > 0.f;
        bool hit_target = (in_windup && e.att_target[i] == target) || dmg[(size_t)i * n + safe];
        float since = target >= 0 ? (hit_target ? 0.f : e.lane_ai_since_attack[i] + dt) : 0.f;
        // Call for Help: a strictly better P1-P4 class switches at once, except holding a turret (not first wave) or
        // mid-windup.
        Best cfh;
        for (int q = 0; q < nrow; ++q)
            if (ign[row[q].c] <= now && row[q].prio <= 4) cfh.offer(row[q].c, row[q].prio, row[q].d);
        bool blocked = target >= 0 && e.kind[safe] == TURRET && !e.lane_ai_first_wave[i];
        bool sw = cfh.c >= 0 && cfh.p < tprio && !blocked && !in_windup;
        if (sw) target = cols[cfh.c], tprio = cfh.p, since = 0.f;
        float timer = e.lane_ai_sweep_timer[i] + dt;
        bool sweep = !sw && (just_lost || timer >= SWEEP_INTERVAL_S);
        if (sweep || sw) timer = 0.f;
        bool give_up = sweep && target >= 0 && since >= GIVE_UP_S;
        if (give_up) {
            int gc = w.col_of[target];
            if (gc >= 0) ign[gc] = now + IGNORE_S;
            target = -1, tprio = NO_PRIORITY;
        }
        tgt[r] = target, tprio_v[r] = tprio, since_v[r] = since, timer_v[r] = timer;
        acquire_v[r] = sweep && target < 0;
    }
    row_off[R] = (int)cells.size();
    sub.lap(S_PASS_A);
    // attackers[team][c]: live minion rows of the team currently targeting column c
    static thread_local std::vector<float> by_team;
    by_team.assign(3 * K, 0.f);
    for (size_t r = 0; r < R; ++r) {
        int i = w.rows_m[r];
        if (!(e.kind[i] == MINION && e.alive[i]) || tgt[r] < 0) continue;
        int c = w.col_of[tgt[r]];
        if (c >= 0) by_team[clampi(e.team[i], 0, 2) * K + c] += 1.f;
    }
    for (size_t r = 0; r < R; ++r) {
        int i = w.rows_m[r];
        bool minion = e.kind[i] == MINION && e.alive[i];
        if (e.kind[i] == MINION) {
            int wp; float lgx, lgy;
            lane_goal(w, e, i, &wp, &lgx, &lgy);
            e.lane_ai_waypoint[i] = wp;
            gx[i] = lgx, gy[i] = lgy;
        }
        if (!minion) {
            gx[i] = e.x[i], gy[i] = e.y[i];
            e.lane_ai_target_priority[i] = NO_PRIORITY, e.lane_ai_since_attack[i] = 0.f;
            continue;
        }
        const Cell* row = cells.data() + row_off[r];
        const int nrow = row_off[r + 1] - row_off[r];
        float* ign = e.lane_ai_ignore_until + r * K;
        int target = tgt[r], tprio = tprio_v[r];
        float since = since_v[r];
        if (acquire_v[r]) {
            // First-wave spread: closest-minion picks go to enemy first-wave melee k mod 3, then the least
            // attacked, then the closest.
            auto fw_melee = [&](int c) {
                int u = cols[c];
                return e.kind[u] == MINION && e.sub[u] == minion::MELEE && e.lane_ai_first_wave[u] && e.alive[u];
            };
            bool restrict_ = false;
            if (e.lane_ai_first_wave[i])
                for (int q = 0; q < nrow && !restrict_; ++q)
                    restrict_ = ign[row[q].c] <= now && fw_melee(row[q].c) && row[q].prio == 5;
            int uk = unit_k(i), want = ((uk % 3) + 3) % 3;
            const float* attackers = by_team.data() + clampi(e.team[i], 0, 2) * K;
            Best acq;
            for (int q = 0; q < nrow; ++q) {
                int c = row[q].c;
                if (!(ign[c] <= now)) continue;
                bool closest_min = row[q].prio == 5, fw = fw_melee(c);
                if (restrict_ && closest_min && !fw) continue;
                float spread = (restrict_ && fw && closest_min)
                                   ? (unit_k(cols[c]) == want ? 0.f : 1e5f) + 1e4f * attackers[c] : 0.f;
                acq.offer(c, row[q].prio, row[q].d + spread);
            }
            if (acq.c >= 0) target = cols[acq.c], tprio = acq.p, since = 0.f;
        }
        int safe = clampi(target, 0, n - 1);
        if (target >= 0 && e.kind[safe] == MINION) e.lane_ai_engaged[i] = 1;
        float d_held = std::sqrt(sq(e.x[i] - e.x[safe]) + sq(e.y[i] - e.y[safe]));
        bool in_range = d_held <= e.attack_range[i] + e.radius[i] + e.radius[safe];
        if (target >= 0) {
            gx[i] = in_range ? e.x[i] : e.x[safe];
            gy[i] = in_range ? e.y[i] : e.y[safe];
        }
        stop[i] = target >= 0 && in_range;
        desired[i] = target;
        e.lane_ai_target_priority[i] = target >= 0 ? tprio : NO_PRIORITY;
        e.lane_ai_sweep_timer[i] = timer_v[r];
        e.lane_ai_since_attack[i] = since;
    }

    sub.lap(S_PASS_B);
    // --- turrets (rows S): one pass over the foes keeps, per priority class, the nearest (first on ties); the lock;
    // and the nearest champion-protection aggressor (towers.select_target).
    for (int i : w.rows_s) {
        bool turret = e.kind[i] == TURRET && e.alive[i];
        if (!turret) {
            e.lane_ai_champion_aggro[i] = 0, e.lane_ai_warm_stacks[i] = 0, e.lane_ai_warm_until[i] = 0.f;
            continue;
        }
        const uint8_t* vis = e.visible + (size_t)clip_team(e.team[i]) * n;
        int held = e.lane_ai_target[i], hs = clampi(held, 0, n - 1);
        int lock = (held >= 0 && e.spawn_seq[hs] == e.lane_ai_target_seq[i]) ? w.col_of[hs] : -1;
        float near_d[7];
        int near_c[7];
        for (int p = 0; p < 7; ++p) near_d[p] = INF, near_c[p] = -1;
        int best_p = 100, aggressor = -1;
        float aggressor_d = INF;
        bool any = false, any_aggr = false, lock_ok = false;
        const float xi = e.x[i], yi = e.y[i], base = e.attack_range[i] + e.radius[i];
        for (const Live& L : sc.foes[e.team[i] == 1 ? 1 : 0]) {
            int k = L.kind;
            if (L.team == e.team[i] || !L.targetable) continue;
            if (!(k == CHAMPION || k == MINION || (k == MONSTER && L.team < 2))) continue;
            float reach = base + L.r;
            float d2 = sq(xi - L.x) + sq(yi - L.y);
            if (d2 > sq(reach + 1.f) || !vis[L.u]) continue;
            float d = std::sqrt(d2);
            if (!(d <= reach)) continue;
            int p = k == CHAMPION ? tower::CHAMPION_P
                  : (L.sub >= minion::CANNON ? tower::CANNON_SUPER
                     : (L.sub == minion::MELEE ? tower::MELEE_P : tower::CASTER_P));
            any = true;
            best_p = std::min(best_p, p);
            if (L.c == lock) lock_ok = true;
            if (d < near_d[p]) near_d[p] = d, near_c[p] = L.c;
            if (k == CHAMPION) {
                // aggressive: an enemy champion that damaged (last tick) an ally champion within 1400
                bool aggr = false;
                for (int q = 0; q < *e.ev_n && !aggr; ++q) {
                    int v = e.ev_dst[q];
                    aggr = e.ev_src[q] == L.u && w.col_of[v] >= 0 && e.alive[v] && e.kind[v] == CHAMPION
                           && e.team[v] == e.team[i] && dist(e, i, v) <= tower::PROTECTION_RADIUS;
                }
                if (aggr) {
                    any_aggr = true;
                    if (d < aggressor_d) aggressor_d = d, aggressor = L.c;
                }
            }
        }
        int pick = !any ? -1 : (any_aggr ? aggressor : (lock >= 0 && lock_ok ? lock : near_c[best_p]));
        int t_target = pick >= 0 ? cols[pick] : -1;
        e.lane_ai_champion_aggro[i] = any_aggr ? 1 : (e.lane_ai_champion_aggro[i] && t_target == held && t_target >= 0);
        bool hit_champ = false;
        for (int q = 0; q < *e.ev_n && !hit_champ; ++q)
            hit_champ = e.ev_src[q] == i && w.col_of[e.ev_dst[q]] >= 0 && e.kind[e.ev_dst[q]] == CHAMPION;
        int stacks_now = now < e.lane_ai_warm_until[i] ? e.lane_ai_warm_stacks[i] : 0;
        if (hit_champ) e.lane_ai_warm_stacks[i] = std::min(stacks_now + 1, 3), e.lane_ai_warm_until[i] = now + 5.f;
        desired[i] = t_target;
    }
    sub.lap(S_TURRETS);
    // Rows outside S keep no turret memory; every row's held target is this tick's choice.
    for (int i = 0; i < n; ++i) {
        if (i < w.struct0) e.lane_ai_champion_aggro[i] = 0, e.lane_ai_warm_stacks[i] = 0, e.lane_ai_warm_until[i] = 0.f;
        if (w.row_m_of[i] < 0) e.lane_ai_target_priority[i] = NO_PRIORITY, e.lane_ai_since_attack[i] = 0.f;
        e.lane_ai_target[i] = desired[i];
        e.lane_ai_target_seq[i] = desired[i] >= 0 ? e.spawn_seq[desired[i]] : 0;
    }
}

// --- movement (mechanics.move_step) ------------------------------------------------------------------------------
// Every slot below ``ward0`` follows its route anchor (inactive ones too: their anchors keep evolving in JAX); a
// slot whose inputs and anchor did not change last tick is skipped (same inputs, same output).
void move_step(const World& w, Env& e, const float* gx, const float* gy, const float* ms, const uint8_t* active,
               float* nx, float* ny) {
    const int m = w.ward0;
    const Routes& routes = w.routes;
    static thread_local std::vector<float> px, py;
    static thread_local std::vector<uint8_t> replan;
    px.resize(m), py.resize(m), replan.resize(m);
    for (int i = 0; i < m; ++i) {
        int team = clip_team(e.team[i]);
        float rr = std::min(e.radius[i], routes.radius);
        float* memo = e.memo_route + 4 * i;
        if (!active[i] && memo[0] == e.x[i] && memo[1] == e.y[i] && memo[2] == gx[i] && memo[3] == gy[i]
            && e.memo_anchor[i] == e.route_anchor[i]) {
            replan[i] = 0;
            continue;
        }
        Routes::Follow f = routes.follow(w.terrain[team], e.x[i], e.y[i], gx[i], gy[i], rr, e.route_anchor[i]);
        if (debug_route) {                                // x, y, gx, gy, rr, anchor in, active, follow x/y/ok/anchor/replan
            float* o = debug_route + 24 * i;
            o[0] = e.x[i], o[1] = e.y[i], o[2] = gx[i], o[3] = gy[i], o[4] = rr, o[5] = (float)e.route_anchor[i];
            o[6] = active[i], o[7] = f.x, o[8] = f.y, o[9] = f.ok, o[10] = (float)f.anchor, o[11] = f.replan;
            o[12] = o[13] = o[14] = o[15] = NAN;
        }
        bool fixed = !active[i] && f.anchor == e.route_anchor[i];
        memo[0] = e.x[i], memo[1] = e.y[i], memo[2] = gx[i], memo[3] = gy[i];
        e.memo_anchor[i] = fixed ? f.anchor : -3;
        px[i] = f.x, py[i] = f.y, replan[i] = f.replan;
        e.route_anchor[i] = f.anchor;
    }
    int budget = std::min(ROUTE_REPLANS_PER_TICK, m);
    for (int i = 0; i < m && budget > 0; ++i) {
        if (!(replan[i] && active[i])) continue;
        --budget;
        int team = clip_team(e.team[i]);
        float rr = std::min(e.radius[i], routes.radius);
        Routes::Follow f = routes.replan(w.terrain[team], e.x[i], e.y[i], gx[i], gy[i], rr);
        px[i] = f.x, py[i] = f.y, e.route_anchor[i] = f.anchor;
        if (debug_route) {                                // replan x/y/ok/anchor
            float* o = debug_route + 24 * i;
            o[12] = f.x, o[13] = f.y, o[14] = f.ok, o[15] = (float)f.anchor;
        }
    }
    for (int i = 0; i < w.n; ++i) nx[i] = e.x[i], ny[i] = e.y[i];
    for (int i = 0; i < m; ++i) {
        if (!active[i]) continue;
        int team = clip_team(e.team[i]);
        float rr = std::min(e.radius[i], routes.radius);
        float dx = px[i] - e.x[i], dy = py[i] - e.y[i];
        float d = std::sqrt(dx * dx + dy * dy);
        float step = std::min(ms[i] * w.dt, d);
        float ox = e.x[i], oy = e.y[i];
        if (d > 1e-6f) {
            float inv = std::max(d, 1e-6f);
            ox = std::fma(dx / inv, step, e.x[i]);     // XLA contracts p + u * step
            oy = std::fma(dy / inv, step, e.y[i]);
        } else {
            ox = e.x[i] + 0.f, oy = e.y[i] + 0.f;
        }
        if (segment_clear(w.terrain[team], e.x[i], e.y[i], ox, oy, rr, 33, 200.f)) nx[i] = ox, ny[i] = oy;
    }
}

// --- unit collision (collision.resolve: avoid, then separate), over slots below ``ward0`` ---------------------------
float pathing_radius(const Env& e, int i) {
    int k = e.kind[i];
    if (k == CHAMPION) return CHAMPION_PATHING_RADIUS;
    if (k == MINION) return MINION_PATHING_RADIUS[clampi(e.sub[i], 0, 3)];
    return e.radius[i];                  // camps use their record (not in the lane slice)
}

void collide(const World& w, Env& e, const float* x1, const float* y1, const float* gx, const float* gy,
             const uint8_t* moving, const uint8_t* solid, float* ox, float* oy) {
    const int m = w.ward0;
    const float h = w.avoid_horizon_ticks;
    static thread_local std::vector<float> rad, vx, vy, step, ux, uy, ovx, ovy, ax_, ay_, clr;
    static thread_local std::vector<uint8_t> act;
    // An obstacle j relevant to mover i, with the heading-independent terms of contact_time.
    struct Rel { int j; float rx, ry, d, c, ovx, ovy; };
    static thread_local std::vector<int32_t> rel_off;
    static thread_local std::vector<Rel> rel;
    for (auto* v : {&rad, &vx, &vy, &step, &ux, &uy, &ovx, &ovy, &ax_, &ay_, &clr}) v->resize(m);
    act.resize(m), rel_off.resize(m + 1);
    const float* x0 = e.x;
    const float* y0 = e.y;
    for (int i = 0; i < m; ++i) {
        rad[i] = pathing_radius(e, i);
        clr[i] = std::min(e.radius[i], w.routes.radius);
        vx[i] = x1[i] - x0[i], vy[i] = y1[i] - y0[i];
        step[i] = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
        act[i] = solid[i] && moving[i] && step[i] > 1e-3f && step[i] <= AVOID_MAX_STEP;
        ux[i] = vx[i] / std::max(step[i], 1e-6f), uy[i] = vy[i] / std::max(step[i], 1e-6f);
        bool walks = solid[i] && step[i] <= AVOID_MAX_STEP;
        ovx[i] = walks ? vx[i] : 0.f, ovy[i] = walks ? vy[i] : 0.f;
        ax_[i] = x0[i] + (x1[i] - x0[i]) * 1.f, ay_[i] = y0[i] + (y1[i] - y0[i]) * 1.f;   // frac 1 (JAX form)
    }
    // Phase 1, avoid: obstacles j relevant to mover i (rel), then the best heading per side.
    static thread_local std::vector<int32_t> solids;
    solids.clear();
    for (int j = 0; j < m; ++j)
        if (solid[j]) solids.push_back(j);
    rel.clear();
    for (int i = 0; i < m; ++i) {
        rel_off[i] = (int)rel.size();
        if (!act[i]) continue;
        float gdist = std::sqrt(sq(gx[i] - x0[i]) + sq(gy[i] - y0[i]));
        for (int j : solids) {
            if (j == i) continue;
            float rx = x0[j] - x0[i], ry = y0[j] - y0[i];
            float rsum = std::max(rad[i], rad[j]);
            if (rx * rx + ry * ry > sq(rsum + (step[i] + AVOID_MAX_STEP) * h + 1.f)) continue;
            float d = std::sqrt(rx * rx + ry * ry);
            bool on_goal = std::sqrt(sq(x0[j] - gx[i]) + sq(y0[j] - gy[i])) < rsum;
            if (!on_goal && d - rsum < gdist && d < rsum + (step[i] + AVOID_MAX_STEP) * h)
                rel.push_back({j, rx, ry, d, d * d - sq(rsum - .5f), ovx[j], ovy[j]});
        }
    }
    rel_off[m] = (int)rel.size();
    auto contact_time = [&](int i, float cx, float cy, int* nearest) {
        float best = INF, nd = INF;
        int near = 0;
        const float sx = cx * step[i], sy = cy * step[i];
        for (int q = rel_off[i]; q < rel_off[i + 1]; ++q) {
            const Rel& o = rel[q];
            float wx = sx - o.ovx, wy = sy - o.ovy;
            float a = o.rx * wx + o.ry * wy;
            if (!(a > 0.f)) continue;
            float ww = std::max(wx * wx + wy * wy, 1e-9f);
            float disc = a * a - ww * o.c;
            if (!(disc > 0.f)) continue;
            float t_hit = o.c <= 0.f ? 0.f : (a - std::sqrt(disc)) / ww;
            if (t_hit <= h) {
                if (t_hit < best) best = t_hit;
                if (o.d < nd) nd = o.d, near = o.j;
            }
        }
        if (nearest) *nearest = near;
        return best;
    };
    const Terrain* ter = w.terrain;
    for (int i = 0; i < m; ++i) {
        if (!act[i]) continue;
        int near = 0;
        float t0 = contact_time(i, ux[i], uy[i], &near);
        float side;
        {
            // argmin over the blocking set; an empty set picks column 0 (argmin of all-inf)
            float rx = x0[near] - x0[i], ry = y0[near] - y0[i];
            float cross = ux[i] * ry - uy[i] * rx;
            side = cross > 0.f ? -1.f : (cross < 0.f ? 1.f : (e.team[i] == 1 ? -1.f : 1.f));
        }
        float sc_[2], cx_[2], cy_[2];
        int k_[2];
        for (int s = 0; s < 2; ++s) {
            float sgn = s == 0 ? 1.f : -1.f;
            int turn = side * sgn > 0.f ? 0 : 1;     // sign of the rotation angle
            float best_s = 0, bx = 0, by = 0;
            int bk = 0;
            for (int j = 0; j < 6; ++j) {
                float c = w.avoid_cos[j], sn = w.avoid_sin[turn][j];
                float cx = ux[i] * c - uy[i] * sn, cy = ux[i] * sn + uy[i] * c;
                float ttc = j ? contact_time(i, cx, cy, nullptr) : t0;
                float score = std::isinf(ttc) ? 1e6f - (float)j : ttc;
                if (j == 0 || score > best_s) best_s = score, bx = cx, by = cy, bk = j;
            }
            sc_[s] = best_s, cx_[s] = bx, cy_[s] = by, k_[s] = bk;
        }
        int team = e.team[i] == 1 ? 1 : 0;
        float pax = cx_[0] * step[i] + x0[i], pay = cy_[0] * step[i] + y0[i];
        float pbx = cx_[1] * step[i] + x0[i], pby = cy_[1] * step[i] + y0[i];
        bool wa = k_[0] == 0 || ter[team].walkable(pax, pay, clr[i], 3);
        bool wb = k_[1] == 0 || ter[team].walkable(pbx, pby, clr[i], 3);
        bool use_b = wb && (!wa || sc_[1] > sc_[0]);
        bool use_a = wa && !use_b;
        float nx = use_a ? pax : (use_b ? pbx : x1[i]);
        float ny = use_a ? pay : (use_b ? pby : y1[i]);
        int k = use_a ? k_[0] : (use_b ? k_[1] : 0);
        if (!(k > 0)) nx = x1[i], ny = y1[i];
        float sc = use_a ? sc_[0] : (use_b ? sc_[1] : (std::isinf(t0) ? 1e6f : t0));
        float frac = sc < 1.f ? std::max(sc, CONTACT_MIN_FRAC) : 1.f;
        ax_[i] = x0[i] + (nx - x0[i]) * frac;
        ay_[i] = y0[i] + (ny - y0[i]) * frac;
    }
    if (debug_route)
        for (int i = 0; i < m; ++i) debug_route[24 * i + 19] = ax_[i], debug_route[24 * i + 20] = ay_[i];
    // Phase 2, separate: Jacobi pushes between overlapping solid pairs.
    static thread_local std::vector<float> mob, px, py;
    static thread_local std::vector<uint8_t> start_ok;
    mob.resize(m), px.resize(m), py.resize(m), start_ok.resize(m);
    float* x = ax_.data();
    float* y = ay_.data();
    for (int i = 0; i < m; ++i) {
        int team = e.team[i] == 1 ? 1 : 0;
        start_ok[i] = solid[i] ? ter[team].walkable(x[i], y[i], clr[i], 3) : 1;
        mob[i] = moving[i] ? 1.f : STATIONARY_MOBILITY;
    }
    // Each round moves a unit at most MAX_PUSH, so a pair can only overlap in some round if it starts within
    // rsum + 2 * MAX_PUSH * (rounds - 1); other pairs contribute exact zeros and are skipped.
    static thread_local std::vector<int32_t> pair_off, pair_j;
    pair_off.resize(m + 1);
    pair_j.clear();
    const float reach = 2.f * MAX_PUSH * (float)(SEPARATION_ITERS - 1) + 1.f;
    for (int i = 0; i < m; ++i) {
        pair_off[i] = (int)pair_j.size();
        if (!solid[i]) continue;
        for (int j : solids) {
            if (j == i) continue;
            float dx = x[i] - x[j], dy = y[i] - y[j], lim = std::max(rad[i], rad[j]) + reach;
            if (dx * dx + dy * dy < lim * lim) pair_j.push_back(j);
        }
    }
    pair_off[m] = (int)pair_j.size();
    for (int it = 0; it < SEPARATION_ITERS; ++it) {
        for (int i = 0; i < m; ++i) {
            px[i] = 0.f, py[i] = 0.f;
            if (!solid[i]) continue;
            for (int q = pair_off[i]; q < pair_off[i + 1]; ++q) {
                int j = pair_j[q];
                float dx = x[i] - x[j], dy = y[i] - y[j];
                float d = std::sqrt(dx * dx + dy * dy);
                float over = std::max(std::max(rad[i], rad[j]) - d, 0.f);
                if (!(over > 0.f)) continue;
                bool tiny = d < 1e-3f;
                float nxv = tiny ? w.sep_fx[(size_t)i * m + j] : dx / std::max(d, 1e-6f);
                float nyv = tiny ? w.sep_fy[(size_t)i * m + j] : dy / std::max(d, 1e-6f);
                float share = mob[i] / std::max(mob[i] + mob[j], 1e-9f);
                px[i] += share * over * nxv;
                py[i] += share * over * nyv;
            }
        }
        for (int i = 0; i < m; ++i) {
            if (!solid[i]) continue;
            float mag = std::sqrt(px[i] * px[i] + py[i] * py[i]);
            float f = std::min(1.f, MAX_PUSH / std::max(mag, 1e-6f));
            float cx = x[i] + px[i] * f, cy = y[i] + py[i] * f;
            int team = e.team[i] == 1 ? 1 : 0;
            bool ok = !start_ok[i] || ter[team].walkable(cx, cy, clr[i], 3);
            if (ok) x[i] = cx, y[i] = cy;
            if (mag > 1e-4f && !ok) mob[i] = 1e-3f;
        }
    }
    for (int i = 0; i < w.n; ++i) ox[i] = i < m ? x[i] : x1[i], oy[i] = i < m ? y[i] : y1[i];
    if (debug_route)
        for (int i = 0; i < m; ++i)
            debug_route[24 * i + 21] = x[i], debug_route[24 * i + 22] = y[i], debug_route[24 * i + 23] = solid[i];
}

// --- attacks (mechanics.attack_step, lane.ai.attack_packets, spawn/advance_missiles) -------------------------------
bool in_range(const Env& e, int i, int target) {
    if (target < 0) return false;
    return dist(e, i, target) <= e.attack_range[i] + e.radius[i] + e.radius[target];
}

bool hostile_ok(const Env& e, int i, int t) {
    return e.alive[t] && e.targetable[t] && e.alive[t] && e.team[t] != e.team[i];
}

void attack_step(const World& w, Env& e, const int32_t* desired, const uint8_t* can_attack, uint8_t* launched) {
    const int n = w.n;
    const float dt = w.dt;
    for (int i = 0; i < n; ++i) {
        int des = desired[i], t = clampi(des, 0, n - 1);
        bool valid = des >= 0 && hostile_ok(e, i, t);
        bool same = des == e.att_target[i] && e.spawn_seq[t] == e.att_target_seq[i];
        bool ready = valid && in_range(e, i, des) && can_attack[i] && e.alive[i];
        bool winding = e.att_windup_left[i] > 0.f;
        int at = e.att_target[i], tt = clampi(at, 0, n - 1);
        bool target_ok = at >= 0 && hostile_ok(e, i, tt) && e.spawn_seq[tt] == e.att_target_seq[i];
        bool grace = winding && (e.att_windup_left[i] - dt <= 1e-5f) && target_ok && in_range(e, i, at) && can_attack[i]
                     && e.alive[i];
        bool cancel = winding && !(same && ready) && !grace;
        float cooldown = std::max(e.att_cooldown_left[i] - dt, 0.f);
        if (cancel) cooldown = 0.f;
        float left = cancel ? 0.f : e.att_windup_left[i];
        bool start = ready && !(winding && !cancel) && cooldown <= 0.f;
        float period = 1.f / std::max(e.attack_speed[i], 1e-3f);
        if (start) left = e.windup[i], cooldown = period;
        bool held = ready || (grace && !cancel);
        bool fire = left > 0.f && (left - dt <= 1e-5f) && held;
        left = (fire || left <= 0.f) ? 0.f : std::max(left - dt, 0.f);
        int target = valid ? des : -1;
        int seq = valid ? e.spawn_seq[t] : e.att_target_seq[i];
        if (grace && !cancel) target = at, seq = e.att_target_seq[i];
        e.att_target[i] = target, e.att_target_seq[i] = seq, e.att_windup_left[i] = left, e.att_cooldown_left[i] = cooldown;
        launched[i] = fire;
    }
}

// --- one tick -------------------------------------------------------------------------------------------------------
}  // namespace

void profile(double* out, bool reset) {
    for (int k = 0; k < N_PHASES; ++k) {
        out[k] = prof_ns[k];
        if (reset) prof_ns[k] = 0;
    }
}

TickStats step(const World& w, Env& e, const Orders& o) {
    TickStats st;
    if (*e.game_over) return st;                       // a fallen Nexus freezes the world
    Scratch& sc = scratch;
    sc.size(w);
    const int n = w.n;
    const float dt = w.dt;
    const float now = *e.t + dt;
    const int tick0 = *e.tick;

    Clock clk;
    // 1. INPUT: waves.
    spawn(w, e, now);
    clk.lap(P_SPAWN);
    // 4. AI: structures, then minion and turret targets.
    turret_tick(w, e, now);
    clk.lap(P_TURRET);
    select_targets(w, e, now, sc);
    clk.lap(P_SELECT);

    // 5. MOVE: lane walking and collision.
    float *gx = sc.gx.data(), *gy = sc.gy.data(), *ms = sc.ms.data();
    uint8_t *active = sc.active.data(), *solid = sc.collide.data();
    for (int i = 0; i < n; ++i) {
        bool can_move = !(e.cc_stun_until[i] > now || e.cc_root_until[i] > now || e.cc_knockup_until[i] > now);
        bool minion = e.kind[i] == MINION && e.alive[i];
        float slow = e.cc_slow_until[i] > now ? e.cc_slow[i] : 0.f;
        float base = e.move_speed[i];
        if (e.kind[i] == MINION) {
            int idx = minion::wave_index_at(e.spawn_time[i]);
            float bonus = minion::sidelane_bonus(idx + 1, e.lane_ai_lane[i], minion::wave_spawn_time(idx), now - e.spawn_time[i]);
            base = minion::soft_cap(minion::base_move_speed(now) + bonus);
        }
        ms[i] = base * (1.f - slow * (1.f - 0.f));
        active[i] = minion && !sc.stop[i] && can_move;
        if (!minion) gx[i] = e.x[i], gy[i] = e.y[i];
        bool ghost = e.kind[i] == MINION && e.alive[i] && minion::wave_index_at(e.spawn_time[i]) == 0
                     && (now - e.spawn_time[i]) < minion::first_wave_ghost_s(e.lane_ai_lane[i]);
        solid[i] = e.alive[i] && e.kind[i] != WARD && !is_structure(e.kind[i]) && !ghost;
    }
    float *mx = sc.nx.data(), *my = sc.ny.data();
    clk.lap(P_MOVE_PREP);
    move_step(w, e, gx, gy, ms, active, mx, my);
    if (debug_route)
        for (int i = 0; i < w.ward0; ++i)
            debug_route[24 * i + 16] = ms[i], debug_route[24 * i + 17] = mx[i], debug_route[24 * i + 18] = my[i];
    clk.lap(P_ROUTE);
    float *cx = sc.start_x.data(), *cy = sc.start_y.data();
    collide(w, e, mx, my, gx, gy, active, solid, cx, cy);
    clk.lap(P_COLLIDE);
    std::memcpy(e.x, cx, n * sizeof(float));
    std::memcpy(e.y, cy, n * sizeof(float));

    // 6. ATTACK: the attack machine on the moved positions, packets at launch, missiles.
    uint8_t* can_attack = sc.can_move.data();
    for (int i = 0; i < n; ++i) can_attack[i] = !(e.cc_stun_until[i] > now || e.cc_knockup_until[i] > now) && e.alive[i];
    uint8_t* launched = sc.launched.data();
    attack_step(w, e, sc.desired.data(), can_attack, launched);
    // Minion Pushing (attack.minion_pushing): level and lane-turret leads.
    float alive_t[2][3] = {};
    for (int i = 0; i < n; ++i)
        if (e.kind[i] == TURRET && e.alive[i] && w.unit_lane[i] >= 0 && w.unit_lane[i] < 3 && e.team[i] >= 0
            && e.team[i] < 2)
            alive_t[e.team[i]][w.unit_lane[i]] += 1.f;
    float* push_div = sc.push_div.data();
    float* raw_all = sc.raw_all.data();
    int32_t* dtype_all = sc.dtype_all.data();
    for (int i = 0; i < n; ++i) {
        float bonus = 0.f, div = 1.f;
        if (e.kind[i] == MINION && e.lane_ai_lane[i] >= 0) {
            int t = clip_team(e.team[i]), l = clampi(e.lane_ai_lane[i], 0, 2);
            minion::pushing((float)e.econ_level[t] - (float)e.econ_level[1 - t], alive_t[t][l] - alive_t[1 - t][l],
                            std::floor(now), &bonus, &div);
        }
        push_div[i] = div;
    }
    for (int i = 0; i < n; ++i) {
        raw_all[i] = 0.f, dtype_all[i] = PHYSICAL;
        if (!launched[i] || e.kind[i] == CHAMPION) continue;
        int tgt = e.att_target[i], t = clampi(tgt, 0, n - 1);
        int sub = clampi(e.sub[i], 0, 3), t_kind = e.kind[t], t_sub = clampi(e.sub[t], 0, 3);
        bool is_minion = e.kind[i] == MINION, is_turret = e.kind[i] == TURRET;
        bool t_minion = t_kind == MINION, t_champ = t_kind == CHAMPION;
        bool t_building = t_kind == INHIBITOR || t_kind == NEXUS;
        float m_raw = e.attack_damage[i] + (t_minion ? minion::SLAYER[sub] * e.hp[t] : 0.f);
        m_raw = m_raw * ((sub == minion::CANNON && t_kind == TURRET) ? SIEGE_TURRET_BONUS : 1.f);
        m_raw = m_raw * ((sub == minion::SUPER && t_building) ? SUPER_BUILDING_SCALE : 1.f);
        m_raw = m_raw / ((is_minion && t_minion) ? push_div[t] : 1.f);
        int stacks = now < e.lane_ai_warm_until[i] ? e.lane_ai_warm_stacks[i] : 0;
        float champ_raw = tower::attack_damage(sub, now) * tower::warming(stacks);
        float shot_raw = tower::shot_fraction(t_sub, sub) * e.max_hp[t];
        bool valid = tgt >= 0 && e.alive[i] && ((is_minion && t_kind != NONE) || (is_turret && (t_champ || t_minion)));
        float raw = is_turret ? (t_champ ? champ_raw : shot_raw) : m_raw;
        raw_all[i] = valid ? raw : 0.f;
        dtype_all[i] = is_turret ? (t_champ ? PHYSICAL : TRUE_DMG) : PHYSICAL;
        // The pushing bonus (``amp``) is computed in JAX but never reaches the packets (direct and missile packets
        // are built without it), so it is not applied here either.
    }
    auto& pk = sc.packets;
    pk.clear();
    // Direct (melee) packets in unit order, then missiles.
    for (int i = 0; i < n; ++i) {
        int tgt = e.att_target[i];
        bool ranged = launched[i] && e.missile_speed[i] > 0.f;
        bool on_ward = tgt >= 0 && e.kind[clampi(tgt, 0, n - 1)] == WARD;
        if (launched[i] && !ranged && tgt >= 0 && !on_ward)
            pk.push_back({i, std::max(tgt, 0), raw_all[i], dtype_all[i], BASIC_ATTACK, 0.f});
    }
    {   // spawn_missiles: launchers in unit order take free slots in slot order
        int slot = 0, M = w.missiles;
        for (int i = 0; i < n; ++i) {
            int tgt = e.att_target[i];
            bool on_ward = tgt >= 0 && e.kind[clampi(tgt, 0, n - 1)] == WARD;
            if (!(launched[i] && e.missile_speed[i] > 0.f && !on_ward)) continue;
            while (slot < M && e.missiles_alive[slot]) ++slot;
            if (slot >= M) { ++st.missile_overflow; continue; }
            int t = clampi(tgt, 0, n - 1);
            e.missiles_alive[slot] = 1, e.missiles_src[slot] = i, e.missiles_dst[slot] = tgt, e.missiles_dst_seq[slot] = e.spawn_seq[t];
            e.missiles_x[slot] = e.x[i], e.missiles_y[slot] = e.y[i], e.missiles_speed[slot] = e.missile_speed[i];
            e.missiles_raw[slot] = raw_all[i], e.missiles_dtype[slot] = dtype_all[i], e.missiles_flags[slot] = BASIC_ATTACK;
            e.missiles_cast_id[slot] = tick0 * CAST_ID_STRIDE + i + 1, e.missiles_crit[slot] = 0;
            ++slot;
        }
        for (int s = 0; s < M; ++s) {      // advance_missiles (every slot, as in JAX)
            int t = clampi(e.missiles_dst[s], 0, n - 1);
            bool gone = !e.alive[t] || e.spawn_seq[t] != e.missiles_dst_seq[s];
            float dx = e.x[t] - e.missiles_x[s], dy = e.y[t] - e.missiles_y[s];
            float d = std::sqrt(dx * dx + dy * dy);
            float stp = e.missiles_speed[s] * dt;
            bool arrive = e.missiles_alive[s] && !gone && (d - e.radius[t] <= stp);
            float f = d > 0.f ? std::min(stp / std::max(d, 1e-6f), 1.f) : 1.f;
            e.missiles_x[s] = std::fma(dx, f, e.missiles_x[s]), e.missiles_y[s] = std::fma(dy, f, e.missiles_y[s]);
            e.missiles_alive[s] = e.missiles_alive[s] && !gone && !arrive;
            if (arrive) pk.push_back({e.missiles_src[s], e.missiles_dst[s], e.missiles_raw[s], e.missiles_dtype[s], e.missiles_flags[s], 0.f});
        }
    }
    st.packets = (int)pk.size();
    if ((int)pk.size() > w.packet_capacity) {
        st.packet_overflow = (int)pk.size() - w.packet_capacity;
        pk.resize(w.packet_capacity);
    }

    clk.lap(P_ATTACK);
    // 7. DAMAGE: structure defense, mitigation, parallel resolution (no shields or champions in the slice).
    float* armor = sc.armor.data();
    float* t_mult = sc.t_mult.data();
    uint8_t* invuln = sc.invuln.data();
    for (int i = 0; i < n; ++i) {
        armor[i] = e.armor[i];
        t_mult[i] = 1.f;
        invuln[i] = e.towers_is_structure[i] && !e.towers_targetable[i];
        if (!e.towers_is_structure[i]) continue;
        if (e.kind[i] == TURRET) {
            int n850 = 0;
            if (w.col_of[i] >= 0)
                for (int c = 0; c < N_CHAMPIONS; ++c)        // champions are units [0, C), all in the columns
                    if (e.kind[c] == CHAMPION && e.alive[c] && e.towers_team[i] != e.team[c] && dist(e, i, c) <= tower::BULWARK_RADIUS)
                        ++n850;
            int count = clampi(n850, 1, 5);
            float per_stack = 30.f + 5.f * (float)(count - 1);
            int stacks = 0;
            for (int k = 0; k < 4; ++k) stacks += e.towers_turret_bulwark_until[i * 4 + k] > now;
            armor[i] = 60.f - (e.towers_turret_tier[i] == tower::OUTER ? 15.f * tower::decay_steps(now) : 0.f)
                       + per_stack * (float)stacks;
            if (e.towers_turret_hp[i] > 0.f && now >= e.towers_turret_backdoor_until[i]) t_mult[i] = .2f;
        } else if (e.kind[i] == INHIBITOR || e.kind[i] == NEXUS) {
            armor[i] = tower::BUILDING_ARMOR;
        }
    }
    float* hp_new = sc.hp_after.data();
    std::memcpy(hp_new, e.hp, n * sizeof(float));
    static thread_local std::vector<float> total;
    total.assign(n, 0.f);
    for (const Packet& p : pk) {
        int s = p.src, d = p.dst;
        float ratio = UNIT_CLASS_RATIO[damage_class(e.kind[s])][damage_class(e.kind[d])];
        bool is_true = p.dtype == TRUE_DMG;
        float dealt = std::max(1.f + p.amp - 0.f, 0.f);
        float raw = p.raw * dealt * ratio;
        float pen = e.kind[s] == TURRET ? tower::ARMOR_PEN : 0.f;
        float r = armor[d] - 0.f;
        r = r > 0.f ? r * (1.f - 0.f) : r;
        r = r > 0.f ? r * (1.f - pen) : r;
        r = r > 0.f ? std::max(0.f, r - 0.f) : r;
        float mult = p.dtype == PHYSICAL ? mitigation(r) : (p.dtype == MAGIC ? mitigation(e.magic_resist[d]) : 1.f);
        float post = raw * mult;
        post = is_true ? post : std::max(post - 0.f - 0.f, 0.f);
        post = std::max(post - 0.f, 0.f);
        post = post * t_mult[d];
        total[d] += invuln[d] ? 0.f : std::max(post, 0.f);
    }
    for (int i = 0; i < n; ++i)
        if (e.hp[i] > 0.f) hp_new[i] = e.hp[i] - total[i];

    clk.lap(P_DAMAGE);
    // 9. DEATH: deaths, plates and structure kills, the damage matrix the AI reads next tick.
    uint8_t* died = sc.died.data();
    for (int i = 0; i < n; ++i) died[i] = e.alive[i] && hp_new[i] <= 0.f;
    {
        static thread_local std::vector<float> after;
        after.resize(n);
        for (int i = 0; i < n; ++i) after[i] = is_structure(e.kind[i]) ? hp_new[i] : e.hp[i];
        structure_damage_events(w, e, e.hp, after.data(), now);
    }
    for (int q = 0; q < *e.ev_n; ++q) e.prev_damage_matrix[(size_t)e.ev_src[q] * n + e.ev_dst[q]] = 0;
    *e.ev_n = 0;
    for (const Packet& p : pk) {
        uint8_t& m = e.prev_damage_matrix[(size_t)p.src * n + p.dst];
        if (!m) m = 1, e.ev_src[*e.ev_n] = p.src, e.ev_dst[*e.ev_n] = p.dst, ++*e.ev_n;
    }

    clk.lap(P_DEATH);
    // 10. TIMERS: dead minions free their slot; units left inside closed terrain step out (dynamic_terrain.eject).
    for (int i = 0; i < n; ++i) {
        if (died[i]) {
            e.alive[i] = 0;
            if (e.kind[i] == MINION) e.kind[i] = NONE;
            e.cc_stun_until[i] = e.cc_root_until[i] = e.cc_silence_until[i] = e.cc_knockup_until[i] = 0.f;
            e.cc_slow[i] = e.cc_slow_until[i] = e.cc_champion_cc_until[i] = 0.f;
        }
        e.hp[i] = e.alive[i] ? hp_new[i] : std::min(hp_new[i], 0.f);
    }
    for (int i = w.ward0; i < w.struct0; ++i)            // empty ward slots (wards.ward_view): seq 2^24 + 0
        e.spawn_seq[i] = (1 << 24), e.radius[i] = 1.f;
    {
        int budget = 16;
        for (int i = 0; i < n && budget > 0; ++i) {
            if (!(e.alive[i] && (e.kind[i] == CHAMPION || e.kind[i] == MINION))) continue;
            int team = clip_team(e.team[i]);
            float r = std::min(std::min(e.radius[i], w.routes.radius), 150.f);
            if (w.terrain[team].walkable(e.x[i], e.y[i], r, 3)) continue;
            --budget;
            int best = -1;
            float best_r = INF;
            for (size_t k = 0; k < w.eject_r.size(); ++k)
                if (w.eject_r[k] < best_r && w.terrain[team].walkable(e.x[i] + w.eject_dx[k], e.y[i] + w.eject_dy[k], r, 3))
                    best_r = w.eject_r[k], best = (int)k;
            if (best >= 0) e.x[i] = e.x[i] + w.eject_dx[best], e.y[i] = e.y[i] + w.eject_dy[best];
        }
    }

    clk.lap(P_TIMERS);
    // 11. FOG: attack-reveal circles, then next tick's team visibility (vision.visibility, early exit per team).
    for (int c = 0; c < N_CHAMPIONS; ++c) {
        bool hidden = !e.visible[(size_t)(1 - clip_team(e.team[c])) * n + c] && e.alive[c];
        if (launched[c] && hidden) e.reveal_x[c] = e.x[c], e.reveal_y[c] = e.y[c], e.reveal_until[c] = now + REVEAL_DURATION;
    }
    {
        static thread_local std::vector<float> sight;
        sight.resize(n);
        for (int i = 0; i < n; ++i) {
            bool live = e.alive[i] && e.kind[i] != NONE;
            int k = e.kind[i];
            float r = k == CHAMPION ? CHAMPION_SIGHT
                    : k == MINION ? (e.sub[i] == 3 ? SUPER_MINION_SIGHT : MINION_SIGHT)
                    : k == TURRET ? TURRET_SIGHT : k == NEXUS ? NEXUS_SIGHT : k == INHIBITOR ? 0.f
                    : k == WARD ? (e.sub[i] == 2 ? FARSIGHT_SIGHT : WARD_SIGHT) : 0.f;
            sight[i] = live ? r : 0.f;
        }
        const int nf = w.struct0;
        int pairs = 0;
        static thread_local std::vector<int32_t> viewers;
        viewers.clear();
        for (int i = 0; i < n; ++i)
            if (sight[i] > 0.f) viewers.push_back(i);
        for (int j = 0; j < n; ++j) {
            bool live_j = e.alive[j] && e.kind[j] != NONE;
            for (int t = 0; t < 2; ++t) {
                bool seen = false;
                if (j >= nf || e.team[j] == t) seen = true;
                else if (live_j) {
                    // Seen if any in-range viewer of the team has a clear ray: try the nearest first (short rays).
                    static thread_local std::vector<std::pair<float, int>> cand;
                    cand.clear();
                    for (int i : viewers) {
                        if (e.team[i] != t) continue;
                        float d2 = sq(e.x[i] - e.x[j]) + sq(e.y[i] - e.y[j]);
                        if (d2 <= sight[i] * sight[i]) cand.push_back({d2, i});
                    }
                    pairs += (int)cand.size();
                    if (!w.fog) seen = !cand.empty();
                    else if (!cand.empty()) {
                        std::sort(cand.begin(), cand.end());
                        for (const auto& [d2, i] : cand)
                            if (w.vision.clear(e.x[i], e.y[i], e.x[j], e.y[j])) { seen = true; break; }
                    }
                    for (int c = 0; c < N_CHAMPIONS && !seen; ++c)
                        seen = t != e.team[c] && now < e.reveal_until[c]
                               && sq(e.x[j] - e.reveal_x[c]) + sq(e.y[j] - e.reveal_y[c]) <= REVEAL_RADIUS * REVEAL_RADIUS;
                }
                e.visible[(size_t)t * n + j] = seen && live_j;
            }
        }
        st.rays = pairs;
        st.ray_overflow = std::max(pairs - w.ray_capacity, 0);
    }

    clk.lap(P_FOG);
    // commit
    *e.t = now;
    *e.tick = tick0 + 1;
    bool lost[2] = {false, false};
    for (int i = 0; i < n; ++i)
        if (e.towers_is_structure[i] && e.towers_turret_tier[i] == tower::NEXUS_BUILDING && e.towers_turret_hp[i] <= 0.f && e.towers_team[i] >= 0
            && e.towers_team[i] < 2)
            lost[e.towers_team[i]] = true;
    *e.game_over = lost[0] || lost[1];
    *e.winner = (lost[0] && !lost[1]) ? 1 : ((lost[1] && !lost[0]) ? 0 : -1);
    return st;
}

void scripted_orders(const World& w, const Env& e, Orders& o, int mode) {
    for (int c = 0; c < N_CHAMPIONS; ++c) {          // world.state.no_orders
        o.move[c] = 0, o.move_x[c] = 0.f, o.move_y[c] = 0.f, o.attack[c] = -1, o.stop[c] = 0;
        o.cast_slot[c] = -1, o.cast_target[c] = -1, o.cast_x[c] = 0.f, o.cast_y[c] = 0.f;
        o.summoner_slot[c] = -1, o.summoner_target[c] = -1, o.summoner_x[c] = 0.f, o.summoner_y[c] = 0.f;
        o.item_active[c] = 0, o.buy[c] = 0, o.sell[c] = 0, o.recall[c] = 0, o.level_up[c] = -1;
        o.attack_move[c] = 0, o.ward_kind[c] = -1, o.ward_x[c] = 0.f, o.ward_y[c] = 0.f;
    }
    if (mode != 1) return;
    // ops/modern/bench.scripted_orders: attack the nearest live enemy (not Nexus or inhibitor) within 700, else
    // walk to the lane midpoint.
    for (int c = 0; c < N_CHAMPIONS; ++c) {
        float best = INF;
        int pick = -1;
        for (int j = 0; j < w.n; ++j) {
            int k = e.kind[j];
            if (e.team[j] == e.team[c] || !e.alive[j] || k == NONE || k == NEXUS || k == INHIBITOR) continue;
            float d = std::sqrt(sq(e.x[j] - e.x[c]) + sq(e.y[j] - e.y[c]));
            if (d < 700.f && d < best) best = d, pick = j;
        }
        o.attack[c] = pick;
        o.move[c] = pick < 0;
        o.move_x[c] = w.lane_mid[0], o.move_y[c] = w.lane_mid[1];
    }
}

}  // namespace lanesim
