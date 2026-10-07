// The static world and one env's state as plain arrays, laid out like the JAX ModernState slice it mirrors
// (lanerl_jax/modern/world/state.py): (N,) unit columns, the attack machine, missiles, lane AI and structures.
// ``Env`` only points at storage, so the same tick runs on numpy arrays (differential tests) or on a batch the
// library owns. ``ENV_FIELDS`` is the one list of fields: it generates the struct, its byte sizes and the
// description the Python binding reads.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "geom.hpp"

namespace lanesim {

enum Kind : int32_t { NONE, CHAMPION, MINION, TURRET, INHIBITOR, NEXUS, MONSTER, WARD };
constexpr int N_CHAMPIONS = 2;
constexpr int NO_PRIORITY = 99;

inline bool is_structure(int k) { return k == TURRET || k == INHIBITOR || k == NEXUS; }

struct World {
    // Layout (world.config.Layout) and static unit columns.
    int n = 0, minion0 = 0, monster0 = 0, epic0 = 0, ward0 = 0, struct0 = 0;
    int n_lanes = 0, lanes[3] = {0, 0, 0};
    std::vector<int32_t> unit_lane;                // (N,)
    std::vector<int32_t> rows_m, rows_s, cols;     // lane-AI slot sets (AISlots)
    std::vector<int32_t> col_of;                   // (N,) column position or -1
    std::vector<int32_t> row_m_of;                 // (N,) minion-row position or -1
    float dt = 1.f / 30.f;
    int missiles = 64, packet_capacity = 0, ray_capacity = 0;
    Terrain terrain[2];
    Routes routes;
    VisionGrid vision;
    bool fog = true;
    // map.lanes: LANE_PATHS[team][lane] (L, 2), padded with the last point; BARRACKS[team][lane].
    int path_cap = 0;
    std::vector<float> lane_paths;
    int lane_len[3] = {0, 0, 0};
    float barracks[2][3][2] = {};
    // Constants evaluated by XLA (float32 cos/sin), so both sides use the same bits.
    float avoid_cos[6] = {}, avoid_sin[2][6] = {};   // [sign of the turn][angle]
    std::vector<float> sep_fx, sep_fy;             // (ward0, ward0) coincident-pair fallback directions
    std::vector<float> eject_dx, eject_dy, eject_r; // dynamic_terrain.eject ring offsets (7 rings x 16 directions)
    float avoid_horizon_ticks = 9.f;

    const float* path(int team, int lane) const {
        return lane_paths.data() + ((size_t)(team * 3 + lane) * path_cap) * 2;
    }
};

// Sizes: S scalar, N units, M missiles, L (team, lane) cursors, C champions, RK minion rows x cols, KK cols x cols,
// N4 bulwark stacks, NN unit pairs, TN (team, unit), P packets. ``memo_*`` (route-follow memo) and ``ev_*`` (last
// tick's damage events as a list: the nonzeros of ``damage_matrix``) and ``rec*`` (last_attack entries within the
// attack memory, as flat (cols, cols) indices) are native-only caches.
#define LANESIM_ENV_FIELDS(X)                                                                                   \
    X(float, t, S) X(int32_t, tick, S) X(int32_t, next_seq, S) X(uint8_t, game_over, S) X(int32_t, winner, S)    \
    X(int32_t, kind, N) X(int32_t, sub, N) X(int32_t, team, N) X(int32_t, spawn_seq, N)                          \
    X(int32_t, bounty_level, N) X(uint8_t, alive, N) X(uint8_t, targetable, N)                                   \
    X(float, x, N) X(float, y, N) X(float, hp, N) X(float, max_hp, N) X(float, radius, N) X(float, armor, N)      \
    X(float, mr, N) X(float, ad, N) X(float, range, N) X(float, aspd, N) X(float, ms, N) X(float, windup, N)     \
    X(float, spawn_time, N) X(float, missile_speed, N) X(float, bounty_gold, N) X(float, bounty_xp, N)           \
    X(int32_t, wave, L) X(int32_t, unit, L) X(int32_t, supers, L)                                                \
    X(int32_t, att_target, N) X(int32_t, att_seq, N) X(float, windup_left, N) X(float, cooldown_left, N)         \
    X(uint8_t, m_alive, M) X(int32_t, m_src, M) X(int32_t, m_dst, M) X(int32_t, m_dst_seq, M)                    \
    X(float, m_x, M) X(float, m_y, M) X(float, m_speed, M) X(float, m_raw, M) X(int32_t, m_dtype, M)             \
    X(int32_t, m_flags, M) X(int32_t, m_cast, M) X(uint8_t, m_crit, M)                                           \
    X(float, stun_until, N) X(float, root_until, N) X(float, silence_until, N) X(float, knockup_until, N)        \
    X(float, slow, N) X(float, slow_until, N) X(float, champion_cc_until, N)                                     \
    X(int32_t, level, C)                                                                                         \
    X(int32_t, ai_seq, N) X(int32_t, ai_target, N) X(int32_t, ai_target_seq, N) X(int32_t, ai_priority, N)      \
    X(float, ai_sweep, N) X(float, ai_since, N) X(float, ignore_until, RK) X(float, last_attack, KK)            \
    X(int32_t, ai_lane, N) X(int32_t, ai_waypoint, N) X(uint8_t, ai_first_wave, N) X(uint8_t, ai_engaged, N)     \
    X(uint8_t, ai_champion_aggro, N) X(int32_t, ai_warm_stacks, N) X(float, ai_warm_until, N)                    \
    X(float, tw_hp, N) X(float, tw_max_hp, N) X(int32_t, tw_tier, N) X(float, tw_respawn_at, N)                  \
    X(int32_t, tw_plates, N) X(float, tw_bulwark, N4) X(float, tw_backdoor, N) X(float, tw_growth_since, N)      \
    X(uint8_t, tw_growth_active, N) X(int32_t, tw_warm_stacks, N) X(float, tw_warm_until, N)                     \
    X(uint8_t, tw_is_structure, N) X(int32_t, tw_team, N) X(int32_t, tw_lane, N) X(int32_t, tw_prereq, N)        \
    X(uint8_t, tw_targetable, N) X(uint8_t, tw_first_turret, S)                                                  \
    X(uint8_t, damage_matrix, NN) X(uint8_t, visible, TN)                                                        \
    X(float, reveal_x, C) X(float, reveal_y, C) X(float, reveal_until, C)                                        \
    X(int32_t, route_anchor, N)                                                                                  \
    X(float, memo_route, N4) X(int32_t, memo_anchor, N)                                                          \
    X(int32_t, ev_n, S) X(int32_t, ev_src, P) X(int32_t, ev_dst, P) X(int32_t, rec_n, S) X(int32_t, rec, KK)

struct Env {
#define LANESIM_PTR(type, name, size) type* name;
    LANESIM_ENV_FIELDS(LANESIM_PTR)
#undef LANESIM_PTR
};

// Element count of a field of the given size class.
inline size_t field_count(const World& w, const char* size) {
    size_t n = w.n, k = w.cols.size(), r = w.rows_m.size();
    std::string s(size);
    if (s == "S") return 1;
    if (s == "N") return n;
    if (s == "M") return (size_t)w.missiles;
    if (s == "L") return 6;
    if (s == "C") return N_CHAMPIONS;
    if (s == "RK") return r * k;
    if (s == "KK") return k * k;
    if (s == "N4") return n * 4;
    if (s == "NN") return n * n;
    if (s == "TN") return 2 * n;
    if (s == "P") return (size_t)w.packet_capacity;
    return 0;
}

struct TickStats {
    int32_t packet_overflow = 0, missile_overflow = 0, ray_overflow = 0, packets = 0, rays = 0;
};

TickStats step(const World& w, Env& e);
// Per-phase nanoseconds of the calling thread: spawn, turret, select, move prep, route, collide, attack, damage,
// death, timers, fog.
void profile(double* out, bool reset);
extern thread_local float* debug_route;

}  // namespace lanesim
