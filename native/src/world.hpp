// The static world and one env's state as plain arrays, laid out like the JAX ModernState slice it mirrors
// (lanerl_jax/modern/world/state.py): (N,) unit columns, the attack machine, missiles, lane AI and structures.
// ``Env`` only points at storage, so the same tick runs on numpy arrays (differential tests) or on a batch the
// library owns. ``ENV_FIELDS`` is the one list of fields: it generates the struct, its byte sizes and the
// description the Python binding reads.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
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
    float lane_mid[2] = {0.f, 0.f};                // cfg.lane_path midpoint (benchmark orders)
    // Constants evaluated by XLA (float32 cos/sin), so both sides use the same bits.
    float avoid_cos[6] = {}, avoid_sin[2][6] = {};   // [sign of the turn][angle]
    std::vector<float> sep_fx, sep_fy;             // (ward0, ward0) coincident-pair fallback directions
    std::vector<float> eject_dx, eject_dy, eject_r; // dynamic_terrain.eject ring offsets (7 rings x 16 directions)
    std::vector<size_t> env_counts;                // element count of every Env field
    // Per-world champion data from the binding (champion bases, rune pages, allow-lists, skill orders, shards...):
    // name -> flat float32 table (ints stored as floats are exact below 2^24).
    std::unordered_map<std::string, std::vector<float>> tables;
    const std::vector<float>& tab(const std::string& k) const { return tables.at(k); }
    float avoid_horizon_ticks = 9.f;
    // The world's initial state (ModernState at 0:00, every Env field's bytes; empty if the binding gave none):
    // the reference of the champion layer's dormant item modules.
    std::vector<std::vector<uint8_t>> initial;

    const float* path(int team, int lane) const {
        return lane_paths.data() + ((size_t)(team * 3 + lane) * path_cap) * 2;
    }
};

// One env's state: a pointer per ModernState leaf (generated, pytree order) plus native-only caches
// (``memo_*`` route-follow memo, ``ev_*`` last tick's damage events as a list, ``rec*`` recent last_attack
// entries). Element counts are per world (``World::env_counts``, from the Python binding).
struct Env {
#define X(type, name) type* name;
#include "gen/env_fields.inc"
#undef X
};

// One tick's champion orders (ModernOrders leaves, (C,) each).
struct Orders {
#define X(type, name) type* name;
#include "gen/orders_fields.inc"
#undef X
};

struct TickStats {
    int32_t packet_overflow = 0, missile_overflow = 0, ray_overflow = 0, packets = 0, rays = 0;
};

TickStats step(const World& w, Env& e, const Orders& o);
// Champion orders for benchmarks: mode 0 none, 1 ops/modern/bench.scripted_orders (nearest enemy within 700, else
// walk to the lane midpoint).
void scripted_orders(const World& w, const Env& e, Orders& o, int mode);
// Per-phase nanoseconds of the calling thread: spawn, turret, select, move prep, route, collide, attack, damage,
// death, timers, fog.
void profile(double* out, bool reset);
extern thread_local float* debug_route;

}  // namespace lanesim
