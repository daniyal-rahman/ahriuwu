// C API for the Python binding (native/python/lanesim.py): build a World from named values and arrays, step one
// env whose fields live in caller memory, or run a batch of envs the library owns across threads.
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "world.hpp"

using namespace lanesim;

namespace {

struct FieldInfo { const char* type; const char* name; size_t elem; };

const std::vector<FieldInfo>& fields() {
    static const std::vector<FieldInfo> f = {
#define X(type, name) {#type, #name, sizeof(type)},
#include "gen/env_fields.inc"
#undef X
    };
    return f;
}

const std::vector<FieldInfo>& order_fields() {
    static const std::vector<FieldInfo> f = {
#define X(type, name) {#type, #name, sizeof(type)},
#include "gen/orders_fields.inc"
#undef X
    };
    return f;
}

template <class T>
T from_ptrs(void* const* ptrs, size_t count) {
    T out;
    void** dst = reinterpret_cast<void**>(&out);
    for (size_t k = 0; k < count; ++k) dst[k] = ptrs[k];
    return out;
}

// Order fields are (C,) each and at most 4 bytes wide.
struct OrderStore {
    uint8_t bytes[32][N_CHAMPIONS * 4];
    Orders view() {
        Orders o;
        void** dst = reinterpret_cast<void**>(&o);
        for (size_t k = 0; k < order_fields().size(); ++k) dst[k] = bytes[k];
        return o;
    }
};

struct Batch {
    const World* world;
    int n_envs;
    int order_mode;
    size_t env_bytes;
    std::vector<uint8_t> storage;
    std::vector<Env> envs;
};

size_t align8(size_t v) { return (v + 7) & ~size_t(7); }

}  // namespace

extern "C" {

// "type name;" per Env field, in struct order.
const char* ls_env_fields() {
    static std::string s;
    if (s.empty())
        for (const auto& f : fields()) s += std::string(f.type) + " " + f.name + ";";
    return s.c_str();
}

// "type name;" per Orders field.
const char* ls_order_fields() {
    static std::string s;
    if (s.empty())
        for (const auto& f : order_fields()) s += std::string(f.type) + " " + f.name + ";";
    return s.c_str();
}

// Element count of every Env field (ls_env_fields order), from the JAX state shapes.
void ls_world_env_counts(void* wp, const long* counts, long n) {
    auto& v = static_cast<World*>(wp)->env_counts;
    v.assign(counts, counts + n);
}

void* ls_world_new() { return new World(); }
void ls_world_free(void* w) { delete static_cast<World*>(w); }

int ls_world_int(void* wp, const char* name, long v) {
    World& w = *static_cast<World*>(wp);
    std::string k(name);
    if (k == "n") w.n = (int)v;
    else if (k == "minion0") w.minion0 = (int)v;
    else if (k == "monster0") w.monster0 = (int)v;
    else if (k == "epic0") w.epic0 = (int)v;
    else if (k == "ward0") w.ward0 = (int)v;
    else if (k == "struct0") w.struct0 = (int)v;
    else if (k == "missiles") w.missiles = (int)v;
    else if (k == "packet_capacity") w.packet_capacity = (int)v;
    else if (k == "ray_capacity") w.ray_capacity = (int)v;
    else if (k == "fog") w.fog = v != 0;
    else if (k == "path_cap") w.path_cap = (int)v;
    else if (k == "terrain_width") w.terrain[0].width = w.terrain[1].width = (int)v;
    else if (k == "terrain_height") w.terrain[0].height = w.terrain[1].height = (int)v;
    else if (k == "route_points") w.routes.n_points = (int)v;
    else if (k == "route_cells_h") w.routes.cells_h = (int)v;
    else if (k == "route_cells_w") w.routes.cells_w = (int)v;
    else if (k == "vision_height") w.vision.height = (int)v;
    else if (k == "vision_width") w.vision.width = (int)v;
    else return -1;
    return 0;
}

int ls_world_float(void* wp, const char* name, double v) {
    World& w = *static_cast<World*>(wp);
    std::string k(name);
    float f = (float)v;
    if (k == "dt") w.dt = f;
    else if (k == "avoid_horizon_ticks") w.avoid_horizon_ticks = f;
    else if (k == "terrain_cell_size") w.terrain[0].cell_size = w.terrain[1].cell_size = f;
    else if (k == "terrain_min_x") w.terrain[0].min_x = w.terrain[1].min_x = f;
    else if (k == "terrain_min_z") w.terrain[0].min_z = w.terrain[1].min_z = f;
    else if (k == "terrain_max_x") w.terrain[0].max_x = w.terrain[1].max_x = f;
    else if (k == "terrain_max_z") w.terrain[0].max_z = w.terrain[1].max_z = f;
    else if (k == "route_spacing") w.routes.spacing = f;
    else if (k == "route_min_x") w.routes.min_x = f;
    else if (k == "route_min_z") w.routes.min_z = f;
    else if (k == "route_radius") w.routes.radius = f;
    else if (k == "vision_cell_size") w.vision.cell_size = f;
    else if (k == "vision_min_x") w.vision.min_x = f;
    else if (k == "vision_min_y") w.vision.min_y = f;
    else return -1;
    return 0;
}

// Arrays: small ones are copied; the large static tables (gaps, route next-hop, vision flags) are borrowed and must
// outlive the world.
int ls_world_array(void* wp, const char* name, const void* p, long count) {
    World& w = *static_cast<World*>(wp);
    std::string k(name);
    auto ints = [&](std::vector<int32_t>& v) { v.assign((const int32_t*)p, (const int32_t*)p + count); };
    auto floats = [&](std::vector<float>& v) { v.assign((const float*)p, (const float*)p + count); };
    const float* fp = (const float*)p;
    if (k == "lanes") { w.n_lanes = (int)count; for (long i = 0; i < count; ++i) w.lanes[i] = ((const int32_t*)p)[i]; }
    else if (k == "unit_lane") ints(w.unit_lane);
    else if (k == "rows_m") ints(w.rows_m);
    else if (k == "rows_s") ints(w.rows_s);
    else if (k == "cols") ints(w.cols);
    else if (k == "lane_paths") floats(w.lane_paths);
    else if (k == "lane_len") for (long i = 0; i < count; ++i) w.lane_len[i] = ((const int32_t*)p)[i];
    else if (k == "barracks") std::memcpy(w.barracks, p, sizeof(w.barracks));
    else if (k == "lane_mid") std::memcpy(w.lane_mid, p, sizeof(w.lane_mid));
    else if (k == "avoid_cos") std::memcpy(w.avoid_cos, fp, sizeof(w.avoid_cos));
    else if (k == "avoid_sin") std::memcpy(w.avoid_sin, fp, sizeof(w.avoid_sin));
    else if (k == "sep_fx") floats(w.sep_fx);
    else if (k == "sep_fy") floats(w.sep_fy);
    else if (k == "eject_dx") floats(w.eject_dx);
    else if (k == "eject_dy") floats(w.eject_dy);
    else if (k == "eject_r") floats(w.eject_r);
    else if (k == "gaps0") w.terrain[0].gaps = (const int8_t*)p;
    else if (k == "gaps1") w.terrain[1].gaps = (const int8_t*)p;
    else if (k == "route_points_xy") w.routes.points = fp;
    else if (k == "route_cells") w.routes.cells = (const int32_t*)p;
    else if (k == "route_next") w.routes.next = (const int16_t*)p;
    else if (k == "vision_flags") w.vision.flags = (const int32_t*)p;
    else return -1;
    return 0;
}

void ls_world_finish(void* wp) {
    World& w = *static_cast<World*>(wp);
    for (auto& t : w.terrain) t.build_cheb();
    w.col_of.assign(w.n, -1);
    for (size_t c = 0; c < w.cols.size(); ++c) w.col_of[w.cols[c]] = (int)c;
    w.row_m_of.assign(w.n, -1);
    for (size_t r = 0; r < w.rows_m.size(); ++r) w.row_m_of[w.rows_m[r]] = (int)r;
}

// One tick of an env whose fields are ``ptrs`` (ls_env_fields order). ``stats``: packet, missile, ray overflow,
// packets, rays.
void ls_step(void* wp, void* const* ptrs, void* const* order_ptrs, int32_t* stats) {
    Env e = from_ptrs<Env>(ptrs, fields().size());
    Orders o = from_ptrs<Orders>(order_ptrs, order_fields().size());
    TickStats st = step(*static_cast<World*>(wp), e, o);
    if (stats) {
        stats[0] = st.packet_overflow, stats[1] = st.missile_overflow, stats[2] = st.ray_overflow;
        stats[3] = st.packets, stats[4] = st.rays;
    }
}

// ``n_envs`` copies of the env at ``ptrs``, owned by the library.
// ``order_mode``: champion orders each tick (scripted_orders: 0 none, 1 the JAX bench's walk-and-attack).
void* ls_batch_new(void* wp, int n_envs, void* const* ptrs, int order_mode) {
    const World& w = *static_cast<World*>(wp);
    auto* b = new Batch{&w, n_envs, order_mode, 0, {}, {}};
    std::vector<size_t> offset;
    for (const auto& f : fields()) {
        offset.push_back(b->env_bytes);
        b->env_bytes += align8(w.env_counts[&f - fields().data()] * f.elem);
    }
    b->env_bytes = (b->env_bytes + 63) & ~size_t(63);
    b->storage.assign(b->env_bytes * n_envs, 0);
    b->envs.resize(n_envs);
    for (int k = 0; k < n_envs; ++k) {
        uint8_t* base = b->storage.data() + b->env_bytes * k;
        void** dst = reinterpret_cast<void**>(&b->envs[k]);
        for (size_t f = 0; f < fields().size(); ++f) {
            dst[f] = base + offset[f];
            std::memcpy(dst[f], ptrs[f], w.env_counts[f] * fields()[f].elem);
        }
    }
    return b;
}

void ls_batch_free(void* b) { delete static_cast<Batch*>(b); }

long ls_batch_env_bytes(void* bp) { return (long)static_cast<Batch*>(bp)->env_bytes; }

// ``ticks`` ticks of every env on ``threads`` threads (0: OpenMP default); ``stats``: max packet, missile and ray
// overflow over the run.
void ls_batch_run(void* bp, int ticks, int threads, int32_t* stats) {
    Batch& b = *static_cast<Batch*>(bp);
    int po = 0, mo = 0, ro = 0;
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(dynamic, 4) reduction(max : po, mo, ro)
#endif
    for (int k = 0; k < b.n_envs; ++k) {
        OrderStore store;
        Orders o = store.view();
        for (int t = 0; t < ticks; ++t) {
            scripted_orders(*b.world, b.envs[k], o, b.order_mode);
            TickStats st = step(*b.world, b.envs[k], o);
            po = std::max(po, st.packet_overflow), mo = std::max(mo, st.missile_overflow);
            ro = std::max(ro, st.ray_overflow);
        }
    }
    if (stats) stats[0] = po, stats[1] = mo, stats[2] = ro;
}

// Copy env ``k``'s fields out to ``ptrs``.
void ls_batch_get(void* bp, int k, void* const* ptrs) {
    Batch& b = *static_cast<Batch*>(bp);
    const World& w = *b.world;
    void* const* src = reinterpret_cast<void* const*>(&b.envs[k]);
    for (size_t f = 0; f < fields().size(); ++f)
        std::memcpy(ptrs[f], src[f], w.env_counts[f] * fields()[f].elem);
}

void ls_profile(double* out, int reset) { profile(out, reset != 0); }

// Record route inputs/outputs of the movers (ward0 x 24 floats) during the calling thread's next steps.
void ls_debug_route(float* out) { debug_route = out; }

}  // extern "C"
