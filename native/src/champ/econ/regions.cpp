// Map11 region queries (lanerl_jax/modern/map/regions.py) used by the DEATH phase (role-quest lane, Homeguard
// endpoint and jungle flags) and views (river). Region masks and lane paths: native/python/consts/econ_regions.py.
#include <algorithm>
#include <cmath>

#include "../marshal.hpp"
#include "econ.hpp"

namespace lanesim::econ {

using namespace champ;

namespace {

enum MainRegion { SPAWN, BASE, TOP_LANE, MID_LANE, BOT_LANE, TOP_JUNGLE, BOT_JUNGLE, TOP_RIVER, BOT_RIVER,
                  TOP_BASE_PERIMETER, BOT_BASE_PERIMETER, TOP_ALCOVE, BOT_ALCOVE };
constexpr int OUTSIDE = -1, LANE_BOT = 0, LANE_MID = 1, LANE_TOP = 2;
constexpr float HOMEGUARD_TURRET_MARGIN = 500.f, HOMEGUARD_MINION_LEAD = 2000.f, HOMEGUARD_LATE_S = 840.f;

struct Regions {
    std::vector<float> main, side, geom, paths, shape;
    Regions() {
        auto t = [](const char* n) { return data::table(std::string("econ.regions.") + n); };
        main = t("main"), side = t("side"), geom = t("geom"), paths = t("paths"), shape = t("paths_shape");
    }
};
const Regions& R() {
    static const Regions r;
    return r;
}

// regions._cell: (iz, ix, inside)
bool cell(float x, float y, int& iz, int& ix) {
    const Regions& r = R();
    int h = (int)r.geom[0], w = (int)r.geom[1];
    float cs = r.geom[2], min_x = r.geom[3], min_z = r.geom[4];
    ix = (int)std::floor((x - min_x) / cs), iz = (int)std::floor((y - min_z) / cs);
    bool inside = ix >= 0 && ix < w && iz >= 0 && iz < h && std::isfinite(x) && std::isfinite(y);
    iz = clampi(iz, 0, h - 1), ix = clampi(ix, 0, w - 1);
    return inside;
}

int region_lane(int code) {
    switch (code) {
        case BOT_LANE: case BOT_ALCOVE: return LANE_BOT;
        case MID_LANE: return LANE_MID;
        case TOP_LANE: case TOP_ALCOVE: return LANE_TOP;
        default: return -1;
    }
}

}  // namespace

// regions.region_of
int region_of(float x, float y) {
    int iz, ix;
    bool inside = cell(x, y, iz, ix);
    return inside ? (int)R().main[(size_t)iz * (int)R().geom[1] + ix] : OUTSIDE;
}

// regions.lane_of
int lane_of(float x, float y) {
    int code = region_of(x, y);
    return code >= 0 ? region_lane(clampi(code, 0, 12)) : -1;
}

bool in_quest_lane(float x, float y, int quest_lane) { return lane_of(x, y) == quest_lane; }
bool in_jungle(float x, float y) {
    int code = region_of(x, y);
    return code == TOP_JUNGLE || code == BOT_JUNGLE;
}
bool in_river(float x, float y) {
    int code = region_of(x, y);
    return code == TOP_RIVER || code == BOT_RIVER;
}

// regions.lane_progress: arc length of the projection on ``team``'s minion path of ``lane``
float lane_progress(float x, float y, int team, int lane) {
    const Regions& r = R();
    int L = (int)r.shape[2];
    team = clampi(team, 0, 1), lane = clampi(lane, 0, 2);
    const float* p = &r.paths[((size_t)team * (int)r.shape[1] + lane) * L * 2];
    float cum = 0.f, best = INF, best_cum = 0.f, best_t = 0.f, best_sl = 0.f;
    for (int k = 0; k + 1 < L; ++k) {
        float ax = p[2 * k], ay = p[2 * k + 1];
        float sx = p[2 * k + 2] - ax, sy = p[2 * k + 3] - ay;
        float sl = std::sqrt(sx * sx + sy * sy);
        float t = std::min(std::max(((x - ax) * sx + (y - ay) * sy) / std::max(sl * sl, 1e-6f), 0.f), 1.f);
        float d = sq(ax + t * sx - x) + sq(ay + t * sy - y);
        d = sl > 1e-6f ? d : INF;
        if (d < best) best = d, best_cum = cum, best_t = t, best_sl = sl;   // argmin: first minimum
        if (k == 0 && !(d < best)) best_cum = cum, best_t = t, best_sl = sl;
        cum = cum + sl;
    }
    return best_cum + best_t * best_sl;
}

// regions.homeguard_flags (with homeguard_endpoint)
std::tuple<Arr<uint8_t>, Arr<uint8_t>> homeguard_flags(const Arr<float>& x, const Arr<float>& y,
                                                       const Arr<int32_t>& team, float now, const WorldUnits& units,
                                                       const Arr<int32_t>& structure_lane,
                                                       const Arr<int32_t>& minion_lane) {
    size_t c = x.size(), n = units.kind.size();
    Arr<uint8_t> reached(c), jungle(c);
    for (size_t i = 0; i < c; ++i) {
        int lane = std::max(lane_of(x[i], y[i]), 0);
        float tur = -INF, inhib = -INF, front = -INF;
        bool turret_down = false;
        for (size_t j = 0; j < n; ++j) {
            bool own = units.team[j] == team[i];
            bool slane = structure_lane[j] == lane;
            bool lane_turret = own && slane && units.kind[j] == KIND_TURRET && units.sub[j] <= 2;
            bool alive = units.alive[j];
            bool inh = own && slane && units.kind[j] == KIND_INHIBITOR;
            bool mins = own && alive && units.kind[j] == KIND_MINION && minion_lane[j] == lane;
            turret_down = turret_down || (lane_turret && !alive);
            if (!((lane_turret && alive) || inh || mins)) continue;
            float prog = lane_progress(units.x[j], units.y[j], team[i], lane);
            if (lane_turret && alive) tur = std::max(tur, prog);
            if (inh) inhib = std::max(inhib, prog);
            if (mins) front = std::max(front, prog);
        }
        float end = std::max(tur - HOMEGUARD_TURRET_MARGIN, inhib);
        bool late = now >= HOMEGUARD_LATE_S || turret_down;
        end = late ? std::max(end, front - HOMEGUARD_MINION_LEAD) : end;
        float prog = lane_progress(x[i], y[i], team[i], lane);
        reached[i] = lane_of(x[i], y[i]) >= 0 && prog >= end;
        jungle[i] = in_jungle(x[i], y[i]);
    }
    return {reached, jungle};
}

namespace {
struct I1 {
    Arr<int32_t> v;
    template <class F> void visit(F&& f) { f(v); }
};
struct B1 {
    Arr<uint8_t> v;
    template <class F> void visit(F&& f) { f(v); }
};
struct F1 {
    Arr<float> v;
    template <class F> void visit(F&& f) { f(v); }
};
std::tuple<I1, I1, B1, B1, B1> regions_test(Arr<float> x, Arr<float> y) {
    size_t k = x.size();
    I1 a{Arr<int32_t>(k)}, b{Arr<int32_t>(k)};
    B1 q{Arr<uint8_t>(k)}, j{Arr<uint8_t>(k)}, r{Arr<uint8_t>(k)};
    for (size_t i = 0; i < k; ++i) {
        a.v[i] = region_of(x[i], y[i]), b.v[i] = lane_of(x[i], y[i]), q.v[i] = in_quest_lane(x[i], y[i], LANE_TOP);
        j.v[i] = in_jungle(x[i], y[i]), r.v[i] = in_river(x[i], y[i]);
    }
    return {a, b, q, j, r};
}
F1 lane_progress_test(Arr<float> x, Arr<float> y, Arr<int32_t> team, Arr<int32_t> lane) {
    F1 o{Arr<float>(x.size())};
    for (size_t i = 0; i < x.size(); ++i) o.v[i] = lane_progress(x[i], y[i], team[i], lane[i]);
    return o;
}
std::tuple<Arr<uint8_t>, Arr<uint8_t>> homeguard_flags_test(Arr<float> x, Arr<float> y, Arr<int32_t> team, float now,
                                                            WorldUnits units, Arr<int32_t> structure_lane,
                                                            Arr<int32_t> minion_lane) {
    return homeguard_flags(x, y, team, now, units, structure_lane, minion_lane);
}
}  // namespace
LANESIM_TEST(regions_queries, "regions.queries", regions_test);
LANESIM_TEST(regions_lane_progress, "regions.lane_progress", lane_progress_test);
LANESIM_TEST(regions_homeguard_flags, "regions.homeguard_flags", homeguard_flags_test);

}  // namespace lanesim::econ
