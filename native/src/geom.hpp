// Static map queries: navgrid walkability, swept segments, route-graph steering and sight rays.
// Ports of lanerl_jax/modern/map/{terrain,pathing}.py and rays.py (clear_ray_reference), float32 throughout.
#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>

namespace lanesim {

constexpr float INF = std::numeric_limits<float>::infinity();
constexpr int GAP_PAD = 8;

// One team's walkability: per column x and padded row z, the distance in cells to the nearest blocked cell at or
// left/right of x (terrain.row_gaps), shape (W, H + 2*GAP_PAD, 2) int8.
struct Terrain {
    const int8_t* gaps = nullptr;
    int width = 0, height = 0;
    float cell_size = 0, min_x = 0, min_z = 0, max_x = 0, max_z = 0;
    // Chebyshev distance in cells from each cell to the nearest blocked cell (off-grid counts as blocked), capped
    // at CHEB_CAP: a disk of radius r < cheb - 1 cells around any point of the cell meets no blocked cell, so the
    // exact window test can be skipped (same answer).
    static constexpr int CHEB_CAP = 4;
    std::vector<uint8_t> cheb;
    // Lower bound, in cells, on the distance from any point of each cell to any blocked cell (off-grid blocked),
    // capped at CLEAR_CAP: a segment's samples within (clear - r) of a sample are clear too (1-Lipschitz).
    static constexpr int CLEAR_CAP = 12;
    std::vector<float> clear;

    const int8_t* gap(int x, int zp) const { return gaps + ((size_t)x * (height + 2 * GAP_PAD) + zp) * 2; }

    void build_cheb() {
        cheb.assign((size_t)width * height, 0);
        for (int x = 0; x < width; ++x)
            for (int z = 0; z < height; ++z) {
                int g = CHEB_CAP;
                for (int dz = -CHEB_CAP; dz <= CHEB_CAP; ++dz) {
                    int zp = z + dz + GAP_PAD;             // pad rows read gap 0: blocked rows off the grid
                    int row = (zp < 0 || zp >= height + 2 * GAP_PAD) ? 0 : std::min(gap(x, zp)[0], gap(x, zp)[1]);
                    g = std::min(g, std::max(std::abs(dz), row));
                }
                cheb[(size_t)z * width + x] = (uint8_t)g;
            }
        clear.assign((size_t)width * height, (float)CLEAR_CAP);
        auto blocked = [&](int x, int z) {
            if (x < 0 || z < 0 || x >= width || z >= height) return true;
            return gap(x, z + GAP_PAD)[0] == 0;
        };
        for (int x = 0; x < width; ++x)
            for (int z = 0; z < height; ++z) {
                float best = (float)CLEAR_CAP;
                for (int dz = -CLEAR_CAP - 1; dz <= CLEAR_CAP + 1; ++dz)
                    for (int dx = -CLEAR_CAP - 1; dx <= CLEAR_CAP + 1; ++dx) {
                        if (!blocked(x + dx, z + dz)) continue;
                        float gx = (float)std::max(std::abs(dx) - 1, 0), gz = (float)std::max(std::abs(dz) - 1, 0);
                        best = std::min(best, std::sqrt(gx * gx + gz * gz));
                    }
                clear[(size_t)z * width + x] = best;
            }
    }

    // terrain.is_walkable with a static max_radius_cells (the window the pinned callers use is 1 cell).
    bool walkable(float x, float z, float radius, int max_radius_cells = 1) const {
        float nx = (x - min_x) / cell_size, nz = (z - min_z) / cell_size, r = radius / cell_size;
        if (!(std::isfinite(x) && std::isfinite(z) && std::isfinite(r) && r >= 0.f && r <= (float)max_radius_cells))
            return false;
        int bx = (int)std::floor(nx), bz = (int)std::floor(nz);
        int m = max_radius_cells + 1, k = 2 * max_radius_cells + 3;
        int sx = bx < 0 ? 0 : (bx > width - 1 ? width - 1 : bx);
        int sz = (bz < 0 ? 0 : (bz > height - 1 ? height - 1 : bz)) + GAP_PAD - m;
        if (radius == 0.f) {
            bool inb = x >= min_x && x < max_x && z >= min_z && z < max_z;
            return inb && gap(sx, sz + m)[0] > 0;
        }
        if (!(x - radius > min_x && x + radius < max_x && z - radius > min_z && z + radius < max_z)) return false;
        if (bx >= 0 && bz >= 0 && bx < width && bz < height && (float)(cheb[(size_t)bz * width + bx] - 1) > r)
            return true;
        float rr = r * r;
        for (int j = 0; j < k; ++j) {
            const int8_t* g = gap(sx, sz + j);
            int iz = bz + (j - m);
            float dz = std::fabs(nz - ((float)iz + .5f)) - .5f;
            dz = dz > 0.f ? dz : 0.f;
            for (int side = 0; side < 2; ++side) {
                int ix = bx + (side ? (int)g[1] : -(int)g[0]);
                float dx = std::fabs(nx - ((float)ix + .5f)) - .5f;
                dx = dx > 0.f ? dx : 0.f;
                if (dx * dx + dz * dz <= rr) return false;
            }
        }
        return true;
    }
};

// pathing.segment_clear: half-step-inflated disks at ``samples`` evenly spaced points (exact i/(samples-1)).
// Samples in open ground are skipped by the clearance bound: from a sample whose cell is ``clear`` cells from any
// blocked cell, every sample within ``clear - r`` (minus a float margin) is clear as well; the others take the exact
// test. Map bounds are checked per sample unless both ends are inside with a margin (then every sample is).
inline bool segment_clear(const Terrain& t, float x0, float y0, float x1, float y1, float radius, int samples,
                          float max_length) {
    float ex = x1 - x0, ey = y1 - y0;
    float length = std::sqrt(ex * ex + ey * ey);
    if (!(length <= max_length)) return false;
    float inflated = radius + length / (float)(samples - 1) / 2.f;
    float rc = inflated / t.cell_size;
    if (!(std::isfinite(x0) && std::isfinite(y0) && std::isfinite(x1) && std::isfinite(y1) && std::isfinite(rc)
          && rc >= 0.f && rc <= 1.f && inflated > 0.f))
        return [&] {
            for (int i = 0; i < samples; ++i) {
                float f = (float)i / (float)(samples - 1);
                if (!t.walkable(x0 + f * ex, y0 + f * ey, inflated)) return false;
            }
            return true;
        }();
    const float margin = .01f;
    auto inside = [&](float x, float z) {
        return x - inflated > t.min_x + margin && x + inflated < t.max_x - margin && z - inflated > t.min_z + margin
               && z + inflated < t.max_z - margin;
    };
    bool ends_inside = inside(x0, y0) && inside(x1, y1);
    float step_cells = length / (float)(samples - 1) / t.cell_size;
    int i = 0;
    while (i < samples) {
        float f = (float)i / (float)(samples - 1);
        float x = x0 + f * ex, z = y0 + f * ey;
        int bx = (int)std::floor((x - t.min_x) / t.cell_size), bz = (int)std::floor((z - t.min_z) / t.cell_size);
        float room = (bx >= 0 && bz >= 0 && bx < t.width && bz < t.height)
                         ? t.clear[(size_t)bz * t.width + bx] - rc - 1e-3f : -1.f;
        if (room > 0.f && ends_inside) {
            int skip = step_cells > 0.f ? (int)std::floor(room / step_cells) : samples;
            i += 1 + std::max(skip, 0);
            continue;
        }
        if (!t.walkable(x, z, inflated)) return false;
        ++i;
    }
    return true;
}

// FlowRoutes: 100-unit node grid, goal-rooted next-hop table next[goal][source] (int16).
struct Routes {
    const float* points = nullptr;      // (P, 2)
    const int32_t* cells = nullptr;     // (CH, CW) node id or -1
    const int16_t* next = nullptr;      // (P, P)
    int n_points = 0, cells_h = 0, cells_w = 0;
    float spacing = 0, min_x = 0, min_z = 0, radius = 0;

    float px(int i) const { return points[2 * i]; }
    float py(int i) const { return points[2 * i + 1]; }
    int hop(int goal, int source) const { return next[(size_t)goal * n_points + source]; }

    // pathing.nearest_node: the closest node of the 5x5 cell block (with a clear segment if ``check``); ties and
    // the empty case keep argmin's first index.
    int nearest(const Terrain& t, float x, float z, bool check, bool* ok) const {
        int cx = (int)std::floor((x - min_x) / spacing), cz = (int)std::floor((z - min_z) / spacing);
        float best = INF;
        int choice = -1, first_id = 0;
        bool any = false;
        for (int a = 0; a < 25; ++a) {
            int xx = cx + (a % 5) - 2, zz = cz + (a / 5) - 2;
            int id = cells[(size_t)(zz < 0 ? 0 : (zz >= cells_h ? cells_h - 1 : zz)) * cells_w
                           + (xx < 0 ? 0 : (xx >= cells_w ? cells_w - 1 : xx))];
            if (a == 0) first_id = id;
            bool valid = xx >= 0 && zz >= 0 && xx < cells_w && zz < cells_h && id >= 0;
            if (!valid) continue;
            float dx = px(id) - x, dz = py(id) - z;
            float d = dx * dx + dz * dz;
            if (!(d < best)) continue;  // keep the first minimum; check_connection only where it can win
            if (check && !segment_clear(t, x, z, px(id), py(id), radius, 33, 350.f)) continue;
            best = d;
            choice = id;
            any = true;
        }
        *ok = any;
        return any ? choice : first_id;
    }

    struct Steer { float x, y; bool valid; int anchor; };

    // pathing._steer
    Steer steer(const Terrain& t, float x, float y, float gx, float gy, float r, int source, bool source_ok) const {
        bool dest_ok;
        int dest = nearest(t, gx, gy, false, &dest_ok);
        int s = source > 0 ? source : 0;
        int nxt = hop(dest > 0 ? dest : 0, s);
        int nn = nxt > 0 ? nxt : 0;
        bool next_clear = nxt >= 0 && segment_clear(t, x, y, px(nn), py(nn), r, 65, 600.f);
        Steer o;
        o.x = next_clear ? px(nn) : px(s);
        o.y = next_clear ? py(nn) : py(s);
        o.valid = source_ok && dest_ok && (nxt >= 0 || source == dest) && r <= radius;
        o.anchor = next_clear ? nxt : source;
        return o;
    }

    struct Follow { float x, y; bool ok; int anchor; bool replan; };

    // pathing.route_follow
    Follow follow(const Terrain& t, float x, float y, float gx, float gy, float r, int anchor) const {
        bool direct = segment_clear(t, x, y, gx, gy, r, 65, 600.f) && r <= radius;
        int a = anchor > 0 ? anchor : 0;
        bool seen = anchor >= 0 && segment_clear(t, x, y, px(a), py(a), r, 65, 600.f);
        Steer s = steer(t, x, y, gx, gy, r, a, seen);
        float dx = x - px(a), dy = y - py(a);
        bool stuck = seen && s.anchor == a && (dx * dx + dy * dy) < 1.f;
        Follow f;
        f.x = direct ? gx : (s.valid ? s.x : x);
        f.y = direct ? gy : (s.valid ? s.y : y);
        f.ok = direct || s.valid;
        f.anchor = seen ? s.anchor : -1;
        f.replan = !direct && (!seen || stuck);
        return f;
    }

    // pathing.route_replan
    Follow replan(const Terrain& t, float x, float y, float gx, float gy, float r) const {
        bool source_ok;
        int source = nearest(t, x, y, true, &source_ok);
        Steer s = steer(t, x, y, gx, gy, r, source, source_ok);
        return {s.valid ? s.x : x, s.valid ? s.y : y, s.valid, source_ok ? s.anchor : -1, false};
    }
};

// rays.clear_ray_reference: supercover walk over navgrid flags, both side cells at corners, at most 64 steps.
struct VisionGrid {
    const int32_t* flags = nullptr;     // (H, W)
    int height = 0, width = 0;
    float cell_size = 0, min_x = 0, min_y = 0;

    bool at(int x, int y, int* f) const {
        bool valid = x >= 0 && y >= 0 && x < width && y < height;
        int cx = x < 0 ? 0 : (x > width - 1 ? width - 1 : x), cy = y < 0 ? 0 : (y > height - 1 ? height - 1 : y);
        *f = flags[(size_t)cy * width + cx];
        return valid;
    }

    bool clear(float ax, float ay, float bx, float by) const {
        float x0 = (ax - min_x) / cell_size, y0 = (ay - min_y) / cell_size;
        float x1 = (bx - min_x) / cell_size, y1 = (by - min_y) / cell_size;
        int ix = (int)std::floor(x0), iy = (int)std::floor(y0), ex = (int)std::floor(x1), ey = (int)std::floor(y1);
        float dx = std::fabs(x1 - x0), dy = std::fabs(y1 - y0);
        int sx = (x1 > x0) - (x1 < x0), sy = (y1 > y0) - (y1 < y0);
        float err = (sx > 0 ? (float)(ix + 1) - x0 : x0 - (float)ix) * dy
                  - (sy > 0 ? (float)(iy + 1) - y0 : y0 - (float)iy) * dx;
        if (dx == 0.f) err = INF;
        if (dy == 0.f) err = -INF;
        int left = 1 + std::abs(ex - ix) + std::abs(ey - iy);
        int fs, fe;
        bool ok = at(ix, iy, &fs) & at(ex, ey, &fe);
        bool start_grass = fs & 1, end_grass = fe & 1;
        auto cell_clear = [&](int x, int y) {
            int f;
            bool valid = at(x, y, &f);
            bool transparent = (f & 2) == 0 || (f & (0x40 | 0x100)) != 0;
            bool grass = f & 1;
            bool brush_ok = start_grass ? (!end_grass || grass) : !grass;
            return valid && transparent && brush_ok;
        };
        int x = ix, y = iy;
        for (int it = 0; it < 64 && ok && left > 0; ++it) {
            bool corner = std::fabs(err) <= 1e-3f;
            bool c = cell_clear(x, y) && (!corner || (cell_clear(x + sx, y) && cell_clear(x, y + sy)));
            if (!c) return false;
            bool mx = err < 0.f || corner, my = err > 0.f || corner;
            if (mx) x += sx;
            if (my) y += sy;
            err += (mx && my) ? dy - dx : (mx ? dy : -dx);
            left -= 1 + (int)corner;
        }
        return ok && left <= 0;
    }
};

}  // namespace lanesim
