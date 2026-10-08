// Rune effect modules (runes/effects/{precision,domination,sorcery,resolve,inspiration}.py) and their area-local
// helpers (runes/effects/core.py kernels, runes/catalog.py level scaling, core/damage.py packet scans).
// Hooks take the JAX arguments in order: (state, page, ctx[, units], ev[, packets]); ``page`` is the (C, R) rune
// count matrix, (C,) per holder, (N,) per unit, (C, N) row-major.
#pragma once
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "../core.hpp"

namespace lanesim::runes {

using namespace lanesim::champ;
using Page = Arr<int32_t>;

// --- runes.catalog ---------------------------------------------------------------------------------------------
inline int n_runes(const Page& page, int c) { return c ? (int)(page.size() / c) : 0; }
inline bool hasr(const Page& page, int perk, int c, int nc) { return has_rune(page, perk, c, n_runes(page, nc)); }

// A host constant ``"runes.<module>.<name>"`` (consts/runes_<module>.py).
inline const std::vector<float>& K(const std::string& key) { return data::table(key); }

// lin(start, end, level): the consts table is [start, end - start] (the difference is a Python float).
inline float lin(const std::vector<float>& t, float level, bool scale_past_18 = true) {
    float lv = std::max(level, 1.f);
    if (!scale_past_18) lv = std::min(lv, 18.f);
    return t[0] + t[1] * (lv - 1.f) / 17.f;
}
// lin_growth(start, end, level), table [start, end - start].
inline float lin_growth(const std::vector<float>& t, float level) {
    float n = std::max(level - 1.f, 0.f);
    return t[0] + t[1] * n * (0.7025f + 0.0175f * n) / 17.f;
}
// level_table(values, level): index clip(int(level), 0, len - 1).
inline float level_table(const std::vector<float>& t, float level) {
    return t[clampi((int)level, 0, (int)t.size() - 1)];
}

// --- runes.effects.core ----------------------------------------------------------------------------------------
inline int rune_item(int perk) { return -perk; }
inline float by_range(const Ctx& ctx, int c, float melee, float ranged) { return ctx.is_ranged[c] ? ranged : melee; }
inline int adaptive_damage_type(const RuneEvents& ev, int c) {
    return ev.bonus_ad[c] > ev.ap[c] ? PHYSICAL
         : (ev.ap[c] > ev.bonus_ad[c] ? MAGIC : (ev.adaptive_physical[c] ? PHYSICAL : MAGIC));
}
inline int variable_damage_type(float ad_term, float ap_term) { return ad_term > ap_term ? PHYSICAL : MAGIC; }
inline int clip_unit(int idx, int n) { return clampi(idx, 0, n - 1); }

// (C, P) bool matrices, row-major.
using Mask = std::vector<uint8_t>;
// core.damage.first_per_key: per row, keep only the first selected packet of each distinct key tuple.
template <class... Keys>
Mask first_per_key(const Mask& sel, size_t c, size_t p, const Keys&... keys) {
    Mask out(sel.size(), 0);
    auto same = [&](size_t i, size_t j) { return ((keys[i] == keys[j]) && ...); };
    for (size_t h = 0; h < c; ++h)
        for (size_t i = 0; i < p; ++i) {
            if (!sel[h * p + i]) continue;
            bool first = true;
            for (size_t j = 0; j < i && first; ++j)
                if (sel[h * p + j] && same(i, j)) first = false;
            out[h * p + i] = first;
        }
    return out;
}
// runes.effects.core.first_instance: first packet of each (cast_id, src, dst) instance; cast_id 0 is its own.
inline Mask first_instance(const Packets& p, const Mask& sel, size_t c) {
    size_t np = size(p);
    Mask f = first_per_key(sel, c, np, p.cast_id, p.src, p.dst);
    Mask out(sel.size());
    for (size_t h = 0; h < c; ++h)
        for (size_t i = 0; i < np; ++i) out[h * np + i] = sel[h * np + i] && (p.cast_id[i] == 0 || f[h * np + i]);
    return out;
}
// (C,) argmax of a bool row (first true, 0 when none).
inline int argmax_row(const Mask& m, size_t h, size_t p) {
    for (size_t i = 0; i < p; ++i)
        if (m[h * p + i]) return (int)i;
    return 0;
}
inline bool any_row(const Mask& m, size_t h, size_t p) {
    for (size_t i = 0; i < p; ++i)
        if (m[h * p + i]) return true;
    return false;
}

// ItemStats with every field the JAX default (a () zero leaf); hooks then set their (C,) fields.
inline ItemStats default_stats() {
    ItemStats s;
    s.visit([](auto& m) { m.assign(1, 0.f); });
    return s;
}

// --- hooks -----------------------------------------------------------------------------------------------------
namespace precision {
using State = runes_precision_State;
ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev);
std::tuple<State, Effects> on_attack(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_hit(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
Arr<float> packet_amp(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev, const Packets& p);
std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_takedown(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
}  // namespace precision

namespace domination {
using State = runes_domination_State;
std::tuple<State, Effects> on_cc(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_cast(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_attack(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_hit(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_takedown(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev);
}  // namespace domination

namespace sorcery {
using State = runes_sorcery_State;
ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev);
Arr<float> packet_amp(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev, const Packets& p);
std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_cc(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_cast(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_takedown(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
RuneOutputs outputs(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev);
}  // namespace sorcery

namespace resolve {
using State = runes_resolve_State;
ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev);
Arr<float> heal_mult(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev);
std::tuple<State, Effects> on_cast(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_hit(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_cc(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
Arr<float> packet_block(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev, const Packets& p);
std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
State post_tick(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
}  // namespace resolve

namespace inspiration {
using State = runes_inspiration_State;
ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev);
std::tuple<State, Effects> on_cc(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
Arr<float> packet_amp(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev, const Packets& p);
std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
State post_tick(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
RuneOutputs outputs(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev);
}  // namespace inspiration

}  // namespace lanesim::runes
