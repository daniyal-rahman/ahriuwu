// Item effect modules A (items/effects/{consumables,starters,spellblade,hydra,boots,actives}.py): hook
// declarations and area-local helpers. Hooks take the JAX arguments in order and return the JAX result; only
// items Garen or Jax can hold are ported (``holds`` is false for the rest), but every padded packet / grant the
// JAX hook always emits is emitted with the same values.
#pragma once
#include <tuple>

#include "../core.hpp"

namespace lanesim::items {

using namespace lanesim::champ;

// items.catalog.ItemStats() defaults: unset fields are the python scalar 0.0 (one-element leaves); hooks
// replace the fields they compute with (C,) arrays. Combining must broadcast one-element fields.
inline ItemStats default_stats() {
    ItemStats s;
    s.visit([](auto& m) { m.assign(1, 0.f); });
    return s;
}
inline Arr<float> zeros_c(size_t c) { return Arr<float>(c, 0.f); }
inline bool holds_any(const Owned& own, std::initializer_list<int> ids, int c) {
    for (int id : ids)
        if (holds(own, id, c)) return true;
    return false;
}
// core.damage.per_unit(..., "add") of dealt_by_holder: (C, N) post-mitigation damage of holder c's packets
// selected by ``sel`` (P,) on each unit, in packet order.
inline Arr<float> dealt_by_holder(const Report& r, const Ctx& ctx, int n, const Arr<uint8_t>& sel) {
    size_t c = ctx.unit.size();
    Arr<float> out(c * n, 0.f);
    const Packets& p = r.packets;
    for (size_t h = 0; h < c; ++h)
        for (size_t k = 0; k < size(p); ++k)
            if (p.src[k] == ctx.unit[h] && sel[k] && p.dst[k] >= 0 && p.dst[k] < n)
                out[h * n + p.dst[k]] = out[h * n + p.dst[k]] + r.resolved.final[k];
    return out;
}
// hit_by_holder: (C, N) a selected packet of holder c reached unit n.
inline Arr<uint8_t> hit_by_holder(const Report& r, const Ctx& ctx, int n, const Arr<uint8_t>& sel) {
    size_t c = ctx.unit.size();
    Arr<uint8_t> out(c * n, 0);
    const Packets& p = r.packets;
    for (size_t h = 0; h < c; ++h)
        for (size_t k = 0; k < size(p); ++k)
            if (p.src[k] == ctx.unit[h] && sel[k] && p.dst[k] >= 0 && p.dst[k] < n) out[h * n + p.dst[k]] = 1;
    return out;
}
// items.effects.core.nearest_k on one row: mask of the k smallest ``dist`` within ``mask`` (ties: lower index).
inline void nearest_k(const float* dist, const uint8_t* mask, int n, int k, uint8_t* out) {
    Arr<float> key(n);
    Arr<uint8_t> pick(n, 0);
    for (int j = 0; j < n; ++j) key[j] = mask[j] ? dist[j] : INF;
    for (int r = 0; r < std::min(k, n); ++r) {
        int best = 0;                     // argmin: first index of the minimum
        for (int j = 1; j < n; ++j)
            if (key[j] < key[best]) best = j;
        pick[best] = 1, key[best] = INF;
    }
    for (int j = 0; j < n; ++j) out[j] = mask[j] && pick[j];
}
// catalog.level_bp
inline float level_bp(float start, float per_level, float from_level, float level) {
    return start + per_level * std::max(0.f, level - from_level + 1.f);
}
// (C, N) packets from per-holder src, per-unit dst = j.
inline void push_cn(Packets& p, bool valid, int src, int j, float raw, int dtype, int flags, int item) {
    push(p, valid, src, j, raw, dtype, flags, 0.f, item);
}

namespace consumables {
using State = items_consumables_State;
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx);
std::tuple<State, Effects, ActiveOut> active(State s, const Owned& own, const Ctx& ctx, const Units& u,
                                             const Arr<int32_t>& request);
std::tuple<State, Effects> periodic(State s, const Owned& own, const Ctx& ctx, const Units& u);
std::tuple<State, Effects> on_hit(State s, const Owned& own, const Ctx& ctx, const Units& u, const Attack& a);
State on_shop(State s, const Owned& own, const Ctx& ctx);
std::tuple<State, Effects> on_damage(State s, const Owned& own, const Ctx& ctx, const Units& u, const Report& r);
}  // namespace consumables

namespace starters {
using State = items_starters_State;
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx);
HolderDefense defense(const State& s, const Owned& own, const Ctx& ctx);
std::tuple<State, Effects> on_hit(State s, const Owned& own, const Ctx& ctx, const Units& u, const Attack& a);
std::tuple<State, Effects> on_cast(State s, const Owned& own, const Ctx& ctx, const Units& u, const Cast& cast);
std::tuple<State, Effects> on_damage(State s, const Owned& own, const Ctx& ctx, const Units& u, const Report& r);
std::tuple<State, Effects> periodic(State s, const Owned& own, const Ctx& ctx, const Units& u);
std::tuple<State, Effects> on_takedown(State s, const Owned& own, const Ctx& ctx, const Units& u, const Kills& k);
std::tuple<State, Effects> on_cc(State s, const Owned& own, const Ctx& ctx, const Units& u, const CC& cc);
}  // namespace starters

namespace spellblade {
using State = items_spellblade_State;
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx);
std::tuple<State, Effects> on_cast(State s, const Owned& own, const Ctx& ctx, const Units& u, const Cast& cast);
std::tuple<State, Effects> on_hit(State s, const Owned& own, const Ctx& ctx, const Units& u, const Attack& a);
std::tuple<State, Effects> periodic(State s, const Owned& own, const Ctx& ctx, const Units& u);
Debuffs debuffs(const State& s, const Owned& own, const Ctx& ctx, const Units& u);
}  // namespace spellblade

namespace hydra {
using State = items_hydra_State;
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx);
std::tuple<State, Effects> on_hit(State s, const Owned& own, const Ctx& ctx, const Units& u, const Attack& a);
std::tuple<State, Effects, ActiveOut> active(State s, const Owned& own, const Ctx& ctx, const Units& u,
                                             const Arr<int32_t>& request);
}  // namespace hydra

namespace boots {
using State = items_boots_State;
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx);
Arr<float> dealt_amp(const State& s, const Owned& own, const Ctx& ctx, const Units& u);
HolderDefense defense(const State& s, const Owned& own, const Ctx& ctx);
std::tuple<State, Effects> on_damage(State s, const Owned& own, const Ctx& ctx, const Units& u, const Report& r);
std::tuple<State, Effects> on_takedown(State s, const Owned& own, const Ctx& ctx, const Units& u, const Kills& k);
std::tuple<State, Effects> periodic(State s, const Owned& own, const Ctx& ctx, const Units& u);
}  // namespace boots

namespace actives {
using State = items_actives_State;
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx);
StatusFlags status(const State& s, const Owned& own, const Ctx& ctx);
Arr<float> packet_amp(const State& s, const Owned& own, const Ctx& ctx, const Units& u, const Packets& p);
std::tuple<State, Effects, ActiveOut> active(State s, const Owned& own, const Ctx& ctx, const Units& u,
                                             const Arr<int32_t>& request);
// Called by the world phases outside the hook dispatch.
// actives.with_aim(state, unit, x, y): empty ``unit`` = None (-1), empty ``x`` = None (no aim point).
State with_aim(State s, const Arr<int32_t>& unit, const Arr<float>& x, const Arr<float>& y);
// actives.request_allowed(request, disabled=, in_stasis=): empty ``in_stasis`` = None.
Arr<int32_t> request_allowed(const Arr<int32_t>& request, const Arr<uint8_t>& disabled, const Arr<uint8_t>& in_stasis);
// actives.ActiveWorld (hand-written: its ``transform`` field is a plain tuple, flattened here in JAX order).
struct ActiveWorld {
    Arr<uint8_t> stasis{};
    Arr<float> stasis_until{};
    Arr<uint8_t> cleanse{};
    Dash dash{};
    Arr<float> mana_cost_mult{};
    Arr<float> basic_cd_rate{};
    Arr<int32_t> transform_from{};     // transform[0]: Seeker's row
    Arr<int32_t> transform_to{};       // transform[1]: Shattered Armguard row
    Arr<uint8_t> transform_do{};       // transform[2]
    template <class F> void visit(F&& f) {
        f(stasis); f(stasis_until); f(cleanse); f(dash); f(mana_cost_mult); f(basic_cd_rate); f(transform_from);
        f(transform_to); f(transform_do);
    }
};
ActiveWorld world(const State& s, float now);
}  // namespace actives

}  // namespace lanesim::items
