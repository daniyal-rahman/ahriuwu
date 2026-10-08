// Item effect modules B: items.effects.fighter / defense / mage / marksman (hook protocol in items/effects/core.py).
// Hooks take the JAX arguments in order and return the JAX result; state is the module's own State. Helpers below
// are local to these modules (items.effects.core / core.damage helpers not in champ/core.hpp).
#pragma once
#include <string>
#include <tuple>

#include "../core.hpp"

namespace lanesim::items {

using champ::C;

namespace itemsb {

inline int n_units(const Units& u) { return (int)u.x.size(); }
inline float where(bool c, float a, float b) { return c ? a : b; }
// items.effects.core.by_range: ranged value for ranged holders, 1 for melee.
inline float by_range(const Ctx& ctx, int c, float ranged_value) { return ctx.is_ranged[c] ? ranged_value : 1.0f; }
inline bool holds_any(const Owned& own, std::initializer_list<int> ids, int c) {
    for (int id : ids)
        if (champ::holds(own, id, c)) return true;
    return false;
}
// Data value ``items.<module>.dv.<id>.<name>`` (native/python/consts/items_<module>.py).
inline float dv(const char* module, int id, const char* name) {
    return data::f(std::string("items.") + module + ".dv." + std::to_string(id) + "." + name);
}
inline float k(const char* module, const char* name) { return data::f(std::string("items.") + module + "." + name); }

// core.damage.per_unit over (C, P) selections onto (C, N) by packet destination (negative / >= N dropped).
template <class Sel>
Arr<uint8_t> per_unit_any(const Packets& p, int n, Sel sel) {
    Arr<uint8_t> out((size_t)C * n, 0);
    for (int c = 0; c < C; ++c)
        for (size_t i = 0; i < champ::size(p); ++i) {
            int d = p.dst[i];
            if (d >= 0 && d < n && sel(c, (int)i)) out[(size_t)c * n + d] = 1;
        }
    return out;
}
// ``"add"`` in packet order (val(c, i) for every packet, selected or not, like the masked JAX scatter).
template <class Val>
Arr<float> per_unit_add(const Packets& p, int n, Val val) {
    Arr<float> out((size_t)C * n, 0.f);
    for (int c = 0; c < C; ++c)
        for (size_t i = 0; i < champ::size(p); ++i) {
            int d = p.dst[i];
            if (d >= 0 && d < n) out[(size_t)c * n + d] = out[(size_t)c * n + d] + val(c, (int)i);
        }
    return out;
}
template <class Val>
Arr<float> per_unit_max(const Packets& p, int n, Val val) {
    Arr<float> out((size_t)C * n, 0.f);
    for (int c = 0; c < C; ++c)
        for (size_t i = 0; i < champ::size(p); ++i) {
            int d = p.dst[i];
            if (d >= 0 && d < n) out[(size_t)c * n + d] = std::max(out[(size_t)c * n + d], val(c, (int)i));
        }
    return out;
}
// core.damage.shield_value of slot (unit, k), (N, K) row-major.
inline float shield_value(const Shields& sh, size_t idx, float now) {
    float span = std::max(sh.expires_at[idx] - sh.decay_start[idx], 1e-6f);
    float frac = std::min(std::max((sh.expires_at[idx] - now) / span, 0.f), 1.f);
    float cap = now > sh.decay_start[idx] ? sh.initial[idx] * frac : sh.initial[idx];
    bool live = sh.amount[idx] > 0.f && now < sh.expires_at[idx];
    return live ? std::min(sh.amount[idx], cap) : 0.f;
}
inline int shield_slots(const Shields& sh, int n) { return n ? (int)(sh.amount.size() / n) : 0; }
// (C, N) one-hot of per-holder indices (negative = none).
inline bool onehot(int idx, int j) { return idx >= 0 && j == idx; }

// A module ``stats`` result: the fields the JAX hook sets are (C,) zeros to fill in; the others keep the ItemStats
// default, a Python 0.0 that JAX returns as a scalar (one element here, so the leaves match the JAX pytree; callers
// combining stats broadcast size-1 fields).
inline ItemStats stats_out(std::initializer_list<Arr<float> ItemStats::*> fields) {
    ItemStats s;
    s.visit([&](auto& m) { m.assign(1, 0.f); });
    for (auto f : fields) (s.*f).assign(C, 0.f);
    return s;
}

// items.effects.core.nearest_k for one holder row: mask of the ``k`` smallest ``dist`` within ``mask`` (k rounds of
// argmin, ties to the lower index).
inline std::vector<uint8_t> nearest_k(const std::vector<float>& dist, const std::vector<uint8_t>& mask, int k) {
    int n = (int)dist.size();
    std::vector<float> key(n);
    std::vector<uint8_t> pick(n, 0);
    for (int j = 0; j < n; ++j) key[j] = mask[j] ? dist[j] : std::numeric_limits<float>::infinity();
    for (int r = 0; r < std::min(k, n); ++r) {
        int am = 0;
        for (int j = 1; j < n; ++j)
            if (key[j] < key[am]) am = j;
        pick[am] = 1, key[am] = std::numeric_limits<float>::infinity();
    }
    for (int j = 0; j < n; ++j) pick[j] = pick[j] && mask[j];
    return pick;
}

// packets() over a (C, N) grid: src = holder unit, dst = arange(N).
template <class Valid, class Raw, class Flags, class Item>
void push_grid(Packets& out, const Ctx& ctx, int n, Valid valid, Raw raw, int dtype, Flags flags, Item item) {
    for (int c = 0; c < C; ++c)
        for (int j = 0; j < n; ++j) champ::push(out, valid(c, j), ctx.unit[c], j, raw(c, j), dtype, flags(c, j), 0.f, item(c, j));
}

}  // namespace itemsb

namespace fighter {
using State = items_fighter_State;
ItemStats stats(State state, const Owned& own, const Ctx& ctx);
HolderDefense defense(State state, const Owned& own, const Ctx& ctx);
Debuffs debuffs(State state, const Owned& own, const Ctx& ctx, const Units& units);
AttackMods attack_mods(State state, const Owned& own, const Ctx& ctx, const Units& units, Arr<int32_t> target);
Arr<float> packet_amp(State state, const Owned& own, const Ctx& ctx, const Units& units, const Packets& p);
std::tuple<State, Effects> on_cast(State state, const Owned& own, const Ctx& ctx, const Units& units, const Cast& cast);
std::tuple<State, Effects> on_hit(State state, const Owned& own, const Ctx& ctx, const Units& units, const Attack& attack);
std::tuple<State, Effects> on_damage(State state, const Owned& own, const Ctx& ctx, const Units& units, const Report& report);
std::tuple<State, Effects> periodic(State state, const Owned& own, const Ctx& ctx, const Units& units);
std::tuple<State, Effects> on_takedown(State state, const Owned& own, const Ctx& ctx, const Units& units, const Kills& kills);
}  // namespace fighter

namespace defense {
using State = items_defense_State;
ItemStats stats(State state, const Owned& own, const Ctx& ctx);
HolderDefense defense(State state, const Owned& own, const Ctx& ctx);
Debuffs debuffs(State state, const Owned& own, const Ctx& ctx, const Units& units);
std::tuple<State, Effects> on_hit(State state, const Owned& own, const Ctx& ctx, const Units& units, const Attack& attack);
std::tuple<State, Effects> on_damage(State state, const Owned& own, const Ctx& ctx, const Units& units, const Report& report);
std::tuple<State, Effects> periodic(State state, const Owned& own, const Ctx& ctx, const Units& units);
std::tuple<State, Effects> on_takedown(State state, const Owned& own, const Ctx& ctx, const Units& units, const Kills& kills);
}  // namespace defense

namespace mage {
using State = items_mage_State;
ItemStats stats(State state, const Owned& own, const Ctx& ctx);
Arr<float> dealt_amp(State state, const Owned& own, const Ctx& ctx, const Units& units);
Debuffs debuffs(State state, const Owned& own, const Ctx& ctx, const Units& units);
std::tuple<State, Effects> on_hit(State state, const Owned& own, const Ctx& ctx, const Units& units, const Attack& attack);
std::tuple<State, Effects> on_cast(State state, const Owned& own, const Ctx& ctx, const Units& units, const Cast& cast);
std::tuple<State, Effects> on_damage(State state, const Owned& own, const Ctx& ctx, const Units& units, const Report& report);
std::tuple<State, Effects> periodic(State state, const Owned& own, const Ctx& ctx, const Units& units);
std::tuple<State, Effects> on_takedown(State state, const Owned& own, const Ctx& ctx, const Units& units, const Kills& kills);
}  // namespace mage

namespace marksman {
using State = items_marksman_State;
ItemStats stats(State state, const Owned& own, const Ctx& ctx);
StatusFlags status(State state, const Owned& own, const Ctx& ctx);
Arr<float> dealt_amp(State state, const Owned& own, const Ctx& ctx, const Units& units);
AttackMods attack_mods(State state, const Owned& own, const Ctx& ctx, const Units& units, Arr<int32_t> target);
Arr<float> packet_amp(State state, const Owned& own, const Ctx& ctx, const Units& units, const Packets& p);
std::tuple<State, Effects> on_cast(State state, const Owned& own, const Ctx& ctx, const Units& units, const Cast& cast);
std::tuple<State, Effects> on_attack(State state, const Owned& own, const Ctx& ctx, const Units& units, const Attack& attack);
std::tuple<State, Effects> on_hit(State state, const Owned& own, const Ctx& ctx, const Units& units, const Attack& attack);
std::tuple<State, Effects> on_damage(State state, const Owned& own, const Ctx& ctx, const Units& units, const Report& report);
std::tuple<State, Effects> periodic(State state, const Owned& own, const Ctx& ctx, const Units& units);
std::tuple<State, Effects> on_takedown(State state, const Owned& own, const Ctx& ctx, const Units& units, const Kills& kills);
}  // namespace marksman

}  // namespace lanesim::items
