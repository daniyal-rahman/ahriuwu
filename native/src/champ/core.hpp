// Shared helpers of the champion layer, ported from core/damage.py (packets), items/effects/core.py,
// runes/effects/core.py and champions/core.py. Shapes follow JAX: (C,) per champion holder, (N,) per unit,
// (C, N) row-major [c * n + j]; packet arrays are padded with ``valid`` like the JAX ones, so concatenation and
// compaction give the same order.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "../gen/types.hpp"
#include "data.hpp"

namespace lanesim::champ {

constexpr int C = 2;
constexpr float BIG = 1e9f;
constexpr float INF = std::numeric_limits<float>::infinity();

// core.damage constants
enum DType { PHYSICAL = 0, MAGIC = 1, TRUE_DMG = 2 };
constexpr int TAG_AOE = 1 << 0, TAG_PERIODIC = 1 << 1, TAG_INDIRECT = 1 << 2, TAG_BASIC_ATTACK = 1 << 3,
              TAG_ACTIVE_SPELL = 1 << 4, TAG_PROC = 1 << 5, TAG_PET = 1 << 6, TAG_ITEM = 1 << 8,
              TAG_DOES_NOT_AGGRO_JUNGLE = 1 << 9, TAG_ON_HIT = 1 << 10, TAG_BURN = 1 << 12,
              TAG_NON_AMPABLE = 1 << 13, PROP_LIFESTEAL = 1 << 16, PROP_NO_OMNIVAMP = 1 << 17,
              PROP_NO_DAMAGE_MOD = 1 << 18, PROP_REACTIVE = 1 << 19, PROP_CRIT = 1 << 20, PROP_EXECUTE = 1 << 21,
              PROP_ULTIMATE = 1 << 22, PROP_SUMMONER = 1 << 23;
constexpr int BASIC_ATTACK = TAG_BASIC_ATTACK | PROP_LIFESTEAL;
constexpr int ON_HIT_ITEM = TAG_ON_HIT | TAG_PROC | TAG_ITEM;
enum ShieldKind { SHIELD_ALL = 0, SHIELD_PHYSICAL = 1, SHIELD_MAGIC = 2 };
enum UnitClass { CLASS_CHAMPION = 0, CLASS_MINION = 1, CLASS_STRUCTURE = 2, CLASS_MONSTER = 3 };
// core.types kinds
enum Kind : int32_t { KIND_NONE, KIND_CHAMPION, KIND_MINION, KIND_TURRET, KIND_INHIBITOR, KIND_NEXUS, KIND_MONSTER,
                      KIND_WARD };
inline bool has(int flags, int bit) { return (flags & bit) != 0; }
inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float sq(float v) { return v * v; }

// --- packets -----------------------------------------------------------------------------------------------------
inline Packets empty_packets(size_t p = 0) {
    Packets out;
    out.visit([&](auto& m) { m.assign(p, 0); });
    return out;
}
inline size_t size(const Packets& p) { return p.valid.size(); }
inline void append(Packets& a, const Packets& b) {
    a.valid.append(b.valid), a.src.append(b.src), a.dst.append(b.dst), a.raw.append(b.raw);
    a.dtype.append(b.dtype), a.flags.append(b.flags), a.amp.append(b.amp), a.item.append(b.item);
    a.cast_id.append(b.cast_id), a.block.append(b.block);
}
// One packet appended (core.damage.packets with scalar fields).
inline void push(Packets& p, bool valid, int src, int dst, float raw, int dtype, int flags = 0, float amp = 0.f,
                 int item = 0, int cast_id = 0, float block = 0.f) {
    p.valid.push_back(valid), p.src.push_back(src), p.dst.push_back(dst), p.raw.push_back(raw);
    p.dtype.push_back(dtype), p.flags.push_back(flags), p.amp.push_back(amp), p.item.push_back(item);
    p.cast_id.push_back(cast_id), p.block.push_back(block);
}
// compact_packets: valid packets first in emission order, padded to ``capacity``; returns the dropped count.
int compact(const Packets& in, size_t capacity, Packets& out);

// --- ownership (items.effects.core) ------------------------------------------------------------------------------
int item_row(int item_id);                              // catalog row (consts "catalog.ids")
int n_items();
// Static: some holder may ever hold one of the items (an empty ``allowed`` is no restriction).
bool can_hold(const Owned& own, std::initializer_list<int> item_ids);
// (C,) holder owns ``item_id`` (false everywhere when no holder can hold it).
inline bool holds(const Owned& own, int item_id, int c) {
    if (!can_hold(own, {item_id})) return false;
    return own.counts[(size_t)c * n_items() + item_row(item_id)] > 0;
}
inline int count(const Owned& own, int item_id, int c) { return own.counts[(size_t)c * n_items() + item_row(item_id)]; }

// --- Effects / defense / debuffs ---------------------------------------------------------------------------------
Effects no_effects(int c, int n);
void concat_shields(ShieldGrant& a, const ShieldGrant& b, size_t c);   // along the grant axis
void merge_into(Effects& acc, const Effects& p);       // merge_effects, one part at a time (same result)
Effects merge(const std::vector<const Effects*>& parts, int c, int n);
ShieldGrant shield_grants(const Arr<float>& amount, int kind = SHIELD_ALL, float duration = 0.f, float decay_hold = INF);
HolderDefense neutral_defense(int c);
HolderDefense combine_defense(const std::vector<const HolderDefense*>& parts, int c);
Debuffs neutral_debuffs(int n);
Debuffs combine_debuffs(const std::vector<const Debuffs*>& parts, int n);

// --- unit helpers (items.effects.core) ---------------------------------------------------------------------------
inline bool enemy(const Ctx& ctx, const Units& u, int c, int j) {
    return u.team[j] != ctx.team[c] && u.alive[j] && u.targetable[j];
}
inline float dist_to_point(const Units& u, int j, float px, float py) {
    return std::sqrt(sq(u.x[j] - px) + sq(u.y[j] - py));
}
inline bool in_circle(const Units& u, int j, float px, float py, float radius, bool edge = true) {
    return dist_to_point(u, j, px, py) <= radius + (edge ? u.radius[j] : 0.f);
}
inline int target_class(const Units& u, int idx) { return u.cls[clampi(idx, 0, (int)u.cls.size() - 1)]; }

// --- runes (runes.effects.core / runes.catalog) ------------------------------------------------------------------
int rune_row(int perk_id);                              // consts "runes.ids"
inline bool has_rune(const Arr<int32_t>& page, int perk_id, int c, int n_runes) {
    return page[(size_t)c * n_runes + rune_row(perk_id)] > 0;
}
RuneOutputs no_outputs(int c, int n_items);
void merge_outputs_into(RuneOutputs& acc, const RuneOutputs& p);

// --- kits (champions.core) ---------------------------------------------------------------------------------------
constexpr int KIT_ID_BASE = 1 << 30, ID_STRIDE = 8, CODE_GAREN_E_TICK = 4, CODE_JAX_R_PASSIVE = 5;
constexpr float EPS = 1e-6f, NEVER = 1e9f;
CCOut no_cc(int c, int n);
void merge_cc_into(CCOut& acc, const CCOut& b);        // core.types.merge_cc
Dash no_dash(int c);
KitOut no_out(int c, int n);
void merge_out_into(KitOut& acc, const KitOut& p, int c, int n);
KitDefense neutral_kit_defense(int c);
KitAttackMods neutral_attack_mods(int c);
inline int tick_index(const KitCtx& k) {
    return (int)std::nearbyint(k.now / std::max(k.dt, EPS));
}
inline int make_cast_id(const KitCtx& k, int holder, int code) {
    return KIT_ID_BASE + (tick_index(k) * (int)k.unit.size() + holder) * ID_STRIDE + code;
}
inline bool due(const KitCtx& k, float until) { return k.now + k.dt >= until - EPS; }
inline float later(const KitCtx& k, float seconds) { return k.now + seconds; }
inline float later_after_tick(const KitCtx& k, float seconds) { return k.now + k.dt + seconds; }
inline float pulses(const KitCtx& k, float period) {
    return std::floor(k.now / period + 1e-6f) - std::floor((k.now - k.dt) / period + 1e-6f);
}

}  // namespace lanesim::champ
