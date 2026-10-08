// Champion kits (champions/garen.py, champions/jax.py), the kit dispatch of champions/__init__.py and the summoner
// spells (champions/summoners.py). Every kit hook runs for both holders and gates itself on KitCtx.champion_id.
#pragma once
#include <string>
#include <tuple>
#include <vector>

#include "../core.hpp"

namespace lanesim::kits {

// --- area-local helpers (champions/core.py) ----------------------------------------------------------------------
namespace detail {
// A float table of native/python/consts/kits_*.py, copied once.
std::vector<float> table(const std::string& key);
// core.ranked: JSON value at ``rank`` (index 0 is rank 0; clipped to 0..6 and to the table).
inline float ranked(const std::vector<float>& t, int rank) {
    int i = champ::clampi(rank, 0, 6);
    return t[std::min<size_t>((size_t)i, t.size() - 1)];
}
// core.cooldown_row / core.mana_row entry: table[clip(rank - 1, 0, len - 1)] (0 for an empty table).
inline float by_rank(const std::vector<float>& t, int rank) {
    if (t.empty()) return 0.f;
    return t[champ::clampi(rank - 1, 0, (int)t.size() - 1)];
}
inline bool is_structure(int kind) {
    return kind == champ::KIND_TURRET || kind == champ::KIND_INHIBITOR || kind == champ::KIND_NEXUS;
}
inline int gather_i(const Arr<int32_t>& a, int idx) { return a[champ::clampi(idx, 0, (int)a.size() - 1)]; }
inline bool gather_b(const Arr<uint8_t>& a, int idx) { return a[champ::clampi(idx, 0, (int)a.size() - 1)]; }
// core.enemies: living, targetable enemy non-structure, non-ward unit j of holder c.
inline bool enemy(const KitCtx& k, const WorldUnits& u, int c, int j) {
    return u.team[j] != k.team[c] && u.alive[j] && u.targetable[j] && u.kind[j] != champ::KIND_NONE &&
           u.kind[j] != champ::KIND_WARD && !is_structure(u.kind[j]);
}
// core.center_dist
inline float center_dist(const KitCtx& k, const WorldUnits& u, int c, int j) {
    return std::sqrt((u.x[j] - k.x[c]) * (u.x[j] - k.x[c]) + (u.y[j] - k.y[c]) * (u.y[j] - k.y[c]));
}
// core.target_dist
inline float target_dist(const KitCtx& k, const WorldUnits& u, int c, int target) {
    int t = champ::clampi(target, 0, (int)u.x.size() - 1);
    return std::sqrt((u.x[t] - k.x[c]) * (u.x[t] - k.x[c]) + (u.y[t] - k.y[c]) * (u.y[t] - k.y[c]));
}
// core.holder_rows of a (C,) or (N,) array
template <class T> inline T holder_row(const Arr<T>& x, const KitCtx& k, int c) {
    return x.size() == k.unit.size() ? x[c] : x[k.unit[c]];
}
// ItemStats with every field a 0-d zero (the Python default ``0.0``); kits set their (C,) fields.
ItemStats scalar_zero_stats();
}  // namespace detail

namespace garen {
constexpr int ID = 86;
using State = champions_garen_State;
std::tuple<State, KitOut> cast(State s, const KitCtx& k, const WorldUnits& u, const CastOrder& order);
std::tuple<State, KitOut> periodic(State s, const KitCtx& k, const WorldUnits& u);
std::tuple<State, KitOut> on_attack(State s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& launch);
std::tuple<State, KitOut> on_hit(State s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& launch,
                                 const Arr<uint8_t>& dodging);
std::tuple<State, KitOut> on_damage(State s, const KitCtx& k, const WorldUnits& u, const Report& report);
State on_takedown(State s, const KitCtx& k, const WorldUnits& u, const Kills& kills);
ItemStats stats(const State& s, const KitCtx& k);
KitDefense defense(const State& s, const KitCtx& k);
KitAttackMods attack_mods(const State& s, const KitCtx& k);
Arr<uint8_t> ghosted(const State& s, const KitCtx& k);
Debuffs debuffs(const State& s, const KitCtx& k, const WorldUnits& u);
float regen_rate(float level);
float q_attack_time(float bonus_attack_speed);
float e_crit_multiplier(float crit_damage);
}  // namespace garen

namespace jax {
constexpr int ID = 24;
using State = champions_jax_State;
std::tuple<State, KitOut> cast(State s, const KitCtx& k, const WorldUnits& u, const CastOrder& order);
std::tuple<State, KitOut> periodic(State s, const KitCtx& k, const WorldUnits& u);
std::tuple<State, KitOut> on_attack(State s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& launch);
std::tuple<State, KitOut> on_hit(State s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& launch,
                                 const Arr<uint8_t>& dodging_units);
std::tuple<State, KitOut> on_damage(State s, const KitCtx& k, const WorldUnits& u, const Report& report);
State on_takedown(State s, const KitCtx& k, const WorldUnits& u, const Kills& kills);
ItemStats stats(const State& s, const KitCtx& k);
KitDefense defense(const State& s, const KitCtx& k);
KitAttackMods attack_mods(const State& s, const KitCtx& k);
Arr<uint8_t> dodging(const State& s, const KitCtx& k);
Debuffs debuffs(const State& s, const KitCtx& k, const WorldUnits& u);
}  // namespace jax

// --- dispatch (champions/__init__.py): every kit for every holder, merged ---------------------------------------
Arr<float> unit_target_ranges(const Arr<int32_t>& champion_ids);       // (C, 4)
std::tuple<ChampionState, KitOut> cast(ChampionState s, const KitCtx& k, const WorldUnits& u, const CastOrder& order);
std::tuple<ChampionState, KitOut> periodic(ChampionState s, const KitCtx& k, const WorldUnits& u);
std::tuple<ChampionState, KitOut> on_attack(ChampionState s, const KitCtx& k, const WorldUnits& u,
                                            const AttackLaunch& launch);
Arr<uint8_t> dodging_units(const ChampionState& s, const KitCtx& k, int n_units);  // (N,)
std::tuple<ChampionState, KitOut> on_hit(ChampionState s, const KitCtx& k, const WorldUnits& u,
                                         const AttackLaunch& launch);
std::tuple<ChampionState, KitOut> on_damage(ChampionState s, const KitCtx& k, const WorldUnits& u, const Report& report);
ChampionState on_takedown(ChampionState s, const KitCtx& k, const WorldUnits& u, const Kills& kills);
ItemStats stats(const ChampionState& s, const KitCtx& k);              // (C,) fields (0-d defaults broadcast)
KitDefense defense(const ChampionState& s, const KitCtx& k);
KitAttackMods attack_mods(const ChampionState& s, const KitCtx& k);
Arr<uint8_t> ghosted(const ChampionState& s, const KitCtx& k);
Debuffs debuffs(const ChampionState& s, const KitCtx& k, const WorldUnits& u);

// --- summoner spells (champions/summoners.py) --------------------------------------------------------------------
namespace summoners {
using State = champions_summoners_State;
enum Spell : int32_t { FLASH = 4, TELEPORT = 12, IGNITE = 14, EXHAUST = 3, BARRIER = 21, HEAL = 7, GHOST = 6,
                       CLEANSE = 1, SMITE = 11 };
constexpr int QUEST_SLOT = 2;
enum Phase : int32_t { IDLE = 0, CHANNEL = 1, DASHING = 2 };
float ignite_total(float level);
float barrier_amount(float level);
float heal_amount(float level);
float ghost_ms(float level);
float tp_dash_time(float dist, bool unleashed);
// (C,) remaining Flash cooldown at ``now`` (the JAX default ``now=None`` is ``state.now``).
Arr<float> flash_cooldown(const State& s, float now);
inline Arr<float> flash_cooldown(const State& s) { return flash_cooldown(s, s.now); }
// step(state, ctx, units, *, request, now, dt, summoner_haste, can_cast, channel_interrupted, quest_complete,
// rooted): keyword arguments in the order world/phases/casts.py passes them (suppressed/nearsighted: None).
std::tuple<State, Effects, SummonerOut> step(State s, const Ctx& ctx, const WorldUnits& u, const CastOrder& request,
                                             float now, float dt, const Arr<float>& summoner_haste,
                                             const Arr<uint8_t>& can_cast, const Arr<uint8_t>& channel_interrupted,
                                             const Arr<uint8_t>& quest_complete, const Arr<uint8_t>& rooted);
}  // namespace summoners

}  // namespace lanesim::kits
