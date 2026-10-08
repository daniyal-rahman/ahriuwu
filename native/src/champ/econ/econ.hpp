// Economy, Top role quest, wards/trinkets, inventory/shop and the map-region queries of the DEATH phase, ported
// from lanerl_jax/modern/economy.py, role_quest.py, wards.py, items/inventory.py and map/regions.py. Shapes follow
// JAX: (C,) per champion holder, (N,) per world unit, (S,) ward slots, (C, N) row-major; per-champion inventories
// are an ``Inventory`` with 7-slot ``item``/``stack``. Tables come from native/python/consts/econ_*.py.
#pragma once
#include <cstdint>
#include <tuple>

#include "../core.hpp"

namespace lanesim::econ {

using champ::C;

// ---- economy.py -----------------------------------------------------------------------------------------------
constexpr int LEVEL_CAP = 18, QUEST_LEVEL_CAP = 20, SKILL_POINT_LEVELS = 18;
constexpr float BIG = 1e9f;
constexpr float RECALL_CHANNEL = 8.f, RECALL_CAST = .5f, RECALL_DAMAGE_GRACE = .1f;

float ambient_payments(float t0, float t1);
int level_for_xp(float xp, int cap = LEVEL_CAP);
float decimal_level(float xp, int cap = LEVEL_CAP);
inline int skill_points(int level) { return std::min(level, SKILL_POINT_LEVELS); }
int max_rank(int level, bool ultimate = false);
bool in_fountain(float x, float y, float fountain_x, float fountain_y);
// fountain_regen: (hp, mana) after the fountain (and Homeguard) pulses in (t0, t1].
std::pair<float, float> fountain_regen(float hp, float max_hp, float mana, float max_mana, bool in_fountain,
                                       float t0, float t1, bool homeguard = false);
float starting_gold();
EconomyState init_economy(int n_champions, int n_units, const Arr<int32_t>& roles);
// One tick of gold, XP, bounty, death and quest bookkeeping (§13). Reads ``inp.report.packets`` only.
EconomyOut economy_step(const EconomyState& state, const EconomyInputs& inp);

// ---- role_quest.py --------------------------------------------------------------------------------------------
enum Role { ROLE_NONE, ROLE_TOP, ROLE_JUNGLE, ROLE_MID, ROLE_BOT, ROLE_SUPPORT };
QuestState init_quest(const Arr<int32_t>& roles);
float out_of_lane_mult(float points);
QuestStep quest_step(const QuestState& state, const QuestEvents& ev, float now, float dt, const Arr<uint8_t>& in_lane,
                     const Arr<uint8_t>& alive, const Arr<int32_t>& level, const Arr<uint8_t>& recalled);
Arr<float> xp_bonus(const QuestState& state);
Arr<float> takedown_xp(const QuestState& state);
Arr<float> minion_penalty(const QuestState& state, const Arr<int32_t>& level, const Arr<uint8_t>& minion_in_lane);
float unleashed_tp_cooldown(float level, bool quest_complete = false);
float tp_arrival_shield(float max_hp, bool quest_complete, bool has_teleport);

// ---- wards.py -------------------------------------------------------------------------------------------------
constexpr int MAX_WARDS_PER_TEAM = 8, S = 2 * MAX_WARDS_PER_TEAM;
Wards init_wards(int n_champions, const Arr<int32_t>& trinket_ids);
// ``rune_pages`` (C, R) and ``ward_visible`` (2, S) may be empty (None); the WardGrid is world data (consts).
std::tuple<Wards, WardEvents> ward_step(const Wards& w, float now, float dt, const WardRequest& request,
                                        const Arr<float>& x, const Arr<float>& y, const Arr<int32_t>& team,
                                        const Arr<uint8_t>& alive, const Arr<int32_t>& level,
                                        const Arr<int32_t>& trinket_id, const Arr<int32_t>& control_count,
                                        const Arr<uint8_t>& can_use, const Arr<float>& trinket_haste,
                                        const Arr<int32_t>& hits, const Arr<int32_t>& hitter,
                                        const Arr<int32_t>& rune_pages, const Arr<uint8_t>& ward_visible);
std::tuple<WardView, Arr<float>> ward_view(const Wards& w, float now, const Arr<float>& x, const Arr<float>& y,
                                           const Arr<int32_t>& team, const Arr<uint8_t>& alive,
                                           const Arr<int32_t>& level);
// vision_kwargs: the optional vision.visibility inputs (N,).
struct VisionKw {
    Arr<float> radius{};
    Arr<uint8_t> stealthed{};
    Arr<float> true_sight{};
    Arr<uint8_t> unobstructed{};
    Arr<uint8_t> exposed{};
    template <class F> void visit(F&& f) { f(radius); f(stealthed); f(true_sight); f(unobstructed); f(exposed); }
};
VisionKw vision_kwargs(const WardView& view, const Arr<float>& oracle, const Arr<int32_t>& kind,
                       const Arr<int32_t>& sub, const Arr<uint8_t>& alive, int ward_start);

// ---- items/inventory.py (one champion's 7-slot Inventory) -----------------------------------------------------
constexpr int EMPTY = -1, N_SLOTS = 7, TRINKET_SLOT = 6;
enum ShopCode { OK, ERR_NOT_IN_SHOP, ERR_NOT_PURCHASABLE, ERR_LEVEL, ERR_GOLD, ERR_GROUP, ERR_NO_SLOT, ERR_COOLDOWN,
                ERR_EMPTY_SLOT, ERR_NOT_SELLABLE };
struct ShopResult {
    Inventory inv{};
    float gold{};
    Arr<float> group_cd_until{};   // empty for sell (None)
    uint8_t ok{};
    int32_t code{};
    float spent{};
    template <class F> void visit(F&& f) { f(inv); f(gold); f(group_cd_until); f(ok); f(code); f(spent); }
};
// ``ids`` (C, K) item ids, 0 = none (the host lists padded); ``trinket``: Stealth Ward in the trinket slot.
Inventory inventory_from_ids(const Arr<int32_t>& ids, int k, bool trinket = true);
Arr<int32_t> owned_counts(const Inventory& inv);           // (..., 7) -> (..., I)
ItemStats inventory_stats(const Inventory& inv);            // (..., 7) -> ItemStats of (...,)
bool in_shop_area(float x, float z, int team, bool dead);
ShopResult buy(const Inventory& inv, float gold, int row, bool can_shop, int level, bool is_ranged, float now,
               const Arr<float>& group_cd_until, const Arr<uint8_t>& buff_currency);
ShopResult sell(const Inventory& inv, float gold, int slot, bool can_shop);
Inventory replace_item(const Inventory& inv, int from_row, int to_row, bool enabled = true);
Inventory consume_one(const Inventory& inv, int slot, bool enabled = true);

// ---- map/regions.py (the queries the DEATH phase and views make) ----------------------------------------------
int region_of(float x, float y);
int lane_of(float x, float y);
bool in_quest_lane(float x, float y, int quest_lane);
bool in_jungle(float x, float y);
bool in_river(float x, float y);
float lane_progress(float x, float y, int team, int lane);
// (reached_endpoint, in_jungle) (C,) for Homeguard.
std::tuple<Arr<uint8_t>, Arr<uint8_t>> homeguard_flags(const Arr<float>& x, const Arr<float>& y,
                                                       const Arr<int32_t>& team, float now, const WorldUnits& units,
                                                       const Arr<int32_t>& structure_lane,
                                                       const Arr<int32_t>& minion_lane);

}  // namespace lanesim::econ
