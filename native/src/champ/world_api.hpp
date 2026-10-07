// The module functions the full tick calls, with the JAX dispatch-level signatures (champions/__init__.py,
// champions/summoners.py, economy.py, wards.py, items/inventory.py, map/regions.py). Implemented in world_api.cpp
// over the area ports (kits, econ).
#pragma once
#include <tuple>

#include "core.hpp"

namespace lanesim::champ::api {

// champions/__init__.py
std::tuple<ChampionState, KitOut> K_cast(const ChampionState& s, const KitCtx& k, const WorldUnits& u, const CastOrder& o);
std::tuple<ChampionState, KitOut> K_periodic(const ChampionState& s, const KitCtx& k, const WorldUnits& u);
std::tuple<ChampionState, KitOut> K_on_attack(const ChampionState& s, const KitCtx& k, const WorldUnits& u,
                                              const AttackLaunch& l);
std::tuple<ChampionState, KitOut> K_on_hit(const ChampionState& s, const KitCtx& k, const WorldUnits& u,
                                           const AttackLaunch& l);
std::tuple<ChampionState, KitOut> K_on_damage(const ChampionState& s, const KitCtx& k, const WorldUnits& u,
                                              const Report& r);
ChampionState K_on_takedown(const ChampionState& s, const KitCtx& k, const WorldUnits& u, const Kills& kills);
ItemStats K_stats(const ChampionState& s, const KitCtx& k);
KitDefense K_defense(const ChampionState& s, const KitCtx& k);
KitAttackMods K_attack_mods(const ChampionState& s, const KitCtx& k);
Arr<uint8_t> K_ghosted(const ChampionState& s, const KitCtx& k);
Debuffs K_debuffs(const ChampionState& s, const KitCtx& k, const WorldUnits& u);

// champions/summoners.py
std::tuple<champions_summoners_State, Effects, SummonerOut> S_step(
    const champions_summoners_State& s, const Ctx& ctx, const WorldUnits& units, const CastOrder& request, float now,
    float dt, const Arr<float>& summoner_haste, const Arr<uint8_t>& can_cast, const Arr<uint8_t>& channel_interrupted,
    const Arr<uint8_t>& quest_complete, const Arr<uint8_t>& rooted);
Arr<float> S_flash_cooldown(const champions_summoners_State& s, float now);
float S_flash_range();

// economy.py, role_quest.py, map/regions.py
EconomyOut E_economy_step(const EconomyState& s, const EconomyInputs& in);
int E_skill_points(int level);
int E_max_rank(int level, bool ultimate);
float E_decimal_level(float xp, int cap);
bool E_in_fountain(float x, float y, float fx, float fy);
void E_fountain_regen(float& hp, float max_hp, float& mana, float max_mana, bool in_fountain, float t0, float t1,
                      bool homeguard);
float E_recall_channel();
int E_quest_level_cap();
int E_level_cap();
bool REG_in_quest_lane(float x, float y, int lane);
void REG_homeguard_flags(const Arr<float>& x, const Arr<float>& y, const Arr<int32_t>& team, float now,
                         const WorldUnits& units, const Arr<int32_t>& structure_lane, const Arr<int32_t>& minion_lane,
                         Arr<uint8_t>& reached, Arr<uint8_t>& in_jungle);
bool REG_in_river(float x, float y);

// items/inventory.py
Arr<int32_t> I_owned_counts(const Inventory& inv);                          // (C, I)
ItemStats I_inventory_stats(const Inventory& inv);
bool I_in_shop_area(float x, float y, int team, bool dead);
// buy/sell for one champion (rows of ``inv`` are that champion's 7 slots): returns the result code; ``ok`` set.
int I_buy(Inventory& inv1, float& gold, int row, bool can_shop, int level, bool is_ranged, float now,
          Arr<float>& group_cd_until, bool* ok);
int I_sell(Inventory& inv1, float& gold, int slot, bool can_shop, bool* ok);
void I_replace_item(Inventory& inv1, int from_row, int to_row, bool enabled);
void I_consume_one(Inventory& inv1, int slot, bool enabled);

// wards.py
std::tuple<Wards, WardEvents> W_ward_step(const Wards& w, float now, float dt, const WardRequest& req,
                                          const Arr<float>& x, const Arr<float>& y, const Arr<int32_t>& team,
                                          const Arr<uint8_t>& alive, const Arr<int32_t>& level,
                                          const Arr<int32_t>& trinket_id, const Arr<int32_t>& control_count,
                                          const Arr<uint8_t>& can_use, const Arr<float>& trinket_haste,
                                          const Arr<int32_t>& hits, const Arr<int32_t>& hitter,
                                          const Arr<int32_t>& rune_pages, const Arr<uint8_t>& ward_visible);
std::tuple<WardView, Arr<float>> W_ward_view(const Wards& w, float now, const Arr<float>& x, const Arr<float>& y,
                                             const Arr<int32_t>& team, const Arr<uint8_t>& alive,
                                             const Arr<int32_t>& level);

}  // namespace lanesim::champ::api
