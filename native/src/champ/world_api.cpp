// PLACEHOLDER world_api implementations until the area ports are merged: each throws if called.
#include "world_api.hpp"

#include <stdexcept>

namespace lanesim::champ::api {

std::tuple<ChampionState, KitOut> K_cast(const ChampionState& s, const KitCtx& k, const WorldUnits& u, const CastOrder& o) { throw std::runtime_error("world_api::K_cast not merged yet"); }
std::tuple<ChampionState, KitOut> K_periodic(const ChampionState& s, const KitCtx& k, const WorldUnits& u) { throw std::runtime_error("world_api::K_periodic not merged yet"); }
std::tuple<ChampionState, KitOut> K_on_attack(const ChampionState& s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& l) { throw std::runtime_error("world_api::K_on_attack not merged yet"); }
std::tuple<ChampionState, KitOut> K_on_hit(const ChampionState& s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& l) { throw std::runtime_error("world_api::K_on_hit not merged yet"); }
std::tuple<ChampionState, KitOut> K_on_damage(const ChampionState& s, const KitCtx& k, const WorldUnits& u, const Report& r) { throw std::runtime_error("world_api::K_on_damage not merged yet"); }
ChampionState K_on_takedown(const ChampionState& s, const KitCtx& k, const WorldUnits& u, const Kills& kills) { throw std::runtime_error("world_api::K_on_takedown not merged yet"); }
ItemStats K_stats(const ChampionState& s, const KitCtx& k) { throw std::runtime_error("world_api::K_stats not merged yet"); }
KitDefense K_defense(const ChampionState& s, const KitCtx& k) { throw std::runtime_error("world_api::K_defense not merged yet"); }
KitAttackMods K_attack_mods(const ChampionState& s, const KitCtx& k) { throw std::runtime_error("world_api::K_attack_mods not merged yet"); }
Arr<uint8_t> K_ghosted(const ChampionState& s, const KitCtx& k) { throw std::runtime_error("world_api::K_ghosted not merged yet"); }
Debuffs K_debuffs(const ChampionState& s, const KitCtx& k, const WorldUnits& u) { throw std::runtime_error("world_api::K_debuffs not merged yet"); }
std::tuple<champions_summoners_State, Effects, SummonerOut> S_step( const champions_summoners_State& s, const Ctx& ctx, const WorldUnits& units, const CastOrder& request, float now, float dt, const Arr<float>& summoner_haste, const Arr<uint8_t>& can_cast, const Arr<uint8_t>& channel_interrupted, const Arr<uint8_t>& quest_complete, const Arr<uint8_t>& rooted) { throw std::runtime_error("world_api::S_step not merged yet"); }
Arr<float> S_flash_cooldown(const champions_summoners_State& s, float now) { throw std::runtime_error("world_api::S_flash_cooldown not merged yet"); }
float S_flash_range() { throw std::runtime_error("world_api::S_flash_range not merged yet"); }
EconomyOut E_economy_step(const EconomyState& s, const EconomyInputs& in) { throw std::runtime_error("world_api::E_economy_step not merged yet"); }
int E_skill_points(int level) { throw std::runtime_error("world_api::E_skill_points not merged yet"); }
int E_max_rank(int level, bool ultimate) { throw std::runtime_error("world_api::E_max_rank not merged yet"); }
float E_decimal_level(float xp, int cap) { throw std::runtime_error("world_api::E_decimal_level not merged yet"); }
bool E_in_fountain(float x, float y, float fx, float fy) { throw std::runtime_error("world_api::E_in_fountain not merged yet"); }
void E_fountain_regen(float& hp, float max_hp, float& mana, float max_mana, bool in_fountain, float t0, float t1, bool homeguard) { throw std::runtime_error("world_api::E_fountain_regen not merged yet"); }
float E_recall_channel() { throw std::runtime_error("world_api::E_recall_channel not merged yet"); }
int E_quest_level_cap() { throw std::runtime_error("world_api::E_quest_level_cap not merged yet"); }
int E_level_cap() { throw std::runtime_error("world_api::E_level_cap not merged yet"); }
bool REG_in_quest_lane(float x, float y, int lane) { throw std::runtime_error("world_api::REG_in_quest_lane not merged yet"); }
void REG_homeguard_flags(const Arr<float>& x, const Arr<float>& y, const Arr<int32_t>& team, float now, const WorldUnits& units, const Arr<int32_t>& structure_lane, const Arr<int32_t>& minion_lane, Arr<uint8_t>& reached, Arr<uint8_t>& in_jungle) { throw std::runtime_error("world_api::REG_homeguard_flags not merged yet"); }
bool REG_in_river(float x, float y) { throw std::runtime_error("world_api::REG_in_river not merged yet"); }
Arr<int32_t> I_owned_counts(const Inventory& inv) { throw std::runtime_error("world_api::I_owned_counts not merged yet"); }
ItemStats I_inventory_stats(const Inventory& inv) { throw std::runtime_error("world_api::I_inventory_stats not merged yet"); }
bool I_in_shop_area(float x, float y, int team, bool dead) { throw std::runtime_error("world_api::I_in_shop_area not merged yet"); }
int I_buy(Inventory& inv1, float& gold, int row, bool can_shop, int level, bool is_ranged, float now, Arr<float>& group_cd_until, bool* ok) { throw std::runtime_error("world_api::I_buy not merged yet"); }
int I_sell(Inventory& inv1, float& gold, int slot, bool can_shop, bool* ok) { throw std::runtime_error("world_api::I_sell not merged yet"); }
void I_replace_item(Inventory& inv1, int from_row, int to_row, bool enabled) { throw std::runtime_error("world_api::I_replace_item not merged yet"); }
void I_consume_one(Inventory& inv1, int slot, bool enabled) { throw std::runtime_error("world_api::I_consume_one not merged yet"); }
std::tuple<Wards, WardEvents> W_ward_step(const Wards& w, float now, float dt, const WardRequest& req, const Arr<float>& x, const Arr<float>& y, const Arr<int32_t>& team, const Arr<uint8_t>& alive, const Arr<int32_t>& level, const Arr<int32_t>& trinket_id, const Arr<int32_t>& control_count, const Arr<uint8_t>& can_use, const Arr<float>& trinket_haste, const Arr<int32_t>& hits, const Arr<int32_t>& hitter, const Arr<int32_t>& rune_pages, const Arr<uint8_t>& ward_visible) { throw std::runtime_error("world_api::W_ward_step not merged yet"); }
std::tuple<WardView, Arr<float>> W_ward_view(const Wards& w, float now, const Arr<float>& x, const Arr<float>& y, const Arr<int32_t>& team, const Arr<uint8_t>& alive, const Arr<int32_t>& level) { throw std::runtime_error("world_api::W_ward_view not merged yet"); }

}  // namespace lanesim::champ::api
