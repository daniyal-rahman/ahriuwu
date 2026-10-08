// The world's calls into the area ports (kits, summoners, economy, regions, inventory, wards): thin adapters
// from the world_api signatures to the ported JAX functions.
#include "world_api.hpp"

#include "econ/econ.hpp"
#include "kits/kits.hpp"

namespace lanesim::champ::api {

// champions/__init__.py
std::tuple<ChampionState, KitOut> K_cast(const ChampionState& s, const KitCtx& k, const WorldUnits& u, const CastOrder& o) {
    return kits::cast(s, k, u, o);
}
std::tuple<ChampionState, KitOut> K_periodic(const ChampionState& s, const KitCtx& k, const WorldUnits& u) {
    return kits::periodic(s, k, u);
}
std::tuple<ChampionState, KitOut> K_on_attack(const ChampionState& s, const KitCtx& k, const WorldUnits& u,
                                              const AttackLaunch& l) {
    return kits::on_attack(s, k, u, l);
}
std::tuple<ChampionState, KitOut> K_on_hit(const ChampionState& s, const KitCtx& k, const WorldUnits& u,
                                           const AttackLaunch& l) {
    return kits::on_hit(s, k, u, l);
}
std::tuple<ChampionState, KitOut> K_on_damage(const ChampionState& s, const KitCtx& k, const WorldUnits& u,
                                              const Report& r) {
    return kits::on_damage(s, k, u, r);
}
ChampionState K_on_takedown(const ChampionState& s, const KitCtx& k, const WorldUnits& u, const Kills& kills) {
    return kits::on_takedown(s, k, u, kills);
}
ItemStats K_stats(const ChampionState& s, const KitCtx& k) { return kits::stats(s, k); }
KitDefense K_defense(const ChampionState& s, const KitCtx& k) { return kits::defense(s, k); }
KitAttackMods K_attack_mods(const ChampionState& s, const KitCtx& k) { return kits::attack_mods(s, k); }
Arr<uint8_t> K_ghosted(const ChampionState& s, const KitCtx& k) { return kits::ghosted(s, k); }
Debuffs K_debuffs(const ChampionState& s, const KitCtx& k, const WorldUnits& u) { return kits::debuffs(s, k, u); }

// champions/summoners.py
std::tuple<champions_summoners_State, Effects, SummonerOut> S_step(
    const champions_summoners_State& s, const Ctx& ctx, const WorldUnits& units, const CastOrder& request, float now,
    float dt, const Arr<float>& summoner_haste, const Arr<uint8_t>& can_cast, const Arr<uint8_t>& channel_interrupted,
    const Arr<uint8_t>& quest_complete, const Arr<uint8_t>& rooted) {
    return kits::summoners::step(s, ctx, units, request, now, dt, summoner_haste, can_cast, channel_interrupted,
                                 quest_complete, rooted);
}
Arr<float> S_flash_cooldown(const champions_summoners_State& s, float now) { return kits::summoners::flash_cooldown(s, now); }
float S_flash_range() { return data::f("world.flash_range"); }

// economy.py, map/regions.py
EconomyOut E_economy_step(const EconomyState& s, const EconomyInputs& in) { return econ::economy_step(s, in); }
int E_skill_points(int level) { return econ::skill_points(level); }
int E_max_rank(int level, bool ultimate) { return econ::max_rank(level, ultimate); }
float E_decimal_level(float xp, int cap) { return econ::decimal_level(xp, cap); }
bool E_in_fountain(float x, float y, float fx, float fy) { return econ::in_fountain(x, y, fx, fy); }
void E_fountain_regen(float& hp, float max_hp, float& mana, float max_mana, bool in_fountain, float t0, float t1,
                      bool homeguard) {
    std::tie(hp, mana) = econ::fountain_regen(hp, max_hp, mana, max_mana, in_fountain, t0, t1, homeguard);
}
bool REG_in_quest_lane(float x, float y, int lane) { return econ::in_quest_lane(x, y, lane); }
void REG_homeguard_flags(const Arr<float>& x, const Arr<float>& y, const Arr<int32_t>& team, float now,
                         const WorldUnits& units, const Arr<int32_t>& structure_lane, const Arr<int32_t>& minion_lane,
                         Arr<uint8_t>& reached, Arr<uint8_t>& in_jungle) {
    std::tie(reached, in_jungle) = econ::homeguard_flags(x, y, team, now, units, structure_lane, minion_lane);
}
bool REG_in_river(float x, float y) { return econ::in_river(x, y); }

// items/inventory.py
Arr<int32_t> I_owned_counts(const Inventory& inv) { return econ::owned_counts(inv); }
ItemStats I_inventory_stats(const Inventory& inv) { return econ::inventory_stats(inv); }
bool I_in_shop_area(float x, float y, int team, bool dead) { return econ::in_shop_area(x, y, team, dead); }
int I_buy(Inventory& inv1, float& gold, int row, bool can_shop, int level, bool is_ranged, float now,
          Arr<float>& group_cd_until, bool* ok) {
    econ::ShopResult r = econ::buy(inv1, gold, row, can_shop, level, is_ranged, now, group_cd_until, {});
    inv1 = r.inv, gold = r.gold, group_cd_until = r.group_cd_until, *ok = r.ok;
    return r.code;
}
int I_sell(Inventory& inv1, float& gold, int slot, bool can_shop, bool* ok) {
    econ::ShopResult r = econ::sell(inv1, gold, slot, can_shop);
    inv1 = r.inv, gold = r.gold, *ok = r.ok;
    return r.code;
}
void I_replace_item(Inventory& inv1, int from_row, int to_row, bool enabled) {
    inv1 = econ::replace_item(inv1, from_row, to_row, enabled);
}
void I_consume_one(Inventory& inv1, int slot, bool enabled) { inv1 = econ::consume_one(inv1, slot, enabled); }

// wards.py
std::tuple<Wards, WardEvents> W_ward_step(const Wards& w, float now, float dt, const WardRequest& req,
                                          const Arr<float>& x, const Arr<float>& y, const Arr<int32_t>& team,
                                          const Arr<uint8_t>& alive, const Arr<int32_t>& level,
                                          const Arr<int32_t>& trinket_id, const Arr<int32_t>& control_count,
                                          const Arr<uint8_t>& can_use, const Arr<float>& trinket_haste,
                                          const Arr<int32_t>& hits, const Arr<int32_t>& hitter,
                                          const Arr<int32_t>& rune_pages, const Arr<uint8_t>& ward_visible) {
    return econ::ward_step(w, now, dt, req, x, y, team, alive, level, trinket_id, control_count, can_use,
                           trinket_haste, hits, hitter, rune_pages, ward_visible);
}
std::tuple<WardView, Arr<float>> W_ward_view(const Wards& w, float now, const Arr<float>& x, const Arr<float>& y,
                                             const Arr<int32_t>& team, const Arr<uint8_t>& alive,
                                             const Arr<int32_t>& level) {
    return econ::ward_view(w, now, x, y, team, alive, level);
}

}  // namespace lanesim::champ::api
