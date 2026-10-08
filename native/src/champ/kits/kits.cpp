// Kit dispatch (champions/__init__.py): every kit's hook runs for every holder (each gates on champion_id) and the
// results merge in KITS order (garen, jax). Also the area-local helpers of kits.hpp.
#include "kits.hpp"

#include "../stats.hpp"

namespace lanesim::kits {

using namespace champ;

namespace detail {

std::vector<float> table(const std::string& key) { return data::table(key); }

ItemStats scalar_zero_stats() {
    ItemStats s;
    for (Arr<float>* f : stats::fields(s)) f->assign(1, 0.f);
    return s;
}

}  // namespace detail

namespace {

// (C, 4) UNIT_TARGET_RANGE rows per kit (consts kits.<name>.UNIT_TARGET_RANGE).
const std::vector<float>& target_range(int kit) {
    static const std::vector<float> g = detail::table("kits.garen.UNIT_TARGET_RANGE"),
                                    j = detail::table("kits.jax.UNIT_TARGET_RANGE");
    return kit == 0 ? g : j;
}

KitOut merged(const KitOut& a, const KitOut& b, int c, int n) {
    KitOut acc = no_out(c, n);
    merge_out_into(acc, a, c, n);
    merge_out_into(acc, b, c, n);
    return acc;
}

// A 0-d ItemStats field broadcast to (C,).
Arr<float> broadcast(const Arr<float>& v, size_t c) { return v.size() == c ? v : Arr<float>(c, v.size() ? v[0] : 0.f); }

}  // namespace

Arr<float> unit_target_ranges(const Arr<int32_t>& ids) {
    Arr<float> out(ids.size() * 4, 0.f);
    const int kit_ids[2] = {garen::ID, jax::ID};
    for (int kit = 0; kit < 2; ++kit)
        for (size_t h = 0; h < ids.size(); ++h)
            if (ids[h] == kit_ids[kit])
                for (int s = 0; s < 4; ++s) out[h * 4 + s] = target_range(kit)[s];
    return out;
}

std::tuple<ChampionState, KitOut> cast(ChampionState s, const KitCtx& k, const WorldUnits& u, const CastOrder& order) {
    auto [g, go] = garen::cast(s.garen, k, u, order);
    auto [j, jo] = jax::cast(s.jax, k, u, order);
    s.garen = g, s.jax = j;
    return {s, merged(go, jo, (int)k.unit.size(), (int)u.x.size())};
}

std::tuple<ChampionState, KitOut> periodic(ChampionState s, const KitCtx& k, const WorldUnits& u) {
    auto [g, go] = garen::periodic(s.garen, k, u);
    auto [j, jo] = jax::periodic(s.jax, k, u);
    s.garen = g, s.jax = j;
    return {s, merged(go, jo, (int)k.unit.size(), (int)u.x.size())};
}

std::tuple<ChampionState, KitOut> on_attack(ChampionState s, const KitCtx& k, const WorldUnits& u,
                                            const AttackLaunch& launch) {
    auto [g, go] = garen::on_attack(s.garen, k, u, launch);
    auto [j, jo] = jax::on_attack(s.jax, k, u, launch);
    s.garen = g, s.jax = j;
    return {s, merged(go, jo, (int)k.unit.size(), (int)u.x.size())};
}

Arr<uint8_t> dodging_units(const ChampionState& s, const KitCtx& k, int n_units) {
    Arr<uint8_t> dodge = jax::dodging(s.jax, k), out(n_units, 0);       // only Jax defines ``dodging``
    for (size_t h = 0; h < k.unit.size(); ++h) {
        int j = k.unit[h];
        if (j >= 0 && j < n_units) out[j] = std::max(out[j], dodge[h]);
    }
    return out;
}

std::tuple<ChampionState, KitOut> on_hit(ChampionState s, const KitCtx& k, const WorldUnits& u,
                                         const AttackLaunch& launch) {
    Arr<uint8_t> dodge = dodging_units(s, k, (int)u.x.size());
    auto [g, go] = garen::on_hit(s.garen, k, u, launch, dodge);
    auto [j, jo] = jax::on_hit(s.jax, k, u, launch, dodge);
    s.garen = g, s.jax = j;
    return {s, merged(go, jo, (int)k.unit.size(), (int)u.x.size())};
}

std::tuple<ChampionState, KitOut> on_damage(ChampionState s, const KitCtx& k, const WorldUnits& u, const Report& report) {
    auto [g, go] = garen::on_damage(s.garen, k, u, report);
    auto [j, jo] = jax::on_damage(s.jax, k, u, report);
    s.garen = g, s.jax = j;
    return {s, merged(go, jo, (int)k.unit.size(), (int)u.x.size())};
}

ChampionState on_takedown(ChampionState s, const KitCtx& k, const WorldUnits& u, const Kills& kills) {
    s.garen = garen::on_takedown(s.garen, k, u, kills);
    s.jax = jax::on_takedown(s.jax, k, u, kills);
    return s;
}

ItemStats stats(const ChampionState& s, const KitCtx& k) {
    size_t c = k.unit.size();
    ItemStats a = garen::stats(s.garen, k), b = jax::stats(s.jax, k);
    for (auto* x : {&a, &b})
        for (Arr<float>* f : stats::fields(*x)) *f = broadcast(*f, c);
    return stats::combine2(a, b);
}

KitDefense defense(const ChampionState& s, const KitCtx& k) {
    size_t c = k.unit.size();
    KitDefense out = neutral_kit_defense((int)c);
    KitDefense parts[2] = {garen::defense(s.garen, k), jax::defense(s.jax, k)};
    for (const KitDefense& d : parts)
        for (size_t h = 0; h < c; ++h) {
            out.received_mult[h] = out.received_mult[h] * d.received_mult[h];
            out.dodge_basic[h] = out.dodge_basic[h] | d.dodge_basic[h];
            out.aoe_received_mult[h] = out.aoe_received_mult[h] * d.aoe_received_mult[h];
            out.tenacity_bonus[h] = 1.f - (1.f - out.tenacity_bonus[h]) * (1.f - d.tenacity_bonus[h]);
        }
    return out;
}

KitAttackMods attack_mods(const ChampionState& s, const KitCtx& k) {
    size_t c = k.unit.size();
    KitAttackMods out = neutral_attack_mods((int)c);
    KitAttackMods parts[2] = {garen::attack_mods(s.garen, k), jax::attack_mods(s.jax, k)};
    for (const KitAttackMods& m : parts)       // core.combine_attack_mods
        for (size_t h = 0; h < c; ++h) {
            out.extra_range[h] = out.extra_range[h] + m.extra_range[h];
            out.attack_reset[h] = out.attack_reset[h] | m.attack_reset[h];
            out.cannot_attack[h] = out.cannot_attack[h] | m.cannot_attack[h];
            out.cannot_crit[h] = out.cannot_crit[h] | m.cannot_crit[h];
            out.windup[h] = std::max(out.windup[h], m.windup[h]);
            out.period[h] = std::max(out.period[h], m.period[h]);
            out.uncancellable[h] = out.uncancellable[h] | m.uncancellable[h];
        }
    return out;
}

Arr<uint8_t> ghosted(const ChampionState& s, const KitCtx& k) {
    return garen::ghosted(s.garen, k);                // only Garen defines ``ghosted``
}

Debuffs debuffs(const ChampionState& s, const KitCtx& k, const WorldUnits& u) {
    Debuffs g = garen::debuffs(s.garen, k, u), j = jax::debuffs(s.jax, k, u);
    return combine_debuffs({&g, &j}, (int)u.x.size());
}

}  // namespace lanesim::kits
