// items/effects/boots.py. Reachable for Garen/Jax: Berserker's, Swiftness, Mercury's, Gunmetal (stats only),
// Plated Steelcaps, Swiftmarch, Chainlaced Crushers, Armored Advance. Slay boots, Ionian, Crimson Lucidity and
// Immortal Path are not holdable; the Slay stack clamp/reset that runs regardless is kept.
#include "../marshal.hpp"
#include "items.hpp"

namespace lanesim::items::boots {

namespace {
constexpr int STEELCAPS = 3047, SWIFTMARCH = 3170, CHAINLACED = 3173, ARMORED = 3174;

float K(const char* name) { return data::f(std::string("items.boots.") + name); }
}  // namespace

// boots.stats: Swiftmarch Noxian Fervor (Slay omnivamp, summoner haste, Immortal Path, Crimson not holdable)
ItemStats stats(const State& s, const Owned& own, const Ctx& ctx) {
    static const float af_ratio = K("swiftmarch_af"), ad_per_af = K("adaptive_ad_per_af");
    size_t c = ctx.unit.size();
    ItemStats o = default_stats();
    o.omnivamp = o.summoner_haste = o.incoming_heal = o.attack_damage = o.ability_power = o.percent_move_speed =
        zeros_c(c);
    for (size_t h = 0; h < c; ++h) {
        float af = holds(own, SWIFTMARCH, h) ? af_ratio * ctx.move_speed[h] : 0.f;
        bool to_ad = ctx.bonus_ad[h] >= ctx.ap[h];
        o.attack_damage[h] = to_ad ? ad_per_af * af : 0.f;
        o.ability_power[h] = to_ad ? 0.f : af;
    }
    return o;
}

// boots.dealt_amp: Immortal Path (not holdable)
Arr<float> dealt_amp(const State& s, const Owned& own, const Ctx& ctx, const Units& u) {
    return Arr<float>(ctx.unit.size() * u.x.size(), 0.f);
}

// boots.defense: Plating
HolderDefense defense(const State& s, const Owned& own, const Ctx& ctx) {
    static const float steel = K("steelcaps_mult"), armored = K("armored_mult");
    int c = (int)ctx.unit.size();
    HolderDefense d = neutral_defense(c);
    for (int h = 0; h < c; ++h)
        d.basic_attack_mult[h] = (holds(own, STEELCAPS, h) ? steel : 1.f) * (holds(own, ARMORED, h) ? armored : 1.f);
    return d;
}

// boots.on_damage: Noxian Endurance / Persistence shields (Crimson Lucidity not holdable)
std::tuple<State, Effects> on_damage(State s, const Owned& own, const Ctx& ctx, const Units& u, const Report& r) {
    static const std::vector<float> nox[2] = {data::table("items.boots.noxian_armored"),
                                              data::table("items.boots.noxian_chainlaced")};
    static const int items[2] = {ARMORED, CHAINLACED};
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    const Packets& p = r.packets;
    size_t np = size(p);
    Arr<float> amount(c, 0.f), duration(c, 0.f);
    Arr<int32_t> kind(c, 0);
    for (int h = 0; h < c; ++h) {
        bool ready = ctx.alive[h] && ctx.now >= s.noxian_cd_until[h];
        float cd_until = s.noxian_cd_until[h];
        for (int q = 0; q < 2; ++q) {
            const std::vector<float>& v = nox[q];   // l1, per, at, bonus ratio, cd, duration, dtype, kind
            bool took = false;
            for (size_t k = 0; k < np; ++k) {
                int src = clampi(p.src[k], 0, n - 1);
                bool from_enemy_champ = p.valid[k] && r.resolved.final[k] > 0.f && u.cls[src] == CLASS_CHAMPION
                                        && p.src[k] != p.dst[k];
                took = took || (p.dst[k] == ctx.unit[h] && u.team[src] != ctx.team[h] && from_enemy_champ
                                 && p.dtype[k] == (int)v[6]);
            }
            if (!(ready && holds(own, items[q], h) && took)) continue;
            amount[h] = level_bp(v[0], v[1], v[2], ctx.level[h]) + v[3] * (ctx.max_hp[h] - ctx.base_hp[h]);
            kind[h] = (int)v[7];
            duration[h] = v[5];
            cd_until = ctx.now + v[4];
        }
        s.noxian_cd_until[h] = cd_until;
    }
    Effects e = no_effects(c, n);
    e.shields = shield_grants(amount, SHIELD_ALL, 0.f);
    for (int h = 0; h < c; ++h) e.shields.kind[h] = kind[h], e.shields.duration[h] = duration[h];
    return {s, e};
}

// boots.on_takedown: Slay stacks (no Slay boot holdable: the clamp still runs)
std::tuple<State, Effects> on_takedown(State s, const Owned& own, const Ctx& ctx, const Units& u, const Kills& k) {
    static const float slay_max = K("slay_max");
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    for (int h = 0; h < c; ++h) s.slay_stacks[h] = std::min(s.slay_stacks[h] + 0.f, slay_max);
    return {s, no_effects(c, n)};
}

// boots.periodic: selling every Slay boot clears the stacks
std::tuple<State, Effects> periodic(State s, const Owned& own, const Ctx& ctx, const Units& u) {
    int c = (int)ctx.unit.size(), n = (int)u.x.size();
    for (int h = 0; h < c; ++h) s.slay_stacks[h] = 0.f;
    return {s, no_effects(c, n)};
}

LANESIM_TEST(items_boots_stats, "items.boots.stats", stats);
LANESIM_TEST(items_boots_dealt_amp, "items.boots.dealt_amp", dealt_amp);
LANESIM_TEST(items_boots_defense, "items.boots.defense", defense);
LANESIM_TEST(items_boots_on_damage, "items.boots.on_damage", on_damage);
LANESIM_TEST(items_boots_on_takedown, "items.boots.on_takedown", on_takedown);
LANESIM_TEST(items_boots_periodic, "items.boots.periodic", periodic);

}  // namespace lanesim::items::boots
