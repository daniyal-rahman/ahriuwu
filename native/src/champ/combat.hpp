// combat.combat_tick and the item-side folding (items/effects/runtime.py).
#pragma once
#include "core.hpp"

namespace lanesim::champ {

// combat.CombatTickOut (``transforms`` is the (from_row, to_row, do) tuple).
struct CombatTickOut {
    CombatState state{};
    Arr<float> hp{}, max_hp{};
    Shields shields{};
    UnitStatus status{};
    int32_t packet_overflow = 0;
    Report report{}, follow_up{};
    Effects effects{};
    ActiveOut active{};
    ItemStats dynamic_stats{};
    Arr<int32_t> transform_from{}, transform_to{};
    Arr<uint8_t> transform_do{};
    Arr<int32_t> consume_row{};
    RuneOutputs rune_outputs{};
    RuneEvents events{};
    template <class F> void visit(F&& f) {
        f(state); f(hp); f(max_hp); f(shields); f(status); f(packet_overflow); f(report); f(follow_up); f(effects);
        f(active); f(dynamic_stats); f(transform_from); f(transform_to); f(transform_do); f(consume_row);
        f(rune_outputs); f(events);
    }
};

Defense fold_defense(const Defense& base, const Ctx& ctx, const HolderDefense& holder, const Debuffs& debuffs,
                     float unused, const Arr<float>& shield_power, const Arr<float>& incoming_heal);
Offense fold_offense(const Offense& base, const Ctx& ctx, const ItemStats& st);
Report resolve_tick(const Packets& p, const Offense& off, const Defense& dfn, const Arr<float>& hp,
                    const Arr<float>& max_hp, const Shields& shields, float now, const Vamp& vamp);
void apply_effects(const Effects& eff, const Ctx& ctx, Arr<float>& hp, const Arr<float>& max_hp, Shields& shields,
                   UnitStatus& status, const Arr<float>& heal_power, const Arr<float>& incoming_heal,
                   const Arr<float>* vamp_heal, const Arr<float>* heal_mult);
CombatClocks update_clocks(const CombatClocks& clocks, const Report& report, const Ctx& ctx, const Units& units,
                           const CC* cc);
CombatTickOut combat_tick(CombatState state, Owned own, Arr<int32_t> page, Ctx ctx, Units units, Attack attack,
                          Cast cast, Arr<int32_t> request, Packets base_packets, Offense base_offense,
                          Defense base_defense, Arr<float> hp, Arr<float> max_hp, Shields shields, UnitStatus status,
                          Kills kills, ItemStats holder_stats, CC cc, RuneEvents ev, int32_t main_capacity,
                          int32_t follow_up_capacity);

}  // namespace lanesim::champ
