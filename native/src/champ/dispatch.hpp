// Item and rune effect dispatch (items/effects/__init__.py, runes/effects/__init__.py): every module's hook in
// MODULES order, state updated in place, Effects merged in emission order. Implemented in dispatch.cpp over the
// ported modules (native/src/champ/items, runes).
#pragma once
#include "core.hpp"

namespace lanesim::champ::items {

ItemStats dynamic_stats(const ItemEffectState& s, const Owned& own, const Ctx& ctx);
HolderDefense holder_defense(const ItemEffectState& s, const Owned& own, const Ctx& ctx);
StatusFlags status(const ItemEffectState& s, const Owned& own, const Ctx& ctx);
Debuffs target_debuffs(const ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units);
Arr<float> packet_amp(const ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units,
                      const Packets& p);
AttackMods attack_mods(const ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units,
                       const Arr<int32_t>& target);
Effects on_attack(ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units, const Attack& a);
Effects on_hit(ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units, const Attack& a);
Effects on_cast(ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units, const Cast& cast);
Effects on_cc(ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units, const CC& cc);
Effects on_damage(ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units, const Report& r);
Effects periodic(ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units);
Effects on_takedown(ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units, const Kills& k);
Effects active(ItemEffectState& s, const Owned& own, const Ctx& ctx, const Units& units,
               const Arr<int32_t>& request, ActiveOut& out);
void on_shop(ItemEffectState& s, const Owned& own, const Ctx& ctx);

// Module helpers combat_tick reads directly.
Attack spellblade_extra_on_hit_attack(const items_spellblade_State& s);
Arr<uint8_t> marksman_phantom_hit_due(const items_marksman_State& s, const Ctx& ctx);          // (C,)
Arr<uint8_t> marksman_extra_on_hit_targets(const items_marksman_State& s, const Ctx& ctx);     // (C, N)
void starters_pending_transforms(const items_starters_State& s, const Owned& own, Arr<int32_t>& from,
                                 Arr<int32_t>& to, Arr<uint8_t>& go);

}  // namespace lanesim::champ::items

namespace lanesim::champ::runes {

ItemStats stats(const RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const RuneEvents& ev);
Debuffs debuffs(const RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& units,
                const RuneEvents& ev);
Arr<float> packet_amp(const RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& units,
                      const RuneEvents& ev, const Packets& p);
Arr<float> packet_block(const RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& units,
                        const RuneEvents& ev, const Packets& p);
Arr<float> heal_mult(const RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const RuneEvents& ev);
Effects on_cast(RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
Effects on_attack(RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
Effects on_hit(RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
Effects on_cc(RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
Effects periodic(RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
Effects on_damage(RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
Effects on_takedown(RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& u,
                    const RuneEvents& ev);
void post_tick(RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const Units& u, const RuneEvents& ev);
RuneOutputs outputs(const RuneEffectState& s, const Arr<int32_t>& page, const Ctx& ctx, const RuneEvents& ev);

}  // namespace lanesim::champ::runes
