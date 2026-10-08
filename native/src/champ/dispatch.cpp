// Item and rune dispatch over the ported modules. PLACEHOLDER until the module ports are merged: every hook is
// neutral (no effects, unchanged state), so combat_tick runs and its non-module parts can be tested.
#include "dispatch.hpp"

#include "stats.hpp"

namespace lanesim::champ::items {
ItemStats dynamic_stats(const ItemEffectState&, const Owned&, const Ctx& ctx) { return stats::zero(ctx.unit.size()); }
HolderDefense holder_defense(const ItemEffectState&, const Owned&, const Ctx& ctx) { return neutral_defense((int)ctx.unit.size()); }
StatusFlags status(const ItemEffectState&, const Owned&, const Ctx& ctx) { StatusFlags s; s.ghosted.assign(ctx.unit.size(), 0); return s; }
Debuffs target_debuffs(const ItemEffectState&, const Owned&, const Ctx&, const Units& u) { return neutral_debuffs((int)u.x.size()); }
Arr<float> packet_amp(const ItemEffectState&, const Owned&, const Ctx&, const Units&, const Packets& p) { return Arr<float>(size(p), 0.f); }
AttackMods attack_mods(const ItemEffectState&, const Owned&, const Ctx& ctx, const Units&, const Arr<int32_t>&) {
    AttackMods m; m.force_crit.assign(ctx.unit.size(), 0); m.crit_scale.assign(ctx.unit.size(), 1.f); return m;
}
#define NEUTRAL(name, ...) Effects name(ItemEffectState&, const Owned&, const Ctx& ctx, const Units& u __VA_ARGS__) { return no_effects((int)ctx.unit.size(), (int)u.x.size()); }
NEUTRAL(on_attack, , const Attack&)
NEUTRAL(on_hit, , const Attack&)
NEUTRAL(on_cast, , const Cast&)
NEUTRAL(on_cc, , const CC&)
NEUTRAL(on_damage, , const Report&)
NEUTRAL(periodic)
NEUTRAL(on_takedown, , const Kills&)
#undef NEUTRAL
Effects active(ItemEffectState&, const Owned&, const Ctx& ctx, const Units& u, const Arr<int32_t>&, ActiveOut& out) {
    size_t c = ctx.unit.size();
    out.used.assign(c, 0), out.cast_time.assign(c, 0.f), out.can_move.assign(c, 1), out.attack_reset.assign(c, 0);
    return no_effects((int)c, (int)u.x.size());
}
void on_shop(ItemEffectState&, const Owned&, const Ctx&) {}
}  // namespace lanesim::champ::items

namespace lanesim::champ::runes {
ItemStats stats(const RuneEffectState&, const Arr<int32_t>&, const Ctx& ctx, const RuneEvents&) { return stats::zero(ctx.unit.size()); }
Debuffs debuffs(const RuneEffectState&, const Arr<int32_t>&, const Ctx&, const Units& u, const RuneEvents&) { return neutral_debuffs((int)u.x.size()); }
Arr<float> packet_amp(const RuneEffectState&, const Arr<int32_t>&, const Ctx&, const Units&, const RuneEvents&, const Packets& p) { return Arr<float>(size(p), 0.f); }
Arr<float> packet_block(const RuneEffectState&, const Arr<int32_t>&, const Ctx&, const Units&, const RuneEvents&, const Packets& p) { return Arr<float>(size(p), 0.f); }
Arr<float> heal_mult(const RuneEffectState&, const Arr<int32_t>&, const Ctx& ctx, const RuneEvents&) { return Arr<float>(ctx.unit.size(), 1.f); }
#define NEUTRAL(name) Effects name(RuneEffectState&, const Arr<int32_t>&, const Ctx& ctx, const Units& u, const RuneEvents&) { return no_effects((int)ctx.unit.size(), (int)u.x.size()); }
NEUTRAL(on_cast) NEUTRAL(on_attack) NEUTRAL(on_hit) NEUTRAL(on_cc) NEUTRAL(periodic) NEUTRAL(on_damage) NEUTRAL(on_takedown)
#undef NEUTRAL
void post_tick(RuneEffectState&, const Arr<int32_t>&, const Ctx&, const Units&, const RuneEvents&) {}
RuneOutputs outputs(const RuneEffectState&, const Arr<int32_t>&, const Ctx& ctx, const RuneEvents&) { return no_outputs((int)ctx.unit.size(), n_items()); }
}  // namespace lanesim::champ::runes
