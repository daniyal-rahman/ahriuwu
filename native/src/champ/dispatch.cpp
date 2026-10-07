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
Attack spellblade_extra_on_hit_attack(const items_spellblade_State& s) {
    Attack a; size_t c = s.dd_extra_due.size();
    a.launched.assign(c, 0), a.hit = s.dd_extra_due, a.target = s.dd_due_target, a.raw.assign(c, 0.f), a.is_crit.assign(c, 0);
    return a;
}
Arr<uint8_t> marksman_phantom_hit_due(const items_marksman_State& s, const Ctx& ctx) {
    Arr<uint8_t> out(ctx.unit.size(), 0);
    for (size_t h = 0; h < out.size(); ++h) out[h] = s.phantom_at[h] == ctx.now;
    return out;
}
Arr<uint8_t> marksman_extra_on_hit_targets(const items_marksman_State& s, const Ctx& ctx) {
    size_t c = ctx.unit.size(), n = c ? s.extra_hits.size() / c : 0;
    Arr<uint8_t> out(c * n, 0);
    for (size_t h = 0; h < c; ++h)
        for (size_t j = 0; j < n; ++j) out[h * n + j] = s.extra_hits[h * n + j] && s.extra_at[h] == ctx.now;
    return out;
}
void starters_pending_transforms(const items_starters_State& s, const Owned& own, Arr<int32_t>& from, Arr<int32_t>& to,
                                 Arr<uint8_t>& go) {
    size_t c = s.tear_mana.size();
    from.assign(c, -1), to.assign(c, -1), go.assign(c, 0);
    const auto& tr = data::table("combat.transforms");        // (from id, to id, max mana) triples
    for (size_t t = 0; t + 2 < tr.size(); t += 3)
        for (size_t h = 0; h < c; ++h)
            if (holds(own, (int)tr[t], (int)h) && s.tear_mana[h] >= tr[t + 2])
                from[h] = item_row((int)tr[t]), to[h] = item_row((int)tr[t + 1]);
    for (size_t h = 0; h < c; ++h) go[h] = from[h] >= 0;
}
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
