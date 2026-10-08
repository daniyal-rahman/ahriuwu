// Item module helpers combat_tick reads directly (spellblade.extra_on_hit_attack, marksman.phantom_hit_due,
// marksman.extra_on_hit_targets, starters.pending_transforms).
#include "dispatch.hpp"

namespace lanesim::champ::items {

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
