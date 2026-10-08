// Precision tree 8000 (runes/effects/precision.py). Every perk's state machine is ported; the only effects
// without a page in the Garen/Jax world are cheap and kept so the hooks stay literal.
#include <algorithm>
#include <numeric>

#include "../marshal.hpp"
#include "runes.hpp"

namespace lanesim::runes::precision {

namespace {
enum : int { PTA = 8005, LETHAL_TEMPO = 8008, FLEET = 8021, CONQUEROR = 8010, ABSORB_LIFE = 9101, TRIUMPH = 9111,
             PRESENCE_OF_MIND = 8009, ALACRITY = 9104, HASTE = 9105, BLOODLINE = 9103, COUP_DE_GRACE = 8014,
             CUT_DOWN = 8017, LAST_STAND = 8299 };
constexpr int CONQ_RING = 8, QUEUE = 4;
constexpr float CONQ_MELEE_HIT = 2.f, CONQ_RANGED_HIT = 1.f, CONQ_SPELL = 2.f;

float k(const char* name) { return data::f(std::string("runes.precision.") + name); }
const std::vector<float>& t(const char* name) { return data::table(std::string("runes.precision.") + name); }

struct Consts {
    float conq_max_stacks = k("CONQ_MAX_STACKS"), conq_duration = k("CONQ_DURATION"),
          conq_same_spell = k("CONQ_SAME_SPELL"), conq_heal = k("CONQ_HEAL"), conq_heal_ranged = k("CONQ_HEAL_RANGED");
    float pta_hits = k("PTA_HITS"), pta_stack_time = k("PTA_STACK_TIME"), pta_amp = k("PTA_AMP"),
          pta_ooc = k("PTA_OOC"), pta_cooldown = k("PTA_COOLDOWN");
    float lt_duration = k("LT_DURATION"), lt_max = k("LT_MAX"), lt_as = k("LT_AS"), lt_as_ranged = k("LT_AS_RANGED"),
          lt_bolt_ranged = k("LT_BOLT_RANGED"), lt_decay = k("LT_DECAY");
    float fleet_ad = k("FLEET_AD"), fleet_ap = k("FLEET_AP"), fleet_ranged_heal = k("FLEET_RANGED_HEAL"),
          fleet_minion = k("FLEET_MINION"), fleet_ms = k("FLEET_MS"), fleet_ms_time = k("FLEET_MS_TIME"),
          fleet_ranged_ms = k("FLEET_RANGED_MS"), fleet_full = k("FLEET_FULL"), fleet_per_hit = k("FLEET_PER_HIT"),
          fleet_units_per_charge = k("FLEET_UNITS_PER_CHARGE");
    float absorb_l1 = k("ABSORB_L1"), triumph_missing = k("TRIUMPH_MISSING"), triumph_max = k("TRIUMPH_MAX"),
          triumph_gold = k("TRIUMPH_GOLD"), delay = k("DELAY");
    float pom_cd = k("POM_CD"), pom_energy = k("POM_ENERGY"), pom_takedown = k("POM_TAKEDOWN"),
          pom_ranged = k("POM_RANGED");
    float legend_takedown = k("LEGEND_TAKEDOWN"), legend_minion = k("LEGEND_MINION"),
          legend_large = k("LEGEND_LARGE"), legend_per_stack = k("LEGEND_PER_STACK");
    float coup_below = k("COUP_BELOW"), coup_amp = k("COUP_AMP"), cut_above = k("CUT_ABOVE"), cut_amp = k("CUT_AMP"),
          ls_min = k("LS_MIN"), ls_start = k("LS_START"), ls_span = k("LS_SPAN"), ls_width = k("LS_WIDTH");
    float alacrity_max = k("ALACRITY_MAX_STACKS"), haste_max = k("HASTE_MAX_STACKS"),
          bloodline_max = k("BLOODLINE_MAX_STACKS"), alacrity_base = k("ALACRITY_BASE"),
          alacrity_per = k("ALACRITY_PER"), haste_base = k("HASTE_BASE"), haste_per = k("HASTE_PER"),
          bloodline_base = k("BLOODLINE_BASE"), bloodline_per = k("BLOODLINE_PER"), bloodline_hp = k("BLOODLINE_HP");
    std::vector<float> conq = t("CONQ"), pta = t("PTA"), lt_bolt = t("LT_BOLT"), fleet_heal = t("FLEET_HEAL"),
                       absorb_marks = t("ABSORB_MARKS"), pom_table = t("POM_TABLE");
};
const Consts& K() {
    static const Consts c;
    return c;
}

// _enemy_champion_target (no alive check).
bool enemy_champion_target(const Ctx& ctx, const Units& u, int c, int idx) {
    int i = clip_unit(idx, (int)u.cls.size());
    return idx >= 0 && u.cls[i] == CLASS_CHAMPION && u.team[i] != ctx.team[c];
}

// _to_enemy_champions: (C, P) holder's packets on enemy champions (dead or alive).
Mask to_enemy_champions(const Ctx& ctx, const Units& u, const Packets& p) {
    size_t nc = ctx.unit.size(), np = size(p);
    int n = (int)u.cls.size();
    Mask m(nc * np);
    for (size_t c = 0; c < nc; ++c)
        for (size_t i = 0; i < np; ++i) {
            int d = clip_unit(p.dst[i], n);
            m[c * np + i] = p.valid[i] && u.cls[d] == CLASS_CHAMPION && p.src[i] == ctx.unit[c] && u.team[d] != ctx.team[c];
        }
    return m;
}

float lt_current(const State& s, int c, float now) {
    float lost = now >= s.lt_expire[c] ? 1.f + std::floor((now - s.lt_expire[c]) / K().lt_decay) : 0.f;
    return std::max(s.lt_stacks[c] - lost, 0.f);
}

float legend_stacks(float points, float max_stacks) {
    return std::min(std::floor(points / K().legend_per_stack), max_stacks);
}

// catalog.breakpoints with marks [start, per, stop - start]...
float breakpoints(float level1, const std::vector<float>& marks, float level) {
    float total = level1;
    for (size_t i = 0; i + 2 < marks.size(); i += 3)
        total = total + marks[i + 1] * std::min(std::max(level - marks[i] + 1.f, 0.f), marks[i + 2]);
    return total;
}

// _enqueue: put ``add`` at the first free slot (if any) where add > 0.
void enqueue(Arr<float>& due, Arr<float>& cnt, int c, float add, float at) {
    int slot = 0;
    bool any_free = false;
    for (int q = 0; q < QUEUE; ++q)
        if (due[c * QUEUE + q] >= BIG / 2) { slot = q, any_free = true; break; }
    if (add > 0.f && any_free) due[c * QUEUE + slot] = at, cnt[c * QUEUE + slot] = add;
}

// _pop: fire due entries; returns the fired count.
float pop(Arr<float>& due, Arr<float>& cnt, int c, float now) {
    float total = 0.f;
    for (int q = 0; q < QUEUE; ++q) {
        bool fire = due[c * QUEUE + q] <= now;
        total = total + (fire ? cnt[c * QUEUE + q] : 0.f);
        if (fire) due[c * QUEUE + q] = BIG, cnt[c * QUEUE + q] = 0.f;
    }
    return total;
}
}  // namespace

ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    float now = ctx.now;
    ItemStats o = default_stats();
    o.adaptive_force.assign(nc, 0.f), o.attack_speed.assign(nc, 0.f), o.percent_move_speed.assign(nc, 0.f);
    o.basic_ability_haste.assign(nc, 0.f), o.life_steal.assign(nc, 0.f), o.health.assign(nc, 0.f);
    for (int c = 0; c < nc; ++c) {
        bool conq = hasr(page, CONQUEROR, c, nc) && now < s.conq_expire[c];
        float af = conq ? s.conq_stacks[c] * lin(k.conq, s.conq_level[c]) : 0.f;
        float lt = hasr(page, LETHAL_TEMPO, c, nc)
                 ? lt_current(s, c, now) * k.lt_as * by_range(ctx, c, 1.f, k.lt_as_ranged) : 0.f;
        float ms = hasr(page, FLEET, c, nc) && now < s.fleet_ms_until[c]
                 ? k.fleet_ms * by_range(ctx, c, 1.f, k.fleet_ranged_ms) : 0.f;
        float pts = s.legend_points[c];
        float a_n = legend_stacks(pts, k.alacrity_max), h_n = legend_stacks(pts, k.haste_max),
              b_n = legend_stacks(pts, k.bloodline_max);
        float alacrity = hasr(page, ALACRITY, c, nc) ? k.alacrity_base + k.alacrity_per * a_n : 0.f;
        float haste = hasr(page, HASTE, c, nc) ? k.haste_base + k.haste_per * h_n : 0.f;
        bool blood = hasr(page, BLOODLINE, c, nc);
        float ls = blood ? (k.bloodline_base + k.bloodline_per * b_n) / 100.f : 0.f;
        float hp = blood && b_n >= k.bloodline_max ? k.bloodline_hp : 0.f;
        o.adaptive_force[c] = af, o.attack_speed[c] = lt + alacrity, o.percent_move_speed[c] = ms;
        o.basic_ability_haste[c] = haste, o.life_steal[c] = ls, o.health[c] = hp;
    }
    return o;
}

std::tuple<State, Effects> on_attack(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    const Attack& a = ev.attack;
    float now = ctx.now;
    for (int c = 0; c < nc; ++c) {
        bool launched = a.launched[c] && ctx.alive[c];
        // Lethal Tempo.
        bool lt_go = launched && hasr(page, LETHAL_TEMPO, c, nc) && enemy_champion_target(ctx, u, c, a.target[c]);
        float lt_new = std::min(lt_current(s, c, now) + 1.f, k.lt_max);
        bool bolt = lt_go && lt_new >= k.lt_max;
        float raw = lin(k.lt_bolt, ctx.level[c]) * (1.f + ev.bonus_attack_speed[c]) * by_range(ctx, c, 1.f, k.lt_bolt_ranged);
        // Fleet.
        bool fleet = launched && hasr(page, FLEET, c, nc);
        bool armed = fleet ? s.fleet_energy[c] >= k.fleet_full : (bool)s.fleet_armed[c];
        float energy = fleet ? std::min(s.fleet_energy[c] + k.fleet_per_hit, k.fleet_full) : s.fleet_energy[c];
        if (lt_go) s.lt_stacks[c] = lt_new, s.lt_expire[c] = now + k.lt_duration, s.lt_bolt[c] = bolt;
        if (bolt) {
            s.lt_bolt_target[c] = a.target[c], s.lt_bolt_raw[c] = raw;
            s.lt_bolt_dtype[c] = adaptive_damage_type(ev, c);
        }
        s.fleet_energy[c] = energy, s.fleet_armed[c] = armed;
    }
    return {s, no_effects(nc, n)};
}

std::tuple<State, Effects> on_hit(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    const Attack& a = ev.attack;
    float now = ctx.now;
    Effects e = no_effects(nc, n);
    Packets p_pta = empty_packets(0), p_lt = empty_packets(0);
    State o = s;
    for (int c = 0; c < nc; ++c) {
        bool hit = a.hit[c] && ctx.alive[c];
        int tgt = std::max(a.target[c], 0);
        // Press the Attack.
        bool pta_go = hit && hasr(page, PTA, c, nc) && enemy_champion_target(ctx, u, c, a.target[c]) && now >= s.pta_cd[c];
        bool keep = s.pta_target[c] == a.target[c] && now < s.pta_expire[c];
        float stacks = (keep ? s.pta_stacks[c] : 0.f) + 1.f;
        bool burst = pta_go && stacks >= k.pta_hits;
        push(p_pta, burst, ctx.unit[c], tgt, lin(k.pta, ctx.level[c]), adaptive_damage_type(ev, c), TAG_PROC, 0.f,
             rune_item(PTA));
        // Lethal Tempo bolt.
        bool bolt = hit && s.lt_bolt[c] && hasr(page, LETHAL_TEMPO, c, nc);
        push(p_lt, bolt, ctx.unit[c], std::max(s.lt_bolt_target[c], 0), s.lt_bolt_raw[c], s.lt_bolt_dtype[c], TAG_PROC,
             0.f, rune_item(LETHAL_TEMPO));
        // Fleet energized hit.
        bool fleet = hit && s.fleet_armed[c] && hasr(page, FLEET, c, nc);
        bool minion = target_class(u, a.target[c]) == CLASS_MINION;
        float heal = (lin_growth(k.fleet_heal, ctx.level[c]) + k.fleet_ad * ev.bonus_ad[c] + k.fleet_ap * ev.ap[c])
                   * by_range(ctx, c, 1.f, k.fleet_ranged_heal) * (minion ? k.fleet_minion : 1.f);
        if (pta_go) {
            o.pta_target[c] = a.target[c], o.pta_stacks[c] = burst ? 0.f : stacks;
            o.pta_expire[c] = now + k.pta_stack_time;
        }
        if (burst) o.pta_cd[c] = now + k.pta_cooldown, o.pta_amp_since[c] = now;
        o.pta_amp[c] = s.pta_amp[c] | burst;
        if (hit) o.lt_bolt[c] = 0;
        if (fleet) o.fleet_armed[c] = 0, o.fleet_energy[c] = 0.f, o.fleet_ms_until[c] = now + k.fleet_ms_time;
        e.heal[c] = fleet ? heal : 0.f;
    }
    append(p_pta, p_lt);
    e.packets = p_pta;
    return {o, e};
}

std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) {
        bool conq_end = now >= s.conq_expire[c];
        bool pta_live = now - std::max(ev.clocks.last_champion_combat[c], s.pta_amp_since[c]) < k.pta_ooc;
        bool move = hasr(page, FLEET, c, nc) && ctx.alive[c];
        if (move) s.fleet_energy[c] = std::min(s.fleet_energy[c] + ctx.moved[c] / k.fleet_units_per_charge, k.fleet_full);
        float tri = pop(s.tri_due, s.tri_n, c, now);
        tri = hasr(page, TRIUMPH, c, nc) ? tri : 0.f;
        float tri_heal = tri * (k.triumph_max * ctx.max_hp[c] + k.triumph_missing * s.tri_missing[c]);
        float pom = pop(s.pom_due, s.pom_n, c, now);
        float pom_mana = hasr(page, PRESENCE_OF_MIND, c, nc) ? pom * k.pom_takedown * ctx.max_mana[c] : 0.f;
        if (conq_end) s.conq_stacks[c] = 0.f;
        if (now >= s.pta_expire[c]) s.pta_stacks[c] = 0.f;
        s.pta_amp[c] = s.pta_amp[c] && pta_live;
        e.heal[c] = tri_heal, e.gold[c] = tri * k.triumph_gold, e.mana[c] = pom_mana;
    }
    return {s, e};
}

Arr<float> packet_amp(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev, const Packets& p) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.cls.size();
    size_t np = size(p);
    Mask sel = to_enemy_champions(ctx, u, p);
    Arr<float> out(np, 0.f);
    std::vector<float> rows(nc * np, 0.f);
    for (int c = 0; c < nc; ++c) {
        float own = ctx.hp[c] / std::max(ctx.max_hp[c], 1.f);
        bool pta_on = s.pta_amp[c] && ctx.now > s.pta_amp_since[c]
                   && ctx.now - std::max(ev.clocks.last_champion_combat[c], s.pta_amp_since[c]) < k.pta_ooc;
        float ls = own < k.ls_start
                 ? k.ls_min + k.ls_span * std::min(std::max((k.ls_start - own) / k.ls_width, 0.f), 1.f) : 0.f;
        bool pta = hasr(page, PTA, c, nc) && pta_on, coup = hasr(page, COUP_DE_GRACE, c, nc),
             cut = hasr(page, CUT_DOWN, c, nc), last = hasr(page, LAST_STAND, c, nc);
        for (size_t i = 0; i < np; ++i) {
            bool ok = !has(p.flags[i], PROP_NO_DAMAGE_MOD) && !has(p.flags[i], TAG_NON_AMPABLE)
                   && !has(p.flags[i], PROP_SUMMONER);
            int d = clip_unit(p.dst[i], n);
            float frac = u.hp[d] / std::max(u.max_hp[d], 1.f);
            float amp = (pta ? k.pta_amp : 0.f) + (coup && frac < k.coup_below ? k.coup_amp : 0.f)
                      + (cut && frac > k.cut_above ? k.cut_amp : 0.f) + (last ? ls : 0.f);
            rows[c * np + i] = sel[c * np + i] && ok ? amp : 0.f;
        }
    }
    for (size_t i = 0; i < np; ++i) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) acc = acc + rows[c * np + i];
        out[i] = acc;
    }
    return out;
}

namespace {
// _conqueror: stacks from basic attacks and per-cast-instance spells, heal at max stacks; returns (C,) heal.
std::vector<float> conqueror(State& s, const Page& page, const Ctx& ctx, const Units& u, const Report& rep) {
    const Consts& k = K();
    const Packets& p = rep.packets;
    const Resolved& r = rep.resolved;
    int nc = (int)ctx.unit.size();
    size_t np = size(p);
    float now = ctx.now;
    Mask sel = to_enemy_champions(ctx, u, p);
    for (int c = 0; c < nc; ++c) {
        bool on = hasr(page, CONQUEROR, c, nc) && ctx.alive[c];
        for (size_t i = 0; i < np; ++i) sel[c * np + i] = sel[c * np + i] && on;
    }
    std::vector<uint8_t> proc(np), basic(np);
    for (size_t i = 0; i < np; ++i) {
        proc[i] = has(p.flags[i], TAG_PROC) && !has(p.flags[i], TAG_PET);
        basic[i] = has(p.flags[i], TAG_BASIC_ATTACK) && !proc[i];
    }
    Mask spell(nc * np);
    for (int c = 0; c < nc; ++c)
        for (size_t i = 0; i < np; ++i) spell[c * np + i] = sel[c * np + i] && !basic[i] && !proc[i];
    Mask fpk = first_per_key(spell, nc, np, p.cast_id);
    std::vector<float> heal(nc);
    for (int c = 0; c < nc; ++c) {
        Mask gain_spell(np), ring(np);
        std::vector<float> gain_p(np);
        for (size_t i = 0; i < np; ++i) {
            int cid = p.cast_id[i];
            bool first = spell[c * np + i] && (cid == 0 || fpk[c * np + i]);
            bool seen = false;
            for (int q = 0; q < CONQ_RING; ++q) {
                bool recent = now - s.conq_seen_t[c * CONQ_RING + q] < k.conq_same_spell || has(p.flags[i], PROP_SUMMONER);
                seen = seen || (s.conq_seen_id[c * CONQ_RING + q] == cid && recent);
            }
            seen = seen && cid != 0;
            gain_spell[i] = first && !seen;
            gain_p[i] = (sel[c * np + i] && basic[i] ? by_range(ctx, c, CONQ_MELEE_HIT, CONQ_RANGED_HIT) : 0.f)
                      + (gain_spell[i] ? CONQ_SPELL : 0.f);
            ring[i] = gain_spell[i] && cid != 0;
        }
        float stacks0 = now < s.conq_expire[c] ? s.conq_stacks[c] : 0.f;
        float run = 0.f, total = 0.f, h = 0.f;
        bool refresh_any = false;
        for (size_t i = 0; i < np; ++i) {
            run = run + gain_p[i];
            float cum = std::min(stacks0 + run, k.conq_max_stacks);
            h = h + (sel[c * np + i] && cum >= k.conq_max_stacks ? r.final[i] : 0.f);
            refresh_any = refresh_any || (sel[c * np + i] && !proc[i]);
        }
        total = run;
        float nw = std::min(stacks0 + total, k.conq_max_stacks);
        bool refresh = refresh_any && nw > 0.f;
        heal[c] = h * by_range(ctx, c, k.conq_heal, k.conq_heal_ranged);
        // Write the r-th new instance into the r-th oldest ring slot.
        int order[CONQ_RING];
        std::iota(order, order + CONQ_RING, 0);
        const float* ts0 = &s.conq_seen_t[c * CONQ_RING];
        std::stable_sort(order, order + CONQ_RING, [&](int a, int b) { return ts0[a] < ts0[b]; });
        std::vector<int> ids(&s.conq_seen_id[c * CONQ_RING], &s.conq_seen_id[c * CONQ_RING] + CONQ_RING);
        std::vector<float> ts(ts0, ts0 + CONQ_RING);
        int rank = -1;
        std::vector<int> rank_of(np);
        for (size_t i = 0; i < np; ++i) rank += ring[i], rank_of[i] = rank;
        for (int q = 0; q < CONQ_RING; ++q) {
            bool any = false;
            int val = 0;
            for (size_t i = 0; i < np; ++i)
                if (ring[i] && rank_of[i] == q) any = true, val += p.cast_id[i];
            if (any) ids[order[q]] = val, ts[order[q]] = now;
        }
        std::copy(ids.begin(), ids.end(), &s.conq_seen_id[c * CONQ_RING]);
        std::copy(ts.begin(), ts.end(), &s.conq_seen_t[c * CONQ_RING]);
        if (stacks0 == 0.f && total > 0.f) s.conq_level[c] = ctx.level[c];
        if (refresh) s.conq_expire[c] = now + k.conq_duration;
        s.conq_stacks[c] = nw;
    }
    return heal;
}
}  // namespace

std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    const Packets& p = ev.report.packets;
    size_t np = size(p);
    std::vector<float> heal = conqueror(s, page, ctx, u, ev.report);
    Mask champ = to_enemy_champions(ctx, u, p);
    Mask own(nc * np);
    for (int c = 0; c < nc; ++c)
        for (size_t i = 0; i < np; ++i)
            own[c * np + i] = p.valid[i] && p.src[i] == ctx.unit[c] && has(p.flags[i], TAG_ACTIVE_SPELL)
                           && has(p.flags[i], TAG_ON_HIT);
    Mask inst = first_instance(p, own, nc);
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) {
        // Presence of Mind.
        bool pom = any_row(champ, c, np) && hasr(page, PRESENCE_OF_MIND, c, nc) && ctx.alive[c] && ctx.now >= s.pom_cd[c];
        float restore = ev.uses_energy[c] ? k.pom_energy
                      : level_table(k.pom_table, ctx.level[c]) * by_range(ctx, c, 1.f, k.pom_ranged);
        // Fleet: per ability instance that applies on-hit.
        int cnt = 0;
        for (size_t i = 0; i < np; ++i) cnt += inst[c * np + i];
        bool fleet = hasr(page, FLEET, c, nc) && ctx.alive[c];
        if (pom) s.pom_cd[c] = ctx.now + k.pom_cd;
        if (fleet) s.fleet_energy[c] = std::min(s.fleet_energy[c] + k.fleet_per_hit * (float)cnt, k.fleet_full);
        e.heal[c] = heal[c], e.mana[c] = pom ? restore : 0.f;
    }
    return {s, e};
}

std::tuple<State, Effects> on_takedown(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    const Kills& kl = ev.kills;
    float now = ctx.now;
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) {
        float takedowns = kl.champion_kill[c] + kl.champion_assist[c];
        float kills = kl.champion_kill[c] + kl.minion_kill[c] + ev.large_monster_kill[c];
        float absorb = hasr(page, ABSORB_LIFE, c, nc) ? kills * breakpoints(k.absorb_l1, k.absorb_marks, ctx.level[c]) : 0.f;
        float tri = hasr(page, TRIUMPH, c, nc) ? takedowns : 0.f;
        enqueue(s.tri_due, s.tri_n, c, tri, now + k.delay);
        float pom = hasr(page, PRESENCE_OF_MIND, c, nc) ? takedowns : 0.f;
        enqueue(s.pom_due, s.pom_n, c, pom, now + k.delay);
        bool legend = hasr(page, ALACRITY, c, nc) || hasr(page, HASTE, c, nc) || hasr(page, BLOODLINE, c, nc);
        float pts = k.legend_takedown * (takedowns + ev.epic_takedown[c]) + k.legend_large * ev.large_monster_kill[c]
                  + k.legend_minion * kl.minion_kill[c];
        if (tri > 0.f) s.tri_missing[c] = std::max(ctx.max_hp[c] - ctx.hp[c], 0.f);
        if (legend) s.legend_points[c] = s.legend_points[c] + pts;
        e.heal[c] = absorb;
    }
    return {s, e};
}

LANESIM_TEST(runes_precision_stats, "runes.precision.stats", stats);
LANESIM_TEST(runes_precision_on_attack, "runes.precision.on_attack", on_attack);
LANESIM_TEST(runes_precision_on_hit, "runes.precision.on_hit", on_hit);
LANESIM_TEST(runes_precision_periodic, "runes.precision.periodic", periodic);
LANESIM_TEST(runes_precision_packet_amp, "runes.precision.packet_amp", packet_amp);
LANESIM_TEST(runes_precision_on_damage, "runes.precision.on_damage", on_damage);
LANESIM_TEST(runes_precision_on_takedown, "runes.precision.on_takedown", on_takedown);

}  // namespace lanesim::runes::precision
