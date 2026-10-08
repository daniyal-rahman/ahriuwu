// Domination tree 8100 (runes/effects/domination.py). No Domination perk is on a Garen/Jax page, but the
// Electrocute instance ring and Bounty Hunter takedowns are tracked for every holder, and the proc packets are
// always emitted (invalid), so the whole module is ported.
#include <algorithm>

#include "../marshal.hpp"
#include "runes.hpp"

namespace lanesim::runes::domination {

namespace {
enum : int { ELECTROCUTE = 8112, DARK_HARVEST = 8128, HAIL_OF_BLADES = 9923, CHEAP_SHOT = 8126,
             TASTE_OF_BLOOD = 8139, SUDDEN_IMPACT = 8143, GRISLY_MEMENTOS = 8140, TREASURE_HUNTER = 8135,
             RELENTLESS_HUNTER = 8105, ULTIMATE_HUNTER = 8106 };
constexpr int ELEC_SEEN = 8, DH_SOUL_SLOTS = 2;

float ea(int perk, const char* name) { return data::f("runes.domination." + std::to_string(perk) + "." + name); }
float k(const char* name) { return data::f(std::string("runes.domination.") + name); }
const std::vector<float>& lt(const char* name) { return data::table(std::string("runes.domination.lin.") + name); }

struct Consts {
    float elec_ad = ea(ELECTROCUTE, "BonusADRatio"), elec_ap = ea(ELECTROCUTE, "APRatio"),
          elec_window = ea(ELECTROCUTE, "WindowDuration"), elec_cd = ea(ELECTROCUTE, "Cooldown");
    float dh_threshold = ea(DARK_HARVEST, "HarvestThreshold"), dh_base = ea(DARK_HARVEST, "BaseDamage"),
          dh_per_soul = ea(DARK_HARVEST, "DamagePerSoulEssence"), dh_ad = ea(DARK_HARVEST, "ADRatio"),
          dh_ap = ea(DARK_HARVEST, "APRatio"), dh_cd = ea(DARK_HARVEST, "Cooldown"),
          dh_reset = ea(DARK_HARVEST, "CooldownResetValue");
    float hob_duration = ea(HAIL_OF_BLADES, "Duration"), hob_cd = ea(HAIL_OF_BLADES, "Cooldown"),
          hob_ad = ea(HAIL_OF_BLADES, "BonusADRatio"), hob_ap = ea(HAIL_OF_BLADES, "APRatio"),
          hob_as = ea(HAIL_OF_BLADES, "ASBoost"), hob_as_ranged = ea(HAIL_OF_BLADES, "ASBoostRanged");
    int hob_hits = (int)ea(HAIL_OF_BLADES, "NumHits"), hob_max_bonus = (int)ea(HAIL_OF_BLADES, "MaxBonusHits");
    float cs_cd = ea(CHEAP_SHOT, "Cooldown"), si_cd = ea(SUDDEN_IMPACT, "Cooldown"),
          si_armed = ea(SUDDEN_IMPACT, "ArmedDuration");
    float tob_ad = ea(TASTE_OF_BLOOD, "ADRatio"), tob_ap = ea(TASTE_OF_BLOOD, "APRatio"),
          tob_cd = ea(TASTE_OF_BLOOD, "Cooldown");
    float th_base = ea(TREASURE_HUNTER, "BaseGoldAmount"), th_growth = ea(TREASURE_HUNTER, "GoldGrowth");
    float gm_max = ea(GRISLY_MEMENTOS, "MaxStacks"), gm_ah = ea(GRISLY_MEMENTOS, "TrinketAH");
    float rh_start = ea(RELENTLESS_HUNTER, "StartingOOCMS"), rh_per = ea(RELENTLESS_HUNTER, "OOCMS");
    float uh_start = ea(ULTIMATE_HUNTER, "StartingUltAH"), uh_per = ea(ULTIMATE_HUNTER, "AdditionalUltAH");
    int elec_stacks = (int)k("ELEC_STACKS");
    float elec_delay = k("ELEC_DELAY"), dh_soul_delay = k("DH_SOUL_DELAY"), dh_min_damage = k("DH_MIN_DAMAGE"),
          hob_cancel = k("HOB_CANCEL_LOCKOUT"), bounty_max = k("BOUNTY_MAX"), combat_timeout = k("COMBAT_TIMEOUT");
    std::vector<float> elec = lt("ELEC"), hob = lt("HOB"), cs = lt("CS"), si = lt("SI"), tob = lt("TOB");
};
const Consts& K() {
    static const Consts c;
    return c;
}

bool enemy_champ_unit(const Ctx& ctx, const Units& u, int c, int j) {
    return u.cls[j] == CLASS_CHAMPION && u.team[j] != ctx.team[c];
}

// _to_enemy_champ: (C, P) valid packet from holder c onto an enemy champion.
Mask to_enemy_champ(const Packets& p, const Ctx& ctx, const Units& u) {
    size_t nc = ctx.unit.size(), np = size(p);
    int n = (int)u.x.size();
    Mask m(nc * np);
    for (size_t c = 0; c < nc; ++c)
        for (size_t i = 0; i < np; ++i) {
            int d = clip_unit(p.dst[i], n);
            m[c * np + i] = p.valid[i] && p.src[i] == ctx.unit[c] && enemy_champ_unit(ctx, u, (int)c, d);
        }
    return m;
}

bool non_proc(int flags) { return !has(flags, TAG_PROC) || has(flags, TAG_PET); }

float bounty_stacks(const State& s, int c, int n) {
    int cnt = 0;
    for (int j = 0; j < n; ++j) cnt += s.bounty[c * n + j];
    return (float)std::min(cnt, (int)K().bounty_max);
}

// _proc: rune packet onto max(dst, 0).
void proc(Packets& out, bool valid, const Ctx& ctx, int c, int dst, float raw, int dtype, int perk, int flags = TAG_PROC) {
    push(out, valid, ctx.unit[c], std::max(dst, 0), raw, dtype, flags, 0.f, rune_item(perk));
}

// _elec_add: add (C, N) stacks, expire windows, queue the delayed hit.
void elec_add(State& s, const Page& page, const Ctx& ctx, const RuneEvents& ev, const std::vector<int>& add, int n) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    float now = ctx.now;
    for (int c = 0; c < nc; ++c) {
        bool ready = hasr(page, ELECTROCUTE, c, nc) && now >= s.elec_cd_until[c];
        bool fire = false;
        int tgt = 0;
        std::vector<int> stacks(n);
        std::vector<float> first(n);
        for (int j = 0; j < n; ++j) {
            int nw = ready ? add[c * n + j] : 0;
            int st = s.elec_stacks[c * n + j];
            bool expired = st > 0 && now - s.elec_first_t[c * n + j] > k.elec_window;
            st = expired ? 0 : st;
            float f = expired ? -BIG : s.elec_first_t[c * n + j];
            f = st == 0 && nw > 0 ? now : f;
            st = st + nw;
            stacks[j] = st, first[j] = f;
            if (st >= k.elec_stacks && !fire) fire = true, tgt = j;
        }
        float ad = k.elec_ad * ev.bonus_ad[c], ap = k.elec_ap * ev.ap[c];
        float raw = lin(k.elec, ctx.level[c]) + ad + ap;
        for (int j = 0; j < n; ++j) {
            s.elec_stacks[c * n + j] = fire ? 0 : stacks[j];
            s.elec_first_t[c * n + j] = fire ? -BIG : first[j];
        }
        if (fire) {
            s.elec_cd_until[c] = now + k.elec_cd, s.elec_due[c] = now + k.elec_delay, s.elec_dst[c] = tgt;
            s.elec_raw[c] = raw, s.elec_dtype[c] = variable_damage_type(ad, ap);
        }
    }
}

// _elec_from_packets: per-instance stacks from this damage pass, the (cast_id, dst) ring, CC pairing.
void elec_from_packets(State& s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev, const Packets& p) {
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    size_t np = size(p);
    Mask sel = to_enemy_champ(p, ctx, u);
    for (int c = 0; c < nc; ++c)
        for (size_t i = 0; i < np; ++i) sel[c * np + i] = sel[c * np + i] && non_proc(p.flags[i]);
    Mask first = first_instance(p, sel, nc);
    Mask inst(nc * np);
    for (int c = 0; c < nc; ++c)
        for (size_t i = 0; i < np; ++i) {
            bool seen = false;
            for (int q = 0; q < ELEC_SEEN; ++q)
                seen = seen || (s.elec_seen_id[c * ELEC_SEEN + q] == p.cast_id[i]
                                && s.elec_seen_dst[c * ELEC_SEEN + q] == p.dst[i]);
            seen = seen && p.cast_id[i] != 0;
            inst[c * np + i] = first[c * np + i] && !seen;
        }
    Mask fdst = first_per_key(inst, nc, np, p.dst);
    std::vector<int> add(nc * n, 0);
    std::vector<uint8_t> used(nc * n, 0);
    for (int c = 0; c < nc; ++c) {
        for (size_t i = 0; i < np; ++i) {
            int d = p.dst[i];
            bool in_range = d >= 0 && d < n;
            bool cc_pkt = s.elec_cc_t[c * n + clip_unit(d, n)] == ctx.now && in_range;
            bool paired = fdst[c * np + i] && cc_pkt;
            if (in_range && inst[c * np + i] && !paired) add[c * n + d] += 1;
            if (in_range && inst[c * np + i]) used[c * n + d] = 1;
        }
        // Record new non-zero cast ids in the ring.
        int ptr = s.elec_seen_ptr[c], order = -1, n_rec = 0;
        for (size_t i = 0; i < np; ++i) {
            bool rec = inst[c * np + i] && p.cast_id[i] != 0;
            if (!rec) continue;
            ++order, ++n_rec;
            if (order < ELEC_SEEN) {
                int slot = (ptr + order) % ELEC_SEEN;
                s.elec_seen_id[c * ELEC_SEEN + slot] = p.cast_id[i];
                s.elec_seen_dst[c * ELEC_SEEN + slot] = p.dst[i];
            }
        }
        s.elec_seen_ptr[c] = (ptr + std::min(n_rec, ELEC_SEEN)) % ELEC_SEEN;
        for (int j = 0; j < n; ++j)
            if (s.elec_cc_t[c * n + j] == ctx.now && used[c * n + j]) s.elec_cc_t[c * n + j] = -BIG;
    }
    elec_add(s, page, ctx, ev, add, n);
}

void hob_end(State& s, int c, bool ended, float at) {
    if (!ended) return;
    s.hob_active[c] = 0, s.hob_stacks[c] = 0, s.hob_bonus_used[c] = 0;
    s.hob_cd_until[c] = at + K().hob_cd;
}
}  // namespace

std::tuple<State, Effects> on_cc(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    std::vector<int> hit(nc * n);
    for (int c = 0; c < nc; ++c) {
        bool ready = hasr(page, ELECTROCUTE, c, nc) && ctx.now >= s.elec_cd_until[c];
        for (int j = 0; j < n; ++j) {
            size_t q = (size_t)c * n + j;
            hit[q] = (ev.cc.slowed[q] || ev.cc.immobilized[q]) && enemy_champ_unit(ctx, u, c, j) && u.alive[j];
            if (hit[q] && ready) s.elec_cc_t[q] = ctx.now;
        }
    }
    elec_add(s, page, ctx, ev, hit, n);
    return {s, no_effects(nc, n)};
}

std::tuple<State, Effects> on_cast(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    for (int c = 0; c < nc; ++c) {
        bool armed = s.si_armed_until[c] > -BIG / 2;
        bool lapsed = armed && now >= s.si_armed_until[c];
        float cd = lapsed ? s.si_armed_until[c] + k.si_cd : s.si_cd_until[c];
        float until = lapsed ? -BIG : s.si_armed_until[c];
        bool arm = hasr(page, SUDDEN_IMPACT, c, nc) && ev.blinked[c] && !(armed && !lapsed) && now >= cd;
        s.si_armed_until[c] = arm ? now + k.si_armed : until;
        s.si_cd_until[c] = cd;
    }
    return {s, no_effects(nc, n)};
}

std::tuple<State, Effects> on_attack(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    for (int c = 0; c < nc; ++c) {
        bool own = hasr(page, HAIL_OF_BLADES, c, nc);
        // Timeout without an attack.
        hob_end(s, c, s.hob_active[c] && now >= s.hob_expire[c], s.hob_expire[c]);
        // Cancelled triggering windup.
        bool cancel = s.hob_pending[c] && ev.attack_cancelled[c];
        s.hob_pending[c] = s.hob_pending[c] && !cancel;
        if (cancel) s.hob_cd_until[c] = now + k.hob_cancel;
        // Windup start on an enemy champion arms the trigger.
        int tgt = clip_unit(ev.attack_start_target[c], n);
        bool on_champ = enemy_champ_unit(ctx, u, c, tgt) && ev.attack_start_target[c] >= 0;
        bool start = own && ev.attack_started[c] && on_champ && !s.hob_active[c] && now >= s.hob_cd_until[c];
        bool pending = s.hob_pending[c] || start;
        // Launch.
        bool launched = ev.attack.launched[c];
        bool at_champ = enemy_champ_unit(ctx, u, c, clip_unit(ev.attack.target[c], n));
        bool activate = launched && pending;
        int stacks = activate ? k.hob_hits : s.hob_stacks[c];
        bool active = s.hob_active[c] || activate;
        bool empowered = launched && active && stacks > 0;
        stacks = empowered ? stacks - 1 : stacks;
        if (activate || (empowered && at_champ)) s.hob_expire[c] = now + k.hob_duration;
        s.hob_pending[c] = pending && !activate, s.hob_active[c] = active, s.hob_stacks[c] = stacks;
        s.hob_inflight[c] = s.hob_inflight[c] || empowered;
        if (activate) s.hob_bonus_used[c] = 0;
        // Trait_AttackReset bonus stack.
        bool bonus = s.hob_active[c] && ev.attack_reset[c] && s.hob_stacks[c] > 0 && s.hob_bonus_used[c] < k.hob_max_bonus;
        if (bonus) s.hob_stacks[c] += 1, s.hob_bonus_used[c] += 1;
        hob_end(s, c, s.hob_active[c] && s.hob_stacks[c] <= 0, now);
    }
    return {s, no_effects(nc, n)};
}

std::tuple<State, Effects> on_hit(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) {
        bool land = s.hob_inflight[c] && ev.attack.hit[c] && hasr(page, HAIL_OF_BLADES, c, nc);
        float raw = lin(k.hob, ctx.level[c]) + k.hob_ad * ev.bonus_ad[c] + k.hob_ap * ev.ap[c];
        proc(e.packets, land, ctx, c, ev.attack.target[c], raw, TRUE_DMG, HAIL_OF_BLADES, TAG_PROC | TAG_ON_HIT);
        s.hob_inflight[c] = s.hob_inflight[c] && !ev.attack.hit[c];
    }
    return {s, e};
}

std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) {
        bool due = now >= s.elec_due[c];
        int dst = clip_unit(s.elec_dst[c], n);
        proc(e.packets, due && u.alive[dst], ctx, c, s.elec_dst[c], s.elec_raw[c], s.elec_dtype[c], ELECTROCUTE);
        if (due) s.elec_due[c] = BIG;
        int souls = 0;
        for (int q = 0; q < DH_SOUL_SLOTS; ++q)
            if (now >= s.dh_soul_due[c * DH_SOUL_SLOTS + q]) ++souls, s.dh_soul_due[c * DH_SOUL_SLOTS + q] = BIG;
        s.dh_souls[c] = s.dh_souls[c] + (float)souls;
    }
    return {s, e};
}

std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    const Packets& p = ev.report.packets;
    const Resolved& r = ev.report.resolved;
    size_t np = size(p);
    Mask champ = to_enemy_champ(p, ctx, u);

    elec_from_packets(s, page, ctx, u, ev, p);

    Packets p_dh = empty_packets(0), p_cs = empty_packets(0), p_si = empty_packets(0);
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) {
        Mask dh_sel(np), cs_sel(np), si_sel(np);
        for (size_t i = 0; i < np; ++i) {
            int d = clip_unit(p.dst[i], n);
            bool ch = champ[c * np + i], np_ok = non_proc(p.flags[i]);
            bool below = u.hp[d] < k.dh_threshold * u.max_hp[d] && u.alive[d];
            dh_sel[i] = ch && np_ok && r.final[i] >= k.dh_min_damage && below;
            size_t q = (size_t)c * n + d;
            bool same_hit = ev.cc_on_hit[q] && (ev.cc.slowed[q] || ev.cc.immobilized[q])
                         && ev.cc_cast_id[q] == p.cast_id[i] && p.cast_id[i] != 0;
            cs_sel[i] = ch && np_ok && (ev.impaired[d] || same_hit);
            si_sel[i] = ch && p.item[i] != rune_item(SUDDEN_IMPACT);
        }
        // Dark Harvest.
        bool dh_any = any_row(dh_sel, 0, np);
        int dh_dst = np ? p.dst[argmax_row(dh_sel, 0, np)] : 0;
        bool dh_go = dh_any && hasr(page, DARK_HARVEST, c, nc) && now >= s.dh_cd_until[c];
        float dh_raw = k.dh_base + k.dh_per_soul * s.dh_souls[c] + k.dh_ad * ev.bonus_ad[c] + k.dh_ap * ev.ap[c];
        proc(p_dh, dh_go, ctx, c, dh_dst, dh_raw, adaptive_damage_type(ev, c), DARK_HARVEST);
        int slot = 0;
        for (int q = 0; q < DH_SOUL_SLOTS; ++q)
            if (s.dh_soul_due[c * DH_SOUL_SLOTS + q] >= BIG / 2) { slot = q; break; }
        if (dh_go) s.dh_cd_until[c] = now + k.dh_cd, s.dh_soul_due[c * DH_SOUL_SLOTS + slot] = now + k.dh_soul_delay;
        // Cheap Shot.
        bool cs_any = any_row(cs_sel, 0, np);
        int cs_dst = np ? p.dst[argmax_row(cs_sel, 0, np)] : 0;
        bool cs_go = cs_any && hasr(page, CHEAP_SHOT, c, nc) && now >= s.cs_cd_until[c];
        proc(p_cs, cs_go, ctx, c, cs_dst, lin(k.cs, ctx.level[c]), TRUE_DMG, CHEAP_SHOT);
        if (cs_go) s.cs_cd_until[c] = now + k.cs_cd;
        // Sudden Impact.
        bool si_any = any_row(si_sel, 0, np);
        int si_dst = np ? p.dst[argmax_row(si_sel, 0, np)] : 0;
        bool armed = s.si_armed_until[c] > -BIG / 2 && now < s.si_armed_until[c];
        bool si_go = si_any && armed && hasr(page, SUDDEN_IMPACT, c, nc);
        proc(p_si, si_go, ctx, c, si_dst, lin(k.si, ctx.level[c]), TRUE_DMG, SUDDEN_IMPACT);
        if (si_go) s.si_armed_until[c] = -BIG, s.si_cd_until[c] = now + k.si_cd;
        // Taste of Blood.
        bool tob_go = any_row(champ, c, np) && hasr(page, TASTE_OF_BLOOD, c, nc) && now >= s.tob_cd_until[c]
                   && ctx.alive[c] && ctx.hp[c] < ctx.max_hp[c];
        float heal = lin(k.tob, ctx.level[c]) + k.tob_ad * ev.bonus_ad[c] + k.tob_ap * ev.ap[c];
        if (tob_go) s.tob_cd_until[c] = now + k.tob_cd;
        e.heal[c] = tob_go ? heal : 0.f;
    }
    append(p_dh, p_cs);
    append(p_dh, p_si);
    e.packets = p_dh;
    return {s, e};
}

std::tuple<State, Effects> on_takedown(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) {
        int n_took_i = 0;
        float before = bounty_stacks(s, c, n);
        for (int j = 0; j < n; ++j) {
            bool took = ev.kills.killed_units[(size_t)c * n + j] && enemy_champ_unit(ctx, u, c, j);
            n_took_i += took;
            s.bounty[(size_t)c * n + j] = s.bounty[(size_t)c * n + j] | took;
        }
        float n_took = (float)n_took_i;
        bool dh = hasr(page, DARK_HARVEST, c, nc);
        bool ready = now >= s.dh_cd_until[c];
        float extra = dh && ready ? ev.execute_credit[c] : 0.f;
        bool reset = dh && n_took > 0.f;
        if (reset) s.dh_cd_until[c] = std::min(s.dh_cd_until[c], now + k.dh_reset);
        float after = bounty_stacks(s, c, n);
        float kk = after - before;
        float gold = k.th_base * kk + k.th_growth * (before * kk + kk * (kk - 1.f) / 2.f);
        e.gold[c] = hasr(page, TREASURE_HUNTER, c, nc) ? gold : 0.f;
        float memento = std::min(s.mementos[c] + n_took, k.gm_max);
        s.dh_souls[c] = s.dh_souls[c] + extra;
        if (hasr(page, GRISLY_MEMENTOS, c, nc)) s.mementos[c] = memento;
    }
    return {s, e};
}

ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    int n = nc ? (int)(s.bounty.size() / nc) : 0;
    float now = ctx.now;
    ItemStats o = default_stats();
    o.attack_speed.assign(nc, 0.f), o.attack_speed_cap_lift.assign(nc, 0.f), o.move_speed.assign(nc, 0.f);
    o.ultimate_haste.assign(nc, 0.f), o.trinket_haste.assign(nc, 0.f);
    for (int c = 0; c < nc; ++c) {
        bool hob = hasr(page, HAIL_OF_BLADES, c, nc) && s.hob_active[c] && s.hob_stacks[c] > 0 && now < s.hob_expire[c];
        float hob_as = by_range(ctx, c, k.hob_as, k.hob_as_ranged);
        float stacks = bounty_stacks(s, c, n);
        bool ooc = now - ev.clocks.last_combat_modern[c] >= k.combat_timeout;
        o.attack_speed[c] = hob ? hob_as : 0.f;
        o.attack_speed_cap_lift[c] = hob ? 1.f : 0.f;
        o.move_speed[c] = hasr(page, RELENTLESS_HUNTER, c, nc) && ooc ? k.rh_start + k.rh_per * stacks : 0.f;
        o.ultimate_haste[c] = hasr(page, ULTIMATE_HUNTER, c, nc) ? k.uh_start + k.uh_per * stacks : 0.f;
        o.trinket_haste[c] = hasr(page, GRISLY_MEMENTOS, c, nc) ? k.gm_ah * s.mementos[c] : 0.f;
    }
    return o;
}

LANESIM_TEST(runes_domination_on_cc, "runes.domination.on_cc", on_cc);
LANESIM_TEST(runes_domination_on_cast, "runes.domination.on_cast", on_cast);
LANESIM_TEST(runes_domination_on_attack, "runes.domination.on_attack", on_attack);
LANESIM_TEST(runes_domination_on_hit, "runes.domination.on_hit", on_hit);
LANESIM_TEST(runes_domination_periodic, "runes.domination.periodic", periodic);
LANESIM_TEST(runes_domination_on_damage, "runes.domination.on_damage", on_damage);
LANESIM_TEST(runes_domination_on_takedown, "runes.domination.on_takedown", on_takedown);
LANESIM_TEST(runes_domination_stats, "runes.domination.stats", stats);

}  // namespace lanesim::runes::domination
