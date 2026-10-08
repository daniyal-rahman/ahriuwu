// Resolve tree 8400 (runes/effects/resolve.py). Guardian's damage buckets are kept for every holder and the proc
// packets (Grasp, Shield Bash, Demolish, Aftershock) are always emitted, so the whole module is ported.
#include <algorithm>
#include <cmath>

#include "../marshal.hpp"
#include "runes.hpp"

namespace lanesim::runes::resolve {

namespace {
enum : int { GRASP = 8437, AFTERSHOCK = 8439, GUARDIAN = 8465, DEMOLISH = 8446, FONT_OF_LIFE = 8463,
             SHIELD_BASH = 8401, CONDITIONING = 8429, SECOND_WIND = 8444, BONE_PLATING = 8473, OVERGROWTH = 8451,
             REVITALIZE = 8453, UNFLINCHING = 8242 };

float k(const char* name) { return data::f(std::string("runes.resolve.") + name); }
const std::vector<float>& lt(const char* name) { return data::table(std::string("runes.resolve.lin.") + name); }

struct Consts {
    float grasp_pct_damage = k("GRASP_PCT_DAMAGE"), grasp_pct_heal = k("GRASP_PCT_HEAL"),
          grasp_hp_melee = k("GRASP_HP_MELEE"), grasp_hp_ranged = k("GRASP_HP_RANGED"),
          grasp_ranged_mod = k("GRASP_RANGED_MOD"), grasp_stacks = k("GRASP_STACKS"), grasp_window = k("GRASP_WINDOW"),
          grasp_gen_after = k("GRASP_GEN_AFTER"), eps = k("_EPS");
    float as_flat = k("AS_FLAT"), as_pct = k("AS_PCT"), as_delay = k("AS_DELAY"), as_hp_ratio = k("AS_HP_RATIO"),
          as_radius = k("AS_RADIUS"), as_cooldown = k("AS_COOLDOWN");
    float gd_range = k("GD_RANGE"), gd_guard = k("GD_GUARD"), gd_ap = k("GD_AP"), gd_hp = k("GD_HP"),
          gd_shield_duration = k("GD_SHIELD_DURATION"), gd_bucket = k("GD_BUCKET");
    int gd_buckets = (int)k("GD_BUCKETS");
    float demo_base_melee = k("DEMO_BASE_MELEE"), demo_base_ranged = k("DEMO_BASE_RANGED"),
          demo_hp_melee = k("DEMO_HP_MELEE"), demo_hp_ranged = k("DEMO_HP_RANGED"), demo_cooldown = k("DEMO_COOLDOWN"),
          demo_lock = k("DEMO_LOCK");
    int demo_stacks = (int)k("DEMO_STACKS");
    float font_ranged = k("FONT_RANGED"), font_cooldown = k("FONT_COOLDOWN"), font_range = k("FONT_RANGE");
    float sb_hp = k("SB_HP"), sb_shield = k("SB_SHIELD"), sb_linger = k("SB_LINGER"),
          sb_assumed = k("SB_ASSUMED_SHIELD_LIFE");
    float cond_time = k("COND_TIME"), cond_armor = k("COND_ARMOR"), cond_mr = k("COND_MR"), cond_pct = k("COND_PCT");
    float sw_duration = k("SW_DURATION"), sw_rate = k("SW_RATE");
    int bp_count = (int)k("BP_COUNT");
    float bp_duration = k("BP_DURATION"), bp_cooldown = k("BP_COOLDOWN");
    float og_range_sq = k("OG_RANGE_SQ"), og_per_tier = k("OG_PER_TIER"), og_hp_per_tier = k("OG_HP_PER_TIER"),
          og_threshold = k("OG_THRESHOLD"), og_pct = k("OG_PCT");
    float rev_hsp = k("REV_HSP"), rev_cutoff = k("REV_CUTOFF"), rev_amp = k("REV_AMP");
    float unf_resist = k("UNF_RESIST"), unf_linger = k("UNF_LINGER");
    std::vector<float> as_cap = lt("AS_CAP"), as_dmg = lt("AS_DMG"), gd_shield = lt("GD_SHIELD"), gd_cd = lt("GD_CD"),
                       gd_thr = lt("GD_THR"), font = lt("FONT"), sb = lt("SB"), bp = lt("BP");
};
const Consts& K() {
    static const Consts c;
    return c;
}

inline float bonus_hp(const Ctx& ctx, int c) { return ctx.max_hp[c] - ctx.base_hp[c]; }

// _enemy_champion_target (alive).
bool enemy_champion_target(const Ctx& ctx, const Units& u, int c, int idx) {
    int i = clip_unit(idx, (int)u.cls.size());
    return idx >= 0 && u.cls[i] == CLASS_CHAMPION && u.team[i] != ctx.team[c] && u.alive[i];
}

// _bp_select: (C, P) packets Bone Plating reduces for each holder.
Mask bp_select(const State& s, const Page& page, const Ctx& ctx, const Packets& p) {
    int nc = (int)ctx.unit.size(), bc = K().bp_count;
    size_t np = size(p);
    Mask sel(nc * np);
    for (int c = 0; c < nc; ++c) {
        bool active = hasr(page, BONE_PLATING, c, nc) && ctx.now < s.bp_until[c] && s.bp_left[c] > 0;
        for (size_t i = 0; i < np; ++i) {
            bool seen = false;
            for (int q = 0; q < bc; ++q) seen = seen || (p.cast_id[i] == s.bp_seen[c * bc + q] && p.cast_id[i] != 0);
            sel[c * np + i] = active && p.valid[i] && p.dst[i] == ctx.unit[c] && p.src[i] == s.bp_source[c] && !seen;
        }
    }
    sel = first_instance(p, sel, nc);
    for (int c = 0; c < nc; ++c) {
        int rank = 0;
        for (size_t i = 0; i < np; ++i) {
            rank += sel[c * np + i];
            sel[c * np + i] = sel[c * np + i] && rank <= s.bp_left[c];
        }
    }
    return sel;
}

// _guardian.
std::tuple<State, Effects> guardian(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size(), KB = k.gd_buckets;
    float now = ctx.now;
    const Packets& p = ev.report.packets;
    const Resolved& r = ev.report.resolved;
    size_t np = size(p);
    std::vector<float> per_unit(n, 0.f);
    for (size_t i = 0; i < np; ++i) {
        int si = clip_unit(p.src[i], n), di = clip_unit(p.dst[i], n);
        int scls = u.cls[si];
        bool src_ok = scls == CLASS_CHAMPION || scls == CLASS_MONSTER || (scls == CLASS_STRUCTURE && ev.is_turret[si]);
        bool ok = p.valid[i] && src_ok && u.team[si] != u.team[di] && u.cls[di] == CLASS_CHAMPION;
        per_unit[di] = per_unit[di] + (ok ? r.final[i] : 0.f);
    }
    int epoch = (int)std::floor(now / k.gd_bucket);
    int b = ((epoch % KB) + KB) % KB;
    Effects e = no_effects(nc, n);
    e.shields.amount.assign(nc * 2, 0.f), e.shields.kind.assign(nc * 2, 0);
    e.shields.duration.assign(nc * 2, k.gd_shield_duration), e.shields.decay_hold.assign(nc * 2, INF);
    std::vector<uint8_t> trig(nc), guarded(nc * n);
    std::vector<float> amount(nc);
    for (int c = 0; c < nc; ++c) {
        bool stale = s.gd_epoch[c * KB + b] != epoch;
        for (int j = 0; j < n; ++j) {
            bool friendly = u.team[j] == ctx.team[c] && u.cls[j] == CLASS_CHAMPION;
            float& buf = s.gd_buf[((size_t)c * n + j) * KB + b];
            if (stale) buf = 0.f;
            buf = buf + (friendly ? per_unit[j] : 0.f);
        }
        s.gd_epoch[c * KB + b] = epoch;
        float thr = lin(k.gd_thr, ctx.level[c]);
        bool over = false, any_guarded = false;
        for (int j = 0; j < n; ++j) {
            float window = 0.f;
            for (int q = 0; q < KB; ++q) {
                int ep = s.gd_epoch[c * KB + q];
                bool live = ep > epoch - KB && ep >= 0;
                window = window + (live ? s.gd_buf[((size_t)c * n + j) * KB + q] : 0.f);
            }
            bool friendly = u.team[j] == ctx.team[c] && u.cls[j] == CLASS_CHAMPION;
            bool self_n = j == ctx.unit[c] && ctx.unit[c] >= 0;
            float d = std::sqrt(sq(u.x[j] - ctx.x[c]) + sq(u.y[j] - ctx.y[c]));
            bool allies = friendly && !self_n && u.alive[j];
            bool g = allies && (d <= k.gd_range || now < s.gd_guard_until[(size_t)c * n + j]);
            guarded[(size_t)c * n + j] = g;
            any_guarded = any_guarded || g;
            over = over || ((self_n || g) && window >= thr);
        }
        trig[c] = hasr(page, GUARDIAN, c, nc) && ctx.alive[c] && now >= s.gd_cd_until[c] && any_guarded && over;
        amount[c] = lin(k.gd_shield, ctx.level[c]) + k.gd_ap * ev.ap[c] + k.gd_hp * bonus_hp(ctx, c);
    }
    for (int c2 = 0; c2 < nc; ++c2) {
        float ally_amt = 0.f;
        int col = clip_unit(ctx.unit[c2], n);
        for (int c = 0; c < nc; ++c)
            ally_amt = std::max(ally_amt, guarded[(size_t)c * n + col] && trig[c] ? amount[c] : 0.f);
        e.shields.amount[c2 * 2] = trig[c2] ? amount[c2] : 0.f;
        e.shields.amount[c2 * 2 + 1] = ally_amt;
    }
    for (int c = 0; c < nc; ++c)
        if (trig[c]) {
            for (int j = 0; j < n * KB; ++j) s.gd_buf[(size_t)c * n * KB + j] = 0.f;
            s.gd_cd_until[c] = now + lin(k.gd_cd, ctx.level[c]);
        }
    return {s, e};
}
}  // namespace

ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    float now = ctx.now;
    ItemStats o = default_stats();
    o.health.assign(nc, 0.f), o.armor.assign(nc, 0.f), o.magic_resist.assign(nc, 0.f), o.percent_armor.assign(nc, 0.f);
    o.percent_magic_resist.assign(nc, 0.f), o.percent_health.assign(nc, 0.f), o.heal_shield_power.assign(nc, 0.f);
    for (int c = 0; c < nc; ++c) {
        bool cond = hasr(page, CONDITIONING, c, nc) && ev.game_time >= k.cond_time;
        bool as_on = hasr(page, AFTERSHOCK, c, nc) && now < s.as_until[c];
        bool unf = hasr(page, UNFLINCHING, c, nc) && now < s.unf_until[c];
        bool og = hasr(page, OVERGROWTH, c, nc);
        float tiers = std::floor((float)s.og_count[c] / k.og_per_tier);
        o.health[c] = (hasr(page, GRASP, c, nc) ? s.grasp_hp[c] : 0.f) + (og ? k.og_hp_per_tier * tiers : 0.f);
        o.percent_health[c] = og && (float)s.og_count[c] >= k.og_threshold ? k.og_pct : 0.f;
        o.armor[c] = (cond ? k.cond_armor : 0.f) + (as_on ? s.as_armor[c] : 0.f) + (unf ? k.unf_resist : 0.f);
        o.magic_resist[c] = (cond ? k.cond_mr : 0.f) + (as_on ? s.as_mr[c] : 0.f) + (unf ? k.unf_resist : 0.f);
        o.percent_armor[c] = o.percent_magic_resist[c] = cond ? k.cond_pct : 0.f;
        o.heal_shield_power[c] = hasr(page, REVITALIZE, c, nc) ? k.rev_hsp : 0.f;
    }
    return o;
}

Arr<float> heal_mult(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    Arr<float> out(nc);
    for (int c = 0; c < nc; ++c) {
        bool low = ctx.hp[c] < k.rev_cutoff * ctx.max_hp[c];
        out[c] = hasr(page, REVITALIZE, c, nc) && low ? k.rev_amp : 1.f;
    }
    return out;
}

std::tuple<State, Effects> on_cast(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    for (int c = 0; c < nc; ++c) {
        int tgt = ev.cast.target[c], ti = clip_unit(tgt, n);
        bool ally = tgt >= 0 && u.cls[ti] == CLASS_CHAMPION && u.team[ti] == ctx.team[c] && tgt != ctx.unit[c];
        bool go = hasr(page, GUARDIAN, c, nc) && ev.cast.started[c] && ally;
        if (go && tgt < n) s.gd_guard_until[(size_t)c * n + tgt] = ctx.now + K().gd_guard;
    }
    return {s, no_effects(nc, n)};
}

std::tuple<State, Effects> on_hit(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    const Attack& atk = ev.attack;
    Effects e = no_effects(nc, n);
    Packets p_grasp = empty_packets(0), p_sb = empty_packets(0), p_demo = empty_packets(0);
    std::vector<uint8_t> can(nc * n), full(nc * n);
    std::vector<int> stacks(nc * n);
    for (int c = 0; c < nc; ++c) {
        bool hit = atk.hit[c] && ctx.alive[c];
        int tgt = std::max(atk.target[c], 0);
        bool vs_champ = hit && enemy_champion_target(ctx, u, c, atk.target[c]);
        // Grasp proc.
        bool primed = s.grasp_acc[c] + k.eps >= k.grasp_stacks && now - ev.clocks.last_combat[c] < k.grasp_window;
        bool g = hasr(page, GRASP, c, nc) && vs_champ && primed;
        float gmod = by_range(ctx, c, 1.f, k.grasp_ranged_mod);
        push(p_grasp, g, ctx.unit[c], tgt, k.grasp_pct_damage * gmod * ctx.max_hp[c], MAGIC, TAG_PROC | TAG_ON_HIT, 0.f,
             rune_item(GRASP));
        e.heal[c] = g ? k.grasp_pct_heal * gmod * ctx.max_hp[c] : 0.f;
        s.grasp_hp[c] = s.grasp_hp[c] + (g ? by_range(ctx, c, k.grasp_hp_melee, k.grasp_hp_ranged) : 0.f);
        // Shield Bash.
        bool sb = hasr(page, SHIELD_BASH, c, nc) && vs_champ && now < s.sb_until[c] && s.sb_amount[c] > 0.f;
        float sb_dmg = lin(k.sb, ctx.level[c]) + k.sb_hp * bonus_hp(ctx, c) + k.sb_shield * s.sb_amount[c];
        push(p_sb, sb, ctx.unit[c], tgt, sb_dmg, adaptive_damage_type(ev, c), TAG_PROC | TAG_ON_HIT, 0.f,
             rune_item(SHIELD_BASH));
        // Demolish (stacks per holder; consumption resolved across holders below).
        int ti = clip_unit(atk.target[c], n);
        bool on_turret = hasr(page, DEMOLISH, c, nc) && hit && atk.target[c] >= 0
                      && target_class(u, atk.target[c]) == CLASS_STRUCTURE && ev.is_turret[ti] && u.team[ti] != ctx.team[c];
        for (int j = 0; j < n; ++j) {
            size_t q = (size_t)c * n + j;
            bool oh = j == atk.target[c] && atk.target[c] >= 0 && on_turret;
            can[q] = oh && now >= s.demo_cd_until[c] && now >= s.demo_lock_until[j];
            stacks[q] = std::min(s.demo_stacks[q] + (int)can[q], k.demo_stacks);
            full[q] = can[q] && stacks[q] >= k.demo_stacks;
        }
        if (g) s.grasp_acc[c] = 0.f;
        s.grasp_procs[c] += g;
        if (sb) s.sb_amount[c] = 0.f, s.sb_until[c] = -BIG;
    }
    // One consumer per turret per tick (the first holder).
    for (int j = 0; j < n; ++j) {
        int cum = 0;
        for (int c = 0; c < nc; ++c) {
            size_t q = (size_t)c * n + j;
            cum += full[q];
            full[q] = full[q] && cum == 1;
        }
    }
    for (int c = 0; c < nc; ++c) {
        bool consumed = false;
        for (int j = 0; j < n; ++j) consumed = consumed || full[(size_t)c * n + j];
        for (int j = 0; j < n; ++j) s.demo_stacks[(size_t)c * n + j] = consumed ? 0 : stacks[(size_t)c * n + j];
        float demo_dmg = by_range(ctx, c, k.demo_base_melee + k.demo_hp_melee * ctx.max_hp[c],
                                  k.demo_base_ranged + k.demo_hp_ranged * ctx.max_hp[c]);
        push(p_demo, consumed, ctx.unit[c], std::max(atk.target[c], 0), demo_dmg, PHYSICAL, TAG_PROC | TAG_BASIC_ATTACK,
             0.f, rune_item(DEMOLISH));
        if (consumed) s.demo_cd_until[c] = now + k.demo_cooldown;
    }
    for (int j = 0; j < n; ++j) {
        bool any = false;
        for (int c = 0; c < nc; ++c) any = any || full[(size_t)c * n + j];
        if (any) s.demo_lock_until[j] = now + k.demo_lock;
    }
    append(p_grasp, p_sb);
    append(p_grasp, p_demo);
    e.packets = p_grasp;
    return {s, e};
}

std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now, dt = ctx.dt, start = now - dt;
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) {
        // Grasp generation.
        float lc = ev.clocks.last_combat[c];
        float gen = std::min(std::max(lc + k.grasp_gen_after - start, 0.f), dt);
        float acc = std::min(s.grasp_acc[c] + (s.grasp_acc[c] + k.eps < k.grasp_stacks ? gen : 0.f), k.grasp_stacks);
        acc = (now - lc >= k.grasp_window) || !hasr(page, GRASP, c, nc) ? 0.f : acc;
        // Second Wind.
        float sw_t = std::min(std::max(s.sw_until[c] - start, 0.f), dt);
        e.heal_plain[c] = hasr(page, SECOND_WIND, c, nc) && ctx.alive[c] && ctx.hp[c] > 0.f
                        ? k.sw_rate * std::max(ctx.max_hp[c] - ctx.hp[c], 0.f) * sw_t : 0.f;
        // Aftershock burst.
        bool burst = hasr(page, AFTERSHOCK, c, nc) && s.as_pending[c] && now >= s.as_until[c];
        bool fire = burst && ctx.alive[c];
        float as_dmg = lin(k.as_dmg, ctx.level[c]) + k.as_hp_ratio * bonus_hp(ctx, c);
        for (int j = 0; j < n; ++j) {
            bool near = in_circle(u, j, ctx.x[c], ctx.y[c], k.as_radius);
            bool tgt = near && u.team[j] != ctx.team[c] && u.alive[j] && u.targetable[j]
                    && (u.cls[j] == CLASS_CHAMPION || u.cls[j] == CLASS_MONSTER) && fire;
            push(e.packets, tgt, ctx.unit[c], j, as_dmg, MAGIC, TAG_PROC | TAG_AOE, 0.f, rune_item(AFTERSHOCK));
        }
        // Overgrowth.
        int counted = 0;
        for (int j = 0; j < n; ++j) {
            float d2 = sq(u.x[j] - ctx.x[c]) + sq(u.y[j] - ctx.y[c]);
            bool farm = u.cls[j] == CLASS_MINION || u.cls[j] == CLASS_MONSTER;
            counted += ev.deaths[j] && farm && u.team[j] != ctx.team[c] && d2 <= k.og_range_sq && ev.sight[(size_t)c * n + j];
        }
        s.og_count[c] = s.og_count[c] + (hasr(page, OVERGROWTH, c, nc) ? counted : 0);
        if (hasr(page, UNFLINCHING, c, nc) && ev.holder_cc_from_champion[c]) s.unf_until[c] = now + k.unf_linger;
        s.grasp_acc[c] = acc;
        s.as_pending[c] = s.as_pending[c] && !burst;
    }
    return {s, e};
}

std::tuple<State, Effects> on_cc(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    std::vector<uint8_t> f(nc);
    std::vector<float> amount(nc);
    for (int c = 0; c < nc; ++c) {
        bool immob = false, impair = false;
        for (int j = 0; j < n; ++j) {
            size_t q = (size_t)c * n + j;
            bool champ = u.cls[j] == CLASS_CHAMPION && u.team[j] != ctx.team[c];
            immob = immob || (ev.cc.immobilized[q] && champ);
            impair = impair || ((ev.cc.immobilized[q] || ev.cc.slowed[q]) && champ);
        }
        immob = immob && ctx.alive[c], impair = impair && ctx.alive[c];
        // Aftershock.
        bool a = hasr(page, AFTERSHOCK, c, nc) && immob && now >= s.as_cd_until[c];
        float cap = lin(k.as_cap, ctx.level[c]);
        if (a) {
            s.as_until[c] = now + k.as_delay, s.as_pending[c] = 1;
            s.as_armor[c] = std::min(k.as_flat + k.as_pct * ctx.bonus_armor[c], cap);
            s.as_mr[c] = std::min(k.as_flat + k.as_pct * ctx.bonus_mr[c], cap);
            s.as_cd_until[c] = now + k.as_cooldown;
        }
        // Font of Life.
        f[c] = hasr(page, FONT_OF_LIFE, c, nc) && impair && now >= s.font_cd_until[c];
        amount[c] = lin(k.font, ctx.level[c]) * by_range(ctx, c, 1.f, k.font_ranged);
    }
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) e.heal[c] = f[c] ? amount[c] : 0.f;
    std::vector<float> extra(nc, 0.f);
    for (int c = 0; c < nc; ++c) {
        int best = -1;
        float best_key = INF;
        for (int c2 = 0; c2 < nc; ++c2) {
            float d = std::sqrt(sq(ctx.x[c2] - ctx.x[c]) + sq(ctx.y[c2] - ctx.y[c]));
            bool ally = ctx.team[c2] == ctx.team[c] && ctx.unit[c2] != ctx.unit[c] && ctx.alive[c2] && d <= k.font_range;
            float frac = ctx.hp[c2] / std::max(ctx.max_hp[c2], 1.f);
            float key = ally ? frac + d * 1e-7f : INF;
            if (best < 0 || key < best_key) best = c2, best_key = key;
        }
        // argsort(argsort(key)) == 0 marks the first minimum; it must also be an ally.
        if (f[c] && best >= 0 && best_key < INF) extra[best] = extra[best] + amount[c];
    }
    for (int c = 0; c < nc; ++c) e.heal[c] = e.heal[c] + extra[c];
    for (int c = 0; c < nc; ++c)
        if (f[c]) s.font_cd_until[c] = now + k.font_cooldown;
    return {s, e};
}

Arr<float> packet_block(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev, const Packets& p) {
    int nc = (int)ctx.unit.size();
    size_t np = size(p);
    Mask sel = bp_select(s, page, ctx, p);
    Arr<float> out(np, 0.f);
    for (size_t i = 0; i < np; ++i) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) acc = acc + (sel[c * np + i] ? lin(K().bp, ctx.level[c]) : 0.f);
        out[i] = acc;
    }
    return out;
}

std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size(), bc = k.bp_count;
    float now = ctx.now;
    const Packets& p = ev.report.packets;
    const Resolved& r = ev.report.resolved;
    size_t np = size(p);
    Mask used = bp_select(s, page, ctx, p);
    for (int c = 0; c < nc; ++c) {
        // _champion_health_hits.
        bool hurt = false;
        int first = 0;
        for (size_t i = 0; i < np; ++i) {
            int si = clip_unit(p.src[i], n);
            bool hit = p.valid[i] && p.dst[i] == ctx.unit[c] && u.cls[si] == CLASS_CHAMPION && u.team[si] != ctx.team[c]
                    && r.health_loss[i] > 0.f;
            if (hit && !hurt) hurt = true, first = (int)i;
        }
        if (hasr(page, SECOND_WIND, c, nc) && hurt) s.sw_until[c] = now + k.sw_duration;
        // Bone Plating: consume the blocks this pass used, then activate.
        int n_used = 0;
        int before = bc - s.bp_left[c];
        std::vector<int> seen(&s.bp_seen[c * bc], &s.bp_seen[c * bc] + bc);
        for (size_t i = 0; i < np; ++i) {
            if (!used[c * np + i]) continue;
            ++n_used;
            int slot = before + n_used - 1;
            if (slot >= 0 && slot < bc) seen[slot] = p.cast_id[i];
        }
        int left = s.bp_left[c] - n_used;
        bool spent = n_used > 0 && left <= 0;
        float bp_until = spent ? now : s.bp_until[c];
        float bp_cd = spent ? now + k.bp_cooldown : s.bp_cd_until[c];
        bool trig = hasr(page, BONE_PLATING, c, nc) && hurt && now >= s.bp_cd_until[c];
        int src = np ? p.src[first] : 0;
        if (trig) s.bp_source[c] = src;
        s.bp_until[c] = trig ? now + k.bp_duration : bp_until;
        s.bp_left[c] = trig ? bc : left;
        for (int q = 0; q < bc; ++q) s.bp_seen[c * bc + q] = trig ? 0 : seen[q];
        s.bp_cd_until[c] = trig ? now + k.bp_duration + k.bp_cooldown : bp_cd;
    }
    return guardian(s, page, ctx, u, ev);
}

State post_tick(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    for (int c = 0; c < nc; ++c) {
        bool gained = hasr(page, SHIELD_BASH, c, nc) && ev.shield_gained[c] > 0.f;
        bool active = ctx.now < s.sb_until[c];
        float current = active ? s.sb_amount[c] : 0.f;
        float life = ev.shield_gained_duration[c] > 0.f ? ev.shield_gained_duration[c] : k.sb_assumed;
        if (gained) {
            s.sb_amount[c] = std::max(current, ev.shield_gained[c]);
            s.sb_until[c] = std::max(s.sb_until[c], ctx.now + life + k.sb_linger);
        }
    }
    return s;
}

LANESIM_TEST(runes_resolve_stats, "runes.resolve.stats", stats);
LANESIM_TEST(runes_resolve_heal_mult, "runes.resolve.heal_mult", heal_mult);
LANESIM_TEST(runes_resolve_on_cast, "runes.resolve.on_cast", on_cast);
LANESIM_TEST(runes_resolve_on_hit, "runes.resolve.on_hit", on_hit);
LANESIM_TEST(runes_resolve_periodic, "runes.resolve.periodic", periodic);
LANESIM_TEST(runes_resolve_on_cc, "runes.resolve.on_cc", on_cc);
LANESIM_TEST(runes_resolve_packet_block, "runes.resolve.packet_block", packet_block);
LANESIM_TEST(runes_resolve_on_damage, "runes.resolve.on_damage", on_damage);
LANESIM_TEST(runes_resolve_post_tick, "runes.resolve.post_tick", post_tick);

}  // namespace lanesim::runes::resolve
