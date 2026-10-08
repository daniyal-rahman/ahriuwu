// Sorcery tree 8200 (runes/effects/sorcery.py). Stormraider's Surge window buckets and Aery's summoner clock are
// kept for every holder; the delayed rune packets are always emitted (invalid when not due).
#include <algorithm>
#include <cmath>

#include "../marshal.hpp"
#include "runes.hpp"

namespace lanesim::runes::sorcery {

namespace {
enum : int { AERY = 8214, COMET = 8229, STORMRAIDER = 8230, DEATHFIRE = 8992, AXIOM = 8224, MANAFLOW = 8226,
             NIMBUS = 8275, TRANSCENDENCE = 8210, CELERITY = 8234, ABSOLUTE_FOCUS = 8233, SCORCH = 8237,
             WATERWALKING = 8232, GATHERING_STORM = 8236 };

float ea(int perk, const char* name) { return data::f("runes.sorcery." + std::to_string(perk) + "." + name); }
float k(const char* name) { return data::f(std::string("runes.sorcery.") + name); }
const std::vector<float>& lt(const char* name) { return data::table(std::string("runes.sorcery.lin.") + name); }

struct Consts {
    float sr_threshold = k("SR_THRESHOLD"), sr_duration = k("SR_DURATION"), sr_haste = k("SR_HASTE"),
          sr_ranged = k("SR_RANGED"), sr_slow_resist = k("SR_SLOW_RESIST"), sr_bucket = k("SR_BUCKET");
    int sr_window_buckets = (int)k("SR_WINDOW_BUCKETS"), sr_slots = (int)k("SR_SLOTS");
    float aery_travel = k("AERY_TRAVEL"), aery_linger = k("AERY_LINGER"), aery_accel = k("AERY_ACCEL"),
          aery_near = k("AERY_NEAR"), aery_gap = k("AERY_SUMMONER_GAP"), aery_k2 = k("AERY_K2");
    std::vector<float> aery_speed = data::table("runes.sorcery.AERY_RETURN_SPEED");
    float comet_delay = k("COMET_DELAY"), comet_radius = k("COMET_RADIUS"), comet_max_range = k("COMET_MAX_RANGE"),
          comet_max_amp = k("COMET_MAX_AMP");
    float dft_tick = k("DFT_TICK"), dft_amp = k("DFT_AMP"), dft_spell = k("DFT_SPELL"), dft_aoe = k("DFT_AOE"),
          dft_dot = k("DFT_DOT"), dft_amp_at = k("DFT_AMP_AT");
    float scorch_delay = k("SCORCH_DELAY"), scorch_cd = k("SCORCH_CD");
    int mf_cap = (int)k("MF_CAP");
    float eps = k("EPS");
    float aery_ad = ea(AERY, "DamageADRatio"), aery_ap = ea(AERY, "DamageAPRatio");
    float comet_ad = ea(COMET, "ADRatio"), comet_ap = ea(COMET, "APRatio");
    float dft_ad = ea(DEATHFIRE, "ADRatio"), dft_ap = ea(DEATHFIRE, "APRatio");
    float axiom_aoe = ea(AXIOM, "AOEAmp"), axiom_amp = ea(AXIOM, "DamageAmp");
    float mf_mana = ea(MANAFLOW, "ManaIncrease"), mf_cd = ea(MANAFLOW, "Cooldown"),
          mf_restore_cd = ea(MANAFLOW, "PercentManaRestoreCooldown"), mf_restore = ea(MANAFLOW, "PercentManaRestore");
    float nim_duration = ea(NIMBUS, "Duration"), nim_lo = ea(NIMBUS, "{2fd68801}"), nim_hi = ea(NIMBUS, "{b0d06764}"),
          nim_low = ea(NIMBUS, "LowCDMSBoost"), nim_mid = ea(NIMBUS, "{1c32110c}"), nim_high = ea(NIMBUS, "HighCDMSBoost");
    float cel_amp = ea(CELERITY, "PercentHasteMod"), cel_ms = k("CEL_MS");
    float ww_ms = ea(WATERWALKING, "MovementSpeed"), ww_decay = k("WW_DECAY");
    float abs_hp = ea(ABSOLUTE_FOCUS, "HealthPercent");
    float gs_period = k("GS_PERIOD"), gs_half = k("GS_HALF");
    float tr_on1 = ea(TRANSCENDENCE, "LevelToTurnOn"), tr_on2 = ea(TRANSCENDENCE, "LevelToTurnOn2"),
          tr_on3 = ea(TRANSCENDENCE, "LevelToTurnOn3"), tr_bonus1 = ea(TRANSCENDENCE, "HasteBonus1"),
          tr_bonus2 = ea(TRANSCENDENCE, "HasteBonus2");
    float axiom_base = k("AXIOM_BASE"), tr_base = k("TR_BASE");
    std::vector<float> sr_cd = lt("SR_CD"), aery = lt("AERY"), comet = lt("COMET"), comet_cd = lt("COMET_CD"),
                       dft = lt("DFT"), scorch = lt("SCORCH"), ww = lt("WW"), abs = lt("ABS");
};
const Consts& K() {
    static const Consts c;
    return c;
}

bool enemy_champ_unit(const Ctx& ctx, const Units& u, int c, int j) {
    return u.cls[j] == CLASS_CHAMPION && u.team[j] != ctx.team[c];
}

// _dealt: (C, P) holder's non-rune packet landed (> 0) on an enemy champion.
Mask dealt_mask(const Packets& p, const Resolved& r, const Ctx& ctx, const Units& u) {
    size_t nc = ctx.unit.size(), np = size(p);
    int n = (int)u.x.size();
    Mask m(nc * np);
    for (size_t c = 0; c < nc; ++c)
        for (size_t i = 0; i < np; ++i) {
            int d = clip_unit(p.dst[i], n);
            bool champ = u.cls[d] == CLASS_CHAMPION && p.item[i] >= 0 && p.valid[i] && r.final[i] > 0.f;
            m[c * np + i] = p.src[i] == ctx.unit[c] && champ && u.team[d] != ctx.team[c] && ctx.alive[c];
        }
    return m;
}

bool ability(int f) {
    bool spell = has(f, TAG_ACTIVE_SPELL) && !has(f, TAG_BASIC_ATTACK) && !has(f, TAG_ITEM) && !has(f, TAG_PROC);
    return spell || has(f, TAG_PET);
}

float aery_return_time(float dist, float level) {
    const Consts& k = K();
    float v0 = k.aery_speed[1];
    for (size_t i = 2; i + 1 < k.aery_speed.size(); i += 2) v0 = level >= k.aery_speed[i] ? k.aery_speed[i + 1] : v0;
    float far = std::max(dist - k.aery_near, 0.f);
    float t1 = std::log1p(k.aery_accel * far / v0) / k.aery_accel;
    float v1 = v0 + k.aery_accel * far;
    float t2 = std::log1p(k.aery_k2 * std::min(dist, k.aery_near) / v1) / k.aery_k2;
    return t1 + t2;
}

float comet_cooldown(float level) { return std::min(std::max(lin(K().comet_cd, level), 0.3f), 20.f); }

// _nimbus_now: (MS fraction now, still active).
std::pair<float, bool> nimbus_now(const State& s, int c, float now) {
    float left = std::min(std::max(1.f - (now - s.nim_start[c]) / K().nim_duration, 0.f), 1.f);
    return {s.nim_ms[c] * left, (now - s.nim_start[c]) < K().nim_duration};
}

// _manaflow_stack.
void manaflow_stack(State& s, const Page& page, const Ctx& ctx, int c, bool trigger) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    bool go = hasr(page, MANAFLOW, c, nc) && ctx.alive[c] && trigger && ctx.now >= s.mf_cd_until[c]
           && s.mf_stacks[c] < k.mf_cap;
    int stacks = s.mf_stacks[c] + (int)go;
    bool capped = go && stacks >= k.mf_cap;
    s.mf_stacks[c] = stacks;
    if (go) s.mf_cd_until[c] = ctx.now + k.mf_cd;
    if (capped) s.mf_next_restore[c] = ctx.now + k.mf_restore_cd;
}
}  // namespace

ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    float now = ctx.now;
    ItemStats o = default_stats();
    o.percent_move_speed.assign(nc, 0.f), o.slow_resist.assign(nc, 0.f), o.move_speed.assign(nc, 0.f);
    o.bonus_ms_amp.assign(nc, 0.f), o.adaptive_force.assign(nc, 0.f), o.ability_haste.assign(nc, 0.f);
    o.mana.assign(nc, 0.f);
    float m = 1.f + std::floor(ev.game_time / k.gs_period);
    for (int c = 0; c < nc; ++c) {
        float lv = ctx.level[c];
        bool sr_on = hasr(page, STORMRAIDER, c, nc) && now < s.sr_until[c];
        float sr_ms = sr_on ? k.sr_haste * (ctx.is_ranged[c] ? k.sr_ranged : 1.f) : 0.f;
        float sr_resist = sr_on ? k.sr_slow_resist : 0.f;
        float nim = hasr(page, NIMBUS, c, nc) ? nimbus_now(s, c, now).first : 0.f;
        bool cel = hasr(page, CELERITY, c, nc);
        float cel_ms = cel ? k.cel_ms : 0.f;
        bool ww = hasr(page, WATERWALKING, c, nc), river = ev.in_river[c];
        float ww_decay = std::min(std::max(1.f - (now - s.ww_last_river[c]) / k.ww_decay, 0.f), 1.f);
        float ww_ms = ww ? k.ww_ms * (river ? 1.f : ww_decay) : 0.f;
        float ww_af = ww && river ? lin(k.ww, lv) : 0.f;
        bool af_on = hasr(page, ABSOLUTE_FOCUS, c, nc) && ctx.hp[c] > k.abs_hp * ctx.max_hp[c];
        float abs_af = af_on ? lin(k.abs, lv) : 0.f;
        float gs_af = hasr(page, GATHERING_STORM, c, nc) ? k.gs_half * m * (m - 1.f) : 0.f;
        bool tr = hasr(page, TRANSCENDENCE, c, nc);
        float tr_ah = 100.f * ((lv >= k.tr_on1 ? k.tr_bonus1 : 0.f) + (lv >= k.tr_on2 ? k.tr_bonus2 : 0.f));
        float mana = hasr(page, MANAFLOW, c, nc) ? k.mf_mana * (float)s.mf_stacks[c] : 0.f;
        o.percent_move_speed[c] = sr_ms + nim + cel_ms, o.slow_resist[c] = sr_resist, o.move_speed[c] = ww_ms;
        o.bonus_ms_amp[c] = cel ? k.cel_amp : 0.f, o.adaptive_force[c] = ww_af + abs_af + gs_af;
        o.ability_haste[c] = tr ? tr_ah : 0.f, o.mana[c] = mana;
    }
    return o;
}

Arr<float> packet_amp(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev, const Packets& p) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    size_t np = size(p);
    Arr<float> out(np, 0.f);
    for (size_t i = 0; i < np; ++i) {
        bool mine = false;
        for (int c = 0; c < nc; ++c) mine = mine || (p.src[i] == ctx.unit[c] && hasr(page, AXIOM, c, nc));
        bool ult = p.valid[i] && has(p.flags[i], PROP_ULTIMATE);
        float amt = has(p.flags[i], TAG_AOE) ? k.axiom_aoe : k.axiom_amp;
        out[i] = mine && ult ? amt : 0.f;
    }
    return out;
}

std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    const Packets& p = ev.report.packets;
    const Resolved& r = ev.report.resolved;
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    size_t np = size(p);
    float now = ctx.now;
    Mask dealt = dealt_mask(p, r, ctx, u);
    int S = k.sr_slots;

    // Stormraider's Surge: sliding window per enemy champion.
    int kb = (int)std::floor(now / k.sr_bucket + k.eps);
    int slot = ((kb % S) + S) % S;
    for (int c = 0; c < nc; ++c) {
        bool stale = s.sr_bucket[c * S + slot] != kb;
        if (stale)
            for (int j = 0; j < n; ++j) s.sr_buf[((size_t)c * n + j) * S + slot] = 0.f;
        s.sr_bucket[c * S + slot] = kb;
        std::vector<float> add(n, 0.f);
        for (size_t i = 0; i < np; ++i) {
            int d = p.dst[i];
            if (d >= 0 && d < n) add[d] = add[d] + (dealt[c * np + i] ? r.final[i] : 0.f);
        }
        for (int j = 0; j < n; ++j) {
            float& b = s.sr_buf[((size_t)c * n + j) * S + slot];
            b = b + add[j];
        }
        bool over = false;
        for (int j = 0; j < n; ++j) {
            float window = 0.f;
            for (int q = 0; q < S; ++q) {
                int bq = s.sr_bucket[c * S + q];
                bool live = bq > kb - k.sr_window_buckets && bq >= 0;
                window = window + (live ? s.sr_buf[((size_t)c * n + j) * S + q] : 0.f);
            }
            over = over || (enemy_champ_unit(ctx, u, c, j) && window >= k.sr_threshold * u.max_hp[j]);
        }
        bool sr_go = hasr(page, STORMRAIDER, c, nc) && ctx.alive[c] && now >= s.sr_cd_until[c] && over;
        if (sr_go) {
            for (int j = 0; j < n * S; ++j) s.sr_buf[(size_t)c * n * S + j] = 0.f;
            s.sr_until[c] = now + k.sr_duration;
            s.sr_cd_until[c] = now + lin(k.sr_cd, ctx.level[c]);
        }
    }

    for (int c = 0; c < nc; ++c) {
        float lv = ctx.level[c];
        int hu = clip_unit(ctx.unit[c], n);
        float hx = u.x[hu], hy = u.y[hu];
        // Summon Aery.
        bool any_a = false, any_summ = false;
        int ia = 0;
        for (size_t i = 0; i < np; ++i) {
            int f = p.flags[i];
            bool summ = has(f, PROP_SUMMONER);
            bool summ_first = now - s.aery_summ_last[c] > k.aery_gap && summ;
            bool kind = (has(f, TAG_BASIC_ATTACK) || has(f, TAG_ACTIVE_SPELL) || has(f, TAG_ITEM)) && !has(f, TAG_PERIODIC)
                     && !summ;
            bool sel = dealt[c * np + i] && (kind || summ_first);
            if (sel && !any_a) any_a = true, ia = (int)i;
            any_summ = any_summ || (dealt[c * np + i] && summ);
        }
        bool a_go = hasr(page, AERY, c, nc) && any_a && now >= s.aery_free_t[c];
        if (a_go) {
            int a_dst = p.dst[ia];
            int tu = clip_unit(a_dst, n);
            float dist = std::sqrt(sq(u.x[tu] - hx) + sq(u.y[tu] - hy));
            float a_raw = lin(k.aery, lv) + k.aery_ad * ev.bonus_ad[c] + k.aery_ap * ev.ap[c];
            s.aery_due[c] = now + k.aery_travel, s.aery_dst[c] = a_dst, s.aery_raw[c] = a_raw;
            s.aery_dtype[c] = adaptive_damage_type(ev, c);
            s.aery_free_t[c] = now + k.aery_travel + k.aery_linger + aery_return_time(dist, lv);
        }
        if (any_summ) s.aery_summ_last[c] = now;

        // Arcane Comet.
        bool any_c = false;
        int ic = 0;
        for (size_t i = 0; i < np; ++i)
            if (dealt[c * np + i] && ability(p.flags[i])) { any_c = true, ic = (int)i; break; }
        bool c_go = hasr(page, COMET, c, nc) && any_c && now >= s.comet_cd_until[c];
        if (c_go) {
            int cu = clip_unit(p.dst[ic], n);
            float cx = u.x[cu], cy = u.y[cu];
            float cdist = std::min(std::sqrt(sq(cx - hx) + sq(cy - hy)), k.comet_max_range);
            float ad_t = k.comet_ad * ev.bonus_ad[c], ap_t = k.comet_ap * ev.ap[c];
            float c_raw = (lin(k.comet, lv) + ad_t + ap_t) * (1.f + k.comet_max_amp * cdist / k.comet_max_range);
            s.comet_due[c] = now + k.comet_delay, s.comet_x[c] = cx, s.comet_y[c] = cy, s.comet_raw[c] = c_raw;
            s.comet_dtype[c] = variable_damage_type(ad_t, ap_t), s.comet_cd_until[c] = now + comet_cooldown(lv);
        }

        // Deathfire Touch: per-target burn, §5.4 refresh rule.
        if (hasr(page, DEATHFIRE, c, nc)) {
            std::vector<float> new_dur(n, 0.f);
            for (size_t i = 0; i < np; ++i) {
                int f = p.flags[i], d = p.dst[i];
                float dur = has(f, TAG_PET) || has(f, TAG_PERIODIC) ? k.dft_dot : (has(f, TAG_AOE) ? k.dft_aoe : k.dft_spell);
                float v = dealt[c * np + i] && ability(f) ? dur : 0.f;
                if (d >= 0 && d < n) new_dur[d] = std::max(new_dur[d], v);
            }
            float tick = (lin(k.dft, lv) + k.dft_ad * ev.bonus_ad[c] + k.dft_ap * ev.ap[c]) * k.dft_tick;
            for (int j = 0; j < n; ++j) {
                size_t q = (size_t)c * n + j;
                bool burning = s.dft_next[q] <= s.dft_end[q] + k.eps;
                float remaining = s.dft_end[q] - now;
                bool apply = new_dur[j] > 0.f && (!burning || new_dur[j] >= s.dft_total[q] || remaining < new_dur[j]);
                if (!apply) continue;
                if (!burning) s.dft_start[q] = now, s.dft_next[q] = now + k.dft_tick;
                s.dft_end[q] = now + new_dur[j], s.dft_total[q] = new_dur[j], s.dft_dmg[q] = tick;
            }
        }

        // Scorch.
        bool s_go = hasr(page, SCORCH, c, nc) && any_c && now >= s.scorch_cd_until[c];
        if (s_go) {
            s.scorch_due[c] = now + k.scorch_delay, s.scorch_dst[c] = p.dst[ic];
            s.scorch_raw[c] = lin(k.scorch, lv), s.scorch_cd_until[c] = now + k.scorch_cd;
        }

        manaflow_stack(s, page, ctx, c, any_c);
    }
    return {s, no_effects(nc, n)};
}

std::tuple<State, Effects> on_cc(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    for (int c = 0; c < nc; ++c) {
        bool hit = false;
        for (int j = 0; j < n; ++j) {
            size_t q = (size_t)c * n + j;
            hit = hit || ((ev.cc.slowed[q] || ev.cc.immobilized[q]) && enemy_champ_unit(ctx, u, c, j) && u.alive[j]);
        }
        manaflow_stack(s, page, ctx, c, hit);
    }
    return {s, no_effects(nc, n)};
}

std::tuple<State, Effects> on_cast(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    for (int c = 0; c < nc; ++c) {
        float cd = ev.summoner_cooldown[c];
        float boost = cd < k.nim_lo ? k.nim_low : (cd <= k.nim_hi ? k.nim_mid : k.nim_high);
        boost = ev.summoner_is_teleport[c] ? k.nim_high : boost;
        float cur = nimbus_now(s, c, ctx.now).first;
        bool go = hasr(page, NIMBUS, c, nc) && ev.summoner_cast[c] && ctx.alive[c] && boost >= cur;
        if (go) s.nim_ms[c] = boost, s.nim_start[c] = ctx.now;
    }
    return {s, no_effects(nc, n)};
}

std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    Effects e = no_effects(nc, n);
    Packets p_aery = empty_packets(0), p_comet = empty_packets(0), p_dft = empty_packets(0), p_scorch = empty_packets(0);
    for (int c = 0; c < nc; ++c) {
        int holder = ctx.unit[c];
        // Aery.
        bool a_due = now >= s.aery_due[c] - k.eps;
        push(p_aery, a_due && u.alive[clip_unit(s.aery_dst[c], n)], holder, s.aery_dst[c], s.aery_raw[c], s.aery_dtype[c],
             TAG_PROC, 0.f, rune_item(AERY));
        if (a_due) s.aery_due[c] = BIG;
        // Comet.
        bool land = now >= s.comet_due[c] - k.eps;
        for (int j = 0; j < n; ++j) {
            bool champ = enemy_champ_unit(ctx, u, c, j) && u.alive[j] && u.targetable[j];
            bool area = in_circle(u, j, s.comet_x[c], s.comet_y[c], k.comet_radius) && champ && land;
            push(p_comet, area, holder, j, s.comet_raw[c], s.comet_dtype[c], TAG_PROC | TAG_AOE, 0.f, rune_item(COMET));
        }
        if (land) s.comet_due[c] = BIG;
        // Deathfire.
        for (int j = 0; j < n; ++j) {
            size_t q = (size_t)c * n + j;
            bool due = now >= s.dft_next[q] - k.eps && s.dft_next[q] <= s.dft_end[q] + k.eps && u.alive[j];
            float amp = s.dft_next[q] - s.dft_start[q] >= k.dft_amp_at ? k.dft_amp : 1.f;
            push(p_dft, due, holder, j, s.dft_dmg[q] * amp, MAGIC, TAG_PROC | TAG_PERIODIC, 0.f, rune_item(DEATHFIRE));
            float next = due ? s.dft_next[q] + k.dft_tick : s.dft_next[q];
            s.dft_next[q] = u.alive[j] ? next : BIG;
        }
        // Scorch.
        bool s_due = now >= s.scorch_due[c] - k.eps;
        push(p_scorch, s_due && u.alive[clip_unit(s.scorch_dst[c], n)], holder, s.scorch_dst[c], s.scorch_raw[c], MAGIC,
             TAG_PROC | TAG_PERIODIC | TAG_INDIRECT, 0.f, rune_item(SCORCH));
        if (s_due) s.scorch_due[c] = BIG;
        // Manaflow restore.
        bool restore = hasr(page, MANAFLOW, c, nc) && ctx.alive[c] && s.mf_stacks[c] >= k.mf_cap
                    && now >= s.mf_next_restore[c] - k.eps;
        float max_mana = ctx.max_mana[c] + k.mf_mana * (float)s.mf_stacks[c];
        e.mana[c] = restore ? k.mf_restore * std::max(max_mana - ctx.mana[c], 0.f) : 0.f;
        if (restore) s.mf_next_restore[c] = s.mf_next_restore[c] + k.mf_restore_cd;
        if (ev.in_river[c]) s.ww_last_river[c] = now;
    }
    append(p_aery, p_comet);
    append(p_aery, p_dft);
    append(p_aery, p_scorch);
    e.packets = p_aery;
    return {s, e};
}

std::tuple<State, Effects> on_takedown(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    for (int c = 0; c < nc; ++c) {
        float kk = ev.kills.champion_kill[c] + ev.kills.champion_assist[c];
        float ult = 1.f - std::pow(k.axiom_base, kk);
        float basic = 1.f - std::pow(k.tr_base, kk);
        bool tr_on = hasr(page, TRANSCENDENCE, c, nc) && ctx.level[c] >= k.tr_on3;
        s.ult_refund[c] = hasr(page, AXIOM, c, nc) ? ult : 0.f;
        s.basic_refund[c] = tr_on ? basic : 0.f;
    }
    return {s, no_effects(nc, n)};
}

RuneOutputs outputs(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev) {
    int nc = (int)ctx.unit.size();
    RuneOutputs o = no_outputs(nc, n_items());
    o.basic_cd_refund = s.basic_refund, o.ult_cd_refund = s.ult_refund;
    for (int c = 0; c < nc; ++c) o.ghosted[c] = hasr(page, NIMBUS, c, nc) && nimbus_now(s, c, ctx.now).second;
    return o;
}

LANESIM_TEST(runes_sorcery_stats, "runes.sorcery.stats", stats);
LANESIM_TEST(runes_sorcery_packet_amp, "runes.sorcery.packet_amp", packet_amp);
LANESIM_TEST(runes_sorcery_on_damage, "runes.sorcery.on_damage", on_damage);
LANESIM_TEST(runes_sorcery_on_cc, "runes.sorcery.on_cc", on_cc);
LANESIM_TEST(runes_sorcery_on_cast, "runes.sorcery.on_cast", on_cast);
LANESIM_TEST(runes_sorcery_periodic, "runes.sorcery.periodic", periodic);
LANESIM_TEST(runes_sorcery_on_takedown, "runes.sorcery.on_takedown", on_takedown);
LANESIM_TEST(runes_sorcery_outputs, "runes.sorcery.outputs", outputs);

}  // namespace lanesim::runes::sorcery
