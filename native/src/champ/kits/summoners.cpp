// Summoner's Rift summoner spells, 26.19: champions/summoners.py (step and the helpers the world calls). Role-quest
// numbers from role_quest.py (unleashed_tp_cooldown, tp_arrival_shield, FREE_TP_COOLDOWN, TP_SHIELD_*).
#include <cmath>

#include "../marshal.hpp"
#include "../stats.hpp"
#include "kits.hpp"

namespace lanesim::kits::summoners {

using namespace champ;

namespace {

constexpr float READY_EPS = 1e-4f;
constexpr float FLASH_RANGE = 400.f, TP_CHANNEL = 3.f, TP_UPGRADE_TIME = 600.f, TP_UPGRADE_FLOOR = 2.f;
constexpr float UTP_MS = .5f, UTP_MS_DURATION = 4.f, TP_FORGIVE_DIST = 2000.f, TP_FORGIVE_RADIUS = 400.f;
constexpr float IGNITE_RANGE = 600.f, IGNITE_FIRST = .25f, IGNITE_PERIOD = 1.056f, GRIEVOUS_DURATION = 5.f;
constexpr int IGNITE_TICKS = 5;
constexpr int IGNITE_FLAGS = TAG_PERIODIC | PROP_SUMMONER | PROP_NO_DAMAGE_MOD | PROP_NO_OMNIVAMP;
constexpr float EXHAUST_RANGE = 650.f, EXHAUST_DURATION = 3.f, EXHAUST_SLOW = .40f, EXHAUST_REDUCTION = .35f;
constexpr float BARRIER_DURATION = 2.5f, HEAL_ALLY_RANGE = 900.f, HEAL_CURSOR = 200.f, HEAL_MS = .30f;
constexpr float HEAL_MS_DURATION = 1.f, HEAL_REPEAT = .5f, HEAL_DEBUFF = 30.f, GHOST_DURATION = 10.f;
constexpr float CLEANSE_TENACITY = .75f, CLEANSE_DURATION = 3.f;
// role_quest
constexpr float FREE_TP_COOLDOWN = 390.f, TP_SHIELD_FRACTION = .35f, TP_SHIELD_DURATION = 10.f,
                QUEST_TP_REDUCTION = 30.f;

float base_cooldown(int spell) {      // COOLDOWN
    switch (spell) {
        case FLASH: case TELEPORT: return 300.f;
        case IGNITE: case BARRIER: return 180.f;
        case EXHAUST: case HEAL: case GHOST: case CLEANSE: return 240.f;
        case SMITE: return 15.f;
        default: return 0.f;
    }
}

// stat_pipeline.cooldown / rescale_cooldown
float cd(float base, float haste) { return base * 100.f / (100.f + std::min(haste, stats::HASTE_CAP)); }
float rescale(float remaining, float old_haste, float new_haste) {
    return remaining * (100.f + std::min(old_haste, stats::HASTE_CAP)) / (100.f + std::min(new_haste, stats::HASTE_CAP));
}

// role_quest.unleashed_tp_cooldown
float unleashed_tp_cooldown(float lv, bool quest_complete) {
    float v = 330.f - 10.f * (std::min(lv, 9.f) - 1.f) - (lv >= 10.f ? 10.f : 0.f);
    return v - (quest_complete ? QUEST_TP_REDUCTION : 0.f);
}

// runes.catalog.lin
float lin(float start, float span, float level) {
    float lv = std::max(level, 1.f);
    return start + span * (lv - 1.f) / 17.f;
}

}  // namespace

float ignite_total(float level) {
    float lv = std::max(level, 1.f);
    return 70.f + 20.f * (std::min(lv, 5.f) - 1.f) + 25.f * std::max(lv - 5.f, 0.f);
}
float barrier_amount(float level) { return lin(100.f, 360.f, level); }
float heal_amount(float level) { return lin(80.f, 238.f, level); }
float ghost_ms(float level) { return lin(.24f, (float)(0.48 - 0.24), level); }   // the unused lin(4, 7) is ignored
float tp_dash_time(float dist, bool unleashed) {
    float b0 = .5f, b1 = unleashed ? 3.5f : 4.5f, cap = unleashed ? 18000.f : 5000.f;
    return b0 + b1 * std::min(dist, cap) / cap;
}

Arr<float> flash_cooldown(const State& s, float now) {
    size_t c = s.haste.size();
    Arr<float> out(c, 0.f);
    for (size_t h = 0; h < c; ++h) {
        float total = 0.f;
        for (int k = 0; k < 2; ++k)
            total = total + (s.spell[h * 2 + k] == FLASH ? std::max(s.ready_at[h * 3 + k] - now, 0.f) : 0.f);
        out[h] = total;
    }
    return out;
}

// summoners.step: haste upkeep, the 10:00 upgrade, TP phases, casts, Ignite ticks.
std::tuple<State, Effects, SummonerOut> step(State s, const Ctx& ctx, const WorldUnits& u, const CastOrder& request,
                                             float now, float dt, const Arr<float>& summoner_haste,
                                             const Arr<uint8_t>& can_cast, const Arr<uint8_t>& channel_interrupted,
                                             const Arr<uint8_t>& quest_complete, const Arr<uint8_t>& rooted_in) {
    const int c = (int)s.haste.size(), n = (int)u.x.size();
    auto at = [](const auto& a, int h) { return a.size() == 1 ? a[0] : a[h]; };        // () or (C,) inputs
    auto rooted = [&](int h) { return rooted_in.size() ? (bool)at(rooted_in, h) : false; };
    const bool suppressed = false, nearsighted = false;                              // world passes None
    Arr<float> ready = s.ready_at;                                                   // (C, 3)
    Arr<float> haste(c), level(c);
    for (int h = 0; h < c; ++h) haste[h] = at(summoner_haste, h) * 1.f, level[h] = ctx.level[h];

    Effects eff = no_effects(c, n);
    SummonerOut out;
    out.dash = no_dash(c);
    auto init = [&](auto& a, size_t len, auto v) { a.assign(len, v); };
    init(out.teleport_start, c, 0), init(out.teleport_channel, c, 0), init(out.teleport_dash, c, 0);
    init(out.teleport_arrive, c, 0), init(out.teleport_x, c, 0.f), init(out.teleport_y, c, 0.f);
    init(out.teleport_target, c, 0), init(out.arrival_shield, c, 0.f), init(out.ghosted, c, 0);
    init(out.bonus_ms_pct, c, 0.f), init(out.tenacity, c, 0.f), init(out.exhaust_reduction, n, 0.f);
    init(out.exhaust_slow, n, 0.f), init(out.exhaust_slow_duration, n, 0.f), init(out.cleanse, c, 0);
    init(out.cast_event, c, 0), init(out.cast_cooldown, c, 0.f), init(out.cast_spell, c, 0);
    init(out.is_teleport, c, 0), init(out.blinked, c, 0), init(out.ignite_target, c, -1);
    init(out.cooldowns, (size_t)c * 2, 0.f), init(out.quest_cooldown, c, 0.f);

    // Per-caster values carried between the phases below.
    std::vector<int> spell(c), slot(c), tgt(c), ti(c), tp_tgt(c), phase(c);
    std::vector<uint8_t> quest_tp(c), upgraded(c), done_channel(c), arrive(c), flash(c), tp(c), ignite(c),
        exhaust(c), barrier(c), heal(c), ghost(c), cleanse(c), instant(c), unleashed_cast(c);
    std::vector<float> t_end(c), shield_tp(c), utp_ms_until(c), spell_cd(c), tp_d(c), tx(c), ty(c), this_cd(c), rx(c),
        ry(c);

    // tp_cd: every Teleport is hasted (INFERRED).
    auto tp_cd = [&](int h, int sl, bool unleashed) {
        float unl = cd(unleashed_tp_cooldown(level[h], quest_complete[h] && sl < QUEST_SLOT), haste[h]);
        return sl == QUEST_SLOT ? cd(FREE_TP_COOLDOWN, haste[h]) : (unleashed ? unl : cd(base_cooldown(TELEPORT), haste[h]));
    };

    for (int h = 0; h < c; ++h) {
        float* rd = &ready[h * 3];
        // ---- haste rescale (U-S1)
        bool seen = s.haste[h] >= 0.f;
        float ratio = seen ? rescale(1.f, s.haste[h], haste[h]) : 1.f;
        for (int k = 0; k < 3; ++k) {
            float rem = std::max(s.ready_at[h * 3 + k] - now, 0.f);
            rd[k] = rem > 0.f ? now + rem * ratio : s.ready_at[h * 3 + k];
        }
        bool own_tp[2] = {s.spell[h * 2] == TELEPORT, s.spell[h * 2 + 1] == TELEPORT};
        bool has_tp = own_tp[0] || own_tp[1];
        // ---- quest free TP (§3.4, U-S13 ready on grant)
        bool grant = quest_complete[h] && !has_tp && !s.quest_tp[h];
        quest_tp[h] = s.quest_tp[h] || grant;
        if (grant) rd[QUEST_SLOT] = now;
        // ---- 10:00 Unleashed transformation (§3.3, U-S9)
        bool upgrade = has_tp && !s.upgraded[h] && now >= TP_UPGRADE_TIME;
        float cap = cd(unleashed_tp_cooldown(1.f, quest_complete[h]), haste[h]);
        for (int k = 0; k < 2; ++k) {
            float rem = std::max(rd[k] - now, 0.f);
            bool idle_tp = own_tp[k] && s.tp_phase[h] == IDLE;
            float up_rem = std::max(TP_UPGRADE_FLOOR, std::min(rem, cap));
            if (upgrade && idle_tp) rd[k] = now + up_rem;
        }
        upgraded[h] = s.upgraded[h] || upgrade;

        // ---- TP phase transitions (§3.1)
        int ph = s.tp_phase[h];
        float te = s.tp_t_end[h];
        this_cd[h] = tp_cd(h, s.tp_slot[h], s.tp_unleashed[h]);
        bool in_channel = ph == CHANNEL;
        done_channel[h] = in_channel && now >= te - READY_EPS;
        bool interrupted = in_channel && !done_channel[h] &&
                           (channel_interrupted[h] || !ctx.alive[h] || rooted(h) || suppressed);
        for (int k = 0; k < 3; ++k)
            if (k == s.tp_slot[h] && interrupted) rd[k] = now + this_cd[h];        // U-S6
        float channel_end = te;
        if (done_channel[h]) te = channel_end + s.tp_dash_time[h];
        ph = interrupted ? IDLE : (done_channel[h] ? DASHING : ph);
        arrive[h] = ph == DASHING && now >= te - READY_EPS;
        for (int k = 0; k < 3; ++k)
            if (k == s.tp_slot[h] && arrive[h]) rd[k] = te + this_cd[h];
        if (arrive[h]) ph = IDLE;
        shield_tp[h] = arrive[h] ? (quest_complete[h] && s.tp_slot[h] < QUEST_SLOT ? TP_SHIELD_FRACTION * ctx.max_hp[h] : 0.f)
                                 : 0.f;
        utp_ms_until[h] = arrive[h] && s.tp_unleashed[h] ? now + UTP_MS_DURATION : s.utp_ms_until[h];

        // ---- casts (§1.4)
        int sl = slot[h] = request.slot[h];
        bool has_quest_slot = quest_tp[h] && sl == QUEST_SLOT;
        spell[h] = sl == QUEST_SLOT ? (quest_tp[h] ? TELEPORT : 0)
                                    : ((sl == 0 || sl == 1) ? s.spell[h * 2 + clampi(sl, 0, 1)] : 0);
        bool slot_ready = rd[clampi(sl, 0, 2)] <= now + READY_EPS;
        bool busy = ph != IDLE;
        bool base_ok = sl >= 0 && (sl < QUEST_SLOT || has_quest_slot) && slot_ready && ctx.alive[h] && !busy && !suppressed;
        bool mobile_ok = base_ok && can_cast[h] && !rooted(h);          // Flash / Teleport

        tgt[h] = request.target[h];
        int t = ti[h] = clampi(tgt[h], 0, n - 1);
        float dist_t = std::sqrt((u.x[t] - ctx.x[h]) * (u.x[t] - ctx.x[h]) + (u.y[t] - ctx.y[h]) * (u.y[t] - ctx.y[h]));
        bool enemy_champ = tgt[h] >= 0 && u.kind[t] == KIND_CHAMPION && u.team[t] != ctx.team[h] && u.alive[t] &&
                           u.targetable[t];

        // Flash: blink min(|cursor|, 400) toward the cursor.
        flash[h] = mobile_ok && spell[h] == FLASH;
        rx[h] = request.x[h], ry[h] = request.y[h];
        float vx = rx[h] - ctx.x[h], vy = ry[h] - ctx.y[h];
        float d = std::sqrt(vx * vx + vy * vy);
        float scale = d > FLASH_RANGE ? FLASH_RANGE / std::max(d, 1e-6f) : 1.f;
        out.dash.active[h] = flash[h];
        out.dash.to_x[h] = flash[h] ? ctx.x[h] + vx * scale : 0.f;
        out.dash.to_y[h] = flash[h] ? ctx.y[h] + vy * scale : 0.f;
        out.dash.speed[h] = flash[h] ? INF : 0.f;
        out.dash.target[h] = -1, out.dash.blink[h] = flash[h];

        // Teleport: allied minion/structure target (forgiveness snap near a far click).
        auto tp_valid = [&](int j) {
            int kd = u.kind[j];
            bool kind_ok = kd == KIND_MINION || kd == KIND_TURRET || kd == KIND_INHIBITOR || kd == KIND_NEXUS;
            return u.team[j] == ctx.team[h] && u.alive[j] && u.targetable[j] && kind_ok;
        };
        bool direct = tgt[h] >= 0 && tp_valid(t);
        bool click_far = std::sqrt(vx * vx + vy * vy) >= TP_FORGIVE_DIST;
        int snap = 0;
        float best = INF;
        for (int j = 0; j < n; ++j) {
            float dclick = std::sqrt((u.x[j] - rx[h]) * (u.x[j] - rx[h]) + (u.y[j] - ry[h]) * (u.y[j] - ry[h]));
            float key = tp_valid(j) && dclick <= TP_FORGIVE_RADIUS ? dclick : INF;
            if (key < best) best = key, snap = j;
        }
        bool can_snap = click_far && std::isfinite(best);
        tp_tgt[h] = direct ? t : (can_snap ? snap : -1);
        tp[h] = mobile_ok && spell[h] == TELEPORT && !nearsighted && tp_tgt[h] >= 0;
        int tpi = clampi(tp_tgt[h], 0, n - 1);
        tx[h] = u.x[tpi], ty[h] = u.y[tpi];
        unleashed_cast[h] = sl == QUEST_SLOT ? true : (bool)upgraded[h];
        tp_d[h] = std::sqrt((tx[h] - ctx.x[h]) * (tx[h] - ctx.x[h]) + (ty[h] - ctx.y[h]) * (ty[h] - ctx.y[h]));
        if (tp[h]) ph = CHANNEL, te = now + TP_CHANNEL;
        phase[h] = ph, t_end[h] = te;

        ignite[h] = base_ok && spell[h] == IGNITE && enemy_champ && dist_t <= IGNITE_RANGE;
        exhaust[h] = base_ok && spell[h] == EXHAUST && enemy_champ && dist_t <= EXHAUST_RANGE;
        barrier[h] = base_ok && spell[h] == BARRIER;
        heal[h] = base_ok && spell[h] == HEAL;
        ghost[h] = base_ok && spell[h] == GHOST;
        cleanse[h] = base_ok && spell[h] == CLEANSE;
        instant[h] = flash[h] || ignite[h] || exhaust[h] || barrier[h] || heal[h] || ghost[h] || cleanse[h];
        spell_cd[h] = 0.f;
        for (int sid : {FLASH, IGNITE, EXHAUST, BARRIER, HEAL, GHOST, CLEANSE})
            if (spell[h] == sid) spell_cd[h] = cd(base_cooldown(sid), haste[h]);
        for (int k = 0; k < 3; ++k)
            if (k == sl && instant[h]) rd[k] = now + spell_cd[h];
    }

    // Ignite: recast overrides; GW 40% for 5 s on cast. Exhaust: refresh, no stacking; slow on the cast tick.
    std::vector<int> ig_target(c), ex_target(c);
    std::vector<float> ig_next(c), ig_left(c), ig_per(c), ex_until(c);
    for (int h = 0; h < c; ++h) {
        ig_target[h] = ignite[h] ? tgt[h] : s.ig_target[h];
        ig_next[h] = ignite[h] ? now + (dt >= IGNITE_FIRST ? 0.f : IGNITE_FIRST) : s.ig_next[h];
        ig_left[h] = ignite[h] ? (float)IGNITE_TICKS : s.ig_left[h];
        ig_per[h] = ignite[h] ? ignite_total(level[h]) / (float)IGNITE_TICKS : s.ig_per_tick[h];
        if (ignite[h]) eff.grievous[ti[h]] = std::max(eff.grievous[ti[h]], GRIEVOUS_DURATION);
        ex_target[h] = exhaust[h] ? tgt[h] : s.ex_target[h];
        ex_until[h] = exhaust[h] ? now + EXHAUST_DURATION : s.ex_until[h];
        if (exhaust[h]) {
            out.exhaust_slow[ti[h]] = std::max(out.exhaust_slow[ti[h]], EXHAUST_SLOW);
            out.exhaust_slow_duration[ti[h]] = std::max(out.exhaust_slow_duration[ti[h]], EXHAUST_DURATION);
        }
    }
    // Cleanse removes Ignite DoTs and Exhaust on the caster (GW stays, §9).
    std::vector<uint8_t> cleansed(n, 0);
    for (int h = 0; h < c; ++h)
        if (cleanse[h] && ctx.unit[h] >= 0 && ctx.unit[h] < n) cleansed[ctx.unit[h]] = 1;
    std::vector<float> cleanse_until(c), ghost_until(c), ghost_pct(c);
    for (int h = 0; h < c; ++h) {
        if (ig_target[h] >= 0 && cleansed[clampi(ig_target[h], 0, n - 1)]) ig_target[h] = -1;
        if (ex_target[h] >= 0 && cleansed[clampi(ex_target[h], 0, n - 1)]) ex_target[h] = -1;
        cleanse_until[h] = cleanse[h] ? now + CLEANSE_DURATION : s.cleanse_until[h];
        ghost_until[h] = ghost[h] ? now + GHOST_DURATION : s.ghost_until[h];
        ghost_pct[h] = ghost[h] ? ghost_ms(level[h]) : s.ghost_pct[h];
    }

    // Heal: self + one ally champion (cursor within 200, else lowest %HP within 900).
    std::vector<float> base_heal(c), rep(c), self_heal(c), ally_heal(c, 0.f);
    std::vector<int> pick(c);
    std::vector<uint8_t> has_ally(c);
    for (int h = 0; h < c; ++h) {
        std::vector<float> key(c);
        std::vector<uint8_t> ally(c), near(c);
        bool any_near = false;
        for (int a = 0; a < c; ++a) {
            bool same = ctx.team[h] == ctx.team[a] && h != a && ctx.alive[a];
            float d_ally = std::sqrt((ctx.x[a] - ctx.x[h]) * (ctx.x[a] - ctx.x[h]) +
                                     (ctx.y[a] - ctx.y[h]) * (ctx.y[a] - ctx.y[h]));
            ally[a] = same && d_ally <= HEAL_ALLY_RANGE;
            float d_cur = std::sqrt((ctx.x[a] - rx[h]) * (ctx.x[a] - rx[h]) + (ctx.y[a] - ry[h]) * (ctx.y[a] - ry[h]));
            near[a] = ally[a] && d_cur <= HEAL_CURSOR;
            key[a] = d_cur;                                    // (overwritten below)
            any_near |= near[a];
        }
        float best = INF;
        pick[h] = 0;
        for (int a = 0; a < c; ++a) {
            float pct = ctx.hp[a] / std::max(ctx.max_hp[a], 1.f);
            float kv = any_near ? (near[a] ? key[a] : INF) : (ally[a] ? pct : INF);
            if (kv < best) best = kv, pick[h] = a;
        }
        has_ally[h] = heal[h] && std::isfinite(best);
        bool debuffed = s.heal_debuff_until[h] > now;
        rep[h] = debuffed ? HEAL_REPEAT : 1.f;
        base_heal[h] = heal_amount(level[h]);
        self_heal[h] = heal[h] ? base_heal[h] * rep[h] : 0.f;
    }
    // Ally heal uses the caster's heal power; the integrator applies the recipient's, so pre-divide by it.
    std::vector<float> heal_debuff_until(c), heal_ms_until(c);
    for (int a = 0; a < c; ++a) {
        float sum = 0.f;
        bool hit_any = false;
        for (int h = 0; h < c; ++h) {
            bool hit = pick[h] == a && has_ally[h];
            sum = sum + (hit ? base_heal[h] * (1.f + ctx.heal_shield_power[h]) : 0.f);
            hit_any |= hit;
        }
        ally_heal[a] = sum * rep[a] / (1.f + ctx.heal_shield_power[a]);
        bool healed = heal[a] || hit_any;
        heal_debuff_until[a] = healed ? now + HEAL_DEBUFF : s.heal_debuff_until[a];
        heal_ms_until[a] = healed ? now + HEAL_MS_DURATION : s.heal_ms_until[a];
    }

    // ---- Ignite ticks (after casts so a dt >= 0.25 cast ticks at once)
    Packets pk = empty_packets(0);
    for (int h = 0; h < c; ++h) {
        int ig_ti = clampi(ig_target[h], 0, n - 1);
        bool ig_alive = ig_target[h] >= 0 && u.alive[ig_ti];
        float left = ig_alive ? ig_left[h] : 0.f;
        bool due_now = left > 0.f && now >= ig_next[h] - READY_EPS;
        float k = due_now ? std::min(std::floor((now - ig_next[h]) / IGNITE_PERIOD + READY_EPS) + 1.f, left) : 0.f;
        push(pk, k > 0.f, ctx.unit[h], ig_ti, k * ig_per[h], TRUE_DMG, IGNITE_FLAGS, 0.f, 0, 0);
        ig_left[h] = left - k;
        ig_next[h] = ig_next[h] + k * IGNITE_PERIOD;
        ig_target[h] = ig_left[h] > 0.f ? ig_target[h] : -1;
    }

    // ---- outputs
    Arr<float> barrier_amt(c), shield_t(c);
    for (int h = 0; h < c; ++h) {
        bool ex_live = ex_target[h] >= 0 && ex_until[h] > now;
        if (ex_live) {
            int j = clampi(ex_target[h], 0, n - 1);
            out.exhaust_reduction[j] = std::max(out.exhaust_reduction[j], EXHAUST_REDUCTION);
        }
        barrier_amt[h] = barrier[h] ? barrier_amount(level[h]) : 0.f;
        shield_t[h] = shield_tp[h];
        eff.heal[h] = self_heal[h] + ally_heal[h];
    }
    eff.packets = pk;
    eff.shields = shield_grants(barrier_amt, SHIELD_ALL, BARRIER_DURATION);
    concat_shields(eff.shields, shield_grants(shield_t, SHIELD_ALL, TP_SHIELD_DURATION), c);

    State ns = s;
    for (int h = 0; h < c; ++h) {
        bool ghosted = now < ghost_until[h];
        float bonus_ms = (ghosted ? ghost_pct[h] : 0.f) + (now < heal_ms_until[h] ? HEAL_MS : 0.f) +
                         (now < utp_ms_until[h] ? UTP_MS : 0.f);
        float tp_event_cd = this_cd[h];
        ns.haste[h] = haste[h], ns.quest_tp[h] = quest_tp[h], ns.upgraded[h] = upgraded[h], ns.tp_phase[h] = phase[h];
        if (tp[h]) {
            ns.tp_slot[h] = slot[h], ns.tp_unleashed[h] = unleashed_cast[h];
            ns.tp_dash_time[h] = tp_dash_time(tp_d[h], unleashed_cast[h]);
            ns.tp_x[h] = tx[h], ns.tp_y[h] = ty[h], ns.tp_target[h] = tp_tgt[h];
        }
        ns.tp_t_end[h] = t_end[h], ns.utp_ms_until[h] = utp_ms_until[h];
        ns.ig_target[h] = ig_target[h], ns.ig_next[h] = ig_next[h], ns.ig_left[h] = ig_left[h];
        ns.ig_per_tick[h] = ig_per[h], ns.ex_target[h] = ex_target[h], ns.ex_until[h] = ex_until[h];
        ns.heal_debuff_until[h] = heal_debuff_until[h], ns.heal_ms_until[h] = heal_ms_until[h];
        ns.ghost_until[h] = ghost_until[h], ns.ghost_pct[h] = ghost_pct[h], ns.cleanse_until[h] = cleanse_until[h];

        out.teleport_start[h] = tp[h], out.teleport_channel[h] = phase[h] == CHANNEL;
        out.teleport_dash[h] = phase[h] == DASHING, out.teleport_arrive[h] = arrive[h];
        out.teleport_x[h] = ns.tp_x[h], out.teleport_y[h] = ns.tp_y[h], out.teleport_target[h] = ns.tp_target[h];
        out.arrival_shield[h] = shield_tp[h], out.ghosted[h] = ghosted, out.bonus_ms_pct[h] = bonus_ms;
        out.tenacity[h] = now < cleanse_until[h] ? CLEANSE_TENACITY : 0.f;
        out.cleanse[h] = cleanse[h];
        out.cast_event[h] = instant[h] || done_channel[h];
        out.cast_cooldown[h] = instant[h] ? spell_cd[h] : (done_channel[h] ? tp_event_cd : 0.f);
        out.cast_spell[h] = instant[h] ? spell[h] : (done_channel[h] ? TELEPORT : 0);
        out.is_teleport[h] = done_channel[h], out.blinked[h] = flash[h] || arrive[h];
        out.ignite_target[h] = ignite[h] ? tgt[h] : -1;
        for (int k = 0; k < 2; ++k) out.cooldowns[h * 2 + k] = std::max(ready[h * 3 + k] - now, 0.f);
        out.quest_cooldown[h] = quest_tp[h] ? std::max(ready[h * 3 + QUEST_SLOT] - now, 0.f) : INF;
    }
    ns.ready_at = ready;
    ns.now = now;
    return {ns, eff, out};
}

namespace {
// Replay adapter: the captured keyword arguments come back in sorted key order (jax pytree flattening of the
// kwargs dict), not in the order world/phases/casts.py passes them.
std::tuple<State, Effects, SummonerOut> step_captured(State s, const Ctx& ctx, const WorldUnits& u,
                                                      const Arr<uint8_t>& can_cast,
                                                      const Arr<uint8_t>& channel_interrupted, float dt, float now,
                                                      const Arr<uint8_t>& quest_complete, const CastOrder& request,
                                                      const Arr<uint8_t>& rooted, const Arr<float>& summoner_haste) {
    return step(s, ctx, u, request, now, dt, summoner_haste, can_cast, channel_interrupted, quest_complete, rooted);
}
}  // namespace

LANESIM_TEST(summoners_step, "summoners.step", step_captured);

}  // namespace lanesim::kits::summoners
