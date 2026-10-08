// Inspiration tree 8300 (runes/effects/inspiration.py): grant queue, Cash Back, Time Warp, Biscuits, First Strike,
// Glacial Augment, Unsealed Spellbook, Hextech Flashtraption, Approach Velocity, Jack of All Trades.
#include <algorithm>
#include <cmath>

#include "../marshal.hpp"
#include "runes.hpp"

namespace lanesim::runes::inspiration {

namespace {
enum : int { GLACIAL = 8351, SPELLBOOK = 8360, FIRST_STRIKE = 8369, FLASHTRAPTION = 8306, FOOTWEAR = 8304,
             CASH_BACK = 8321, TRIPLE_TONIC = 8313, TIME_WARP = 8352, BISCUITS = 8345, COSMIC = 8347,
             APPROACH = 8410, JACK = 8316 };
constexpr int BISCUIT_ITEM = 2010, BOOTS_ITEM = 2422, SKILL = 2150, HEALTH_POTION = 2003, REFILLABLE = 2031;
constexpr int FS = -FIRST_STRIKE;

float k(const char* name) { return data::f(std::string("runes.inspiration.") + name); }
const std::vector<float>& t(const char* name) { return data::table(std::string("runes.inspiration.") + name); }

struct Consts {
    int ga_rays = (int)k("GA_RAYS");
    float ga_length = k("GA_LENGTH"), ga_half_width = k("GA_HALF_WIDTH"), ga_duration = k("GA_DURATION"),
          ga_cc_carry = k("GA_CC_CARRY"), ga_cooldown = k("GA_COOLDOWN"), ga_reduction = k("GA_REDUCTION"),
          ga_indent = k("GA_INDENT"), ga_slow_base = k("GA_SLOW_BASE"), ga_slow_bad = k("GA_SLOW_BAD"),
          ga_slow_ap = k("GA_SLOW_AP"), ga_slow_hsp = k("GA_SLOW_HSP"), ga_ally_range = k("GA_ALLY_RANGE");
    std::vector<float> ga_rot = t("GA_ROT");
    float sb_first = k("SB_FIRST"), sb_base = k("SB_BASE"), sb_per_unique = k("SB_PER_UNIQUE"), sb_min = k("SB_MIN"),
          sb_ooc = k("SB_OOC");
    float fs_duration = k("FS_DURATION"), fs_amp = k("FS_AMP"), fs_gold_flat = k("FS_GOLD_FLAT"),
          fs_gold_melee = k("FS_GOLD_MELEE"), fs_gold_ranged = k("FS_GOLD_RANGED"), fs_mode_cd = k("FS_MODE_CD"),
          fs_delay = k("FS_DELAY"), fs_grace_eps = k("FS_GRACE_EPS");
    int fs_slots = (int)k("FS_SLOTS"), fs_push = (int)k("FS_PUSH");
    std::vector<float> fs_cd = t("lin.FS_CD");
    float hx_ms = k("HX_MS"), hx_ms_duration = k("HX_MS_DURATION"), hx_flash_gate = k("HX_FLASH_GATE"),
          hx_cooldown = k("HX_COOLDOWN"), hx_combat_cd = k("HX_COMBAT_CD"), hx_channel_eps = k("HX_CHANNEL_EPS"),
          hx_min_eps = k("HX_MIN_EPS"), hx_range0 = k("HX_RANGE0"), hx_range_step = k("HX_RANGE_STEP"),
          hx_range_period = k("HX_RANGE_PERIOD"), hx_range_max = k("HX_RANGE_MAX");
    float mf_at = k("MF_AT"), mf_per_takedown = k("MF_PER_TAKEDOWN"), mf_ms = k("MF_MS"), cb_refund = k("CB_REFUND");
    float biscuit_every = k("BISCUIT_EVERY"), biscuit_hp = k("BISCUIT_HP");
    int biscuit_count = (int)k("BISCUIT_COUNT"), grant_slots = (int)k("GRANT_SLOTS");
    float ci_summoner = k("CI_SUMMONER"), ci_item = k("CI_ITEM"), av_own = k("AV_OWN"), av_other = k("AV_OTHER"),
          av_range = k("AV_RANGE"), jack_ah = k("JACK_AH"), jack_af5 = k("JACK_AF5"), jack_af10 = k("JACK_AF10");
    std::vector<float> tonics = t("TONICS"), twt_heal = t("TWT_HEAL");
    std::vector<float> ids = t("tab.ids"), boots = t("tab.boots"), legendary = t("tab.legendary"),
                       total = t("tab.total"), jack = t("tab.jack"), slots = t("tab.slots"),
                       max_stack = t("tab.max_stack");
    int ni = (int)ids.size(), jack_types = ni ? (int)(jack.size() / ni) : 0;
};
const Consts& K() {
    static const Consts c;
    return c;
}

// _row_of: (row, valid) of an item id (searchsorted on the sorted catalog ids).
std::pair<int, bool> row_of(int item_id) {
    const auto& ids = K().ids;
    int r = (int)(std::lower_bound(ids.begin(), ids.end(), (float)item_id) - ids.begin());
    r = clampi(r, 0, (int)ids.size() - 1);
    return {r, item_id > 0 && (int)ids[r] == item_id};
}

float slots_used(const Arr<int32_t>& own, int c) {
    const Consts& k = K();
    float s = 0.f;
    for (int i = 0; i < k.ni; ++i) s = s + std::ceil((float)own[(size_t)c * k.ni + i] / k.max_stack[i]) * k.slots[i];
    return s;
}

float jack_stacks(const Arr<int32_t>& own, int c) {
    const Consts& k = K();
    int cnt = 0;
    for (int ty = 0; ty < k.jack_types; ++ty) {
        bool present = false;
        for (int i = 0; i < k.ni && !present; ++i) present = own[(size_t)c * k.ni + i] > 0 && k.jack[i * k.jack_types + ty] > 0.f;
        cnt += present;
    }
    return (float)cnt;
}

// _push: append ``item`` at the first empty slot.
void qpush(Arr<int32_t>& q, int c, int Q, bool go, int item) {
    if (!go) return;
    for (int s = 0; s < Q; ++s)
        if (q[c * Q + s] == 0) { q[c * Q + s] = item; return; }
}
void qpop(Arr<int32_t>& q, int c, int Q, bool go) {
    if (!go) return;
    for (int s = 0; s + 1 < Q; ++s) q[c * Q + s] = q[c * Q + s + 1];
    q[c * Q + Q - 1] = 0;
}

float first_strike_cooldown(float level) { return lin(K().fs_cd, level, false) * K().fs_mode_cd; }
float spellbook_cooldown(int unique) { return std::max(K().sb_base - K().sb_per_unique * (float)unique, K().sb_min); }
float hexflash_range(float elapsed) {
    const Consts& k = K();
    return std::min(k.hx_range0 + k.hx_range_step * std::floor(elapsed / k.hx_range_period + 1e-4f), k.hx_range_max);
}

bool spellbook_ready(const State& s, const Page& page, const Ctx& ctx, const RuneEvents& ev, int c) {
    int nc = (int)ctx.unit.size();
    bool ooc = ctx.now - ev.clocks.last_combat[c] >= K().sb_ooc;
    return hasr(page, SPELLBOOK, c, nc) && ctx.alive[c] && ev.game_time >= s.sb_ready_at[c] && ooc;
}

bool enemy_champion(const Ctx& ctx, const Units& u, int c, int j) {
    return u.cls[j] == CLASS_CHAMPION && u.team[j] != ctx.team[c] && u.alive[j];
}

// zone_mask: (C, N) units touching an active icy zone.
std::vector<uint8_t> zone_mask(const State& s, const Ctx& ctx, const Units& u) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size(), R = k.ga_rays;
    std::vector<uint8_t> m(nc * n, 0);
    for (int c = 0; c < nc; ++c) {
        if (!(ctx.now < s.ga_until[c])) continue;
        for (int j = 0; j < n; ++j) {
            bool any = false;
            for (int r = 0; r < R && !any; ++r) {
                float rx = u.x[j] - s.ga_x0[c * R + r], ry = u.y[j] - s.ga_y0[c * R + r];
                float dx = s.ga_dx[c * R + r], dy = s.ga_dy[c * R + r];
                float along = rx * dx + ry * dy;
                float across = std::fabs(-rx * dy + ry * dx);
                float rad = u.radius[j];
                any = along >= -rad && along <= k.ga_length + rad && across <= k.ga_half_width + rad;
            }
            m[(size_t)c * n + j] = any;
        }
    }
    return m;
}
}  // namespace

ItemStats stats(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size();
    ItemStats o = default_stats();
    o.move_speed.assign(nc, 0.f), o.percent_move_speed.assign(nc, 0.f), o.ability_haste.assign(nc, 0.f);
    o.adaptive_force.assign(nc, 0.f), o.health.assign(nc, 0.f), o.silent_health.assign(nc, 0.f);
    o.summoner_haste.assign(nc, 0.f), o.item_haste.assign(nc, 0.f);
    bool has_own = ev.own.size() > 0;
    for (int c = 0; c < nc; ++c) {
        bool boots = false;
        float jack = 0.f;
        if (has_own) {
            for (int i = 0; i < k.ni; ++i) boots = boots || (ev.own[(size_t)c * k.ni + i] > 0 && k.boots[i] > 0.f);
            jack = hasr(page, JACK, c, nc) ? jack_stacks(ev.own, c) : 0.f;
        }
        float ms = hasr(page, FOOTWEAR, c, nc) && boots ? k.mf_ms : 0.f;
        float hx_ms = hasr(page, FLASHTRAPTION, c, nc) && ctx.now < s.hx_ms_until[c] ? k.hx_ms : 0.f;
        float af = jack >= 10.f ? k.jack_af10 : (jack >= 5.f ? k.jack_af5 : 0.f);
        float biscuit_hp = hasr(page, BISCUITS, c, nc) ? k.biscuit_hp * s.biscuits_sold[c] : 0.f;
        bool cosmic = hasr(page, COSMIC, c, nc);
        o.move_speed[c] = ms;
        o.percent_move_speed[c] = (hasr(page, APPROACH, c, nc) ? s.av_bonus[c] : 0.f) + hx_ms;
        o.ability_haste[c] = k.jack_ah * jack, o.adaptive_force[c] = af;
        o.health[c] = biscuit_hp, o.silent_health[c] = biscuit_hp;
        o.summoner_haste[c] = cosmic ? k.ci_summoner : 0.f, o.item_haste[c] = cosmic ? k.ci_item : 0.f;
    }
    return o;
}

std::tuple<State, Effects> on_cc(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size(), R = k.ga_rays;
    float now = ctx.now;
    for (int c = 0; c < nc; ++c) {
        bool any_imm = false;
        int tgt = 0;
        for (int j = 0; j < n; ++j)
            if (ev.cc.immobilized[(size_t)c * n + j] && enemy_champion(ctx, u, c, j)) { any_imm = true, tgt = j; break; }
        bool go = hasr(page, GLACIAL, c, nc) && ctx.alive[c] && now >= s.ga_cd_until[c] && any_imm;
        if (!go) continue;
        float tx = u.x[tgt], ty = u.y[tgt];
        float dur = k.ga_duration + k.ga_cc_carry * ev.cc_duration[(size_t)c * n + tgt];
        // Ray 0 aims at the holder; the others at the nearest other allied champions near the target, else fan.
        float hx = u.x[ctx.unit[c]], hy = u.y[ctx.unit[c]];
        float d0x = hx - tx, d0y = hy - ty;
        float norm = std::sqrt(sq(d0x) + sq(d0y));
        d0x = norm > 1e-3f ? d0x / std::max(norm, 1e-3f) : -ctx.facing_x[c];
        d0y = norm > 1e-3f ? d0y / std::max(norm, 1e-3f) : -ctx.facing_y[c];
        std::vector<float> key(n);
        std::vector<int> order(n);
        for (int j = 0; j < n; ++j) {
            bool ally = u.cls[j] == CLASS_CHAMPION && u.team[j] == ctx.team[c] && u.alive[j] && j != ctx.unit[c];
            float dist = std::sqrt(sq(u.x[j] - tx) + sq(u.y[j] - ty));
            key[j] = ally && dist <= k.ga_ally_range && dist > 1e-3f ? dist : INF;
            order[j] = j;
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
        std::vector<float> dx(R), dy(R);
        dx[0] = d0x, dy[0] = d0y;
        for (int r = 1; r < R; ++r) {
            int j = n >= r ? order[r - 1] : 0;
            bool has_ally = std::isfinite(key[j]);
            float ax = u.x[j] - tx, ay = u.y[j] - ty;
            float an = std::max(std::sqrt(sq(ax) + sq(ay)), 1e-3f);
            float cs = k.ga_rot[2 * (r - 1)], sn = k.ga_rot[2 * (r - 1) + 1];
            float fx = d0x * cs - d0y * sn, fy = d0x * sn + d0y * cs;
            dx[r] = has_ally ? ax / an : fx;
            dy[r] = has_ally ? ay / an : fy;
        }
        for (int r = 0; r < R; ++r) {
            s.ga_x0[c * R + r] = tx - k.ga_indent * dx[r], s.ga_y0[c * R + r] = ty - k.ga_indent * dy[r];
            s.ga_dx[c * R + r] = dx[r], s.ga_dy[c * R + r] = dy[r];
        }
        s.ga_until[c] = now + dur, s.ga_cd_until[c] = now + k.ga_cooldown;
    }
    // Slow enemy non-structures inside a zone, for one tick at a time.
    std::vector<uint8_t> zone = zone_mask(s, ctx, u);
    Effects e = no_effects(nc, n);
    for (int j = 0; j < n; ++j) {
        float slow = 0.f;
        for (int c = 0; c < nc; ++c) {
            bool tg = zone[(size_t)c * n + j] && u.team[j] != ctx.team[c] && u.alive[j] && u.cls[j] != CLASS_STRUCTURE
                   && hasr(page, GLACIAL, c, nc);
            float strength = k.ga_slow_base + k.ga_slow_bad * ev.bonus_ad[c] + k.ga_slow_ap * ev.ap[c]
                           + k.ga_slow_hsp * ctx.heal_shield_power[c];
            slow = std::max(slow, tg ? strength : 0.f);
        }
        e.slow[j] = slow;
        e.slow_duration[j] = (slow > 0.f ? ctx.dt : 0.f) * 1.f;
    }
    return {s, e};
}

Arr<float> packet_amp(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev, const Packets& p) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    size_t np = size(p);
    std::vector<uint8_t> zone = zone_mask(s, ctx, u);
    Arr<float> out(np, 0.f);
    for (size_t i = 0; i < np; ++i) {
        int src = clip_unit(p.src[i], n), dst = clip_unit(p.dst[i], n);
        bool hit = false;
        for (int c = 0; c < nc; ++c) {
            bool from_zone = zone[(size_t)c * n + src] && u.team[src] != ctx.team[c] && hasr(page, GLACIAL, c, nc);
            bool to_ally = u.team[dst] == ctx.team[c] && u.cls[dst] == CLASS_CHAMPION && p.dst[i] != ctx.unit[c];
            hit = hit || (from_zone && to_ally);
        }
        out[i] = hit && p.valid[i] ? -k.ga_reduction : 0.f;
    }
    return out;
}

std::tuple<State, Effects> periodic(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size(), Q = k.grant_slots, KS = k.fs_slots, ni = k.ni;
    float now = ctx.now, gt = ev.game_time;
    Effects e = no_effects(nc, n);
    bool has_own = ev.own.size() > 0;
    for (int c = 0; c < nc; ++c) {
        // Grant queue: pop the acknowledged head, push due grants.
        int head = s.grant_q[c * Q];
        bool acked = head != 0 && ev.granted[c] == head;
        s.boots_received[c] = s.boots_received[c] || (acked && head == BOOTS_ITEM);
        qpop(s.grant_q, c, Q, acked);
        bool due_b = hasr(page, BISCUITS, c, nc) && s.biscuits_sched[c] < k.biscuit_count
                  && gt >= k.biscuit_every * (float)(s.biscuits_sched[c] + 1);
        qpush(s.grant_q, c, Q, due_b, BISCUIT_ITEM);
        s.biscuits_sched[c] += due_b;
        bool full = has_own ? slots_used(ev.own, c) >= 6.f : false;
        int bits = s.tonic_bits[c], skill_now = 0;
        for (size_t q = 0; q + 1 < k.tonics.size(); q += 2) {
            int kk = (int)(q / 2), lvl = (int)k.tonics[q], item = (int)k.tonics[q + 1];
            bool go = hasr(page, TRIPLE_TONIC, c, nc) && ctx.level[c] >= (float)lvl && ((bits >> kk) & 1) == 0;
            if (go) bits |= 1 << kk;
            if (item == SKILL) {
                if (go && full) skill_now = 1;
                go = go && !full;
            }
            qpush(s.grant_q, c, Q, go, item);
        }
        float takedowns = s.takedowns[c] + ev.kills.champion_kill[c] + ev.kills.champion_assist[c];
        float boots_due = k.mf_at - k.mf_per_takedown * takedowns;
        bool go_boots = hasr(page, FOOTWEAR, c, nc) && !s.boots_queued[c] && gt >= boots_due;
        qpush(s.grant_q, c, Q, go_boots, BOOTS_ITEM);

        // Cash Back.
        bool cb = hasr(page, CASH_BACK, c, nc);
        auto [rs, vs] = row_of(ev.sold[c]);
        bool sold_back = cb && vs && s.refunds[(size_t)c * ni + rs] > 0;
        if (sold_back) s.refunds[(size_t)c * ni + rs] -= 1;
        auto [rb, vb] = row_of(ev.purchased[c]);
        bool bought = cb && vb && k.legendary[rb] > 0.f;
        if (bought) s.refunds[(size_t)c * ni + rb] += 1;
        float gold = (bought ? k.cb_refund * k.total[rb] : 0.f) - (sold_back ? k.cb_refund * k.total[rs] : 0.f);

        // Time Warp Tonic.
        int pot = ev.potion_drunk[c];
        float heal = pot == HEALTH_POTION ? k.twt_heal[0] : (pot == REFILLABLE ? k.twt_heal[1] : 0.f);
        heal = hasr(page, TIME_WARP, c, nc) && ctx.alive[c] ? heal : 0.f;
        bool sold_biscuit = hasr(page, BISCUITS, c, nc) && ev.sold[c] == BISCUIT_ITEM;

        // First Strike: emit due packets; pay the gold once the buff and its missiles are done.
        bool any_due = false, all_free = true;
        for (int q = 0; q < KS; ++q) {
            size_t i = (size_t)c * KS + q;
            bool due = s.fs_due[i] <= now + 1e-4f;
            bool alive_dst = u.alive[clip_unit(s.fs_dst[i], n)];
            push(e.packets, due && alive_dst && s.fs_amt[i] > 0.f, ctx.unit[c], s.fs_dst[i], s.fs_amt[i], TRUE_DMG,
                 TAG_PROC | TAG_INDIRECT, 0.f, FS);
            if (due) s.fs_due[i] = BIG, s.fs_amt[i] = 0.f;
            any_due = any_due || due;
            all_free = all_free && s.fs_due[i] >= BIG / 2;
        }
        bool done = now > s.fs_until[c] && all_free && !any_due;
        float pct = ctx.is_ranged[c] ? k.fs_gold_ranged : k.fs_gold_melee;
        float pay = done && s.fs_gold_acc[c] > 0.f ? pct * s.fs_gold_acc[c] : 0.f;
        if (done) s.fs_gold_acc[c] = 0.f;

        // Unsealed Spellbook.
        int req = ev.spellbook_request[c];
        bool ready = spellbook_ready(s, page, ctx, ev, c);
        bool recent = false;
        for (int q = 0; q < 3; ++q) recent = recent || s.sb_recent[c * 3 + q] == req;
        bool allowed = ready && req > 0 && !recent;
        if (allowed) {
            s.sb_mask[c] = s.sb_mask[c] | (1 << clampi(req, 0, 30));
            int unique = __builtin_popcount((uint32_t)s.sb_mask[c]);
            s.sb_recent[c * 3 + 2] = s.sb_recent[c * 3 + 1], s.sb_recent[c * 3 + 1] = s.sb_recent[c * 3];
            s.sb_recent[c * 3] = req;
            s.sb_ready_at[c] = now + spellbook_cooldown(unique);
        }

        s.tonic_bits[c] = bits, s.boots_queued[c] = s.boots_queued[c] || go_boots, s.takedowns[c] = takedowns;
        s.skill_now[c] = skill_now, s.biscuits_sold[c] = s.biscuits_sold[c] + (float)sold_biscuit;
        s.fs_gold_tick[c] = pay;
        e.heal_plain[c] = heal, e.gold[c] = gold + pay;
    }
    // Packets come out holder-major: (C, K) flattened, like the JAX (C, K) packets.
    return {s, e};
}

std::tuple<State, Effects> on_damage(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size(), KS = k.fs_slots;
    const Packets& p = ev.report.packets;
    const Resolved& r = ev.report.resolved;
    size_t np = size(p);
    float now = ctx.now;
    const CombatClocks& clk = ev.clocks;
    Effects e = no_effects(nc, n);
    for (int c = 0; c < nc; ++c) {
        bool has_fs = hasr(page, FIRST_STRIKE, c, nc);
        float acc_sum = 0.f;
        bool hit = false;
        std::vector<uint8_t> mine(np), to_champ(np);
        for (size_t i = 0; i < np; ++i) {
            int d = clip_unit(p.dst[i], n);
            mine[i] = p.valid[i] && p.src[i] == ctx.unit[c];
            to_champ[i] = u.cls[d] == CLASS_CHAMPION && u.team[d] != ctx.team[c];
            bool fs_pkt = p.item[i] == FS;
            acc_sum = acc_sum + (mine[i] && fs_pkt ? r.final[i] : 0.f);
            hit = hit || (mine[i] && to_champ[i] && !fs_pkt);
        }
        float acc = s.fs_gold_acc[c] + acc_sum;
        bool ready = has_fs && now >= s.fs_cd_until[c];
        bool new_episode = clk.champion_combat_start[c] >= now - 1e-6f;
        bool lockout = ready && new_episode && !clk.struck_first[c];
        bool fire = ready && !lockout && clk.struck_first[c] && (now - clk.champion_combat_start[c] <= k.fs_grace_eps) && hit;
        float cd = first_strike_cooldown(ctx.level[c]);
        float cd_until = fire || lockout ? now + cd : s.fs_cd_until[c];
        float fs_until = fire ? now + k.fs_duration : s.fs_until[c];
        float flat = fire ? k.fs_gold_flat : 0.f;
        // Queue a share of post-mitigation champion damage while active.
        bool active = has_fs && now <= fs_until;
        std::vector<float> bonus(n, 0.f);
        for (size_t i = 0; i < np; ++i) {
            bool sel = mine[i] && to_champ[i] && p.item[i] != FS && r.final[i] > 0.f && active;
            int d = p.dst[i];
            if (d >= 0 && d < n) bonus[d] = bonus[d] + (sel ? k.fs_amp * r.final[i] : 0.f);
        }
        float* due = &s.fs_due[(size_t)c * KS];
        int* dsts = &s.fs_dst[(size_t)c * KS];
        float* amt = &s.fs_amt[(size_t)c * KS];
        float t_new = now + k.fs_delay;
        for (int it = 0; it < k.fs_push; ++it) {
            int j = (int)(std::max_element(bonus.begin(), bonus.end()) - bonus.begin());
            float b = bonus[j];
            bool go = b > 0.f;
            int same_k = -1, free_k = -1, latest = 0;
            float latest_v = 0.f;
            for (int q = 0; q < KS; ++q) {
                bool free = due[q] >= BIG / 2;
                if (same_k < 0 && std::fabs(due[q] - t_new) < 1e-6f && dsts[q] == j) same_k = q;
                if (free_k < 0 && free) free_k = q;
                float v = dsts[q] == j ? (free ? -BIG : due[q]) : -2 * BIG;
                if (q == 0 || v > latest_v) latest = q, latest_v = v;
            }
            int kq = same_k >= 0 ? same_k : (free_k >= 0 ? free_k : latest);
            if (go) {
                bool fresh = due[kq] >= BIG / 2;
                if (fresh) due[kq] = t_new, dsts[kq] = j;
                if (dsts[kq] == j) amt[kq] = amt[kq] + b;
            }
            bonus[j] = 0.f;
        }
        s.fs_cd_until[c] = cd_until, s.fs_until[c] = fs_until, s.fs_gold_acc[c] = acc;
        s.fs_gold_tick[c] = s.fs_gold_tick[c] + flat;
        e.gold[c] = flat;
    }
    return {s, e};
}

State post_tick(State s, const Page& page, const Ctx& ctx, const Units& u, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), n = (int)u.x.size();
    float now = ctx.now;
    for (int c = 0; c < nc; ++c) {
        // Hextech Flashtraption.
        bool hx = hasr(page, FLASHTRAPTION, c, nc);
        float haste = std::max(ev.summoner_haste[c], hasr(page, COSMIC, c, nc) ? k.ci_summoner : 0.f);
        float haste_mult = 100.f / (100.f + haste);
        bool combat = hx && ev.clocks.last_champion_combat[c] >= now - 1e-6f;
        bool avail = hx && ctx.alive[c] && ev.flash_cooldown[c] > k.hx_flash_gate && now >= s.hx_cd_until[c];
        int req = ev.hexflash_request[c];
        bool start = !s.hx_channel[c] && req == 1 && avail && !combat;
        bool channel = s.hx_channel[c] || start;
        float t0 = start ? now : s.hx_start[c];
        float elapsed = now - t0;
        bool interrupted = channel && (combat || !ctx.alive[c]);
        bool release = channel && !interrupted && !start && (req == 2 || elapsed >= k.hx_channel_eps);
        bool early = release && elapsed < k.hx_min_eps;
        bool blink = release && !early;
        float cd_until = s.hx_cd_until[c];
        cd_until = combat ? std::max(cd_until, now + k.hx_combat_cd * haste_mult) : cd_until;
        cd_until = interrupted || early ? now + k.hx_combat_cd * haste_mult : cd_until;
        cd_until = blink ? now + k.hx_cooldown * haste_mult : cd_until;
        s.hx_channel[c] = channel && !interrupted && !release, s.hx_start[c] = t0, s.hx_cd_until[c] = cd_until;
        s.hx_blink[c] = blink, s.hx_range[c] = blink ? hexflash_range(elapsed) : 0.f;
        if (blink) s.hx_ms_until[c] = now + k.hx_ms_duration;

        // Approach Velocity.
        bool own = false, other = false;
        for (int j = 0; j < n; ++j) {
            size_t q = (size_t)c * n + j;
            bool champs = enemy_champion(ctx, u, c, j);
            float dx = u.x[j] - ctx.x[c], dy = u.y[j] - ctx.y[c];
            bool facing = dx * ctx.facing_x[c] + dy * ctx.facing_y[c] >= 0.f;
            float dist = std::sqrt(sq(dx) + sq(dy));
            own = own || (champs && facing && ev.impaired_by_holder[q]);
            other = other || (champs && facing && ev.visible[q] && ev.movement_impaired[j] && dist <= k.av_range);
        }
        float bonus = own ? k.av_own : (other ? k.av_other : 0.f);
        s.av_bonus[c] = hasr(page, APPROACH, c, nc) && ctx.alive[c] ? bonus : 0.f;
    }
    return s;
}

RuneOutputs outputs(State s, const Page& page, const Ctx& ctx, const RuneEvents& ev) {
    const Consts& k = K();
    int nc = (int)ctx.unit.size(), Q = k.grant_slots;
    RuneOutputs o = no_outputs(nc, n_items());
    for (int c = 0; c < nc; ++c) {
        o.grant_item[c] = s.grant_q[c * Q];
        bool forbid = hasr(page, FOOTWEAR, c, nc) && !s.boots_received[c];
        for (int i = 0; i < k.ni; ++i) o.forbid_purchase[(size_t)c * k.ni + i] = forbid && k.boots[i] > 0.f;
        o.skill_points[c] = s.skill_now[c], o.move_locked[c] = s.hx_channel[c], o.blink[c] = s.hx_blink[c];
        o.blink_range[c] = s.hx_range[c], o.spellbook_swap_ready[c] = spellbook_ready(s, page, ctx, ev, c);
        o.first_strike_gold[c] = s.fs_gold_tick[c];
    }
    return o;
}

LANESIM_TEST(runes_inspiration_stats, "runes.inspiration.stats", stats);
LANESIM_TEST(runes_inspiration_on_cc, "runes.inspiration.on_cc", on_cc);
LANESIM_TEST(runes_inspiration_packet_amp, "runes.inspiration.packet_amp", packet_amp);
LANESIM_TEST(runes_inspiration_periodic, "runes.inspiration.periodic", periodic);
LANESIM_TEST(runes_inspiration_on_damage, "runes.inspiration.on_damage", on_damage);
LANESIM_TEST(runes_inspiration_post_tick, "runes.inspiration.post_tick", post_tick);
LANESIM_TEST(runes_inspiration_outputs, "runes.inspiration.outputs", outputs);

}  // namespace lanesim::runes::inspiration
