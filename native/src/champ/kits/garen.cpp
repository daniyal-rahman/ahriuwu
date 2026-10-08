// Garen (86), 26.19: champions/garen.py. Constants: native/python/consts/kits_garen.py.
#include <cmath>

#include "../marshal.hpp"
#include "../rng.hpp"
#include "kits.hpp"

namespace lanesim::kits::garen {

using namespace champ;
using detail::ranked;

namespace {

constexpr int E_FLAGS = TAG_AOE | TAG_ACTIVE_SPELL | TAG_PERIODIC;
constexpr int Q_FLAGS = TAG_ACTIVE_SPELL;
constexpr int R_FLAGS = TAG_ACTIVE_SPELL | PROP_ULTIMATE;

struct Consts {
    std::vector<float> q_ms_duration, q_base, q_silence, w_shield, w_dr, e_num_ticks, e_as_per_tick, e_base, e_ad,
        r_base, r_execute, cd[4];
    float Q_WINDOW, Q_MS, Q_AD_EXTRA, Q_RANGE_BONUS, Q_ATTACK_TIME, Q_ATTACK_TIME_AS, Q_WINDUP_FRACTION,
        Q_LOCK_FRACTION, W_SHIELD_RATIO, W_UPFRONT, W_TENACITY, W_DR_DURATION, W_RESIST_PER_STACK, W_MAX_STACKS,
        E_DURATION, E_RADIUS, E_MIN_SPIN_EPS, E_NEAREST_MULT, E_CRIT_MOD, E_SHRED, E_SHRED_DURATION, R_RANGE,
        R_CAST_TIME, PASSIVE_DELAY, PASSIVE_PULSE, PULSE_FRACTION;
    int E_SHRED_HITS;
};

const Consts& K() {
    static const Consts k = [] {
        Consts k;
        auto t = [](const char* name) { return detail::table(std::string("kits.garen.") + name); };
        auto f = [](const char* name) { return data::f(std::string("kits.garen.") + name); };
        k.q_ms_duration = t("Q.MovementSpeedDuration"), k.q_base = t("Q.BaseDamage");
        k.q_silence = t("Q.SilenceDuration"), k.w_shield = t("W.BaseShield"), k.w_dr = t("W.DRPercent");
        k.e_num_ticks = t("E.NumTicks"), k.e_as_per_tick = t("E.ASPerTick"), k.e_base = t("E.BaseDamagePerTick");
        k.e_ad = t("E.ADRatioPerTick"), k.r_base = t("R.BaseDamage"), k.r_execute = t("R.ExecuteDamage");
        k.cd[0] = t("cd.Q"), k.cd[1] = t("cd.W"), k.cd[2] = t("cd.E"), k.cd[3] = t("cd.R");
        k.Q_WINDOW = f("Q_WINDOW"), k.Q_MS = f("Q_MS"), k.Q_AD_EXTRA = f("Q_AD_EXTRA");
        k.Q_RANGE_BONUS = f("Q_RANGE_BONUS"), k.Q_ATTACK_TIME = f("Q_ATTACK_TIME");
        k.Q_ATTACK_TIME_AS = f("Q_ATTACK_TIME_AS"), k.Q_WINDUP_FRACTION = f("Q_WINDUP_FRACTION");
        k.Q_LOCK_FRACTION = f("Q_LOCK_FRACTION"), k.W_SHIELD_RATIO = f("W_SHIELD_RATIO");
        k.W_UPFRONT = f("W_UPFRONT"), k.W_TENACITY = f("W_TENACITY"), k.W_DR_DURATION = f("W_DR_DURATION");
        k.W_RESIST_PER_STACK = f("W_RESIST_PER_STACK"), k.W_MAX_STACKS = f("W_MAX_STACKS");
        k.E_DURATION = f("E_DURATION"), k.E_RADIUS = f("E_RADIUS"), k.E_MIN_SPIN_EPS = f("E_MIN_SPIN_EPS");
        k.E_NEAREST_MULT = f("E_NEAREST_MULT"), k.E_CRIT_MOD = f("E_CRIT_MOD"), k.E_SHRED = f("E_SHRED");
        k.E_SHRED_DURATION = f("E_SHRED_DURATION"), k.E_SHRED_HITS = (int)f("E_SHRED_HITS");
        k.R_RANGE = f("R_RANGE"), k.R_CAST_TIME = f("R_CAST_TIME"), k.PASSIVE_DELAY = f("PASSIVE_DELAY");
        k.PASSIVE_PULSE = f("PASSIVE_PULSE"), k.PULSE_FRACTION = f("PULSE_FRACTION");
        return k;
    }();
    return k;
}

bool mine(const KitCtx& k, int c) { return k.champion_id[c] == ID; }

// _base_cd: (C, 4) base cooldowns at the holder's ranks, 0 for other kits' holders.
Arr<float> base_cd(const KitCtx& k) {
    const Consts& G = K();
    size_t c = k.unit.size();
    Arr<float> out(c * 4, 0.f);
    for (size_t h = 0; h < c; ++h)
        if (mine(k, (int)h))
            for (int s = 0; s < 4; ++s) out[h * 4 + s] = detail::by_rank(G.cd[s], k.ranks[h * 4 + s]);
    return out;
}

}  // namespace

float regen_rate(float lv) {
    return 1.5f + 0.2f * std::min(std::max(lv - 1.f, 0.f), 5.f) + 0.8f * std::min(std::max(lv - 6.f, 0.f), 7.f) +
           0.4f * std::max(lv - 13.f, 0.f);
}

float q_attack_time(float bonus_attack_speed) { return K().Q_ATTACK_TIME - K().Q_ATTACK_TIME_AS * bonus_attack_speed; }

float e_crit_multiplier(float crit_damage) { return 1.f + K().E_CRIT_MOD * (crit_damage - 1.f); }

// garen.cast
std::tuple<State, KitOut> cast(State s, const KitCtx& k, const WorldUnits& u, const CastOrder& order) {
    const Consts& G = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    KitOut out = no_out(c, n);
    Arr<float> shield(c, 0.f);
    const float now = k.now;
    for (int h = 0; h < c; ++h) {
        const int* r = &k.ranks[h * 4];
        bool free = mine(k, h) && k.alive[h] && !k.stunned[h] && !k.silenced[h] && !s.r_pending[h];
        bool ready[4];
        for (int i = 0; i < 4; ++i) ready[i] = r[i] > 0 && k.cooldowns[h * 4 + i] <= 0.f;
        auto want = [&](int sl) { return order.slot[h] == sl && free; };
        int target = order.target[h], t = clampi(target, 0, n - 1);
        bool valid_r = target >= 0 && u.alive[t] && u.targetable[t] && u.kind[t] == KIND_CHAMPION &&
                       u.team[t] != k.team[h] && detail::target_dist(k, u, h, target) <= G.R_RANGE + u.radius[t];
        bool q = want(0) && ready[0] && !s.q_on[h];
        bool w = want(1) && ready[1];
        bool e_start = want(2) && ready[2] && !s.e_on[h];
        bool rr = want(3) && ready[3] && valid_r;
        // Recast after 1 s ends Judgment; Demacian Justice interrupts it.
        bool e_cancel = s.e_on[h] && ((want(2) && (k.now - s.e_start[h] >= G.E_MIN_SPIN_EPS)) || rr);
        bool started = q || w || e_start || rr;
        int slot = q ? 0 : w ? 1 : e_start ? 2 : rr ? 3 : -1;
        int cid = started ? make_cast_id(k, h, std::max(slot, 0)) : 0;

        float ticks = ranked(G.e_num_ticks, r[2]) +
                      std::floor(std::max(k.bonus_attack_speed[h], 0.f) / ranked(G.e_as_per_tick, r[2]) + 1e-6f);
        shield[h] = w ? ranked(G.w_shield, r[1]) + G.W_SHIELD_RATIO * k.bonus_hp[h] : 0.f;
        // A new spin keeps the hit count on targets whose shred is still running (GAR.E4).
        if (e_start)
            for (int j = 0; j < n; ++j)
                if (!(k.now < s.shred_until[(size_t)h * n + j])) s.e_hits[(size_t)h * n + j] = 0;
        s.q_on[h] = s.q_on[h] | q;
        if (q) {
            s.q_until[h] = later(k, G.Q_WINDOW);
            s.q_cast_id[h] = cid;
            s.q_haste_until[h] = now + ranked(G.q_ms_duration, r[0]);
            s.reset_at[h] = now;
        }
        s.q_haste_on[h] = s.q_haste_on[h] | q;
        s.w_on[h] = s.w_on[h] | w;
        s.w_ten_on[h] = s.w_ten_on[h] | w;
        if (w) {
            s.w_until[h] = later(k, G.W_DR_DURATION);
            s.w_dr[h] = ranked(G.w_dr, r[1]);
            s.w_ten_until[h] = later(k, G.W_UPFRONT);
        }
        s.e_on[h] = (s.e_on[h] | e_start) & !e_cancel;
        if (e_start) s.e_start[h] = now, s.e_ticks[h] = (int32_t)ticks, s.e_done[h] = 0;
        s.r_pending[h] = s.r_pending[h] | rr;
        if (rr) {
            s.r_fire_at[h] = later(k, G.R_CAST_TIME);
            s.r_target[h] = target;
            s.r_seq[h] = u.spawn_seq[t];
            s.r_cast_id[h] = cid;
        }
        out.cooldown_start[h * 4 + 0] = 0, out.cooldown_start[h * 4 + 1] = w;
        out.cooldown_start[h * 4 + 2] = e_cancel, out.cooldown_start[h * 4 + 3] = rr;
        out.attack_reset[h] = q, out.cast_started[h] = started, out.cast_slot[h] = slot, out.cast_id[h] = cid;
        out.cast_lockout[h] = rr ? G.R_CAST_TIME : 0.f;
        out.cleanse_slow[h] = q;
    }
    out.shield = shield_grants(shield, SHIELD_ALL, G.W_UPFRONT);
    out.base_cooldown = base_cd(k);
    return {s, out};
}

// garen.periodic
std::tuple<State, KitOut> periodic(State s, const KitCtx& k, const WorldUnits& u) {
    const Consts& G = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    KitOut out = no_out(c, n);
    Packets p_e = empty_packets(0), p_r = empty_packets(0);
    rng::Key key{s.key[0], s.key[1]};
    rng::Key tick_key = rng::fold_in(key, (uint32_t)tick_index(k));
    int tick_id_base = make_cast_id(k, 0, CODE_GAREN_E_TICK);
    std::vector<float> d2(n);
    for (int h = 0; h < c; ++h) {
        const int* r = &k.ranks[h * 4];
        bool g = mine(k, h), alive = k.alive[h], dead = !alive;
        // Buff clocks (expire on the tick the countdown reaches 0, or on death).
        bool q_end = s.q_on[h] && (due(k, s.q_until[h]) || dead);
        bool q_haste_on = s.q_haste_on[h] && !(due(k, s.q_haste_until[h]) || dead);
        bool w_on = s.w_on[h] && !(due(k, s.w_until[h]) || dead);
        bool w_ten_on = s.w_ten_on[h] && !(due(k, s.w_ten_until[h]) || dead);

        // Judgment: spin k completes at e_start + k * Duration / n (at most one per sim tick).
        int count = s.e_ticks[h];
        float next_at = s.e_start[h] + (float)(s.e_done[h] + 1) * G.E_DURATION / (float)std::max(count, 1);
        bool fires = s.e_on[h] && g && alive && s.e_done[h] < count && due(k, next_at);
        int nearest = 0;
        float best = INF;
        for (int j = 0; j < n; ++j) {
            d2[j] = detail::center_dist(k, u, h, j);
            bool near = detail::enemy(k, u, h, j) && d2[j] <= G.E_RADIUS + u.radius[j];
            float key_j = near ? d2[j] : INF;
            if (key_j < best) best = key_j, nearest = j;
        }
        float roll = rng::uniform(tick_key, (uint32_t)h);
        bool crit = roll < k.crit_chance[h];
        float power = ranked(G.e_base, r[2]) + ranked(G.e_ad, r[2]) * (k.base_ad[h] + k.bonus_ad[h]);
        float scaled = power * (crit ? e_crit_multiplier(k.crit_damage[h]) : 1.f);
        int flags = E_FLAGS | (crit ? PROP_CRIT : 0);
        int tick_id = tick_id_base + h * ID_STRIDE;
        for (int j = 0; j < n; ++j) {
            size_t cj = (size_t)h * n + j;
            bool near = detail::enemy(k, u, h, j) && d2[j] <= G.E_RADIUS + u.radius[j];
            bool hit = fires && near;
            float bonus = j == nearest ? G.E_NEAREST_MULT : 1.f;
            push(p_e, hit, k.unit[h], j, scaled * bonus, PHYSICAL, flags, 0.f, 0, tick_id);
            int hits = s.e_hits[cj] + (int)hit;
            int kk = G.E_SHRED_HITS;
            bool shred = hit && u.kind[j] == KIND_CHAMPION &&
                         (hits == kk || hits == kk + 1 || (hits > kk + 1 && (hits - (kk + 1)) % kk == 0));
            if (shred) s.shred_until[cj] = later_after_tick(k, G.E_SHRED_DURATION);
            s.e_hits[cj] = hits;
        }
        int e_done = s.e_done[h] + (int)fires;
        bool e_end = s.e_on[h] && (e_done >= count || dead);

        // Demacian Justice: true damage at the end of the cast time.
        int rt = clampi(s.r_target[h], 0, n - 1);
        bool r_due = s.r_pending[h] && (due(k, s.r_fire_at[h]) || dead);
        bool r_fire = r_due && alive && u.alive[rt] && u.spawn_seq[rt] == s.r_seq[h];
        float missing = std::max(u.max_hp[rt] - u.hp[rt], 0.f);
        float r_raw = ranked(G.r_base, r[3]) + ranked(G.r_execute, r[3]) * missing;
        push(p_r, r_fire, k.unit[h], rt, r_raw, TRUE_DMG, R_FLAGS, 0.f, 0, s.r_cast_id[h]);

        // Perseverance: RegenCalc per 5 s, paid in 0.5 s pulses while not disabled.
        bool on = g && alive && k.now >= s.p_block_until[h] - 1e-6f;
        out.heal[h] = on ? k.max_hp[h] * regen_rate((float)k.level[h]) / 100.f * G.PULSE_FRACTION *
                               pulses(k, G.PASSIVE_PULSE)
                         : 0.f;

        s.q_on[h] = s.q_on[h] & !q_end;
        s.q_haste_on[h] = q_haste_on, s.w_on[h] = w_on, s.w_ten_on[h] = w_ten_on;
        s.e_on[h] = s.e_on[h] & !e_end;
        s.e_done[h] = e_done;
        s.r_pending[h] = s.r_pending[h] & !r_due;
        out.cooldown_start[h * 4 + 0] = q_end && g;
        out.cooldown_start[h * 4 + 2] = e_end && g;
    }
    append(p_e, p_r);
    out.packets = p_e;
    out.base_cooldown = base_cd(k);
    return {s, out};
}

// garen.on_attack
std::tuple<State, KitOut> on_attack(State s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& launch) {
    return {s, no_out((int)k.unit.size(), (int)u.x.size())};
}

// garen.on_hit: the empowered Q attack. ``dodging`` (N,): target dodges basic attacks.
std::tuple<State, KitOut> on_hit(State s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& launch,
                                 const Arr<uint8_t>& dodging) {
    const Consts& G = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    KitOut out = no_out(c, n);
    Packets p_q = empty_packets(0);
    for (int h = 0; h < c; ++h) {
        const int* r = &k.ranks[h * 4];
        int target = detail::holder_row(launch.target, k, h);
        bool hit = detail::holder_row(launch.launched, k, h) && mine(k, h) && k.alive[h] && s.q_on[h] && target >= 0;
        int t = clampi(target, 0, n - 1);
        bool land = hit && !detail::gather_b(dodging, target);
        float raw = ranked(G.q_base, r[0]) + G.Q_AD_EXTRA * (k.base_ad[h] + k.bonus_ad[h]);
        push(p_q, land, k.unit[h], t, raw, PHYSICAL, Q_FLAGS, 0.f, 0, s.q_cast_id[h]);
        if (land) {
            out.cc.silence[(size_t)h * n + t] = ranked(G.q_silence, r[0]);
            out.cc.cast_id[(size_t)h * n + t] = s.q_cast_id[h];
        }
        float lock = k.now + G.Q_LOCK_FRACTION * q_attack_time(k.bonus_attack_speed[h]);
        s.q_on[h] = s.q_on[h] & !hit;
        if (hit) s.q_lock_until[h] = lock;
        out.cooldown_start[h * 4 + 0] = hit;
    }
    out.packets = p_q;
    out.base_cooldown = base_cd(k);
    return {s, out};
}

// garen.on_damage: Perseverance lockout on health lost to an enemy champion or turret.
std::tuple<State, KitOut> on_damage(State s, const KitCtx& k, const WorldUnits& u, const Report& report) {
    const Consts& G = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    const Packets& p = report.packets;
    bool resolved = report.resolved.health_loss.size() > 0;
    for (int h = 0; h < c; ++h) {
        bool struck = false;
        for (size_t i = 0; i < size(p); ++i) {
            bool lost = resolved ? report.resolved.health_loss[i] > 0.f : p.raw[i] > 0.f;
            int src_kind = detail::gather_i(u.kind, p.src[i]), src_team = detail::gather_i(u.team, p.src[i]);
            bool counts = p.valid[i] && lost && (src_kind == KIND_CHAMPION || src_kind == KIND_TURRET);
            struck |= counts && p.dst[i] == k.unit[h] && src_team != k.team[h];
        }
        if (mine(k, h) && struck) s.p_block_until[h] = k.now + G.PASSIVE_DELAY;
    }
    return {s, no_out(c, n)};
}

// garen.on_takedown: Courage stacks from champion killing blows, minion last hits and monster kills.
State on_takedown(State s, const KitCtx& k, const WorldUnits& u, const Kills& kills) {
    const Consts& G = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    for (int h = 0; h < c; ++h) {
        int monsters = 0;
        for (int j = 0; j < n; ++j) monsters += kills.killed_units[(size_t)h * n + j] && u.kind[j] == KIND_MONSTER;
        float earned = kills.champion_kill[h] + kills.minion_kill[h] + (float)monsters;
        bool learned = k.ranks[h * 4 + 1] > 0;
        if (mine(k, h) && learned) s.w_stacks[h] = std::min(G.W_MAX_STACKS, s.w_stacks[h] + earned);
    }
    return s;
}

// garen.stats
ItemStats stats(const State& s, const KitCtx& k) {
    const Consts& G = K();
    size_t c = k.unit.size();
    ItemStats o = detail::scalar_zero_stats();
    o.armor.assign(c, 0.f), o.magic_resist.assign(c, 0.f), o.percent_move_speed.assign(c, 0.f);
    for (size_t h = 0; h < c; ++h) {
        bool g = mine(k, (int)h);
        float resist = g ? std::min(s.w_stacks[h], G.W_MAX_STACKS) * G.W_RESIST_PER_STACK : 0.f;
        o.armor[h] = resist, o.magic_resist[h] = resist;
        o.percent_move_speed[h] = g && s.q_haste_on[h] ? G.Q_MS : 0.f;
    }
    return o;
}

// garen.defense
KitDefense defense(const State& s, const KitCtx& k) {
    const Consts& G = K();
    size_t c = k.unit.size();
    KitDefense d = neutral_kit_defense((int)c);
    for (size_t h = 0; h < c; ++h) {
        bool g = mine(k, (int)h);
        d.received_mult[h] = g && s.w_on[h] ? 1.f - s.w_dr[h] : 1.f;
        d.tenacity_bonus[h] = g && s.w_ten_on[h] ? G.W_TENACITY : 0.f;
    }
    return d;
}

// garen.attack_mods
KitAttackMods attack_mods(const State& s, const KitCtx& k) {
    const Consts& G = K();
    size_t c = k.unit.size();
    KitAttackMods m = neutral_attack_mods((int)c);
    for (size_t h = 0; h < c; ++h) {
        bool g = mine(k, (int)h), q = g && s.q_on[h];
        // Lunge range only against champions; without the world input every target gets it.
        bool vs_champ = k.attack_target_kind.size() == 0 || k.attack_target_kind[h] == KIND_CHAMPION;
        float T = q_attack_time(k.bonus_attack_speed[h]);
        m.extra_range[h] = q && vs_champ ? G.Q_RANGE_BONUS : 0.f;
        m.attack_reset[h] = g && std::fabs(k.now - s.reset_at[h]) < 0.5f * k.dt;
        m.cannot_attack[h] = g && (s.e_on[h] || s.r_pending[h] || k.now < s.q_lock_until[h] - 1e-6f);
        m.windup[h] = q ? G.Q_WINDUP_FRACTION * T : 0.f;
        m.period[h] = q ? T : 0.f;
        m.uncancellable[h] = q;
    }
    return m;
}

// garen.ghosted
Arr<uint8_t> ghosted(const State& s, const KitCtx& k) {
    Arr<uint8_t> o(k.unit.size(), 0);
    for (size_t h = 0; h < o.size(); ++h) o[h] = mine(k, (int)h) && s.e_on[h] && k.alive[h];
    return o;
}

// garen.debuffs: Judgment armor shred on enemy champions, (N,).
Debuffs debuffs(const State& s, const KitCtx& k, const WorldUnits& u) {
    const Consts& G = K();
    size_t c = k.unit.size(), n = u.x.size();
    Debuffs d = neutral_debuffs((int)n);
    for (size_t j = 0; j < n; ++j) {
        bool on = false;
        for (size_t h = 0; h < c; ++h) on |= mine(k, (int)h) && k.now < s.shred_until[h * n + j];
        d.percent_armor_reduction[j] = on ? G.E_SHRED : 0.f;
    }
    return d;
}

LANESIM_TEST(kits_garen_cast, "kits.garen.cast", cast);
LANESIM_TEST(kits_garen_periodic, "kits.garen.periodic", periodic);
LANESIM_TEST(kits_garen_on_attack, "kits.garen.on_attack", on_attack);
LANESIM_TEST(kits_garen_on_hit, "kits.garen.on_hit", on_hit);
LANESIM_TEST(kits_garen_on_damage, "kits.garen.on_damage", on_damage);
LANESIM_TEST(kits_garen_on_takedown, "kits.garen.on_takedown", on_takedown);
LANESIM_TEST(kits_garen_stats, "kits.garen.stats", stats);
LANESIM_TEST(kits_garen_defense, "kits.garen.defense", defense);
LANESIM_TEST(kits_garen_attack_mods, "kits.garen.attack_mods", attack_mods);
LANESIM_TEST(kits_garen_ghosted, "kits.garen.ghosted", ghosted);
LANESIM_TEST(kits_garen_debuffs, "kits.garen.debuffs", debuffs);

}  // namespace lanesim::kits::garen
