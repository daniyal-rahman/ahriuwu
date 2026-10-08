// Jax (24), 26.19: champions/jax.py. Constants: native/python/consts/kits_jax.py.
#include <cmath>

#include "../marshal.hpp"
#include "kits.hpp"

namespace lanesim::kits::jax {

using namespace champ;
using detail::ranked;

namespace {

constexpr int SPELL = TAG_ACTIVE_SPELL;
constexpr int AOE_SPELL = TAG_AOE | TAG_ACTIVE_SPELL;
constexpr int R_FLAGS = TAG_AOE | TAG_ACTIVE_SPELL | PROP_ULTIMATE;
constexpr int R_PASSIVE_FLAGS = TAG_ON_HIT | TAG_PROC | PROP_ULTIMATE;

struct Consts {
    std::vector<float> q_damage, w_damage, e_base, r_swing, r_resists, r_resists_extra, r_passive, cd[4], mana[4];
    float P_DURATION, P_FALLOFF, Q_RANGE, Q_SPEED, W_DURATION, W_AP, W_STRUCTURE, RANGE_BONUS, E_DURATION,
        E_MIN_EPS, E_RADIUS, E_AP, E_PCT_HP, E_PER_DODGE, E_MAX_DODGES, E_STUN, E_AOE_MULT, E_MONSTER_CAP, R_DELAY,
        R_RADIUS, R_AP, R_DURATION, R_MR_MULT, R_PASSIVE_AP, R_FALLOFF, R_STRUCTURE, P_LEVEL_CAP;
    int P_MAX, R_PASSIVE_STACKS;
};

const Consts& K() {
    static const Consts k = [] {
        Consts k;
        auto t = [](const char* name) { return detail::table(std::string("kits.jax.") + name); };
        auto f = [](const char* name) { return data::f(std::string("kits.jax.") + name); };
        k.q_damage = t("Q.Damage"), k.w_damage = t("W.Damage"), k.e_base = t("E.BaseDamage");
        k.r_swing = t("R.SwingDamageBase"), k.r_resists = t("R.BaseResists");
        k.r_resists_extra = t("R.ResistsPerExtraTarget"), k.r_passive = t("R.PassiveBaseDamage");
        const char* slots[4] = {"Q", "W", "E", "R"};
        for (int i = 0; i < 4; ++i) {
            k.cd[i] = detail::table(std::string("kits.jax.cd.") + slots[i]);
            k.mana[i] = detail::table(std::string("kits.jax.mana.") + slots[i]);
        }
        k.P_DURATION = f("P_DURATION"), k.P_FALLOFF = f("P_FALLOFF"), k.Q_RANGE = f("Q_RANGE");
        k.Q_SPEED = f("Q_SPEED"), k.W_DURATION = f("W_DURATION"), k.W_AP = f("W_AP");
        k.W_STRUCTURE = f("W_STRUCTURE"), k.RANGE_BONUS = f("RANGE_BONUS"), k.E_DURATION = f("E_DURATION");
        k.E_MIN_EPS = f("E_MIN_EPS"), k.E_RADIUS = f("E_RADIUS"), k.E_AP = f("E_AP"), k.E_PCT_HP = f("E_PCT_HP");
        k.E_PER_DODGE = f("E_PER_DODGE"), k.E_MAX_DODGES = f("E_MAX_DODGES"), k.E_STUN = f("E_STUN");
        k.E_AOE_MULT = f("E_AOE_MULT"), k.E_MONSTER_CAP = f("E_MONSTER_CAP"), k.R_DELAY = f("R_DELAY");
        k.R_RADIUS = f("R_RADIUS"), k.R_AP = f("R_AP"), k.R_DURATION = f("R_DURATION"), k.R_MR_MULT = f("R_MR_MULT");
        k.R_PASSIVE_AP = f("R_PASSIVE_AP"), k.R_FALLOFF = f("R_FALLOFF"), k.R_STRUCTURE = f("R_STRUCTURE");
        k.P_LEVEL_CAP = f("P_LEVEL_CAP"), k.P_MAX = (int)f("P_MAX"), k.R_PASSIVE_STACKS = (int)f("R_PASSIVE_STACKS");
        return k;
    }();
    return k;
}

bool mine(const KitCtx& k, int c) { return k.champion_id[c] == ID; }

Arr<float> base_cd(const KitCtx& k) {
    const Consts& J = K();
    size_t c = k.unit.size();
    Arr<float> out(c * 4, 0.f);
    for (size_t h = 0; h < c; ++h)
        if (mine(k, (int)h))
            for (int s = 0; s < 4; ++s) out[h * 4 + s] = detail::by_rank(J.cd[s], k.ranks[h * 4 + s]);
    return out;
}

int r_needed(const State& s, int h) { return s.r_buff_on[h] ? K().R_PASSIVE_STACKS - 1 : K().R_PASSIVE_STACKS; }

float w_damage(const KitCtx& k, int h) { return ranked(K().w_damage, k.ranks[h * 4 + 1]) + K().W_AP * k.ap[h]; }

// _release: Counter Strike release for holders in ``mask``: (C, N) packets appended to ``p``, stun and cast ids
// into ``cc``; ends the dodge window.
void release(State& s, const KitCtx& k, const WorldUnits& u, const uint8_t* mask, Packets& p, CCOut& cc) {
    const Consts& J = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    for (int h = 0; h < c; ++h) {
        float base = ranked(J.e_base, k.ranks[h * 4 + 2]) + J.E_AP * k.ap[h];
        float mult = 1.f + J.E_PER_DODGE * std::min((float)s.e_dodges[h], J.E_MAX_DODGES);
        for (int j = 0; j < n; ++j) {
            bool hit = mask[h] && detail::enemy(k, u, h, j) && detail::center_dist(k, u, h, j) <= J.E_RADIUS + u.radius[j];
            float pct = J.E_PCT_HP * u.max_hp[j];
            if (u.kind[j] == KIND_MONSTER) pct = std::min(pct, J.E_MONSTER_CAP);
            push(p, hit, k.unit[h], j, (base + pct) * mult, MAGIC, AOE_SPELL, 0.f, 0, s.e_cast_id[h]);
            cc.stun[(size_t)h * n + j] = hit ? J.E_STUN : 0.f;
            cc.cast_id[(size_t)h * n + j] = hit ? s.e_cast_id[h] : 0;
        }
        s.e_on[h] = s.e_on[h] & !mask[h];
    }
}

}  // namespace

// jax.dodging: (C,) holder dodges basic attacks (Counter Strike).
Arr<uint8_t> dodging(const State& s, const KitCtx& k) {
    Arr<uint8_t> o(k.unit.size(), 0);
    for (size_t h = 0; h < o.size(); ++h) o[h] = mine(k, (int)h) && s.e_on[h] && k.alive[h];
    return o;
}

// jax.cast
std::tuple<State, KitOut> cast(State s, const KitCtx& k, const WorldUnits& u, const CastOrder& order) {
    const Consts& J = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    KitOut out = no_out(c, n);
    const float now = k.now;
    std::vector<uint8_t> e_release(c, 0), q(c), w(c), e_start(c), rr(c);
    std::vector<int> cid(c);
    std::vector<float> cost((size_t)c * 4), dist(c);
    for (int h = 0; h < c; ++h) {
        const int* r = &k.ranks[h * 4];
        bool free = mine(k, h) && k.alive[h] && !k.stunned[h] && !k.silenced[h] && !s.r_pending[h];
        bool ready[4];
        for (int i = 0; i < 4; ++i) {
            cost[h * 4 + i] = detail::by_rank(J.mana[i], r[i]);
            ready[i] = r[i] > 0 && k.cooldowns[h * 4 + i] <= 0.f && k.mana[h] >= cost[h * 4 + i];
        }
        auto want = [&](int sl) { return order.slot[h] == sl && free; };
        int target = order.target[h], t = clampi(target, 0, n - 1);
        dist[h] = detail::target_dist(k, u, h, target);
        bool valid_q = target >= 0 && target != k.unit[h] && u.alive[t] && u.targetable[t] && u.kind[t] != KIND_NONE &&
                       !detail::is_structure(u.kind[t]) && dist[h] <= J.Q_RANGE + u.radius[t];
        bool rooted = k.rooted.size() ? k.rooted[h] : false;
        q[h] = want(0) && ready[0] && valid_q && !s.q_pending[h] && !rooted;
        w[h] = want(1) && ready[1] && !s.w_on[h];
        e_start[h] = want(2) && ready[2] && !s.e_on[h];
        e_release[h] = want(2) && s.e_on[h] && (k.now - s.e_start[h] >= J.E_MIN_EPS);
        rr[h] = want(3) && ready[3];
        bool started = q[h] || w[h] || e_start[h] || rr[h];
        int slot = q[h] ? 0 : w[h] ? 1 : e_start[h] ? 2 : rr[h] ? 3 : -1;
        cid[h] = started ? make_cast_id(k, h, std::max(slot, 0)) : 0;
        out.cast_started[h] = started, out.cast_slot[h] = slot, out.cast_id[h] = cid[h];
    }

    Packets p_e = empty_packets(0);
    release(s, k, u, e_release.data(), p_e, out.cc);
    for (int h = 0; h < c; ++h) {
        int target = order.target[h], t = clampi(target, 0, n - 1);
        s.q_pending[h] = s.q_pending[h] | q[h];
        if (q[h]) {
            s.q_land_at[h] = now + std::max(dist[h] / J.Q_SPEED, 1e-3f);
            s.q_target[h] = target, s.q_seq[h] = u.spawn_seq[t], s.q_cast_id[h] = cid[h];
        }
        s.w_on[h] = s.w_on[h] | w[h];
        if (w[h]) s.w_until[h] = later(k, J.W_DURATION), s.w_cast_id[h] = cid[h], s.reset_at[h] = now;
        s.e_on[h] = s.e_on[h] | e_start[h];
        if (e_start[h]) {
            s.e_start[h] = now, s.e_until[h] = later(k, J.E_DURATION);
            s.e_dodges[h] = 0, s.e_cast_id[h] = cid[h];
        }
        s.r_pending[h] = s.r_pending[h] | rr[h];
        if (rr[h]) s.r_fire_at[h] = later(k, J.R_DELAY), s.r_cast_id[h] = cid[h];
        const uint8_t used[4] = {q[h], w[h], e_start[h], rr[h]};
        float mana = 0.f;
        for (int i = 0; i < 4; ++i) mana = mana + (used[i] ? cost[h * 4 + i] : 0.f);
        out.mana_cost[h] = mana;
        out.dash.active[h] = q[h], out.dash.to_x[h] = u.x[t], out.dash.to_y[h] = u.y[t];
        out.dash.speed[h] = J.Q_SPEED, out.dash.target[h] = q[h] ? target : -1, out.dash.blink[h] = 0;
        out.cooldown_start[h * 4 + 0] = q[h], out.cooldown_start[h * 4 + 1] = 0;
        out.cooldown_start[h * 4 + 2] = e_release[h], out.cooldown_start[h * 4 + 3] = rr[h];
        out.attack_reset[h] = w[h];
    }
    out.packets = p_e;
    out.base_cooldown = base_cd(k);
    return {s, out};
}

// jax.periodic
std::tuple<State, KitOut> periodic(State s, const KitCtx& k, const WorldUnits& u) {
    const Consts& J = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    KitOut out = no_out(c, n);
    Packets p_q = empty_packets(0), p_wq = empty_packets(0), p_e = empty_packets(0), p_r = empty_packets(0);
    std::vector<uint8_t> e_expire(c, 0), strike(c), w_end(c);
    for (int h = 0; h < c; ++h) {
        const int* r = &k.ranks[h * 4];
        bool alive = k.alive[h], dead = !alive;
        // Leap landing (target identity guards against a recycled slot).
        int t = clampi(s.q_target[h], 0, n - 1);
        bool target_ok = u.alive[t] && u.spawn_seq[t] == s.q_seq[h];
        bool land = s.q_pending[h] && due(k, s.q_land_at[h]) && alive && target_ok;
        strike[h] = land && u.team[t] != k.team[h] && u.kind[t] != KIND_WARD;
        float q_raw = ranked(J.q_damage, r[0]) + k.bonus_ad[h];
        push(p_q, strike[h], k.unit[h], t, q_raw, PHYSICAL, SPELL, 0.f, 0, s.q_cast_id[h]);
        bool w_on_q = strike[h] && s.w_on[h];
        push(p_wq, w_on_q, k.unit[h], t, w_damage(k, h), MAGIC, SPELL, 0.f, 0, s.w_cast_id[h]);
        s.q_pending[h] = s.q_pending[h] & !(land || dead || !target_ok);
        // Empower: expiry, consumption by Q, death.
        w_end[h] = s.w_on[h] && (due(k, s.w_until[h]) || w_on_q || dead);
        // Counter Strike: expiry releases (alive); death ends it silently.
        e_expire[h] = s.e_on[h] && due(k, s.e_until[h]) && alive;
    }
    release(s, k, u, e_expire.data(), p_e, out.cc);
    for (int h = 0; h < c; ++h) {
        const int* r = &k.ranks[h * 4];
        bool j = mine(k, h), alive = k.alive[h], dead = !alive;
        bool e_dead = s.e_on[h] && dead;
        bool e_end = e_expire[h] || e_dead;

        // Grandmaster's Might swing.
        bool r_due = s.r_pending[h] && (due(k, s.r_fire_at[h]) || dead);
        bool r_fire = r_due && alive;
        float r_raw = ranked(J.r_swing, r[3]) + J.R_AP * k.ap[h];
        int champs = 0;
        for (int i = 0; i < n; ++i) {
            bool hit = r_fire && detail::enemy(k, u, h, i) && detail::center_dist(k, u, h, i) <= J.R_RADIUS + u.radius[i];
            push(p_r, hit, k.unit[h], i, r_raw, MAGIC, R_FLAGS, 0.f, 0, s.r_cast_id[h]);
            champs += hit && u.kind[i] == KIND_CHAMPION;
        }
        float armor = ranked(J.r_resists, r[3]) + 0.4f * k.bonus_ad[h] +
                      (float)std::max(champs - 1, 0) * (ranked(J.r_resists_extra, r[3]) + 0.1f * k.bonus_ad[h]);
        bool gain = r_fire && champs > 0;
        bool r_buff_on = (s.r_buff_on[h] && !(due(k, s.r_buff_until[h]) || dead)) || gain;

        // Passive stacks fall off one at a time once the buff lapses.
        bool expired = s.stacks[h] > 0 && due(k, s.stack_until[h]);
        int stacks = dead ? 0 : std::max(0, s.stacks[h] - (int)expired);
        if (expired) s.stack_until[h] = later_after_tick(k, J.P_FALLOFF);
        if (due(k, s.r_hit_until[h]) || dead) s.r_hits[h] = 0;

        s.w_on[h] = s.w_on[h] & !w_end[h];
        s.e_on[h] = s.e_on[h] & !e_dead;
        s.r_pending[h] = s.r_pending[h] & !r_due;
        s.r_buff_on[h] = r_buff_on;
        if (gain) {
            s.r_buff_until[h] = later_after_tick(k, J.R_DURATION);
            s.r_armor[h] = armor, s.r_mr[h] = J.R_MR_MULT * armor;
        }
        s.stacks[h] = stacks;
        out.cooldown_start[h * 4 + 1] = w_end[h] && j;
        out.cooldown_start[h * 4 + 2] = e_end && j;
        int t = clampi(s.q_target[h], 0, n - 1);
        out.attack_target[h] = strike[h] && u.kind[t] == KIND_CHAMPION ? s.q_target[h] : -1;
    }
    append(p_q, p_wq), append(p_q, p_e), append(p_q, p_r);
    out.packets = p_q;
    out.base_cooldown = base_cd(k);
    return {s, out};
}

// jax.on_attack: passive stacks on attack launch.
std::tuple<State, KitOut> on_attack(State s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& launch) {
    const Consts& J = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    for (int h = 0; h < c; ++h) {
        bool go = detail::holder_row(launch.launched, k, h) && mine(k, h) && k.alive[h];
        if (go) s.stacks[h] = std::min(J.P_MAX, s.stacks[h] + 1), s.stack_until[h] = later_after_tick(k, J.P_DURATION);
    }
    KitOut out = no_out(c, n);
    out.base_cooldown = base_cd(k);
    return {s, out};
}

// jax.on_hit: Empower and the R passive on a landed attack. ``dodging_units`` (N,).
std::tuple<State, KitOut> on_hit(State s, const KitCtx& k, const WorldUnits& u, const AttackLaunch& launch,
                                 const Arr<uint8_t>& dodging_units) {
    const Consts& J = K();
    int c = (int)k.unit.size(), n = (int)u.x.size();
    KitOut out = no_out(c, n);
    Packets p_w = empty_packets(0), p_r = empty_packets(0);
    int r_id_base = make_cast_id(k, 0, CODE_JAX_R_PASSIVE);
    for (int h = 0; h < c; ++h) {
        int target = detail::holder_row(launch.target, k, h);
        bool hit = detail::holder_row(launch.launched, k, h) && mine(k, h) && k.alive[h] && target >= 0;
        int t = clampi(target, 0, n - 1);
        bool landed = hit && !detail::gather_b(dodging_units, target);
        bool structure = detail::is_structure(u.kind[t]);
        float st = structure ? J.W_STRUCTURE : 1.f;
        bool w = landed && s.w_on[h];
        push(p_w, w, k.unit[h], t, w_damage(k, h) * st, MAGIC, SPELL, 0.f, 0, s.w_cast_id[h]);
        int rank_r = k.ranks[h * 4 + 3];
        bool has_r = rank_r > 0;
        bool ward = u.kind[t] == KIND_WARD;
        bool ready = has_r && s.r_hits[h] >= r_needed(s, h);
        bool proc = landed && ready && !ward;          // vs wards: triggers, but neither consumed nor applied
        float r_raw = (ranked(J.r_passive, rank_r) + J.R_PASSIVE_AP * k.ap[h]) * (structure ? J.R_STRUCTURE : 1.f);
        push(p_r, proc, k.unit[h], t, r_raw, MAGIC, R_PASSIVE_FLAGS, 0.f, 0, r_id_base + h * ID_STRIDE);
        s.w_on[h] = s.w_on[h] & !w;
        if (landed && has_r) s.r_hits[h] = proc ? 0 : std::min(s.r_hits[h] + 1, J.R_PASSIVE_STACKS);
        if (landed) s.r_hit_until[h] = later_after_tick(k, J.R_FALLOFF);
        out.cooldown_start[h * 4 + 1] = w;
    }
    append(p_w, p_r);
    out.packets = p_w;
    out.base_cooldown = base_cd(k);
    return {s, out};
}

// jax.on_damage: count dodged attack instances (one per basic-attack cast id) during Counter Strike.
std::tuple<State, KitOut> on_damage(State s, const KitCtx& k, const WorldUnits& u, const Report& report) {
    int c = (int)k.unit.size(), n = (int)u.x.size();
    const Packets& p = report.packets;
    size_t P = size(p);
    std::vector<uint8_t> basic(P), first(P);
    for (size_t i = 0; i < P; ++i)
        basic[i] = p.valid[i] && has(p.flags[i], TAG_BASIC_ATTACK) && !has(p.flags[i], TAG_ON_HIT) &&
                   !detail::is_structure(detail::gather_i(u.kind, p.src[i]));
    // first_per_key(basic, cast_id, dst): the first selected packet of each (cast_id, dst).
    for (size_t i = 0; i < P; ++i) {
        if (!basic[i]) continue;
        bool earlier = false;
        for (size_t q = 0; q < i && !earlier; ++q)
            earlier = basic[q] && p.cast_id[q] == p.cast_id[i] && p.dst[q] == p.dst[i];
        first[i] = p.cast_id[i] == 0 || !earlier;
    }
    Arr<uint8_t> dodge = dodging(s, k);
    for (int h = 0; h < c; ++h) {
        int count = 0;
        for (size_t i = 0; i < P; ++i) count += p.dst[i] == k.unit[h] && first[i];
        if (dodge[h]) s.e_dodges[h] = s.e_dodges[h] + count;
    }
    return {s, no_out(c, n)};
}

// jax.on_takedown
State on_takedown(State s, const KitCtx& k, const WorldUnits& u, const Kills& kills) { return s; }

// jax.stats
ItemStats stats(const State& s, const KitCtx& k) {
    const Consts& J = K();
    size_t c = k.unit.size();
    ItemStats o = detail::scalar_zero_stats();
    o.attack_speed.assign(c, 0.f), o.armor.assign(c, 0.f), o.magic_resist.assign(c, 0.f);
    for (size_t h = 0; h < c; ++h) {
        bool j = mine(k, (int)h);
        float lv = std::min((float)k.level[h], J.P_LEVEL_CAP);
        float per_stack = 0.05f + 0.015f * std::floor((lv - 1.f) / 3.f);
        bool buff = j && s.r_buff_on[h];
        o.attack_speed[h] = j ? per_stack * (float)s.stacks[h] : 0.f;
        o.armor[h] = buff ? s.r_armor[h] : 0.f;
        o.magic_resist[h] = buff ? s.r_mr[h] : 0.f;
    }
    return o;
}

// jax.defense
KitDefense defense(const State& s, const KitCtx& k) {
    const Consts& J = K();
    size_t c = k.unit.size();
    KitDefense d = neutral_kit_defense((int)c);
    Arr<uint8_t> on = dodging(s, k);
    for (size_t h = 0; h < c; ++h) d.dodge_basic[h] = on[h], d.aoe_received_mult[h] = on[h] ? J.E_AOE_MULT : 1.f;
    return d;
}

// jax.attack_mods
KitAttackMods attack_mods(const State& s, const KitCtx& k) {
    const Consts& J = K();
    size_t c = k.unit.size();
    KitAttackMods m = neutral_attack_mods((int)c);
    for (size_t h = 0; h < c; ++h) {
        bool j = mine(k, (int)h);
        bool empowered = j && (s.w_on[h] || (k.ranks[h * 4 + 3] > 0 && s.r_hits[h] >= r_needed(s, (int)h)));
        m.extra_range[h] = j && s.w_on[h] ? J.RANGE_BONUS : 0.f;
        m.attack_reset[h] = j && std::fabs(k.now - s.reset_at[h]) < 0.5f * k.dt;
        m.cannot_attack[h] = j && (s.q_pending[h] || s.r_pending[h]);
        m.uncancellable[h] = empowered;
    }
    return m;
}

// jax.debuffs
Debuffs debuffs(const State& s, const KitCtx& k, const WorldUnits& u) { return neutral_debuffs((int)u.x.size()); }

LANESIM_TEST(kits_jax_cast, "kits.jax.cast", cast);
LANESIM_TEST(kits_jax_periodic, "kits.jax.periodic", periodic);
LANESIM_TEST(kits_jax_on_attack, "kits.jax.on_attack", on_attack);
LANESIM_TEST(kits_jax_on_hit, "kits.jax.on_hit", on_hit);
LANESIM_TEST(kits_jax_on_damage, "kits.jax.on_damage", on_damage);
LANESIM_TEST(kits_jax_on_takedown, "kits.jax.on_takedown", on_takedown);
LANESIM_TEST(kits_jax_stats, "kits.jax.stats", stats);
LANESIM_TEST(kits_jax_defense, "kits.jax.defense", defense);
LANESIM_TEST(kits_jax_attack_mods, "kits.jax.attack_mods", attack_mods);
LANESIM_TEST(kits_jax_dodging, "kits.jax.dodging", dodging);
LANESIM_TEST(kits_jax_debuffs, "kits.jax.debuffs", debuffs);

}  // namespace lanesim::kits::jax
