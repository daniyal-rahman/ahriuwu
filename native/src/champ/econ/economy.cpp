// Patch-26.19 economy and progression (lanerl_jax/modern/economy.py): ambient gold, levels, minion/kill/structure
// rewards, bounties, death timers, recall, Homeguard, fountain, and economy_step (§13 order).
#include <algorithm>
#include <cmath>

#include "../marshal.hpp"
#include "econ.hpp"

namespace lanesim::econ {

using namespace champ;

namespace {

constexpr float AMBIENT_TICK = .5f, STRUCTURE_RADIUS = 1200.f, STRUCTURE_WINDOW = 10.f, FIRST_TURRET_BONUS = 300.f;
constexpr float HOMEGUARD_SWITCH = 840.f, HOMEGUARD_DECAY = 4.f, HOMEGUARD_LOCKOUT = 8.f,
                DEATHGUARD_MS = .75f, COMPLETION_XP = 600.f;

float k_(const char* name) { return data::f(std::string("econ.economy.const.") + name); }
float b_(const char* name) { return data::f(std::string("econ.economy.bounty.") + name); }

// economy._tables and the constants economy_step reads, loaded once.
struct Tables {
    std::vector<float> need, kill_xp, share, death, base_gold, split, scaling;
    float minion_xp_radius, ambient_start, ambient_per, assist_window, first_blood, ld_slope, inc, scaling_cap;
    float exp_radius2, kill_credit_after_death, cb_start, cb_dmin, cb_c1, cb_c1_ub, cb_c2, cb_cap, xp_min, xp_max;
    float respawn_mod_min, gold_max, regen_period, regen_hp, regen_mana, regen_radius, starting_gold;
    float max_above_base, positive_buffer, min_kill_gold, kill_gold_per_bounty, gv_pos, gv_neg, devalue, deferral;
    float early_s, early_e, early_m;
    Tables() {
        auto t = [](const char* n) { return data::table(std::string("econ.economy.") + n); };
        need = t("need"), kill_xp = t("kill_xp"), share = t("share"), death = t("death"), base_gold = t("base_gold");
        split = t("split"), scaling = t("death_scaling_points");
        minion_xp_radius = t("minion_xp_radius")[0], ambient_per = t("ambient_per")[0];
        assist_window = t("assist_window")[0], first_blood = t("first_blood_bonus")[0];
        ld_slope = t("level_difference_slope")[0], inc = t("death_scaling_increment")[0];
        scaling_cap = t("death_scaling_cap")[0];
        ambient_start = k_("mission_AmbientGoldStartTime"), exp_radius2 = k_("ai_ExpRadius2");
        kill_credit_after_death = k_("aiExp_timeForKillCreditAfterDeath");
        cb_start = k_("aiExp_bonusExpLaneLevelStart"), cb_dmin = k_("aiExp_bonusExpLaneLevelDeltaMin");
        cb_c1 = k_("aiExp_bonusExpPercentPerLaneMinionLevelC1");
        cb_c1_ub = k_("aiExp_bonusExpPercentPerLaneMinionLevelC1UBound");
        cb_c2 = k_("aiExp_bonusExpPercentPerLaneMinionLevelC2"), cb_cap = k_("aiExp_bonusExpLevelDeltaCap");
        xp_min = k_("gcd_PercentEXPBonusMinimum"), xp_max = k_("gcd_PercentEXPBonusMaximum");
        respawn_mod_min = k_("gcd_PercentRespawnTimeModMinimum"), gold_max = k_("Gold_Max");
        regen_period = k_("sp_RegenTickInterval"), regen_hp = k_("sp_HealthRegenPercent");
        regen_mana = k_("sp_ManaRegenPercent"), regen_radius = k_("sp_RegenRadius");
        starting_gold = k_("ai_StartingGold");
        max_above_base = b_("max_above_base"), positive_buffer = b_("positive_buffer");
        min_kill_gold = b_("min_kill_gold"), kill_gold_per_bounty = b_("kill_gold_per_bounty");
        gv_pos = b_("gv_gold_per_bounty_positive"), gv_neg = b_("gv_gold_per_bounty_negative");
        devalue = b_("devalue_gold_per_bounty"), deferral = b_("deferral_out_of_combat");
        early_s = b_("early_assist_start"), early_e = b_("early_assist_end"), early_m = b_("early_assist_mult");
    }
};
const Tables& T() {
    static const Tables t;
    return t;
}

int lv_(int level) { return clampi(level, 0, QUEST_LEVEL_CAP); }
float clip(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }

// ---- §4 minion rewards ----
float minion_comeback_mult(float ml, float receiver_dec) {
    const Tables& t = T();
    float d = ml - receiver_dec;
    float bonus = d < t.cb_c1_ub ? t.cb_c1 * d : t.cb_c2 * std::min(d, t.cb_cap);
    return 1.f + ((ml > t.cb_start && d > t.cb_dmin) ? bonus : 0.f);
}
float xp_modifier(float s) { return 1.f + clip(s, T().xp_min, T().xp_max); }

// economy.minion_rewards: (gold, xp, last_hits) (C,); gold_mult/xp_mult (C, M).
void minion_rewards(const MinionDeaths& d, const Arr<float>& cx, const Arr<float>& cy, const Arr<int32_t>& cteam,
                    const Arr<uint8_t>& calive, const Arr<float>& cdec, const Arr<float>& xp_bonus,
                    const Arr<float>& gold_mult, const Arr<float>& xp_mult, Arr<float>& gold, Arr<float>& xp,
                    Arr<int32_t>& last_hits) {
    const Tables& t = T();
    size_t c = cx.size(), m = d.valid.size();
    gold.assign(c, 0.f), xp.assign(c, 0.f), last_hits.assign(c, 0);
    std::vector<int> n(m, 0);
    std::vector<uint8_t> elig(c * m), hitter(c * m);
    for (size_t i = 0; i < c; ++i)
        for (size_t j = 0; j < m; ++j) {
            float dist = std::sqrt(sq(cx[i] - d.x[j]) + sq(cy[i] - d.y[j]));
            bool h = (int)i == d.last_hitter[j] && d.valid[j];
            bool near = cteam[i] != d.team[j] && calive[i] && dist <= t.minion_xp_radius && d.valid[j];
            hitter[i * m + j] = h, elig[i * m + j] = near || h;
            n[j] += near || h;
        }
    for (size_t i = 0; i < c; ++i) {
        float xmod = xp_modifier(xp_bonus[i]);
        for (size_t j = 0; j < m; ++j) {
            float split = t.split[clampi(n[j] - 1, 0, (int)t.split.size() - 1)];
            float per = d.xp[j] * split * minion_comeback_mult((float)d.level[j], cdec[i]) * xmod * xp_mult[i * m + j];
            xp[i] += elig[i * m + j] ? per : 0.f;
            gold[i] += hitter[i * m + j] ? d.gold[j] * gold_mult[i * m + j] : 0.f;
            last_hits[i] += hitter[i * m + j];
        }
    }
}

// ---- §5 kill credit ----
// economy.credit_update
Credit credit_update(const Credit& credit, const Packets& p, const Arr<int32_t>& unit, float now,
                     const Arr<int32_t>& cls, const CC& cc) {
    size_t c = unit.size(), n = cls.size();
    std::vector<uint8_t> touched(c * n, 0);
    for (size_t i = 0; i < c; ++i)
        for (size_t k = 0; k < size(p); ++k)
            if (p.valid[k] && p.src[k] == unit[i] && !has(p.flags[k], PROP_REACTIVE) && p.dst[k] >= 0 &&
                p.dst[k] < (int)n)
                touched[i * n + p.dst[k]] = 1;
    Credit out = credit;
    for (size_t i = 0; i < c * n; ++i) {
        bool t = touched[i] || cc.slowed[i] || cc.immobilized[i];
        if (t) out.last_affect[i] = now;
        if (t && cls[i % n] == CLASS_STRUCTURE) out.last_structure_damage[i] = now;
    }
    return out;
}

// economy.kill_credit: killer (-1), assisters (C,), any credit.
int kill_credit(const Credit& credit, int victim, float now, int killer_hint, Arr<uint8_t>& assisters, bool& any) {
    const Tables& tb = T();
    size_t nc = assisters.size(), n = credit.last_affect.size() / nc;
    std::vector<float> key(nc);
    std::vector<uint8_t> ok(nc);
    any = false;
    for (size_t i = 0; i < nc; ++i) {
        float t = credit.last_affect[i * n + victim];
        bool o = (now - t) <= tb.assist_window;
        float k = o ? t : -BIG;
        if ((int)i == killer_hint) k = o ? k + 1e-3f : now + 1e-3f, o = true;
        key[i] = k, ok[i] = o, any = any || o;
    }
    int killer = -1;
    if (any) killer = (int)(std::max_element(key.begin(), key.end()) - key.begin());
    for (size_t i = 0; i < nc; ++i) assisters[i] = ok[i] && (int)i != killer;
    return killer;
}

float level_difference_xp_mult(float victim_dec, float recipient_dec) {
    float delta = victim_dec - recipient_dec;
    float m = T().ld_slope * std::max(std::fabs(delta) - 1.f, 0.f);
    return delta > 0 ? 1.f + m : 1.f - std::min(m, .6f);
}

// ---- §6 kill gold and bounty ----
float base_kill_gold(int level) { return T().base_gold[lv_(level)]; }

float kill_gold(float b, int level, bool first_blood) {
    const Tables& t = T();
    float base = base_kill_gold(level), cap = t.max_above_base;
    float k = clip(base + std::min(std::max(b, 0.f), cap) + std::min(b, 0.f), t.min_kill_gold, base + cap);
    return k + (first_blood ? t.first_blood : 0.f);
}

float early_assist_factor(float tm) {
    const Tables& t = T();
    float s = t.early_s, e = t.early_e, m = t.early_m;
    return clip(m + (float)(1.0 - (double)m) * (tm - s) / (float)((double)e - (double)s), m, 1.f);
}

void accrue(float b, float buf, float delta, float& nb_out, float& nbuf_out) {
    float total = b + delta;
    float pos_gain = std::max(total, 0.f) - std::max(b, 0.f);
    float fill = std::max(std::min(pos_gain, T().positive_buffer - buf), 0.f);
    float nb = total - fill;
    float nbuf = nb <= 0.f ? 0.f : buf + fill;
    nb_out = nb, nbuf_out = total <= 0.f ? 0.f : nbuf;
}

// economy.champion_kill (bounty updated in place; returns gold (C,)).
Arr<float> champion_kill(Bounty& b, int victim, int victim_level, int killer, const Arr<uint8_t>& assisters, float now,
                         bool fb_done, bool credited) {
    const Tables& t = T();
    size_t c = b.b.size();
    bool fb = credited && !fb_done;
    float k = kill_gold(b.b[victim], victim_level, fb);
    float k_nofb = k - (fb ? t.first_blood : 0.f);
    int n_ast = 0;
    for (size_t i = 0; i < c; ++i) n_ast += assisters[i];
    float base = base_kill_gold(victim_level);
    float total = (std::min(.5f * k_nofb, .5f * base) + .5f * (k - k_nofb)) * early_assist_factor(now);
    float each = n_ast > 0 ? total / (float)std::max(n_ast, 1) : 0.f;
    float total_a = n_ast > 0 ? total : 0.f;
    Arr<float> gold(c);
    for (size_t i = 0; i < c; ++i) {
        bool is_killer = (int)i == killer && credited;
        gold[i] = (is_killer ? k : 0.f) + (assisters[i] && credited ? each : 0.f);
        float shutdown = is_killer ? std::max(k_nofb - base - 100.f, 0.f) : 0.f;
        float counted = gold[i] - (b.b[i] > 0.f ? shutdown : 0.f);           // bounty_champion_gold
        b.pending[i] = b.pending[i] + counted / t.kill_gold_per_bounty;
    }
    float paid = k_nofb + total_a;                                           // bounty_on_death
    for (size_t i = 0; i < c; ++i) {
        bool died = (int)i == victim && credited;
        bool pos = b.b[i] > 0.f;
        float carry = pos ? std::max(b.b[i] - t.max_above_base, 0.f) : b.carry[i];
        float neg_b = std::max(b.b[i] - paid / t.devalue, t.min_kill_gold - base);
        float nb = pos ? 0.f : neg_b;
        if (died) b.b[i] = nb, b.buf[i] = 0.f, b.carry[i] = carry;
    }
    return gold;
}

// ---- §9 death timer ----
float time_increase_factor(float tm) {
    const Tables& t = T();
    float total = 0.f;
    for (size_t i = 0; i + 2 < t.scaling.size(); i += 3) {
        float start = t.scaling[i], end = t.scaling[i + 1], pct = t.scaling[i + 2];
        total = total + pct * std::max(std::min(tm, end) - start, 0.f) / t.inc;
    }
    return std::min(total, t.scaling_cap);
}

float death_time(int level, float tm) {
    float mod = std::max(-0.f, T().respawn_mod_min);
    return T().death[lv_(level)] * (1.f + time_increase_factor(tm)) * (1.f + mod);
}

float homeguard_bonus_ms(float game_time, float since) {
    bool late = game_time >= HOMEGUARD_SWITCH;
    float hi = late ? 1.5f : .8f, lo = late ? .65f : .4f;
    float f = clip(since / HOMEGUARD_DECAY, 0.f, 1.f);
    return hi + (lo - hi) * f;
}

}  // namespace

// ---- public helpers ----
// economy.ambient_payments
float ambient_payments(float t0, float t1) {
    const Tables& t = T();
    auto k = [&](float v) { return v >= t.ambient_start ? std::floor((v - t.ambient_start) / AMBIENT_TICK + 1e-6f) + 1.f : 0.f; };
    return t.ambient_per * (k(t1) - k(t0));
}

// economy.level_for_xp
int level_for_xp(float xp, int cap) {
    const Tables& t = T();
    int n = 0;
    for (int L = 2; L <= QUEST_LEVEL_CAP; ++L) n += xp >= t.need[L] && L <= cap;
    return 1 + n;
}

// economy.decimal_level
float decimal_level(float xp, int cap) {
    const Tables& t = T();
    int lv = level_for_xp(xp, cap);
    float lo = t.need[lv], hi = t.need[std::min(lv + 1, QUEST_LEVEL_CAP + 1)];
    float frac = clip((xp - lo) / std::max(hi - lo, 1.f), 0.f, .999999f);
    return lv >= cap ? (float)lv : (float)lv + frac;
}

// economy.max_rank
int max_rank(int level, bool ultimate) {
    int lv = std::min(level, SKILL_POINT_LEVELS);
    if (ultimate) return (lv >= 6) + (lv >= 11) + (lv >= 16);
    return std::min((lv + 1) / 2, 5);   // lv >= 0 in the world: floor division
}

// economy.in_fountain
bool in_fountain(float x, float y, float fx, float fy) {
    return std::sqrt(sq(x - fx) + sq(y - fy)) <= T().regen_radius;
}

// economy.fountain_regen
std::pair<float, float> fountain_regen(float hp, float max_hp, float mana, float max_mana, bool in_f, float t0,
                                       float t1, bool homeguard) {
    const Tables& t = T();
    auto pulses = [&](float p) { return std::floor(t1 / p + 1e-6f) - std::floor(t0 / p + 1e-6f); };
    float k = in_f ? pulses(t.regen_period) : 0.f;
    hp = std::min(hp + k * t.regen_hp * max_hp, max_hp);
    mana = std::min(mana + k * t.regen_mana * max_mana, max_mana);
    float kh = in_f && homeguard ? pulses(.5f) : 0.f;
    float keep = std::pow((float)(1.0 - 0.08), kh);                       // (1.0 - HOMEGUARD_FOUNTAIN_HEAL) ** kh
    return {max_hp - (max_hp - hp) * keep, max_mana - (max_mana - mana) * keep};
}

float starting_gold() { return T().starting_gold; }

// economy.init_economy
EconomyState init_economy(int c, int n, const Arr<int32_t>& roles) {
    EconomyState s;
    float g = 0.f + starting_gold();
    s.gold.assign(c, g), s.gold_total.assign(c, g), s.xp.assign(c, 0.f), s.level.assign(c, 1);
    s.bounty.b.assign(c, 0.f), s.bounty.buf.assign(c, 0.f), s.bounty.carry.assign(c, 0.f);
    s.bounty.pending.assign(c, 0.f);
    s.credit.last_affect.assign((size_t)c * n, -BIG), s.credit.last_structure_damage.assign((size_t)c * n, -BIG);
    s.recall.channeling.assign(c, 0), s.recall.start.assign(c, -BIG);
    s.homeguard.active.assign(c, 0), s.homeguard.left_at.assign(c, 0.f + BIG);
    s.homeguard.lockout_until.assign(c, 0.f - BIG);
    s.quest = init_quest(roles);
    s.dead.assign(c, 0), s.dead_since.assign(c, 0.f - BIG), s.respawn_at.assign(c, 0.f - BIG);
    s.first_blood_done = 0, s.first_turret_done = 0, s.last_t = 0.f;
    return s;
}

// economy.economy_step
EconomyOut economy_step(const EconomyState& state, const EconomyInputs& inp) {
    const Tables& tb = T();
    const size_t c = state.gold.size();
    const float now = inp.now;
    const float dt = now - state.last_t;
    const size_t n = state.credit.last_affect.size() / c;
    // Unit classes for the credit (structures marked by StructureEvents.is_structure, else valid).
    Arr<int32_t> cls(n, CLASS_MINION);
    for (size_t i = 0; i < c; ++i) cls[inp.unit[i]] = CLASS_CHAMPION;
    const StructureEvents& st = inp.structures;
    const Arr<uint8_t>& st_mark = st.is_structure.size() ? st.is_structure : st.valid;
    for (size_t k = 0; k < st.unit.size(); ++k)
        if (st_mark[k]) cls[st.unit[k]] = CLASS_STRUCTURE;
    Credit credit = credit_update(state.credit, inp.report.packets, inp.unit, now, cls, inp.cc);

    Arr<float> gold_gain(c, ambient_payments(state.last_t, now));          // paid while dead (§2.4)
    // Champion deaths: credit, kill gold, bounty, kill XP.
    Arr<uint8_t> died(c), alive_now(c);
    for (size_t i = 0; i < c; ++i) died[i] = !state.dead[i] && inp.hp[i] <= 0.f;
    for (size_t i = 0; i < c; ++i) alive_now[i] = !state.dead[i] && !died[i];
    Bounty bounty = state.bounty;
    bool fb_done = state.first_blood_done;
    Arr<float> kills(c, 0.f), assists(c, 0.f), xp_gain(c, 0.f);
    Arr<uint8_t> killed_units(c * n, 0);
    Arr<float> dec(c);
    for (size_t i = 0; i < c; ++i) dec[i] = decimal_level(state.xp[i], state.quest.complete[i] ? QUEST_LEVEL_CAP : LEVEL_CAP);
    Arr<float> td_xp = takedown_xp(state.quest);
    for (size_t v = 0; v < c; ++v) {
        Arr<uint8_t> ast(c);
        bool any_credit;
        int killer = kill_credit(credit, inp.unit[v], now, inp.final_blow[v], ast, any_credit);
        for (size_t i = 0; i < c; ++i) ast[i] = ast[i] && inp.team[i] != inp.team[v];
        bool is_dead = died[v];
        bool valid_kill = is_dead && any_credit && killer >= 0 && inp.team[std::max(killer, 0)] != inp.team[v];
        Bounty b = bounty;
        Arr<float> pay = champion_kill(b, (int)v, state.level[v], killer, ast, now, fb_done, valid_kill);
        if (is_dead) bounty = b;
        if (valid_kill) fb_done = true;
        Arr<uint8_t> takedown(c), elig(c);
        int n_elig = 0;
        for (size_t i = 0; i < c; ++i) {
            gold_gain[i] = gold_gain[i] + (is_dead ? pay[i] : 0.f);
            bool is_killer = (int)i == killer && valid_kill;
            kills[i] = kills[i] + (float)is_killer;
            assists[i] = assists[i] + (float)(ast[i] && valid_kill);
            takedown[i] = is_killer || (ast[i] && valid_kill);
            killed_units[i * n + inp.unit[v]] |= takedown[i];
            // kill_xp_eligible
            float dist = std::sqrt(sq(inp.x[i] - inp.x[v]) + sq(inp.y[i] - inp.y[v]));
            bool recently_dead = !alive_now[i] && (now - state.dead_since[i] <= tb.kill_credit_after_death);
            elig[i] = inp.team[i] != inp.team[v] && (((alive_now[i] || recently_dead) && dist <= tb.exp_radius2) ||
                                                    takedown[i]);
            n_elig += elig[i];
        }
        // kill_xp
        int vl = lv_(state.level[v]);
        float pool = tb.kill_xp[vl] * (n_elig >= 2 ? tb.share[vl] : 1.f);
        float each = pool / (float)std::max(n_elig, 1);
        for (size_t i = 0; i < c; ++i) {
            float x = elig[i] ? each * level_difference_xp_mult(dec[v], dec[i]) : 0.f;
            x = x + (takedown[i] ? td_xp[i] : 0.f);
            xp_gain[i] = xp_gain[i] + (is_dead ? x : 0.f);
        }
    }
    // Minion gold (last hit) and XP, with the top-quest early out-of-lane penalty.
    const MinionDeaths& md = inp.minion_deaths;
    const size_t m = md.valid.size();
    Arr<float> penalty = minion_penalty(state.quest, state.level, inp.minion_in_lane);
    Arr<float> xp_mult = penalty;
    if (inp.minion_xp_mult.size())
        for (size_t i = 0; i < c; ++i)
            for (size_t j = 0; j < m; ++j) xp_mult[i * m + j] = penalty[i * m + j] * inp.minion_xp_mult[i];
    Arr<float> m_gold, m_xp;
    Arr<int32_t> last_hits;
    minion_rewards(md, inp.x, inp.y, inp.team, alive_now, dec, xp_bonus(state.quest), penalty, xp_mult, m_gold, m_xp,
                   last_hits);
    if (inp.minion_gold_delta.size())
        for (size_t i = 0; i < c; ++i) m_gold[i] = std::max(m_gold[i] + inp.minion_gold_delta[i] * (float)last_hits[i], 0.f);
    for (size_t i = 0; i < c; ++i) gold_gain[i] = gold_gain[i] + m_gold[i], xp_gain[i] = xp_gain[i] + m_xp[i];
    if (md.unit.size())
        for (size_t j = 0; j < m; ++j)
            for (size_t i = 0; i < c; ++i)
                if ((int)i == md.last_hitter[j] && md.valid[j]) killed_units[i * n + clampi(md.unit[j], 0, (int)n - 1)] = 1;
    for (size_t i = 0; i < c; ++i) {                                      // bounty_gv
        float rate = bounty.b[i] + bounty.pending[i] >= 0.f ? tb.gv_pos : tb.gv_neg;
        bounty.pending[i] = bounty.pending[i] + m_gold[i] / rate;
    }
    // Structures (plates, turrets; first turret +300).
    bool first_turret = state.first_turret_done;
    const size_t ns = st.valid.size();
    Arr<float> s_gold(c, 0.f);
    std::vector<uint8_t> s_elig(c * ns, 0);                               // (C, S)
    for (size_t k = 0; k < ns; ++k) {
        int su = st.unit[k];
        int n_el = 0;
        Arr<uint8_t> el(c);
        for (size_t i = 0; i < c; ++i) {                                  // structure_eligible
            bool enemy = inp.team[i] != st.team[k];
            bool recent = (now - credit.last_structure_damage[i * n + su]) <= STRUCTURE_WINDOW;
            bool near = alive_now[i] && std::sqrt(sq(inp.x[i] - st.x[k]) + sq(inp.y[i] - st.y[k])) <= STRUCTURE_RADIUS;
            el[i] = enemy && (recent || near);
            n_el += el[i];
            s_elig[i * ns + k] = el[i] && st.valid[k];
        }
        bool ft = st.valid[k] && st.is_turret[k] && !first_turret;
        float local = st.local_gold[k] + (ft ? FIRST_TURRET_BONUS : 0.f);  // structure_gold
        for (size_t i = 0; i < c; ++i) {
            float share = el[i] ? local / (float)std::max(n_el, 1) : 0.f;
            float g = share + (inp.team[i] != st.team[k] ? st.global_gold[k] : 0.f);
            s_gold[i] = s_gold[i] + (st.valid[k] ? g : 0.f);
        }
        first_turret = first_turret || (st.valid[k] && st.is_turret[k]);
    }
    for (size_t i = 0; i < c; ++i) gold_gain[i] = gold_gain[i] + s_gold[i];
    if (inp.extra_gold.size())
        for (size_t i = 0; i < c; ++i) gold_gain[i] = gold_gain[i] + inp.extra_gold[i];
    if (inp.extra_xp.size())
        for (size_t i = 0; i < c; ++i) xp_gain[i] = xp_gain[i] + inp.extra_xp[i];
    // Quest points and completion before the level-up.
    QuestEvents qe;
    for (auto* a : {&qe.minions_in_lane, &qe.minions_out, &qe.turrets_in_lane, &qe.turrets_out, &qe.plates_in_lane,
                    &qe.plates_out, &qe.takedowns, &qe.epic})
        a->assign(c, 0.f);
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < m; ++j) {
            bool h = (int)i == md.last_hitter[j] && md.valid[j];
            qe.minions_in_lane[i] += inp.minion_in_lane[j] && h ? 1.f : 0.f;
            qe.minions_out[i] += !inp.minion_in_lane[j] && h ? 1.f : 0.f;
        }
        for (size_t k = 0; k < ns; ++k) {
            float e = (float)s_elig[i * ns + k];
            qe.turrets_in_lane[i] += e * (float)(st.is_turret[k] && st.in_top_lane[k]);
            qe.turrets_out[i] += e * (float)(st.is_turret[k] && !st.in_top_lane[k]);
            qe.plates_in_lane[i] += e * (float)(!st.is_turret[k] && st.in_top_lane[k]);
            qe.plates_out[i] += e * (float)(!st.is_turret[k] && !st.in_top_lane[k]);
        }
        qe.takedowns[i] = kills[i] + assists[i];
        qe.epic[i] = inp.epic.size() ? inp.epic[i] : 0.f;
    }
    // recall_step
    Recall recall;
    recall.channeling.assign(c, 0), recall.start.assign(c, 0.f);
    Arr<uint8_t> recalled(c);
    for (size_t i = 0; i < c; ++i) {
        bool dead = state.dead[i] || died[i];
        bool start = inp.recall_request[i] && !state.recall.channeling[i] && !dead;
        bool ch = state.recall.channeling[i] || start;
        float t0 = start ? now : state.recall.start[i];
        float elapsed = now - t0;
        float total, grace_at;
        if (inp.recall_channel.size()) total = RECALL_CAST + inp.recall_channel[i], grace_at = total - RECALL_DAMAGE_GRACE;
        else total = (float)(0.5 + 8.0), grace_at = (float)(0.5 + 8.0 - 0.1);
        bool grace = elapsed >= grace_at;
        bool interrupted = ch && !start &&
                           (inp.cancel_action[i] || (inp.health_damage[i] && !grace) || inp.disabled[i] || dead);
        bool done = ch && !interrupted && elapsed >= total;
        recall.channeling[i] = ch && !interrupted && !done, recall.start[i] = t0, recalled[i] = done;
    }
    QuestStep qs = quest_step(state.quest, qe, now, dt, inp.in_quest_lane, alive_now, state.level, recalled);
    Arr<float> xp(c);
    Arr<int32_t> level(c);
    for (size_t i = 0; i < c; ++i) {
        xp_gain[i] = xp_gain[i] + (qs.completed_now[i] ? COMPLETION_XP : 0.f);
        xp[i] = state.xp[i] + xp_gain[i];
        level[i] = level_for_xp(xp[i], qs.level_cap[i]);
    }
    // Death timers (level at death), respawn, deferred bounty, Homeguard.
    Arr<float> duration(c), respawn_at(c), dead_since(c), hg_ms(c);
    Arr<uint8_t> dead(c), respawned(c);
    Homeguard hg = state.homeguard;
    for (size_t i = 0; i < c; ++i) {
        duration[i] = died[i] ? death_time(state.level[i], now) : 0.f;
        respawn_at[i] = died[i] ? now + duration[i] : state.respawn_at[i];
        dead_since[i] = died[i] ? now : state.dead_since[i];
        bool d = state.dead[i] || died[i];
        respawned[i] = d && now >= respawn_at[i] && !died[i];
        dead[i] = d && !respawned[i];
        if (respawned[i]) bounty.b[i] = bounty.b[i] + bounty.carry[i], bounty.carry[i] = 0.f;   // bounty_on_respawn
        float nb, nbuf;                                                                        // bounty_apply_pending
        accrue(bounty.b[i], bounty.buf[i], bounty.pending[i], nb, nbuf);
        if (now - inp.last_champion_combat[i] >= tb.deferral)
            bounty.b[i] = nb, bounty.buf[i] = nbuf, bounty.pending[i] = 0.f;
        // homeguard_step(game_time = now)
        bool in_f = inp.in_fountain[i] && !dead[i];
        bool combat = (now - inp.last_champion_combat[i]) <= 0.f;
        float lock = recalled[i] ? -BIG : state.homeguard.lockout_until[i];
        bool gain = in_f && now >= 20.f && now >= lock;
        bool active = state.homeguard.active[i] || gain;
        float left_at = in_f ? BIG : (state.homeguard.left_at[i] >= BIG ? now : state.homeguard.left_at[i]);
        bool lose_combat = active && (combat || inp.in_jungle[i]) && !in_f;
        bool lose = (active && ((inp.reached_endpoint[i] || inp.teleported[i]) && !in_f)) || lose_combat;
        lock = lose_combat ? now + HOMEGUARD_LOCKOUT : lock;
        active = active && !lose;
        float ms = active && !in_f ? homeguard_bonus_ms(now, now - left_at) : 0.f;
        hg.active[i] = active, hg.left_at[i] = left_at, hg.lockout_until[i] = lock;
        hg_ms[i] = respawned[i] ? std::max(ms, now < HOMEGUARD_SWITCH ? DEATHGUARD_MS : 0.f) : ms;
    }
    EconomyOut out;
    EconomyState& ns_ = out.state;
    ns_.gold.assign(c, 0.f), ns_.gold_total.assign(c, 0.f);
    for (size_t i = 0; i < c; ++i) {
        ns_.gold[i] = std::min(state.gold[i] + gold_gain[i], tb.gold_max);
        ns_.gold_total[i] = state.gold_total[i] + gold_gain[i];
    }
    ns_.xp = xp, ns_.level = level, ns_.bounty = bounty, ns_.credit = credit, ns_.recall = recall, ns_.homeguard = hg;
    ns_.quest = qs.state, ns_.dead = dead, ns_.dead_since = dead_since, ns_.respawn_at = respawn_at;
    ns_.first_blood_done = fb_done, ns_.first_turret_done = first_turret, ns_.last_t = now;
    out.gold_gained = gold_gain, out.xp_gained = xp_gain;
    out.levels_gained.assign(c, 0);
    for (size_t i = 0; i < c; ++i) out.levels_gained[i] = level[i] - state.level[i];
    out.kills.champion_kill = kills, out.kills.champion_assist = assists;
    out.kills.minion_kill.assign(c, 0.f);
    for (size_t i = 0; i < c; ++i) out.kills.minion_kill[i] = (float)last_hits[i];
    out.kills.holder_died = died, out.kills.killed_units = killed_units;
    out.respawned = respawned, out.recalled = recalled, out.death_duration = duration, out.homeguard_ms = hg_ms;
    out.quest_completed = qs.completed_now;
    return out;
}

// ---- replay registrations ----------------------------------------------------------------------------------------
namespace {
// The DEATH phase passes ``Report(packets, None, None)``: the capture has the packets and three empty leaves.
struct EconomyInputsArg {
    EconomyInputs v;
    Arr<float> resolved, life_steal_heal, omnivamp_heal;   // None
    template <class F> void visit(F&& f) {
        f(v.now); f(v.unit); f(v.x); f(v.y); f(v.team); f(v.hp); f(v.max_hp); f(v.report.packets); f(resolved);
        f(life_steal_heal); f(omnivamp_heal); f(v.cc); f(v.final_blow); f(v.minion_deaths); f(v.minion_in_lane);
        f(v.structures); f(v.last_champion_combat); f(v.in_fountain); f(v.in_quest_lane); f(v.recall_request);
        f(v.cancel_action); f(v.health_damage); f(v.disabled); f(v.reached_endpoint); f(v.in_jungle);
        f(v.teleported); f(v.extra_gold); f(v.extra_xp); f(v.epic); f(v.recall_channel); f(v.minion_gold_delta);
        f(v.minion_xp_mult);
    }
};
EconomyOut economy_step_test(EconomyState state, EconomyInputsArg inp) { return economy_step(state, inp.v); }

// Scalar helpers over (K,) arrays, for the direct checks in ops/native/test_hooks.py.
struct F1 {
    Arr<float> v;
    template <class F> void visit(F&& f) { f(v); }
};
struct I1 {
    Arr<int32_t> v;
    template <class F> void visit(F&& f) { f(v); }
};
F1 decimal_level_test(Arr<float> xp, Arr<int32_t> cap) {
    F1 o{Arr<float>(xp.size())};
    for (size_t i = 0; i < xp.size(); ++i) o.v[i] = decimal_level(xp[i], cap[i]);
    return o;
}
I1 level_for_xp_test(Arr<float> xp, Arr<int32_t> cap) {
    I1 o{Arr<int32_t>(xp.size())};
    for (size_t i = 0; i < xp.size(); ++i) o.v[i] = level_for_xp(xp[i], cap[i]);
    return o;
}
std::tuple<I1, I1, I1> ranks_test(Arr<int32_t> level) {
    I1 a{Arr<int32_t>(level.size())}, b = a, c = a;
    for (size_t i = 0; i < level.size(); ++i)
        a.v[i] = skill_points(level[i]), b.v[i] = max_rank(level[i]), c.v[i] = max_rank(level[i], true);
    return {a, b, c};
}
struct B1 {
    Arr<uint8_t> v;
    template <class F> void visit(F&& f) { f(v); }
};
B1 in_fountain_test(Arr<float> x, Arr<float> y, Arr<float> fx, Arr<float> fy) {
    B1 o{Arr<uint8_t>(x.size())};
    for (size_t i = 0; i < x.size(); ++i) o.v[i] = in_fountain(x[i], y[i], fx[i], fy[i]);
    return o;
}
std::tuple<F1, F1> fountain_regen_test(Arr<float> hp, Arr<float> max_hp, Arr<float> mana, Arr<float> max_mana,
                                       Arr<uint8_t> in_f, float t0, float t1, Arr<uint8_t> homeguard) {
    F1 h{Arr<float>(hp.size())}, mn{Arr<float>(hp.size())};
    for (size_t i = 0; i < hp.size(); ++i)
        std::tie(h.v[i], mn.v[i]) = fountain_regen(hp[i], max_hp[i], mana[i], max_mana[i], in_f[i], t0, t1, homeguard[i]);
    return {h, mn};
}
F1 starting_gold_test(Arr<float> dummy) { return F1{Arr<float>(1, starting_gold())}; }
EconomyState init_economy_test(int32_t c, int32_t n, Arr<int32_t> roles) { return init_economy(c, n, roles); }
}  // namespace
LANESIM_TEST(economy_economy_step, "economy.economy_step", economy_step_test);
LANESIM_TEST(economy_decimal_level, "economy.decimal_level", decimal_level_test);
LANESIM_TEST(economy_level_for_xp, "economy.level_for_xp", level_for_xp_test);
LANESIM_TEST(economy_ranks, "economy.ranks", ranks_test);
LANESIM_TEST(economy_in_fountain, "economy.in_fountain", in_fountain_test);
LANESIM_TEST(economy_fountain_regen, "economy.fountain_regen", fountain_regen_test);
LANESIM_TEST(economy_starting_gold, "economy.starting_gold", starting_gold_test);
LANESIM_TEST(economy_init_economy, "economy.init_economy", init_economy_test);

}  // namespace lanesim::econ
