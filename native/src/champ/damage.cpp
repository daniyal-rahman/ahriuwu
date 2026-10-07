// Damage resolution for the champion layer (core/damage.py): DMG.15-75 per packet, shields, Lifeline, Death's
// Dance storage, the sequential pass on stateful units and the parallel pass on the others, vamp and heals.
#include "damage.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "marshal.hpp"

namespace lanesim::champ::damage {

namespace {
constexpr float UNIT_CLASS_RATIO[4][4] = {
    {1.f, 1.f, 1.f, 1.f}, {.55f, 1.f, .60f, 1.f}, {1.f, 1.f, 1.f, 1.f}, {1.f, 1.f, 1.f, 1.f}};
constexpr int RESOLVE_CAPACITY = 64;
constexpr float LIFELINE_THRESHOLD = .30f, OMNIVAMP_MODIFIED_RATIO = .333f, GRIEVOUS_WOUNDS = .40f;
inline int nslots(const Shields& sh, int n) { return n ? (int)(sh.amount.size() / n) : 0; }
inline bool opt_b(const Arr<uint8_t>& a, int i) { return a.size() ? a[i] != 0 : false; }
inline float opt_f(const Arr<float>& a, int i, float d) { return a.size() ? a[i] : d; }
}  // namespace

float shield_value(const Shields& sh, int unit, int k, int K, float now) {
    size_t s = (size_t)unit * K + k;
    float span = std::max(sh.expires_at[s] - sh.decay_start[s], 1e-6f);
    float frac = std::min(std::max((sh.expires_at[s] - now) / span, 0.f), 1.f);
    float cap = now > sh.decay_start[s] ? sh.initial[s] * frac : sh.initial[s];
    bool live = sh.amount[s] > 0.f && now < sh.expires_at[s];
    return live ? std::min(sh.amount[s], cap) : 0.f;
}

float total_shield(const Shields& sh, int unit, float now, int n) {
    int K = nslots(sh, n);
    float t = 0.f;
    for (int k = 0; k < K; ++k) t = t + shield_value(sh, unit, k, K, now);
    return t;
}

void grant_shield(Shields& sh, int unit, float amount, int kind, float now, float duration, float decay_hold,
                  bool enabled, int n) {
    int K = nslots(sh, n);
    int slot = 0;
    float best = INF;
    for (int k = 0; k < K; ++k) {              // argmin, first on ties; empty slots score -inf
        float score = shield_value(sh, unit, k, K, now) > 0.f ? sh.expires_at[(size_t)unit * K + k] : -INF;
        if (k == 0 || score < best) best = score, slot = k;
    }
    bool ok = enabled && amount > 0.f;
    if (!ok) return;
    int order = sh.order.size() ? *std::max_element(sh.order.begin(), sh.order.end()) + 1 : 1;
    size_t s = (size_t)unit * K + slot;
    sh.amount[s] = amount, sh.initial[s] = amount, sh.kind[s] = kind, sh.expires_at[s] = now + duration;
    sh.decay_start[s] = now + decay_hold, sh.order[s] = order;
}

float effective_resist(float resist, float flat_red, float pct_red, float pct_pen, float flat_pen) {
    float r = resist - flat_red;
    r = r > 0.f ? r * (1.f - pct_red) : r;
    r = r > 0.f ? r * (1.f - pct_pen) : r;
    return r > 0.f ? std::max(0.f, r - flat_pen) : r;
}

inline float mitigation(float r) { return r < 0.f ? 2.f - 100.f / (100.f - r) : 100.f / (100.f + r); }

bool spell_blocked(const Packets& p, size_t i, const Offense& off, const Defense& dfn) {
    int s = p.src[i], d = p.dst[i];
    return p.valid[i] && dfn.spell_shield[d] && has(p.flags[i], TAG_ACTIVE_SPELL) && off.unit_class[s] == CLASS_CHAMPION
           && s != d;
}

bool dodged(const Packets& p, size_t i, const Offense& off, const Defense& dfn) {
    if (dfn.dodge_basic.size() == 0) return false;
    return p.valid[i] && dfn.dodge_basic[p.dst[i]] && has(p.flags[i], TAG_BASIC_ATTACK) && !off.is_turret[p.src[i]];
}

Arr<float> premitigation_to_final(const Packets& p, const Offense& off, const Defense& dfn) {
    size_t P = size(p);
    Arr<float> out(P, 0.f);
    for (size_t i = 0; i < P; ++i) {
        int s = p.src[i], d = p.dst[i];
        float ratio = UNIT_CLASS_RATIO[off.unit_class[s]][dfn.unit_class[d]];
        bool no_mod = has(p.flags[i], PROP_NO_DAMAGE_MOD), is_true = p.dtype[i] == TRUE_DMG;
        float reduction = is_true ? 0.f : off.dealt_reduction[s];
        float dealt = no_mod ? 1.f : std::max(1.f + p.amp[i] - reduction, 0.f);
        float raw = p.raw[i] * dealt * ratio;
        float armor = effective_resist(dfn.armor[d], dfn.flat_armor_reduction[d], dfn.percent_armor_reduction[d],
                                       off.percent_armor_pen[s], off.lethality[s]);
        float mr = effective_resist(dfn.magic_resist[d], dfn.flat_mr_reduction[d], dfn.percent_mr_reduction[d],
                                    off.percent_magic_pen[s], off.magic_pen[s]);
        float mult = p.dtype[i] == PHYSICAL ? mitigation(armor) : (p.dtype[i] == MAGIC ? mitigation(mr) : 1.f);
        float post = raw * mult;
        float amp = 1.f + dfn.received_amp[d] + (p.dtype[i] == MAGIC ? dfn.magic_received_amp[d] : 0.f);
        bool from_champion = off.unit_class[s] == CLASS_CHAMPION;
        float reduction_mult = dfn.received_mult[d] * (from_champion ? dfn.champion_received_mult[d] : 1.f);
        float received = (is_true ? 1.f : reduction_mult) * amp;
        bool basic = has(p.flags[i], TAG_BASIC_ATTACK) && !off.is_turret[s];
        received = received * ((basic && !is_true) ? dfn.basic_attack_mult[d] : 1.f);
        received = received * ((basic && has(p.flags[i], PROP_CRIT)) ? dfn.crit_taken_mult[d] : 1.f);
        post = no_mod ? post : post * received;
        bool champ_basic = basic && from_champion;
        float block = champ_basic ? std::min(dfn.champion_attack_block[d], .2f * post) : 0.f;
        post = is_true ? post : std::max(post - block - dfn.postmit_flat[d], 0.f);
        post = std::max(post - p.block[i], 0.f);
        if (dfn.aoe_received_mult.size()) post = post * (has(p.flags[i], TAG_AOE) ? dfn.aoe_received_mult[d] : 1.f);
        if (dfn.received_mult_all.size()) post = post * dfn.received_mult_all[d];
        bool ok = p.valid[i] && !dfn.invulnerable[d] && !spell_blocked(p, i, off, dfn) && !dodged(p, i, off, dfn);
        out[i] = ok ? std::max(post, 0.f) : 0.f;
    }
    return out;
}

Resolved resolve(const Packets& p, const Offense& off, const Defense& dfn, const Arr<float>& hp_in,
                 const Arr<float>& max_hp_in, const Shields& shields_in, float now) {
    const size_t P = size(p);
    const int n = (int)hp_in.size();
    Resolved r;
    r.final = premitigation_to_final(p, off, dfn);
    r.absorbed.assign(P, 0.f), r.stored.assign(P, 0.f), r.health_loss.assign(P, 0.f), r.killed.assign(P, 0);
    Arr<float> hp = hp_in, max_hp = max_hp_in;
    Shields sh = shields_in;
    int K = nslots(sh, n);
    Arr<uint8_t> stateful(n, 0), fired(n, 0);
    for (int j = 0; j < n; ++j)
        stateful[j] = dfn.unit_class[j] == CLASS_CHAMPION || total_shield(sh, j, now, n) > 0.f || dfn.lifeline_ready[j]
                      || dfn.store_fraction[j] > 0.f;
    // Sequential pass: the first RESOLVE_CAPACITY packets on stateful units, in emission order.
    size_t cap = std::min(P, (size_t)RESOLVE_CAPACITY);
    size_t seq_total = 0, taken = 0;
    for (size_t i = 0; i < P; ++i) {
        if (!(p.valid[i] && stateful[p.dst[i]])) continue;
        ++seq_total;
        if (taken >= cap) continue;
        ++taken;
        int d = p.dst[i];
        float dmg = r.final[i];
        int dtype = p.dtype[i];
        bool exe = has(p.flags[i], PROP_EXECUTE);
        bool go = hp[d] > 0.f;
        auto typed = [&](int k) {
            int kind = sh.kind[(size_t)d * K + k];
            return kind == SHIELD_ALL || (kind == SHIELD_PHYSICAL && dtype == PHYSICAL)
                   || (kind == SHIELD_MAGIC && dtype == MAGIC);
        };
        float pre_shield = 0.f;
        for (int k = 0; k < K; ++k) pre_shield = pre_shield + (typed(k) ? shield_value(sh, d, k, K, now) : 0.f);
        bool trigger = go && !exe && dfn.lifeline_ready[d] && !fired[d] && dmg > 0.f
                       && (!dfn.lifeline_magic_only[d] || dtype == MAGIC)
                       && (hp[d] - std::max(dmg - pre_shield, 0.f) < LIFELINE_THRESHOLD * max_hp[d]);
        grant_shield(sh, d, dfn.lifeline_shield[d], dfn.lifeline_shield_kind[d], now, dfn.lifeline_duration[d],
                     dfn.lifeline_decay_hold[d], trigger, n);
        float bonus = trigger ? dfn.lifeline_bonus_health[d] : 0.f;
        hp[d] = hp[d] + bonus, max_hp[d] = max_hp[d] + bonus;
        fired[d] = fired[d] | trigger;
        // Typed shields absorb soonest-expiry, earliest-grant first (a stable sort on the JAX key).
        std::vector<int> order(K);
        std::vector<float> usable(K), key(K);
        for (int k = 0; k < K; ++k) {
            float v = shield_value(sh, d, k, K, now);
            usable[k] = (typed(k) && go && !exe) ? v : 0.f;
            key[k] = usable[k] > 0.f ? sh.expires_at[(size_t)d * K + k] * 1e3f + (float)sh.order[(size_t)d * K + k] * 1e-6f
                                     : INF;
        }
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
        float before = 0.f, absorbed = 0.f;
        std::vector<float> take(K, 0.f);
        // cumsum(sorted) - sorted, then clip(dmg - before, 0, avail)
        float run = 0.f;
        for (int q = 0; q < K; ++q) {
            int k = order[q];
            run = run + usable[k];
            before = run - usable[k];
            take[k] = std::min(std::max(dmg - before, 0.f), usable[k]);
        }
        for (int k = 0; k < K; ++k) absorbed = absorbed + take[k];
        for (int k = 0; k < K; ++k) {
            size_t s = (size_t)d * K + k;
            float v = shield_value(sh, d, k, K, now);
            if (take[k] > 0.f) sh.amount[s] = v - take[k];
            if (exe && go) sh.amount[s] = 0.f;
        }
        float through = exe ? hp[d] : dmg - absorbed;
        float stored = (exe || dtype == TRUE_DMG) ? 0.f : through * dfn.store_fraction[d];
        float to_hp = go ? through - stored : 0.f;
        float loss = std::min(to_hp, std::max(hp[d], 0.f));
        float new_hp = hp[d] - to_hp;
        bool killed = go && new_hp <= 0.f;
        if (go) hp[d] = new_hp;
        r.absorbed[i] = go ? absorbed : 0.f, r.stored[i] = go ? stored : 0.f, r.health_loss[i] = loss;
        r.killed[i] = killed;
    }
    // Parallel pass: per target, the damage of earlier packets (JAX: one cumsum over packets sorted by target).
    std::vector<float> f(P, 0.f);
    for (size_t i = 0; i < P; ++i) {
        bool par = p.valid[i] && !stateful[p.dst[i]];
        bool exe = has(p.flags[i], PROP_EXECUTE);
        f[i] = par ? (exe ? std::max(hp_in[p.dst[i]], 0.f) : r.final[i]) : 0.f;
    }
    std::vector<int> order(P);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p.dst[a] < p.dst[b]; });
    std::vector<float> before(P, 0.f);
    float cs = 0.f, base = -INF;
    for (size_t q = 0; q < P; ++q) {
        int i = order[q];
        cs = cs + f[i];
        bool first = q == 0 || p.dst[order[q]] != p.dst[order[q - 1]];
        if (first) base = std::max(base, cs - f[i]);           // lax.cummax of the group starts
        before[i] = cs - f[i] - base;
    }
    Arr<float> total(n, 0.f);
    for (size_t i = 0; i < P; ++i) {
        bool par = p.valid[i] && !stateful[p.dst[i]];
        if (!par) continue;
        float hp0 = hp_in[p.dst[i]];
        float left = hp0 - before[i];
        r.health_loss[i] = r.health_loss[i] + std::min(std::max(left, 0.f), f[i]);
        r.killed[i] = r.killed[i] | (left > 0.f && left - f[i] <= 0.f);
        total[p.dst[i]] = total[p.dst[i]] + f[i];
    }
    r.hp.resize(n);
    for (int j = 0; j < n; ++j) r.hp[j] = stateful[j] ? hp[j] : (hp_in[j] > 0.f ? hp_in[j] - total[j] : hp_in[j]);
    r.max_hp = max_hp;
    r.shields = sh;
    r.lifeline_fired = fired;
    r.dd_pool_add.assign(n, 0.f), r.spell_shield_popped.assign(n, 0);
    for (size_t i = 0; i < P; ++i) {
        r.dd_pool_add[p.dst[i]] = r.dd_pool_add[p.dst[i]] + r.stored[i];
        r.spell_shield_popped[p.dst[i]] = r.spell_shield_popped[p.dst[i]] | spell_blocked(p, i, off, dfn);
    }
    r.overflow = (int32_t)std::max((long)seq_total - (long)cap, 0L);
    return r;
}

void vamp_heal_split(const Packets& p, const Resolved& res, const Vamp& vamp, const Arr<int32_t>& dst_class,
                     const Arr<float>* lifesteal_scale, Arr<float>& ls_out, Arr<float>& ov_out) {
    size_t n = vamp.life_steal.size();
    ls_out.assign(n, 0.f), ov_out.assign(n, 0.f);
    for (size_t i = 0; i < size(p); ++i) {
        int cls = dst_class[p.dst[i]];
        float scale = lifesteal_scale ? (*lifesteal_scale)[i] : 1.f;
        float ls = (has(p.flags[i], PROP_LIFESTEAL) && cls != CLASS_STRUCTURE)
                       ? vamp.life_steal[p.src[i]] * res.final[i] * scale : 0.f;
        bool modified = (cls == CLASS_MINION || cls == CLASS_MONSTER)
                        && (has(p.flags[i], TAG_AOE) || has(p.flags[i], TAG_PET) || has(p.flags[i], TAG_PERIODIC));
        bool ov_ok = !has(p.flags[i], PROP_NO_OMNIVAMP) && !has(p.flags[i], PROP_REACTIVE) && cls != CLASS_STRUCTURE;
        float ov = ov_ok ? vamp.omnivamp[p.src[i]] * res.final[i] * (modified ? OMNIVAMP_MODIFIED_RATIO : 1.f) : 0.f;
        ls_out[p.src[i]] = ls_out[p.src[i]] + (p.valid[i] ? ls : 0.f);
        ov_out[p.src[i]] = ov_out[p.src[i]] + (p.valid[i] ? ov : 0.f);
    }
}

float heal_amount(float base, float source_power, float incoming, bool grievous) {
    float gw = grievous ? 1.f - GRIEVOUS_WOUNDS : 1.f;
    return std::max(base, 0.f) * (1.f + source_power) * (1.f + incoming) * gw;
}

namespace {
Resolved resolve_test(Packets p, Offense off, Defense dfn, Arr<float> hp, Arr<float> max_hp, Shields sh, float now) {
    return resolve(p, off, dfn, hp, max_hp, sh, now);
}
}  // namespace
LANESIM_TEST(damage_resolve, "damage.resolve", resolve_test);

}  // namespace lanesim::champ::damage
