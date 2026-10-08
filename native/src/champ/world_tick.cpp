// The full 26.19 tick with champions (lanerl_jax/modern/world/tick.py and world/phases/*), top-lane world without
// jungle and objectives: the lane-slice building blocks (slice.hpp) for minions, structures and missiles, and the
// champion layer (kits, summoners, items, runes, economy, wards) in the JAX phase order:
// INPUT, STATS, CASTS, AI, MOVE, ATTACK, DAMAGE, CC/HEAL, DEATH, TIMERS, FOG, commit.
#include "world_tick.hpp"

#include <algorithm>
#include <cmath>

#include "../prof.hpp"
#include "../slice.hpp"
#include "combat.hpp"
#include "damage.hpp"
#include "dispatch.hpp"
#include "envio.hpp"
#include "rng.hpp"
#include "stats.hpp"
#include "world_api.hpp"

namespace lanesim::champ {

using slice::Clock;

namespace {

constexpr float CAST_BUFFER_S = .5f, CAST_RANGE_SLACK = 5.f, MOVE_ARRIVE_RADIUS = 5.f;
constexpr float CHAMPION_ACQUISITION_RANGE = 400.f, ATTACK_MOVE_ARRIVE = 10.f;
constexpr int CAST_ID_STRIDE = 256;
constexpr float REVEAL_RADIUS = 300.f, REVEAL_DURATION = 2.f, TURRET_TRUE_SIGHT = 1100.f;

// Indices of the first Env leaf of each state subtree (ModernState flatten order).
struct Roots {
    long champ, kits, summoners, combat, econ, shields, status, prev_kills, prev_death_seen, prev_epic, prev_large,
        prev_pending_dash, sight, wards, amove, reveal;
    Roots() {
        champ = env_index("champ_ranks"), kits = env_index("kits_garen_q_on"), summoners = env_index("summoners_spell");
        combat = env_index("combat_items_consumables_hot_per_tick"), econ = env_index("econ_gold");
        shields = env_index("shields_amount"), status = env_index("status_slow");
        prev_kills = env_index("prev_kills_champion_kill"), prev_death_seen = env_index("prev_death_seen");
        prev_epic = env_index("prev_epic"), prev_large = env_index("prev_large");
        prev_pending_dash = env_index("prev_pending_dash_active"), sight = env_index("sight");
        wards = env_index("wards_slots_alive"), amove = env_index("amove_active"), reveal = env_index("reveal_x");
    }
};
const Roots& roots() {
    static const Roots r;
    return r;
}

// The state subtrees the champion layer works on, loaded at tick start and stored at commit.
struct Layer {
    ChampionLayer champ;
    ChampionState kits;
    champions_summoners_State summ;
    CombatState combat;
    EconomyState econ;
    Shields shields;
    UnitStatus status;
    Kills prev_kills;
    Arr<uint8_t> prev_death_seen;
    Arr<float> prev_epic, prev_large;
    Dash prev_pending_dash;
    Arr<uint8_t> sight;
    Wards wards;
    AttackMove amove;
    Reveal reveal;
};

void load(const World& w, const Env& e, Layer& L) {
    const Roots& r = roots();
    env_load(w, e, r.champ, L.champ), env_load(w, e, r.kits, L.kits), env_load(w, e, r.summoners, L.summ);
    env_load(w, e, r.combat, L.combat), env_load(w, e, r.econ, L.econ), env_load(w, e, r.shields, L.shields);
    env_load(w, e, r.status, L.status), env_load(w, e, r.prev_kills, L.prev_kills);
    env_load(w, e, r.prev_death_seen, L.prev_death_seen), env_load(w, e, r.prev_epic, L.prev_epic);
    env_load(w, e, r.prev_large, L.prev_large), env_load(w, e, r.prev_pending_dash, L.prev_pending_dash);
    env_load(w, e, r.sight, L.sight), env_load(w, e, r.wards, L.wards), env_load(w, e, r.amove, L.amove);
    env_load(w, e, r.reveal, L.reveal);
}

void store(const Env& e, Layer& L) {
    const Roots& r = roots();
    env_store(e, r.champ, L.champ), env_store(e, r.kits, L.kits), env_store(e, r.summoners, L.summ);
    env_store(e, r.combat, L.combat), env_store(e, r.econ, L.econ), env_store(e, r.shields, L.shields);
    env_store(e, r.status, L.status), env_store(e, r.prev_kills, L.prev_kills);
    env_store(e, r.prev_death_seen, L.prev_death_seen), env_store(e, r.prev_epic, L.prev_epic);
    env_store(e, r.prev_large, L.prev_large), env_store(e, r.prev_pending_dash, L.prev_pending_dash);
    env_store(e, r.sight, L.sight), env_store(e, r.wards, L.wards), env_store(e, r.amove, L.amove);
    env_store(e, r.reveal, L.reveal);
}

template <class T>
Arr<T> arr(const T* p, size_t n) {
    Arr<T> a(n);
    std::memcpy(a.data(), p, n * sizeof(T));
    return a;
}

// world.units.units_view (targetable implies alive).
WorldUnits units_view(const World& w, const Env& e) {
    size_t n = w.n;
    WorldUnits u;
    u.kind = arr(e.kind, n), u.sub = arr(e.sub, n), u.team = arr(e.team, n), u.alive = arr(e.alive, n);
    u.targetable = arr(e.targetable, n);
    for (size_t j = 0; j < n; ++j) u.targetable[j] = u.targetable[j] && u.alive[j];
    u.x = arr(e.x, n), u.y = arr(e.y, n), u.radius = arr(e.radius, n), u.hp = arr(e.hp, n);
    u.max_hp = arr(e.max_hp, n), u.armor = arr(e.armor, n), u.magic_resist = arr(e.magic_resist, n);
    u.attack_damage = arr(e.attack_damage, n), u.attack_range = arr(e.attack_range, n);
    u.attack_speed = arr(e.attack_speed, n), u.move_speed = arr(e.move_speed, n), u.spawn_seq = arr(e.spawn_seq, n);
    u.spawn_time = arr(e.spawn_time, n);
    return u;
}

inline int damage_class(int k) {
    return k == KIND_CHAMPION ? CLASS_CHAMPION
         : (k == KIND_TURRET || k == KIND_INHIBITOR || k == KIND_NEXUS) ? CLASS_STRUCTURE
         : k == KIND_MONSTER ? CLASS_MONSTER : CLASS_MINION;
}

// world.units.item_units (wards are not item/on-hit targets).
Units item_units(const World& w, const Env& e) {
    size_t n = w.n;
    Units u;
    u.x = arr(e.x, n), u.y = arr(e.y, n), u.team = arr(e.team, n);
    u.cls.resize(n), u.targetable.resize(n), u.is_siege_or_super.resize(n);
    for (size_t j = 0; j < n; ++j) {
        u.cls[j] = damage_class(e.kind[j]);
        u.targetable[j] = e.targetable[j] && e.alive[j] && e.kind[j] != KIND_WARD;
        u.is_siege_or_super[j] = e.kind[j] == KIND_MINION && e.sub[j] >= 2;
    }
    u.alive = arr(e.alive, n), u.hp = arr(e.hp, n), u.max_hp = arr(e.max_hp, n), u.radius = arr(e.radius, n);
    u.bonus_hp.assign(n, 0.f), u.armor = arr(e.armor, n), u.magic_resist = arr(e.magic_resist, n);
    return u;
}

// The world's champion tables (World::tables).
struct ChampData {
    ChampionBase base;
    Arr<int32_t> ids, pages, skill_order, auto_skill;
    Arr<uint8_t> adaptive_physical, uses_energy, allowed;
    Arr<float> fountain, unit_target_ranges, shard_const, shard_hs;
    int n_runes = 0;
    bool has_initial = false;
    ItemEffectState initial_items;                          // the dormancy references (dispatch.hpp)
    RuneEffectState initial_runes;
    explicit ChampData(const World& w) {
        if (!w.initial.empty()) {
            Env e0;
            void** dst = reinterpret_cast<void**>(&e0);
            for (size_t f = 0; f < w.initial.size(); ++f) dst[f] = const_cast<uint8_t*>(w.initial[f].data());
            CombatState combat;
            env_load(w, e0, roots().combat, combat);
            initial_items = std::move(combat.items), initial_runes = std::move(combat.runes), has_initial = true;
        }
        auto ints = [&](const char* k) { Arr<int32_t> a; for (float v : w.tab(k)) a.push_back((int32_t)v); return a; };
        auto bools = [&](const char* k) { Arr<uint8_t> a; for (float v : w.tab(k)) a.push_back(v != 0.f); return a; };
        auto floats = [&](const char* k) { Arr<float> a; for (float v : w.tab(k)) a.push_back(v); return a; };
        ids = ints("champion_ids"), pages = ints("rune_pages"), skill_order = ints("skill_order");
        auto_skill = ints("auto_skill"), adaptive_physical = bools("adaptive_physical");
        uses_energy = bools("uses_energy"), fountain = floats("fountain"), unit_target_ranges = floats("unit_target_ranges");
        shard_const = floats("shard_const"), shard_hs = floats("shard_hs");
        if (w.tables.count("item_allowed")) allowed = bools("item_allowed");
        n_runes = (int)(pages.size() / N_CHAMPIONS);
        const auto& b = w.tab("champion_base");                  // (21 fields, C)
        size_t k = 0;
        base.visit([&](auto& m) {
            m.resize(N_CHAMPIONS);
            for (int c = 0; c < N_CHAMPIONS; ++c) m[c] = b[k * N_CHAMPIONS + c];
            ++k;
        });
    }
};
const ChampData& champ_data(const World& w) {
    static thread_local const World* cached = nullptr;
    static thread_local std::unique_ptr<ChampData> d;
    if (cached != &w) d = std::make_unique<ChampData>(w), cached = &w;
    return *d;
}

// views.shard_stats: constant shards plus Health Scaling's lin(start, end, level).
ItemStats shard_stats(const ChampData& cd, const Arr<int32_t>& level) {
    ItemStats s = stats::zero(N_CHAMPIONS);
    auto f = stats::fields(s);
    for (int c = 0; c < N_CHAMPIONS; ++c) {
        for (size_t k = 0; k < f.size(); ++k) (*f[k])[c] = cd.shard_const[c * f.size() + k];
        if (cd.shard_hs[c * 3] != 0.f) {
            float lv = std::max((float)level[c], 1.f);
            s.health[c] = 0.f + (cd.shard_hs[c * 3 + 1] + cd.shard_hs[c * 3 + 2] * (lv - 1.f) / 17.f);
        }
    }
    return s;
}

// mechanics.capabilities over every unit.
struct Caps {
    Arr<uint8_t> can_move, can_attack, can_cast, can_summoner, stunned, silenced, impaired, movement_impaired;
    Arr<float> slow;
};
Caps capabilities(const Env& e, int n, float now) {
    Caps c;
    for (auto* a : {&c.can_move, &c.can_attack, &c.can_cast, &c.can_summoner, &c.stunned, &c.silenced, &c.impaired,
                    &c.movement_impaired})
        a->resize(n);
    c.slow.resize(n);
    for (int j = 0; j < n; ++j) {
        bool stun = e.cc_stun_until[j] > now, root = e.cc_root_until[j] > now, sil = e.cc_silence_until[j] > now;
        bool up = e.cc_knockup_until[j] > now, slowed = e.cc_slow_until[j] > now;
        c.can_move[j] = !(stun || root || up), c.can_attack[j] = !(stun || up), c.can_cast[j] = !(stun || up || sil);
        c.can_summoner[j] = !(stun || up), c.stunned[j] = stun || up, c.silenced[j] = sil;
        c.slow[j] = slowed ? e.cc_slow[j] : 0.f;
        c.impaired[j] = stun || root || sil || up || slowed;
        c.movement_impaired[j] = stun || root || up || slowed;
    }
    return c;
}

// --- views.py ------------------------------------------------------------------------------------------------------
// static_stats: items + shards, then kit stats (no monster buffs in the lane world), composed without slows.
void static_stats(const World& w, const Env& e, const ChampData& cd, Layer& L, const Caps& caps, float now,
                  ItemStats& stat_out, ChampionStats& st_out);
KitCtx kit_ctx(const World& w, const Env& e, const ChampData& cd, const Layer& L, const ChampionStats& st,
               const Caps& caps, float now) {
    KitCtx k;
    const int c = N_CHAMPIONS, n = w.n;
    k.unit.resize(c), k.team.resize(c), k.alive.resize(c), k.x.resize(c), k.y.resize(c), k.hp.resize(c);
    k.max_hp.resize(c), k.bonus_hp.resize(c), k.armor.resize(c), k.magic_resist.resize(c), k.silenced.resize(c);
    k.stunned.resize(c), k.in_combat_ms_since_damaged.resize(c), k.attack_target_kind.resize(c), k.rooted.resize(c);
    for (int h = 0; h < c; ++h) {
        k.unit[h] = h, k.team[h] = e.team[h], k.alive[h] = e.alive[h], k.x[h] = e.x[h], k.y[h] = e.y[h];
        k.hp[h] = e.hp[h], k.max_hp[h] = e.max_hp[h], k.bonus_hp[h] = e.max_hp[h] - st.base_hp[h];
        k.armor[h] = st.base_armor[h] + st.bonus_armor[h], k.magic_resist[h] = st.base_mr[h] + st.bonus_mr[h];
        k.silenced[h] = caps.silenced[h], k.stunned[h] = caps.stunned[h];
        k.in_combat_ms_since_damaged[h] = now - L.champ.last_damaged[h];
        int ao = L.champ.attack_order[h];
        k.attack_target_kind[h] = ao >= 0 ? e.kind[clampi(ao, 0, n - 1)] : KIND_NONE;
        k.rooted[h] = e.cc_root_until[h] > now;
    }
    k.champion_id = cd.ids, k.level = L.econ.level, k.ranks = L.champ.ranks, k.mana = L.champ.mana;
    k.max_mana = st.max_mana, k.base_ad = st.base_ad, k.bonus_ad = st.bonus_ad, k.ap = st.ap;
    k.bonus_attack_speed = st.bonus_attack_speed, k.crit_chance = st.crit_chance, k.crit_damage = st.crit_damage;
    k.ability_haste = st.basic_ability_haste, k.ultimate_haste = st.ultimate_haste, k.cooldowns = L.champ.cooldowns;
    k.now = now, k.dt = w.dt;
    return k;
}

Ctx item_ctx(const World& w, const Env& e, const ChampData& cd, const Layer& L, const ChampionStats& st, float now) {
    Ctx x;
    const int c = N_CHAMPIONS;
    x.now = now, x.dt = w.dt;
    for (auto* a : {&x.level, &x.x, &x.y, &x.facing_x, &x.facing_y, &x.moved, &x.max_hp, &x.hp, &x.base_ms, &x.base_mana})
        a->resize(c);
    x.unit.resize(c), x.team.resize(c), x.alive.resize(c), x.is_ranged.resize(c), x.in_combat.resize(c);
    x.in_shop.resize(c);
    for (int h = 0; h < c; ++h) {
        x.unit[h] = h, x.team[h] = e.team[h], x.alive[h] = e.alive[h], x.level[h] = (float)L.econ.level[h];
        x.is_ranged[h] = st.attack_range[h] > 300.f, x.x[h] = e.x[h], x.y[h] = e.y[h];
        x.facing_x[h] = L.champ.facing[h * 2], x.facing_y[h] = L.champ.facing[h * 2 + 1], x.moved[h] = 0.f;
        x.max_hp[h] = e.max_hp[h], x.hp[h] = e.hp[h];
        x.in_combat[h] = (now - L.combat.clocks.last_combat[h]) < 5.f;
        x.in_shop[h] = api::I_in_shop_area(e.x[h], e.y[h], e.team[h], !e.alive[h]);
        x.base_ms[h] = cd.base.base_ms[h], x.base_mana[h] = cd.base.base_mana[h];
    }
    x.base_ad = st.base_ad, x.bonus_ad = st.bonus_ad, x.ap = st.ap, x.base_hp = st.base_hp;
    x.base_armor = st.base_armor, x.bonus_armor = st.bonus_armor, x.base_mr = st.base_mr, x.bonus_mr = st.bonus_mr;
    x.mana = L.champ.mana, x.max_mana = st.max_mana, x.move_speed = st.move_speed, x.crit_chance = st.crit_chance;
    x.crit_damage = st.crit_damage, x.life_steal = st.life_steal, x.bonus_attack_speed = st.bonus_attack_speed;
    x.ability_haste = st.basic_ability_haste, x.lethality = st.lethality, x.heal_shield_power = st.heal_shield_power;
    x.attack_windup = st.attack_windup, x.attack_range = st.attack_range;
    return x;
}

void static_stats(const World& w, const Env& e, const ChampData& cd, Layer& L, const Caps& caps, float now,
                  ItemStats& stat_out, ChampionStats& st_out) {
    const Arr<int32_t>& level = L.econ.level;
    ItemStats inv = api::I_inventory_stats(L.champ.inventory), shards = shard_stats(cd, level);
    ItemStats base_static = stats::combine2(inv, shards);
    ChampionStats st0 = stats::compose(cd.base, level, base_static, cd.adaptive_physical, {}, {}, {}, {});
    base_static = stats::combine2(base_static, stats::zero(N_CHAMPIONS));          // monster_buff_stats: none
    st0 = stats::compose(cd.base, level, base_static, cd.adaptive_physical, {}, {}, {}, {});
    stat_out = stats::combine2(base_static, api::K_stats(L.kits, kit_ctx(w, e, cd, L, st0, caps, now)));
    st_out = stats::compose(cd.base, level, stat_out, cd.adaptive_physical, {}, {}, {}, {});
}

// --- tick scratch (world/scratch.py) ------------------------------------------------------------------------------
struct TS {
    float now = 0.f;
    rng::Key key{}, k_crit{};
    Arr<uint8_t> vis_c, in_stasis, moving, locked, item_casting, moving_eff, t_ok_eff, tp_lock, dstart, in_dash;
    Caps caps;
    Arr<int32_t> attack_order, attack_order_eff;
    Arr<float> goal, reach;
    ItemStats stat, summ_world;
    ChampionStats st_static, st;
    WorldUnits units;
    KitCtx kctx;
    Ctx ictx;
    CastOrder cast_order;
    KitOut kit_out, kit_all;
    KitAttackMods kmods;
    Effects s_eff, extra;
    SummonerOut s_out;
    Arr<int32_t> shop_code, bought, sold;
};

// --- phase 1, INPUT (world/phases/inputs.py) ----------------------------------------------------------------------
void queue_casts(const World& w, const Env& e, const ChampData& cd, Layer& L, Orders& o, const Arr<uint8_t>& new_order,
                 const TS& sc, Arr<uint8_t>& chase, Arr<uint8_t>& walked_in) {
    const int c = N_CHAMPIONS, n = w.n;
    const float now = sc.now;
    auto seen = [&](int h, int u) { return u >= 0 && sc.vis_c[(size_t)h * n + clampi(u, 0, n - 1)]; };
    auto needs = [&](int h, int slot, int target, bool* far, bool* blocked) {
        int sl = clampi(slot, 0, 3);
        float r = cd.unit_target_ranges[h * 4 + sl];
        int t = clampi(target, 0, n - 1);
        float d = std::sqrt(sq(e.x[h] - e.x[t]) + sq(e.y[h] - e.y[t]));
        *far = slot >= 0 && target >= 0 && r > 0.f && d > r + e.radius[t] - CAST_RANGE_SLACK;
        float cd_ = L.champ.cooldowns[h * 4 + sl];
        *blocked = slot >= 0 && (now < L.champ.cast_lock_until[h] || now < L.champ.item_cast_until[h]
                                 || (cd_ > 0.f && cd_ <= CAST_BUFFER_S));
    };
    QueuedCast& q = L.champ.queued_cast;
    chase.assign(c, 0), walked_in.assign(c, 0);
    for (int h = 0; h < c; ++h) {
        bool incoming = o.cast_slot[h] >= 0;
        bool far_in, blocked_in;
        needs(h, o.cast_slot[h], o.cast_target[h], &far_in, &blocked_in);
        bool hold_in = incoming && (far_in || blocked_in);
        int qt = q.target[h];
        bool keep = q.slot[h] >= 0 && !incoming && !new_order[h] && now < q.until[h]
                    && (qt < 0 || (seen(h, qt) && e.alive[clampi(qt, 0, n - 1)]));
        if (!keep) q.slot[h] = -1, q.target[h] = -1, q.x[h] = 0.f, q.y[h] = 0.f, q.until[h] = 0.f;
        if (hold_in) {
            q.slot[h] = o.cast_slot[h], q.target[h] = o.cast_target[h], q.x[h] = o.cast_x[h], q.y[h] = o.cast_y[h];
            q.until[h] = far_in ? INF : now + CAST_BUFFER_S;
        }
        bool far_q, blocked_q;
        needs(h, q.slot[h], q.target[h], &far_q, &blocked_q);
        bool fire = q.slot[h] >= 0 && !hold_in && !far_q && !blocked_q;
        bool go_now = incoming && !hold_in;
        int slot = go_now ? o.cast_slot[h] : (fire ? q.slot[h] : -1);
        int target = go_now ? o.cast_target[h] : (fire ? q.target[h] : o.cast_target[h]);
        float cx = go_now ? o.cast_x[h] : (fire ? q.x[h] : o.cast_x[h]);
        float cy = go_now ? o.cast_y[h] : (fire ? q.y[h] : o.cast_y[h]);
        o.cast_slot[h] = slot, o.cast_target[h] = target, o.cast_x[h] = cx, o.cast_y[h] = cy;
        chase[h] = q.slot[h] >= 0 && far_q && !fire;
        walked_in[h] = fire && std::isinf(q.until[h]);
        if (fire) q.slot[h] = -1, q.target[h] = -1, q.x[h] = 0.f, q.y[h] = 0.f, q.until[h] = 0.f;
    }
}

void skill_up(const ChampData& cd, Layer& L, const Orders& o) {
    for (int h = 0; h < N_CHAMPIONS; ++h) {
        int level = L.econ.level[h];
        int points = api::E_skill_points(level) + L.champ.bonus_points[h];
        int spent = 0;
        for (int s = 0; s < 4; ++s) spent += L.champ.ranks[h * 4 + s];
        int automatic = cd.skill_order[h * 20 + clampi(spent, 0, 19)];
        int slot = o.level_up[h] >= 0 ? o.level_up[h] : automatic;
        int cap = slot == 3 ? api::E_max_rank(level, true) : api::E_max_rank(level, false);
        slot = clampi(slot, 0, 3);
        int cur = L.champ.ranks[h * 4 + slot];
        bool ok = points > spent && cur < cap && (o.level_up[h] >= 0 || cd.auto_skill[h]);
        L.champ.ranks[h * 4 + slot] += ok;
    }
}

void phase_input(const World& w, Env& e, const ChampData& cd, Layer& L, Orders& o, TS& sc) {
    const int c = N_CHAMPIONS, n = w.n;
    const float now = sc.now;
    ChampionLayer& ch = L.champ;
    sc.vis_c.resize((size_t)c * n);
    for (int h = 0; h < c; ++h)
        for (int j = 0; j < n; ++j) sc.vis_c[(size_t)h * n + j] = e.visible[(size_t)e.team[h] * n + j];
    auto seen = [&](int h, int u) { return u >= 0 && sc.vis_c[(size_t)h * n + clampi(u, 0, n - 1)]; };
    for (int h = 0; h < c; ++h) {
        if (!seen(h, o.attack[h])) o.attack[h] = -1;
        if (!seen(h, o.cast_target[h])) o.cast_target[h] = -1;
        if (!seen(h, o.summoner_target[h])) o.summoner_target[h] = -1;
    }
    sc.in_stasis.resize(c);
    Arr<uint8_t> new_order(c), am_req(c), lost(c);
    sc.attack_order.resize(c), sc.moving.resize(c), sc.goal.resize(2 * c);
    for (int h = 0; h < c; ++h) {
        sc.in_stasis[h] = now < L.combat.items.actives.stasis_until[h];
        am_req[h] = o.attack_move[h] && !sc.in_stasis[h];
        int kept = seen(h, ch.attack_order[h]) ? ch.attack_order[h] : -1;
        new_order[h] = o.stop[h] || o.move[h] || o.attack[h] >= 0 || am_req[h];
        sc.attack_order[h] = (o.stop[h] || o.move[h] || am_req[h]) ? -1 : (o.attack[h] >= 0 ? o.attack[h] : kept);
        sc.moving[h] = (o.stop[h] || o.attack[h] >= 0 || am_req[h]) ? 0 : (o.move[h] || ch.moving[h]);
        sc.goal[h * 2] = o.move[h] ? o.move_x[h] : ch.move_goal[h * 2];
        sc.goal[h * 2 + 1] = o.move[h] ? o.move_y[h] : ch.move_goal[h * 2 + 1];
        int t_old = clampi(ch.attack_order[h], 0, n - 1);
        lost[h] = ch.attack_order[h] >= 0 && !seen(h, ch.attack_order[h]) && e.alive[t_old] && !new_order[h];
        // A live target that enters fog: walk to where it was last seen (the previous value).
        if (lost[h]) sc.goal[h * 2] = ch.target_seen_at[h * 2], sc.goal[h * 2 + 1] = ch.target_seen_at[h * 2 + 1];
    }
    Arr<float> seen_at = ch.target_seen_at;
    for (int h = 0; h < c; ++h) {
        int t_now = clampi(sc.attack_order[h], 0, n - 1);
        if (seen(h, sc.attack_order[h])) seen_at[h * 2] = e.x[t_now], seen_at[h * 2 + 1] = e.y[t_now];
    }
    Arr<uint8_t> cast_chase, cast_fired;
    queue_casts(w, e, cd, L, o, new_order, sc, cast_chase, cast_fired);
    for (int h = 0; h < c; ++h) {
        sc.moving[h] = sc.moving[h] || lost[h];
        sc.moving[h] = cast_chase[h] ? 1 : (cast_fired[h] ? 0 : sc.moving[h]);
        if (cast_chase[h]) sc.attack_order[h] = -1;
        int tq = clampi(L.champ.queued_cast.target[h], 0, n - 1);
        if (cast_chase[h]) sc.goal[h * 2] = e.x[tq], sc.goal[h * 2 + 1] = e.y[tq];
        AttackMove& am = L.amove;
        if (new_order[h]) am.active[h] = am_req[h], am.held[h] = -1;
        if (am_req[h]) am.x[h] = o.move_x[h], am.y[h] = o.move_y[h];
    }
    skill_up(cd, L, o);
    ch.target_seen_at = seen_at;
    for (int h = 0; h < c; ++h) {
        ch.attack_order[h] = sc.attack_order[h], ch.moving[h] = sc.moving[h];
        ch.move_goal[h * 2] = sc.goal[h * 2], ch.move_goal[h * 2 + 1] = sc.goal[h * 2 + 1];
    }
    slice::spawn(w, e, now);                       // bounty_level uses max(econ.level): e.level mirrors it
    sc.caps = capabilities(e, n, now);
    for (int h = 0; h < c; ++h) {
        e.targetable[h] = !sc.in_stasis[h];
        if (sc.in_stasis[h])
            sc.caps.can_move[h] = sc.caps.can_attack[h] = sc.caps.can_cast[h] = sc.caps.can_summoner[h] = 0;
    }
}

// --- phase 2, STATS (world/phases/stats.py) ------------------------------------------------------------------------
void phase_stats(const World& w, Env& e, const ChampData& cd, Layer& L, TS& sc) {
    static_stats(w, e, cd, L, sc.caps, sc.now, sc.stat, sc.st_static);
    for (int h = 0; h < N_CHAMPIONS; ++h) {
        float old_total = e.max_hp[h], new_total = sc.st_static.max_hp[h] + L.combat.dyn_health[h];
        float delta = new_total - old_total;                    // stat_pipeline.sync_max_health
        float hp = std::min(std::max(e.hp[h] + std::max(delta, 0.f), 0.f), new_total);
        e.hp[h] = e.alive[h] ? hp : e.hp[h];
        e.max_hp[h] = new_total;
        L.champ.static_max_hp[h] = sc.st_static.max_hp[h];
    }
}

// --- phase 3, CASTS (world/phases/casts.py) ------------------------------------------------------------------------
void shop(const World& w, Env& e, const ChampData& cd, Layer& L, const Orders& o, TS& sc) {
    const int ni = n_items();
    const float t0 = *e.t;
    sc.shop_code.assign(N_CHAMPIONS, 0), sc.bought.assign(N_CHAMPIONS, 0), sc.sold.assign(N_CHAMPIONS, 0);
    for (int h = 0; h < N_CHAMPIONS; ++h) {
        bool can = api::I_in_shop_area(e.x[h], e.y[h], e.team[h], !e.alive[h]);
        Inventory inv;
        inv.item.resize(7), inv.stack.resize(7);
        for (int s = 0; s < 7; ++s) inv.item[s] = L.champ.inventory.item[h * 7 + s], inv.stack[s] = L.champ.inventory.stack[h * 7 + s];
        Arr<float> gcd(L.champ.group_cd.size() / N_CHAMPIONS);
        for (size_t g = 0; g < gcd.size(); ++g) gcd[g] = L.champ.group_cd[h * gcd.size() + g];
        int buy_id = o.buy[h], sell_id = o.sell[h];
        int row = 0;
        bool known = false;
        if (buy_id > 0) {
            const auto& ids = data::table("catalog.ids");
            for (int r = 0; r < ni; ++r)
                if ((int)ids[r] == buy_id) { row = r, known = true; break; }
        }
        bool want = buy_id > 0 && known && !L.champ.forbid[(size_t)h * ni + row];
        if (cd.allowed.size()) want = want && cd.allowed[(size_t)h * ni + row];
        float gold = L.econ.gold[h];
        bool ok_buy = false, ok_sell = false;
        int code_buy = api::I_buy(inv, gold, row, can && want, L.econ.level[h], sc.st_static.attack_range[h] > 300.f, t0,
                                  gcd, &ok_buy);
        int srow = 0;                                           // argmax(ids == sell): row 0 when unknown
        if (sell_id > 0) {
            const auto& ids = data::table("catalog.ids");
            for (int r = 0; r < ni; ++r)
                if ((int)ids[r] == sell_id) { srow = r; break; }
        }
        int slot = 0;
        bool has_it = false;
        for (int s = 0; s < 7; ++s)
            if (inv.item[s] == srow) { slot = s, has_it = true; break; }
        has_it = has_it && sell_id > 0;
        int code_sell = api::I_sell(inv, gold, slot, can && has_it, &ok_sell);
        for (int s = 0; s < 7; ++s) L.champ.inventory.item[h * 7 + s] = inv.item[s], L.champ.inventory.stack[h * 7 + s] = inv.stack[s];
        for (size_t g = 0; g < gcd.size(); ++g) L.champ.group_cd[h * gcd.size() + g] = gcd[g];
        L.econ.gold[h] = gold;
        sc.shop_code[h] = want ? code_buy : (has_it ? code_sell : 0);
        sc.bought[h] = (want && ok_buy) ? buy_id : 0;
        sc.sold[h] = (has_it && ok_sell) ? sell_id : 0;
    }
}

void phase_casts(const World& w, Env& e, const ChampData& cd, Layer& L, const Orders& o, TS& sc) {
    const int c = N_CHAMPIONS, n = w.n;
    const float now = sc.now;
    shop(w, e, cd, L, o, sc);
    sc.summ_world = stats::combine2(sc.stat, L.champ.dyn);
    Arr<float> slow_c(c);
    for (int h = 0; h < c; ++h) slow_c[h] = sc.caps.slow[h];
    sc.st = stats::compose(cd.base, L.econ.level, sc.summ_world, cd.adaptive_physical, slow_c, {}, {}, {});
    sc.units = units_view(w, e);
    sc.kctx = kit_ctx(w, e, cd, L, sc.st, sc.caps, now);
    sc.ictx = item_ctx(w, e, cd, L, sc.st_static, now);
    sc.locked.resize(c), sc.item_casting.resize(c);
    CastOrder order;
    order.slot.resize(c), order.target = arr(o.cast_target, c), order.x = arr(o.cast_x, c), order.y = arr(o.cast_y, c);
    for (int h = 0; h < c; ++h) {
        sc.locked[h] = now < L.champ.cast_lock_until[h];
        sc.item_casting[h] = now < L.champ.item_cast_until[h];
        bool can_cast = sc.caps.can_cast[h] && e.alive[h] && !sc.locked[h] && !sc.item_casting[h];
        order.slot[h] = can_cast ? o.cast_slot[h] : -1;
    }
    sc.cast_order = order;
    KitOut k_cast, k_per;
    std::tie(L.kits, k_cast) = api::K_cast(L.kits, sc.kctx, sc.units, order);
    std::tie(L.kits, k_per) = api::K_periodic(L.kits, sc.kctx, sc.units);
    sc.kit_out = no_out(c, n);
    merge_out_into(sc.kit_out, k_cast, c, n);
    merge_out_into(sc.kit_out, k_per, c, n);
    CastOrder req;
    req.slot = arr(o.summoner_slot, c), req.target = arr(o.summoner_target, c), req.x = arr(o.summoner_x, c);
    req.y = arr(o.summoner_y, c);
    Arr<uint8_t> can_summ(c), interrupted(c), rooted(c);
    for (int h = 0; h < c; ++h) {
        can_summ[h] = sc.caps.can_summoner[h] && e.alive[h];
        interrupted[h] = sc.caps.stunned[h] || !e.alive[h];
        rooted[h] = e.cc_root_until[h] > now;
    }
    std::tie(L.summ, sc.s_eff, sc.s_out) = api::S_step(L.summ, sc.ictx, sc.units, req, now, w.dt, sc.st.summoner_haste,
                                                       can_summ, interrupted, L.econ.quest.complete, rooted);
    sc.kmods = api::K_attack_mods(L.kits, sc.kctx);
    sc.reach.resize(c);
    for (int h = 0; h < c; ++h) sc.reach[h] = sc.st.attack_range[h] + sc.kmods.extra_range[h];
}

// --- phase 4, AI: champion acquisition and attack-move (lane.ai) -----------------------------------------------------
// lane.ai._hostile for champion ``me``: enemy champions, minions, structures (and wards), alive, targetable, seen.
bool hostile(const WorldUnits& u, const TS& sc, int me, int j, bool wards, int n) {
    int k = u.kind[j];
    bool kinds = k == KIND_CHAMPION || k == KIND_MINION || k == KIND_TURRET || k == KIND_INHIBITOR || k == KIND_NEXUS
                 || (wards && k == KIND_WARD);
    bool enemy = u.team[j] != u.team[me] && u.team[j] != 2 && kinds;
    return enemy && u.alive[j] && u.targetable[j] && sc.vis_c[(size_t)me * n + j] && j != me;
}

void phase_ai_champions(const World& w, Env& e, const ChampData& cd, Layer& L, TS& sc, int32_t* desired) {
    const int c = N_CHAMPIONS, n = w.n;
    const WorldUnits& u = sc.units;
    Arr<uint8_t> t_ok(c);
    Arr<float> acq(c);
    for (int h = 0; h < c; ++h) {
        int ao = sc.attack_order[h];
        t_ok[h] = ao >= 0 && e.alive[clampi(ao, 0, n - 1)];
        acq[h] = CHAMPION_ACQUISITION_RANGE + (sc.reach[h] - cd.base.attack_range[h]);
    }
    auto center = [&](int me, int j) { return std::sqrt(sq(u.x[j] - u.x[me]) + sq(u.y[j] - u.y[me])); };
    // _nearest: argmin over the mask by centre distance, -1 if empty.
    auto nearest = [&](int me, bool wards, float reach) {
        int best = -1;
        float bd = INF;
        for (int j = 0; j < n; ++j) {
            if (!hostile(u, sc, me, j, wards, n)) continue;
            float d = center(me, j);
            if (!(d - u.radius[j] - u.radius[me] <= reach)) continue;
            if (d < bd) bd = d, best = j;
        }
        return best;
    };
    AttackMove& am = L.amove;
    sc.attack_order_eff.resize(c), sc.t_ok_eff.resize(c), sc.moving_eff.resize(c);
    for (int h = 0; h < c; ++h) {
        int automatic = nearest(h, false, acq[h]);              // idle_acquire (no wards)
        bool idle = !t_ok[h] && !sc.moving[h] && !am.active[h] && e.alive[h] && automatic >= 0;
        if (idle) sc.attack_order[h] = automatic;
        t_ok[h] = t_ok[h] || idle;
        // attack_move_step
        bool act = am.active[h] && e.alive[h];
        int held = am.held[h], hs = clampi(held, 0, n - 1);
        bool keep = act && held >= 0 && hostile(u, sc, h, hs, true, n) && u.spawn_seq[hs] == am.held_seq[h];
        int scan = nearest(h, true, acq[h]);
        int target = keep ? held : (act ? scan : -1);
        int ts = clampi(target, 0, n - 1);
        float gx = target >= 0 ? u.x[ts] : am.x[h], gy = target >= 0 ? u.y[ts] : am.y[h];
        bool arrived = target < 0 && std::sqrt(sq(u.x[h] - am.x[h]) + sq(u.y[h] - am.y[h])) <= ATTACK_MOVE_ARRIVE;
        am.active[h] = act && !arrived;
        am.held[h] = target, am.held_seq[h] = target >= 0 ? u.spawn_seq[ts] : 0;
        bool am_tgt = am.active[h] && target >= 0 && !t_ok[h];
        sc.attack_order_eff[h] = am_tgt ? target : sc.attack_order[h];
        sc.t_ok_eff[h] = t_ok[h] || am_tgt;
        if (am.active[h] && !sc.t_ok_eff[h]) sc.goal[h * 2] = gx, sc.goal[h * 2 + 1] = gy;
        sc.moving_eff[h] = sc.moving[h] || (am.active[h] && !sc.t_ok_eff[h]);
        desired[h] = sc.t_ok_eff[h] ? sc.attack_order_eff[h] : -1;
        L.champ.attack_order[h] = sc.attack_order[h];
    }
}

// --- phase 5, MOVE (world/phases/move.py) --------------------------------------------------------------------------
// mechanics.blink_point for one champion: clamp to the range, then the farthest walkable sample toward the origin.
void blink_point(const World& w, float x0, float y0, float x1, float y1, float max_range, int team, float radius,
                 float* ox, float* oy) {
    static const std::vector<float>& lin = data::table("world.flash_linspace");
    float dx = x1 - x0, dy = y1 - y0;
    float d = std::sqrt(dx * dx + dy * dy);
    float s = std::min(1.f, max_range / std::max(d, 1e-6f));
    float tx = x0 + dx * s, ty = y0 + dy * s;
    const Terrain& ter = w.terrain[team == 1 ? 1 : 0];
    for (float f : lin) {
        float px = x0 + (tx - x0) * f, py = y0 + (ty - y0) * f;
        if (ter.walkable(px, py, std::min(radius, 50.f), 3)) { *ox = px, *oy = py; return; }
    }
    *ox = x0, *oy = y0;
}

void phase_move(const World& w, Env& e, const ChampData& cd, Layer& L, TS& sc, slice::Scratch& ss) {
    const int c = N_CHAMPIONS, n = w.n;
    const float now = sc.now, dt = w.dt;
    ChampionLayer& ch = L.champ;
    const SummonerOut& so = sc.s_out;
    sc.tp_lock.resize(c);
    Arr<uint8_t> can_move_c(c), chase(c), cact(c);
    Arr<float> cgx(c), cgy(c);
    for (int h = 0; h < c; ++h) {
        sc.tp_lock[h] = so.teleport_channel[h] || so.teleport_dash[h];
        can_move_c[h] = sc.caps.can_move[h] && e.alive[h] && !sc.locked[h] && !sc.tp_lock[h] && now >= ch.dash_until[h];
        int target = sc.t_ok_eff[h] ? sc.attack_order_eff[h] : -1;
        int tgt = clampi(sc.attack_order_eff[h], 0, n - 1);
        bool in_range = target >= 0
                        && std::sqrt(sq(e.x[h] - e.x[target]) + sq(e.y[h] - e.y[target]))
                               <= sc.reach[h] + e.radius[h] + e.radius[target];
        chase[h] = sc.t_ok_eff[h] && !in_range;
        cgx[h] = chase[h] ? e.x[tgt] : sc.goal[h * 2], cgy[h] = chase[h] ? e.y[tgt] : sc.goal[h * 2 + 1];
        cact[h] = can_move_c[h] && (chase[h] || (sc.moving_eff[h] && !sc.t_ok_eff[h]));
    }
    float *gx = ss.gx.data(), *gy = ss.gy.data(), *ms = ss.ms.data();
    uint8_t *active = ss.active.data(), *solid = ss.collide.data();
    // Champions: Ghost/Heal, Homeguard are bonus % MS before the soft caps; slows with slow resist.
    ItemStats sw = sc.summ_world;
    for (int h = 0; h < c; ++h)
        sw.percent_move_speed[h] = sw.percent_move_speed[h] + (so.bonus_ms_pct[h] + 0.f + ch.homeguard_ms[h] + 0.f);
    Arr<float> slow_c(c);
    for (int h = 0; h < c; ++h) slow_c[h] = sc.caps.slow[h];
    ChampionStats cst = stats::compose(cd.base, L.econ.level, sw, cd.adaptive_physical, slow_c, {}, {}, {});
    for (int i = 0; i < n; ++i) {
        bool minion = e.kind[i] == KIND_MINION && e.alive[i];
        float base = e.move_speed[i];
        if (e.kind[i] == KIND_MINION) {
            int idx = minion::wave_index_at(e.spawn_time[i]);
            float bonus = minion::sidelane_bonus(idx + 1, e.lane_ai_lane[i], minion::wave_spawn_time(idx),
                                                 now - e.spawn_time[i]);
            base = minion::soft_cap(minion::base_move_speed(now) + bonus);
        }
        ms[i] = base * (1.f - sc.caps.slow[i] * (1.f - 0.f));
        active[i] = minion && !ss.stop[i] && sc.caps.can_move[i];
        if (!minion) gx[i] = e.x[i], gy[i] = e.y[i];
    }
    for (int h = 0; h < c; ++h) gx[h] = cgx[h], gy[h] = cgy[h], ms[h] = cst.move_speed[h], active[h] = cact[h];
    float *mx = ss.nx.data(), *my = ss.ny.data();
    slice::move_step(w, e, gx, gy, ms, active, mx, my);
    // Kit dashes follow their target unit, ignoring terrain; the item dash (Rocketbelt) starts a tick late.
    Dash dash = sc.kit_out.dash;
    const Dash& pd = L.prev_pending_dash;
    sc.dstart.resize(c), sc.in_dash.resize(c);
    Arr<float> dash_until(c), dxs(c), dys(c);
    Arr<int32_t> dash_target(c);
    for (int h = 0; h < c; ++h) {
        if (pd.active[h] && !dash.active[h]) {
            dash.active[h] = pd.active[h], dash.to_x[h] = pd.to_x[h], dash.to_y[h] = pd.to_y[h];
            dash.speed[h] = pd.speed[h], dash.target[h] = pd.target[h], dash.blink[h] = pd.blink[h];
        }
        sc.dstart[h] = dash.active[h] && e.alive[h];
        int dtg = clampi(dash.target[h], 0, n - 1);
        dxs[h] = dash.target[h] >= 0 ? e.x[dtg] : dash.to_x[h], dys[h] = dash.target[h] >= 0 ? e.y[dtg] : dash.to_y[h];
        float dist = std::sqrt(sq(dxs[h] - e.x[h]) + sq(dys[h] - e.y[h]));
        dash_until[h] = sc.dstart[h] ? now + dist / std::max(dash.speed[h], 1.f) : ch.dash_until[h];
        dash_target[h] = sc.dstart[h] ? dash.target[h] : ch.dash_target[h];
        sc.in_dash[h] = now < dash_until[h];
        int ft = clampi(dash_target[h], 0, n - 1);
        float fx = dash_target[h] >= 0 ? e.x[ft] : ch.dash_to[h * 2], fy = dash_target[h] >= 0 ? e.y[ft] : ch.dash_to[h * 2 + 1];
        float fd = std::sqrt(sq(fx - mx[h]) + sq(fy - my[h]));
        float step_d = std::min((dash.speed[h] > 0.f ? dash.speed[h] : 1400.f) * dt, fd);
        float frac = fd > 1e-6f ? step_d / std::max(fd, 1e-6f) : 0.f;
        float cx = sc.in_dash[h] ? mx[h] + (fx - mx[h]) * frac : mx[h];
        float cy = sc.in_dash[h] ? my[h] + (fy - my[h]) * frac : my[h];
        if (so.dash.active[h] && so.dash.blink[h]) {
            float bx, by;
            blink_point(w, cx, cy, so.dash.to_x[h], so.dash.to_y[h], api::S_flash_range(), e.team[h], e.radius[h], &bx, &by);
            cx = bx, cy = by;
        }
        if (so.teleport_arrive[h]) cx = so.teleport_x[h], cy = so.teleport_y[h];
        mx[h] = cx, my[h] = cy;
    }
    // Collision. Wards don't collide; structures block only through their navgrid pads.
    Arr<uint8_t> kghost = api::K_ghosted(L.kits, sc.kctx);
    for (int i = 0; i < n; ++i) {
        bool ghost = e.kind[i] == KIND_MINION && e.alive[i] && minion::wave_index_at(e.spawn_time[i]) == 0
                     && (now - e.spawn_time[i]) < minion::first_wave_ghost_s(e.lane_ai_lane[i]);
        if (i < c) ghost = ghost || so.ghosted[i] || now < dash_until[i] || kghost[i];
        bool collide = e.alive[i] && e.kind[i] != KIND_WARD && !(e.kind[i] == KIND_TURRET || e.kind[i] == KIND_INHIBITOR
                                                                 || e.kind[i] == KIND_NEXUS);
        solid[i] = collide && !ghost;
    }
    float *cx = ss.start_x.data(), *cy = ss.start_y.data();
    slice::collide(w, e, mx, my, gx, gy, active, solid, cx, cy);
    for (int h = 0; h < c; ++h) {
        float fx = cx[h] - e.x[h], fy = cy[h] - e.y[h];
        float norm = std::sqrt(fx * fx + fy * fy);
        if (norm > 1e-3f) {
            float m = std::max(norm, 1e-6f);
            ch.facing[h * 2] = fx / m, ch.facing[h * 2 + 1] = fy / m;
        }
        float moved = std::sqrt(sq(cx[h] - e.x[h]) + sq(cy[h] - e.y[h]));
        sc.ictx.moved[h] = moved;
        float to_goal = std::sqrt(sq(cx[h] - ch.move_goal[h * 2]) + sq(cy[h] - ch.move_goal[h * 2 + 1]));
        bool stuck = cact[h] && moved < .5f && !sc.in_dash[h];
        ch.moving[h] = ch.moving[h] && !(to_goal <= MOVE_ARRIVE_RADIUS) && !(stuck && !chase[h]);
        ch.dash_until[h] = dash_until[h], ch.dash_target[h] = dash_target[h];
        if (sc.dstart[h]) ch.dash_to[h * 2] = dxs[h], ch.dash_to[h * 2 + 1] = dys[h], ch.dash_speed[h] = dash.speed[h];
    }
    std::memcpy(e.x, cx, n * sizeof(float));
    std::memcpy(e.y, cy, n * sizeof(float));
    for (int h = 0; h < c; ++h) {
        sc.ictx.x[h] = e.x[h], sc.ictx.y[h] = e.y[h];
        sc.ictx.facing_x[h] = ch.facing[h * 2], sc.ictx.facing_y[h] = ch.facing[h * 2 + 1];
    }
    sc.units = units_view(w, e);
    for (int h = 0; h < c; ++h) sc.units.attack_range[h] = sc.reach[h], sc.units.attack_speed[h] = sc.st.attack_speed[h];
}

// --- phase 6, ATTACK (world/phases/attack.py) ----------------------------------------------------------------------
struct AttackOut {
    Arr<uint8_t> launched, started, cancelled, reset;
    Attack attack;
    Packets direct, arrived, og_pk;
    int m_over = 0;
    Arr<int32_t> ward_hits, ward_hitter;
};

// lane.ai.overgrowth_packets: crystals consumed by enemy-champion basic-attack hits (towers in place).
Packets overgrowth_packets(const World& w, Env& e, const Arr<uint8_t>& hit_c, const Arr<int32_t>& target_c,
                           float now, const float team_level[2]) {
    const int n = w.n;
    Packets pk = empty_packets(n);
    for (int j = 0; j < n; ++j) {
        bool any = false;
        int attacker = 0;
        for (int h = 0; h < N_CHAMPIONS; ++h) {
            bool hit = hit_c[h] && target_c[h] == j && e.kind[h] == KIND_CHAMPION && e.team[h] != e.towers_team[j];
            if (hit && !any) any = true, attacker = h;
        }
        bool backdoor = now >= e.towers_turret_backdoor_until[j];
        bool proc = any && e.towers_is_structure[j] && e.towers_turret_tier[j] < tower::NEXUS_TURRET
                    && e.towers_turret_growth_active[j] && !backdoor && e.towers_turret_hp[j] > 0.f
                    && e.towers_targetable[j];
        float level = team_level[clampi(e.team[attacker], 0, 1)];
        float lo = (1.6f + .4f * level) / 100.f;
        float lf = std::min(std::max((level - 1.f) / 17.f, 0.f), 1.f);
        float hi = lo * (1.65f + (float)(2.15 - 1.65) * lf);
        float ramp = std::min(std::max((now - e.towers_turret_growth_since[j] - 150.f) / 240.f, 0.f), 1.f);
        float dmg = e.towers_turret_max_hp[j] * (lo + ramp * (hi - lo));
        if (proc) e.towers_turret_growth_active[j] = 0, e.towers_turret_growth_since[j] = now;
        pk.valid[j] = proc, pk.src[j] = attacker, pk.dst[j] = j, pk.raw[j] = proc ? dmg : 0.f, pk.dtype[j] = TRUE_DMG;
        pk.flags[j] = TAG_PROC | PROP_NO_DAMAGE_MOD | PROP_NO_OMNIVAMP;
    }
    return pk;
}

void phase_attack(const World& w, Env& e, const ChampData& cd, Layer& L, TS& sc, slice::Scratch& ss, AttackOut& ao,
                  TickStats& st_out) {
    const int c = N_CHAMPIONS, n = w.n, M = w.missiles;
    const float now = sc.now, dt = w.dt;
    const int w0 = w.ward0, nw = w.struct0 - w.ward0;
    ChampionLayer& ch = L.champ;
    // The units view of this phase has the champions' reach and attack speed (also what commit writes).
    for (int h = 0; h < c; ++h) e.attack_range[h] = sc.reach[h], e.attack_speed[h] = sc.st.attack_speed[h];
    Arr<uint8_t> can_attack(n);
    Arr<float> windup(n), period(n, 0.f);
    Arr<uint8_t> uncancel(n, 0);
    ao.reset.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        can_attack[i] = sc.caps.can_attack[i] && e.alive[i];
        windup[i] = e.windup[i];
    }
    for (int h = 0; h < c; ++h) {
        can_attack[h] = can_attack[h] && !sc.kmods.cannot_attack[h] && !sc.locked[h] && !sc.item_casting[h]
                        && !sc.tp_lock[h] && !sc.in_dash[h];
        float kw = sc.kmods.windup.size() ? sc.kmods.windup[h] : 0.f;
        windup[h] = kw > 0.f ? kw : sc.st.attack_windup[h];
        period[h] = sc.kmods.period.size() ? sc.kmods.period[h] : 0.f;
        uncancel[h] = sc.kmods.uncancellable.size() ? sc.kmods.uncancellable[h] : 0;
        ao.reset[h] = sc.kit_out.attack_reset[h] || sc.kmods.attack_reset[h] || ch.reset_next[h];
    }
    Arr<float> prev_windup = arr(e.att_windup_left, n);
    ao.launched.assign(n, 0);
    slice::attack_step(w, e, ss.desired.data(), can_attack.data(), ao.launched.data(), windup.data(), period.data(),
                       uncancel.data(), ao.reset.data());
    ao.started.resize(n), ao.cancelled.resize(n);
    for (int i = 0; i < n; ++i) {
        ao.started[i] = e.att_windup_left[i] > 0.f && prev_windup[i] <= 0.f;
        ao.cancelled[i] = prev_windup[i] > 0.f && e.att_windup_left[i] <= 0.f && !ao.launched[i];
    }
    // Champion crit roll at launch.
    Owned own;
    own.counts = api::I_owned_counts(ch.inventory), own.allowed = cd.allowed;
    Arr<int32_t> target_c(c);
    for (int h = 0; h < c; ++h) target_c[h] = e.att_target[h];
    AttackMods imods = items::attack_mods(L.combat.items, own, sc.ictx, item_units(w, e), target_c);
    Arr<uint8_t> crit(c);
    Arr<float> craw(c);
    Arr<int32_t> cdtype(c);
    Arr<uint8_t> ranged_c(c);
    std::vector<int32_t> cast_ids(n);
    for (int i = 0; i < n; ++i) cast_ids[i] = *e.tick * CAST_ID_STRIDE + i + 1;
    for (int h = 0; h < c; ++h) {
        bool no_crit = sc.kmods.cannot_crit.size() ? sc.kmods.cannot_crit[h] : false;
        bool roll = rng::uniform(sc.k_crit, (uint32_t)h) < sc.st.crit_chance[h];
        crit[h] = ao.launched[h] && !no_crit && (roll || imods.force_crit[h]);
        float crit_mult = 1.f + (sc.st.crit_damage[h] - 1.f) * (imods.force_crit[h] ? imods.crit_scale[h] : 1.f);
        int at = clampi(e.att_target[h], 0, n - 1);
        bool vs_struct = e.kind[at] == KIND_TURRET || e.kind[at] == KIND_INHIBITOR || e.kind[at] == KIND_NEXUS;
        float struct_raw = sc.st.base_ad[h] + sc.st.bonus_ad[h] + .6f * sc.st.ap[h];
        bool struct_magic = .6f * sc.st.ap[h] > sc.st.bonus_ad[h];
        float r = vs_struct ? struct_raw : (sc.st.base_ad[h] + sc.st.bonus_ad[h]) * (crit[h] ? crit_mult : 1.f);
        r = r * ((e.kind[at] == KIND_TURRET && e.missile_speed[h] <= 0.f) ? 1.2f : 1.f);
        craw[h] = r;
        cdtype[h] = (vs_struct && struct_magic) ? MAGIC : PHYSICAL;
        ranged_c[h] = e.missile_speed[h] > 0.f;
    }
    Arr<uint8_t> on_ward(n);
    for (int i = 0; i < n; ++i) on_ward[i] = e.kind[clampi(e.att_target[i], 0, n - 1)] == KIND_WARD && e.att_target[i] >= 0;
    Attack& attack = ao.attack;
    attack.launched.resize(c), attack.hit.resize(c), attack.target.resize(c), attack.raw.resize(c), attack.is_crit.resize(c);
    for (int h = 0; h < c; ++h) {
        attack.launched[h] = ao.launched[h];
        attack.hit[h] = ao.launched[h] && !ranged_c[h] && !on_ward[h];
        attack.target[h] = e.att_target[h];
        attack.raw[h] = ao.launched[h] ? craw[h] : 0.f;
        attack.is_crit[h] = crit[h];
    }
    // Minion Pushing and the lane attack packets (lane.ai.attack_packets), as in the slice.
    float alive_t[2][3] = {};
    for (int i = 0; i < n; ++i)
        if (e.kind[i] == KIND_TURRET && e.alive[i] && w.unit_lane[i] >= 0 && w.unit_lane[i] < 3 && e.team[i] >= 0 && e.team[i] < 2)
            alive_t[e.team[i]][w.unit_lane[i]] += 1.f;
    std::vector<float> push_div(n, 1.f), raw_all(n, 0.f);
    std::vector<int32_t> dtype_all(n, PHYSICAL), flags_all(n, BASIC_ATTACK);
    for (int i = 0; i < n; ++i)
        if (e.kind[i] == KIND_MINION && e.lane_ai_lane[i] >= 0) {
            int t = e.team[i] == 1 ? 1 : 0, l = clampi(e.lane_ai_lane[i], 0, 2);
            float bonus, div;
            minion::pushing((float)e.econ_level[t] - (float)e.econ_level[1 - t], alive_t[t][l] - alive_t[1 - t][l],
                            std::floor(now), &bonus, &div);
            push_div[i] = div;
        }
    for (int i = 0; i < n; ++i) {
        if (!ao.launched[i] || e.kind[i] == KIND_CHAMPION) continue;
        int tgt = e.att_target[i], t = clampi(tgt, 0, n - 1);
        int sub = clampi(e.sub[i], 0, 3), t_kind = e.kind[t], t_sub = clampi(e.sub[t], 0, 3);
        bool is_minion = e.kind[i] == KIND_MINION, is_turret = e.kind[i] == KIND_TURRET;
        bool t_minion = t_kind == KIND_MINION, t_champ = t_kind == KIND_CHAMPION;
        bool t_building = t_kind == KIND_INHIBITOR || t_kind == KIND_NEXUS;
        float m_raw = e.attack_damage[i] + (t_minion ? minion::SLAYER[sub] * e.hp[t] : 0.f);
        m_raw = m_raw * ((sub == minion::CANNON && t_kind == KIND_TURRET) ? 1.4f : 1.f);
        m_raw = m_raw * ((sub == minion::SUPER && t_building) ? (float)(.125 / .60) : 1.f);
        m_raw = m_raw / ((is_minion && t_minion) ? push_div[t] : 1.f);
        int stacks = now < e.lane_ai_warm_until[i] ? e.lane_ai_warm_stacks[i] : 0;
        float champ_raw = tower::attack_damage(sub, now) * tower::warming(stacks);
        float shot_raw = tower::shot_fraction(t_sub, sub) * e.max_hp[t];
        bool valid = tgt >= 0 && e.alive[i] && ((is_minion && t_kind != KIND_NONE) || (is_turret && (t_champ || t_minion)));
        float raw = is_turret ? (t_champ ? champ_raw : shot_raw) : m_raw;
        raw_all[i] = valid ? raw : 0.f;
        dtype_all[i] = is_turret ? (t_champ ? PHYSICAL : TRUE_DMG) : PHYSICAL;
    }
    for (int h = 0; h < c; ++h) {
        raw_all[h] = craw[h], dtype_all[h] = cdtype[h];
        flags_all[h] = BASIC_ATTACK | (crit[h] ? PROP_CRIT : 0);
    }
    // Missiles: launchers in unit order take free slots in slot order; then every slot advances.
    {
        int slot = 0;
        for (int i = 0; i < n; ++i) {
            int tgt = e.att_target[i];
            if (!(ao.launched[i] && e.missile_speed[i] > 0.f && !on_ward[i])) continue;
            while (slot < M && e.missiles_alive[slot]) ++slot;
            if (slot >= M) { ++ao.m_over; continue; }
            int t = clampi(tgt, 0, n - 1);
            e.missiles_alive[slot] = 1, e.missiles_src[slot] = i, e.missiles_dst[slot] = tgt;
            e.missiles_dst_seq[slot] = e.spawn_seq[t], e.missiles_x[slot] = e.x[i], e.missiles_y[slot] = e.y[i];
            e.missiles_speed[slot] = e.missile_speed[i], e.missiles_raw[slot] = raw_all[i];
            e.missiles_dtype[slot] = dtype_all[i], e.missiles_flags[slot] = flags_all[i];
            e.missiles_cast_id[slot] = cast_ids[i], e.missiles_crit[slot] = i < c ? crit[i] : 0;
            ++slot;
        }
        st_out.missile_overflow = ao.m_over;
        ao.arrived = empty_packets(M);
        for (int s = 0; s < M; ++s) {
            int t = clampi(e.missiles_dst[s], 0, n - 1);
            bool gone = !e.alive[t] || e.spawn_seq[t] != e.missiles_dst_seq[s];
            float dx = e.x[t] - e.missiles_x[s], dy = e.y[t] - e.missiles_y[s];
            float d = std::sqrt(dx * dx + dy * dy);
            float stp = e.missiles_speed[s] * dt;
            bool arrive = e.missiles_alive[s] && !gone && (d - e.radius[t] <= stp);
            float f = d > 0.f ? std::min(stp / std::max(d, 1e-6f), 1.f) : 1.f;
            e.missiles_x[s] = std::fma(dx, f, e.missiles_x[s]), e.missiles_y[s] = std::fma(dy, f, e.missiles_y[s]);
            e.missiles_alive[s] = e.missiles_alive[s] && !gone && !arrive;
            Packets& a = ao.arrived;
            a.valid[s] = arrive, a.src[s] = e.missiles_src[s], a.dst[s] = e.missiles_dst[s], a.raw[s] = e.missiles_raw[s];
            a.dtype[s] = e.missiles_dtype[s], a.flags[s] = e.missiles_flags[s], a.cast_id[s] = e.missiles_cast_id[s];
        }
    }
    ao.direct = empty_packets(n);
    for (int i = 0; i < n; ++i) {
        bool ranged = ao.launched[i] && e.missile_speed[i] > 0.f;
        Packets& p = ao.direct;
        p.valid[i] = ao.launched[i] && !ranged && e.att_target[i] >= 0 && !on_ward[i];
        p.src[i] = i, p.dst[i] = std::max(e.att_target[i], 0), p.raw[i] = raw_all[i], p.dtype[i] = dtype_all[i];
        p.flags[i] = flags_all[i], p.cast_id[i] = cast_ids[i];
    }
    ao.ward_hits.assign(nw, 0), ao.ward_hitter.assign(nw, -1);
    for (int i = 0; i < n; ++i) {
        bool hit_ward = ao.launched[i] && on_ward[i] && i < c;
        int ws = clampi(e.att_target[i] - w0, 0, nw - 1);
        ao.ward_hits[ws] += hit_ward;
        ao.ward_hitter[ws] = std::max(ao.ward_hitter[ws], hit_ward ? i : -1);
    }
    for (int s = 0; s < M; ++s)
        if (ao.arrived.valid[s] && ao.arrived.src[s] < c) attack.hit[clampi(ao.arrived.src[s], 0, c - 1)] = 1;
    AttackLaunch launch_c;
    launch_c.launched = attack.launched, launch_c.target = attack.target, launch_c.ranged = ranged_c;
    launch_c.is_crit = crit, launch_c.cast_id.resize(c);
    for (int h = 0; h < c; ++h) launch_c.cast_id[h] = cast_ids[h];
    KitOut k_att, k_hit;
    std::tie(L.kits, k_att) = api::K_on_attack(L.kits, sc.kctx, sc.units, launch_c);
    AttackLaunch launch_hit = launch_c;
    launch_hit.launched = attack.hit;
    std::tie(L.kits, k_hit) = api::K_on_hit(L.kits, sc.kctx, sc.units, launch_hit);
    float team_level[2];
    for (int t = 0; t < 2; ++t)
        team_level[t] = api::E_decimal_level(L.econ.xp[t], L.econ.quest.complete[t] ? 20 : 18);
    ao.og_pk = overgrowth_packets(w, e, attack.hit, attack.target, now, team_level);
    sc.kit_all = no_out(c, n);
    merge_out_into(sc.kit_all, sc.kit_out, c, n);
    merge_out_into(sc.kit_all, k_att, c, n);
    merge_out_into(sc.kit_all, k_hit, c, n);
}

// --- item actives the world calls directly (items/effects/actives.py) -----------------------------------------------
void with_aim(items_actives_State& s, const Arr<int32_t>& unit, const Arr<float>& x, const Arr<float>& y) {
    s.aim_unit = unit, s.aim_x = x, s.aim_y = y;
    s.aim_set.assign(unit.size(), 1);
}

Arr<int32_t> request_allowed(const Arr<int32_t>& req, const Arr<uint8_t>& disabled, const Arr<uint8_t>& in_stasis) {
    Arr<int32_t> out(req.size());
    for (size_t h = 0; h < req.size(); ++h) {
        bool qss = req[h] == 3140 || req[h] == 3139;            // Quicksilver Sash, Mercurial Scimitar
        bool ok = (!disabled[h] || qss) && !in_stasis[h];
        out[h] = ok ? req[h] : 0;
    }
    return out;
}

struct ActiveWorld {
    Arr<uint8_t> stasis, cleanse;
    Arr<float> stasis_until, mana_cost_mult, basic_cd_rate;
    Dash dash;
    Arr<int32_t> transform_from, transform_to;
    Arr<uint8_t> transform_do;
};

ActiveWorld actives_world(const items_actives_State& s, float now) {
    static const float mana_up = data::f("world.actualizer_mana"), cd_tick = data::f("world.actualizer_cd");
    const size_t c = s.stasis_until.size();
    ActiveWorld a;
    a.stasis.resize(c), a.cleanse = s.cleanse_now, a.stasis_until = s.stasis_until;
    a.mana_cost_mult.resize(c), a.basic_cd_rate.resize(c);
    a.dash.active = s.dash_now, a.dash.to_x = s.dash_x, a.dash.to_y = s.dash_y, a.dash.speed.assign(c, 1500.f);
    a.dash.target.assign(c, -1), a.dash.blink.assign(c, 0);
    a.transform_from.assign(c, item_row(2420)), a.transform_to.assign(c, item_row(2421)), a.transform_do = s.shatter_now;
    for (size_t h = 0; h < c; ++h) {
        bool on = now < s.actualizer_until[h];
        a.stasis[h] = now < s.stasis_until[h];
        a.mana_cost_mult[h] = on ? 1.f + mana_up : 1.f;
        a.basic_cd_rate[h] = on ? 1.f + cd_tick : 1.f;
    }
    return a;
}

// --- phase 7, DAMAGE (world/phases/damage.py) ----------------------------------------------------------------------
struct DamageOut {
    CombatTickOut out;
    KitDefense kdef;
    KitOut k_dmg;
    CCOut cc_now;
    CC cc_items;
    Arr<float> hp, max_hp;
    Shields shields;
    UnitStatus status;
};

void phase_damage(const World& w, Env& e, const ChampData& cd, Layer& L, const Orders& o, TS& sc, AttackOut& ao,
                  DamageOut& d) {
    const int c = N_CHAMPIONS, n = w.n;
    const float now = sc.now;
    // Base packets: direct, arrived, Overgrowth, kit, summoner (combat_tick prepends the carry).
    Packets base = ao.direct;
    append(base, ao.arrived), append(base, ao.og_pk), append(base, sc.kit_all.packets), append(base, sc.s_eff.packets);
    prof::Laps laps;
    d.kdef = api::K_defense(L.kits, sc.kctx);
    Debuffs kdeb = api::K_debuffs(L.kits, sc.kctx, sc.units);
    // lane.ai.turret_defense / structure_defense_mods
    Defense dfn;
    Arr<float> zero_n(n, 0.f), one_n(n, 1.f);
    Arr<uint8_t> false_n(n, 0);
    dfn.armor.resize(n), dfn.magic_resist.resize(n);
    dfn.flat_armor_reduction = zero_n, dfn.percent_armor_reduction = kdeb.percent_armor_reduction;
    dfn.flat_mr_reduction = zero_n, dfn.percent_mr_reduction = zero_n, dfn.received_mult = one_n;
    dfn.champion_received_mult = one_n, dfn.received_amp = zero_n, dfn.magic_received_amp = zero_n;
    dfn.basic_attack_mult = one_n, dfn.crit_taken_mult = one_n, dfn.champion_attack_block = zero_n;
    dfn.postmit_flat = zero_n, dfn.store_fraction = zero_n, dfn.invulnerable.resize(n), dfn.spell_shield = false_n;
    dfn.unit_class.resize(n), dfn.lifeline_ready = false_n, dfn.lifeline_magic_only = false_n;
    dfn.lifeline_shield = zero_n, dfn.lifeline_shield_kind.assign(n, SHIELD_ALL), dfn.lifeline_duration = zero_n;
    dfn.lifeline_decay_hold.assign(n, INF), dfn.lifeline_bonus_health = zero_n, dfn.dodge_basic = false_n;
    dfn.aoe_received_mult = one_n, dfn.received_mult_all.resize(n);
    Offense off;
    off.lethality = zero_n, off.percent_armor_pen.resize(n), off.magic_pen = zero_n, off.percent_magic_pen = zero_n;
    off.dealt_reduction = sc.s_out.exhaust_reduction, off.unit_class.resize(n), off.is_turret.resize(n);
    for (int j = 0; j < n; ++j) {
        bool s = e.towers_is_structure[j];
        int k = e.kind[j];
        float armor = e.armor[j], mr = e.magic_resist[j];
        if (s && k == KIND_TURRET) {
            int n850 = 0;
            for (int u : w.cols)
                if (e.kind[u] == KIND_CHAMPION && e.towers_team[j] != e.team[u] && e.alive[u]
                    && std::sqrt(sq(e.x[j] - e.x[u]) + sq(e.y[j] - e.y[u])) <= tower::BULWARK_RADIUS)
                    ++n850;
            int count = clampi(n850, 1, 5);
            float per_stack = 30.f + 5.f * (float)(count - 1);
            int stacks = 0;
            for (int q = 0; q < 4; ++q) stacks += e.towers_turret_bulwark_until[j * 4 + q] > now;
            armor = mr = 60.f - (e.towers_turret_tier[j] == tower::OUTER ? 15.f * tower::decay_steps(now) : 0.f)
                         + per_stack * (float)stacks;
        } else if (s && (k == KIND_INHIBITOR || k == KIND_NEXUS)) {
            armor = tower::BUILDING_ARMOR, mr = tower::BUILDING_MR;
        }
        dfn.armor[j] = armor, dfn.magic_resist[j] = mr;
        bool turret_alive = s && k == KIND_TURRET && e.towers_turret_hp[j] > 0.f;
        dfn.received_mult_all[j] = (turret_alive && now >= e.towers_turret_backdoor_until[j]) ? .2f : 1.f;
        dfn.invulnerable[j] = s && !e.towers_targetable[j];
        dfn.unit_class[j] = damage_class(k), off.unit_class[j] = damage_class(k);
        off.percent_armor_pen[j] = k == KIND_TURRET ? tower::ARMOR_PEN : 0.f;
        off.is_turret[j] = k == KIND_TURRET;
    }
    for (int h = 0; h < c; ++h) {
        dfn.armor[h] = sc.st_static.base_armor[h] + sc.st_static.bonus_armor[h];
        dfn.magic_resist[h] = sc.st_static.base_mr[h] + sc.st_static.bonus_mr[h];
        dfn.received_mult[h] = d.kdef.received_mult[h];
        dfn.invulnerable[h] = dfn.invulnerable[h] || sc.s_out.teleport_dash[h] || sc.in_stasis[h];
        dfn.dodge_basic[h] = d.kdef.dodge_basic[h], dfn.aoe_received_mult[h] = d.kdef.aoe_received_mult[h];
    }
    LS_LAP(laps, "damage.0_base_defense");
    d.cc_now = sc.kit_all.cc;
    d.cc_items.slowed.resize((size_t)c * n), d.cc_items.immobilized.resize((size_t)c * n);
    for (size_t k = 0; k < (size_t)c * n; ++k) {
        d.cc_items.slowed[k] = d.cc_now.slow[k] > 0.f;
        d.cc_items.immobilized[k] = d.cc_now.stun[k] > 0.f || d.cc_now.root[k] > 0.f || d.cc_now.knockup[k] > 0.f;
    }
    LS_LAP(laps, "damage.1_cc");
    // rune_events(ictx, n, ...)
    RuneEvents ev;
    const Ctx& ictx = sc.ictx;
    Arr<float> zc(c, 0.f);
    Arr<uint8_t> fc(c, 0);
    Arr<int32_t> ic(c, 0);
    ev.game_time = now;
    ev.attack.launched = fc, ev.attack.hit = fc, ev.attack.target = ic, ev.attack.raw = zc, ev.attack.is_crit = fc;
    ev.attack_started.resize(c), ev.attack_start_target.resize(c), ev.attack_cancelled.resize(c), ev.attack_reset.resize(c);
    for (int h = 0; h < c; ++h) {
        ev.attack_started[h] = ao.started[h], ev.attack_start_target[h] = e.att_target[h];
        ev.attack_cancelled[h] = ao.cancelled[h], ev.attack_reset[h] = ao.reset[h];
    }
    ev.cast.started = fc, ev.cast.slot = ic, ev.cast.target.assign(c, -1);
    ev.cast_id = sc.kit_all.cast_id;
    ev.cc.slowed.assign((size_t)c * n, 0), ev.cc.immobilized.assign((size_t)c * n, 0);
    ev.cc_duration.resize((size_t)c * n), ev.impaired_by_holder.resize((size_t)c * n);
    for (size_t k = 0; k < (size_t)c * n; ++k) {
        ev.cc_duration[k] = std::max(d.cc_now.stun[k], d.cc_now.root[k]);
        ev.impaired_by_holder[k] = d.cc_now.slow[k] > 0.f || d.cc_now.stun[k] > 0.f || d.cc_now.root[k] > 0.f;
    }
    ev.impaired = sc.caps.impaired, ev.movement_impaired = sc.caps.movement_impaired;
    ev.holder_cc_from_champion.resize(c), ev.blinked.resize(c), ev.in_river.resize(c);
    for (int h = 0; h < c; ++h) {
        ev.holder_cc_from_champion[h] = e.cc_champion_cc_until[h] > now;
        ev.blinked[h] = sc.s_out.blinked[h] || sc.dstart[h];
        ev.in_river[h] = api::REG_in_river(e.x[h], e.y[h]);
    }
    ev.summoner_cast = sc.s_out.cast_event, ev.summoner_cooldown = sc.s_out.cast_cooldown;
    ev.summoner_is_teleport = sc.s_out.is_teleport, ev.flash_cooldown = api::S_flash_cooldown(L.summ, now);
    ev.hexflash_request = ic;
    ev.kills.champion_kill = zc, ev.kills.champion_assist = zc, ev.kills.minion_kill = zc, ev.kills.holder_died = fc;
    ev.kills.killed_units.assign((size_t)c * n, 0);
    ev.deaths.assign(n, 0);
    for (int h = 0; h < c; ++h)
        for (int j = 0; j < n; ++j) ev.deaths[j] = ev.deaths[j] | L.prev_death_seen[(size_t)h * n + j];
    ev.sight.resize((size_t)c * n);
    for (size_t k = 0; k < (size_t)c * n; ++k) ev.sight[k] = L.sight[k] | L.prev_death_seen[k];
    ev.visible = sc.vis_c;
    ev.large_monster_kill = L.prev_large, ev.epic_takedown = L.prev_epic, ev.execute_credit = zc;
    ev.shield_gained = zc, ev.shield_gained_duration = zc, ev.summoner_haste = zc;
    ev.cc_cast_id = d.cc_now.cast_id, ev.cc_on_hit.assign((size_t)c * n, 0);
    ev.purchased = sc.bought, ev.sold = sc.sold, ev.potion_drunk = ic, ev.granted = L.champ.granted;
    ev.spellbook_request = ic;
    ev.is_turret.resize(n);
    for (int j = 0; j < n; ++j) ev.is_turret[j] = e.kind[j] == KIND_TURRET;
    ev.uses_energy = cd.uses_energy, ev.adaptive_physical = cd.adaptive_physical;
    ev.bonus_ad = ictx.bonus_ad, ev.ap = ictx.ap, ev.bonus_attack_speed = ictx.bonus_attack_speed;
    ev.clocks.last_combat.assign(c, -BIG), ev.clocks.last_champion_combat.assign(c, -BIG);
    ev.clocks.last_hit_by_champion.assign(c, -BIG), ev.clocks.champion_combat_start.assign(c, -BIG);
    ev.clocks.struck_first.assign(c, 0), ev.clocks.last_combat_modern.assign(c, -BIG);
    LS_LAP(laps, "damage.2_rune_events");
    // Item actives: this tick's aim, then the request (only cleanses while disabled, nothing in stasis).
    CombatState combat = L.combat;
    with_aim(combat.items.actives, arr(o.cast_target, c), arr(o.cast_x, c), arr(o.cast_y, c));
    Arr<uint8_t> stunned_c(c);
    for (int h = 0; h < c; ++h) stunned_c[h] = sc.caps.stunned[h];
    Arr<int32_t> item_req = request_allowed(arr(o.item_active, c), stunned_c, sc.in_stasis);
    Owned own;
    own.counts = api::I_owned_counts(L.champ.inventory), own.allowed = cd.allowed;
    Cast cast;
    cast.started = sc.kit_all.cast_started, cast.slot = sc.kit_all.cast_slot, cast.target = sc.cast_order.target;
    Shields shields = L.shields;
    UnitStatus status = L.status;
    LS_LAP(laps, "damage.3_prep");
    d.out = combat_tick(std::move(combat), std::move(own), cd.pages, ictx, item_units(w, e), ao.attack, std::move(cast),
                        std::move(item_req), std::move(base), std::move(off), std::move(dfn), arr(e.hp, n),
                        arr(e.max_hp, n), std::move(shields), std::move(status), L.prev_kills, sc.stat, d.cc_items,
                        std::move(ev), w.packet_capacity, w.packet_capacity / 2);
    LS_LAP(laps, "damage.4_combat_tick");
    d.hp = d.out.hp, d.max_hp = d.out.max_hp, d.shields = d.out.shields, d.status = d.out.status;
    std::tie(L.kits, d.k_dmg) = api::K_on_damage(L.kits, sc.kctx, sc.units, d.out.report);
    LS_LAP(laps, "damage.5_kits_on_damage");
}

// --- phase 8, CC / HEAL (world/phases/cc_heal.py) ------------------------------------------------------------------
// mechanics.apply_cc: this tick's CC (rows x N), tenacity now (not on knock-ups); the strongest slow wins.
void apply_cc(Env& e, int n, const CCOut& out, int rows, const Arr<float>& ten, const Arr<float>& sres, float now,
              const Arr<uint8_t>& source_is_champion, const Arr<uint8_t>* cleansed) {
    for (int j = 0; j < n; ++j) {
        float stun = 0.f, root = 0.f, sil = 0.f, up = 0.f, strength = 0.f;
        int k = 0;
        bool champ = false;
        for (int r = 0; r < rows; ++r) {
            size_t q = (size_t)r * n + j;
            if (out.stun[q] > 0.f) stun = std::max(stun, stats::cc_duration(out.stun[q], ten[j]));
            if (out.root[q] > 0.f) root = std::max(root, stats::cc_duration(out.root[q], ten[j]));
            if (out.silence[q] > 0.f) sil = std::max(sil, stats::cc_duration(out.silence[q], ten[j]));
            up = std::max(up, out.knockup[q]);
            if (out.slow[q] > strength) strength = out.slow[q], k = r;
            champ = champ || (source_is_champion[r] && (out.stun[q] > 0.f || out.root[q] > 0.f || out.silence[q] > 0.f
                                                         || out.knockup[q] > 0.f || out.slow[q] > 0.f));
        }
        float sdur = strength > 0.f ? stats::cc_duration(out.slow_duration[(size_t)k * n + j], ten[j]) : 0.f;
        float active = e.cc_slow_until[j] > now ? e.cc_slow[j] : 0.f;
        bool take = strength > 0.f && strength >= active;
        float longest = std::max(std::max(stun, root), std::max(std::max(sil, up), sdur));
        e.cc_stun_until[j] = std::max(e.cc_stun_until[j], stun > 0.f ? now + stun : 0.f);
        e.cc_root_until[j] = std::max(e.cc_root_until[j], root > 0.f ? now + root : 0.f);
        e.cc_silence_until[j] = std::max(e.cc_silence_until[j], sil > 0.f ? now + sil : 0.f);
        e.cc_knockup_until[j] = std::max(e.cc_knockup_until[j], up > 0.f ? now + up : 0.f);
        float until = take ? std::max(now + sdur, strength == active ? e.cc_slow_until[j] : 0.f) : e.cc_slow_until[j];
        e.cc_slow[j] = take ? strength : e.cc_slow[j];
        e.cc_slow_until[j] = until;
        e.cc_champion_cc_until[j] = std::max(e.cc_champion_cc_until[j], champ ? now + longest : 0.f);
        if (cleansed && (*cleansed)[j]) {
            e.cc_stun_until[j] = std::min(e.cc_stun_until[j], now), e.cc_root_until[j] = std::min(e.cc_root_until[j], now);
            e.cc_silence_until[j] = std::min(e.cc_silence_until[j], now);
            e.cc_slow_until[j] = std::min(e.cc_slow_until[j], now);
        }
    }
}

ActiveWorld phase_cc_heal(const World& w, Env& e, Layer& L, TS& sc, DamageOut& d) {
    const int c = N_CHAMPIONS, n = w.n;
    const float now = sc.now;
    ShieldGrant kit_shields = sc.kit_all.shield;
    concat_shields(kit_shields, d.k_dmg.shield, c);
    Effects s_eff = sc.s_eff;
    s_eff.packets = empty_packets(0);
    Effects kit_eff = no_effects(c, n);
    for (int h = 0; h < c; ++h) kit_eff.heal[h] = sc.kit_all.heal[h] + d.k_dmg.heal[h];
    kit_eff.shields = kit_shields;
    Effects mon = no_effects(c, n);
    mon.shields = shield_grants(Arr<float>(c, 0.f), SHIELD_ALL, INF);
    Effects extra = no_effects(c, n);
    merge_into(extra, s_eff), merge_into(extra, kit_eff), merge_into(extra, mon);
    apply_effects(extra, sc.ictx, d.hp, d.max_hp, d.shields, d.status, sc.st.heal_shield_power,
                  sc.summ_world.incoming_heal, nullptr, nullptr);
    // Champion slows without a (C, N) source row: Exhaust, item/rune effects.
    CCOut slow_cc = no_cc(3, n);
    for (int j = 0; j < n; ++j) {
        slow_cc.slow[j] = sc.s_out.exhaust_slow[j], slow_cc.slow_duration[j] = sc.s_out.exhaust_slow_duration[j];
        slow_cc.slow[n + j] = d.out.effects.slow[j], slow_cc.slow_duration[n + j] = d.out.effects.slow_duration[j];
        slow_cc.slow[2 * n + j] = extra.slow[j], slow_cc.slow_duration[2 * n + j] = extra.slow_duration[j];
    }
    Arr<float> ten(n, 0.f), sres(n, 0.f);
    for (int h = 0; h < c; ++h) {
        ten[h] = 1.f - (1.f - sc.st.tenacity[h]) * (1.f - d.kdef.tenacity_bonus[h]) * (1.f - sc.s_out.tenacity[h]);
        sres[h] = sc.st.slow_resist[h];
    }
    ActiveWorld aw = actives_world(d.out.state.items.actives, now);
    Arr<uint8_t> cleansed(n, 0);
    for (int h = 0; h < c; ++h) cleansed[h] = sc.s_out.cleanse[h] || aw.cleanse[h];
    apply_cc(e, n, d.cc_now, c, ten, sres, now, Arr<uint8_t>(c, 1), &cleansed);
    apply_cc(e, n, slow_cc, 3, ten, sres, now, Arr<uint8_t>(3, 1), nullptr);
    for (int h = 0; h < c; ++h)
        if (sc.kit_all.cleanse_slow[h]) e.cc_slow_until[h] = now;
    sc.extra = extra;   // the merged heal/mana effects (TIMERS reads ``extra.mana``)
    return aw;
}

// --- phase 9, DEATH (world/phases/death.py) ----------------------------------------------------------------------
struct DeathOut {
    Arr<uint8_t> died, minion_died, took_health, in_f;
    EconomyOut eco;
};

void phase_death(const World& w, Env& e, const ChampData& cd, Layer& L, const Orders& o, TS& sc, DamageOut& d,
                 DeathOut& dd) {
    const int c = N_CHAMPIONS, n = w.n;
    const float now = sc.now;
    Packets rp = d.out.report.packets;
    append(rp, d.out.follow_up.packets);
    Arr<uint8_t> rr_killed = d.out.report.resolved.killed;
    rr_killed.append(d.out.follow_up.resolved.killed);
    Arr<float> rr_loss = d.out.report.resolved.health_loss;
    rr_loss.append(d.out.follow_up.resolved.health_loss);
    Arr<int32_t> killer(n, -1);
    dd.took_health.assign(n, 0);
    // The damage matrix for next tick's AI, as the native event list plus the dense table.
    for (int q = 0; q < *e.ev_n; ++q) e.prev_damage_matrix[(size_t)e.ev_src[q] * n + e.ev_dst[q]] = 0;
    *e.ev_n = 0;
    for (size_t i = 0; i < size(rp); ++i) {
        if (!rp.valid[i]) continue;
        int s = clampi(rp.src[i], 0, n - 1), t = clampi(rp.dst[i], 0, n - 1);
        if (rr_killed[i]) killer[t] = std::max(killer[t], rp.src[i]);
        uint8_t& m = e.prev_damage_matrix[(size_t)s * n + t];
        if (!m) m = 1, e.ev_src[*e.ev_n] = s, e.ev_dst[*e.ev_n] = t, ++*e.ev_n;
        if (rr_loss[i] > 0.f) dd.took_health[t] = 1;
    }
    dd.died.resize(n), dd.minion_died.resize(n);
    for (int j = 0; j < n; ++j) {
        dd.died[j] = e.alive[j] && d.hp[j] <= 0.f;
        dd.minion_died[j] = dd.died[j] && e.kind[j] == KIND_MINION;
    }
    for (int h = 0; h < c; ++h)
        for (int j = 0; j < n; ++j) L.prev_death_seen[(size_t)h * n + j] = dd.died[j] && L.sight[(size_t)h * n + j];
    // Structure plates and kills (lane.ai.structure_damage_events), with the reward events the economy reads.
    Arr<float> before = arr(e.hp, n), after(n);
    Arr<int32_t> plates0 = arr(e.towers_turret_plates, n);
    Arr<uint8_t> first0(1, *e.towers_first_turret_taken);
    for (int j = 0; j < n; ++j) {
        bool s = e.kind[j] == KIND_TURRET || e.kind[j] == KIND_INHIBITOR || e.kind[j] == KIND_NEXUS;
        after[j] = s ? d.hp[j] : e.hp[j];
    }
    slice::structure_damage_events(w, e, before.data(), after.data(), now);
    MinionDeaths md;
    md.valid = dd.minion_died, md.x = arr(e.x, n), md.y = arr(e.y, n), md.team = arr(e.team, n);
    md.gold = arr(e.bounty_gold, n), md.xp = arr(e.bounty_xp, n), md.level = arr(e.bounty_level, n);
    md.last_hitter.resize(n), md.unit.resize(n);
    for (int j = 0; j < n; ++j) md.last_hitter[j] = killer[j] < c ? killer[j] : -1, md.unit[j] = j;
    StructureEvents sev;
    sev.valid.resize(n), sev.unit.resize(n), sev.x = md.x, sev.y = md.y, sev.team = md.team;
    sev.local_gold.resize(n), sev.global_gold.resize(n), sev.is_turret.resize(n), sev.in_top_lane.resize(n);
    sev.is_structure.resize(n);
    static const float TURRET_GLOBAL_GOLD[4] = {50.f, 25.f, 25.f, 50.f};
    bool first_done = first0[0];
    for (int j = 0; j < n; ++j) {
        bool s = e.towers_is_structure[j];
        int tier = e.towers_turret_tier[j];
        int gained = e.towers_turret_plates[j] - plates0[j];
        bool destroyed = s && before[j] > 0.f && std::max(after[j], 0.f) <= 0.f;
        bool turret_kill = destroyed && tier <= tower::NEXUS_TURRET;
        float plate_value = tier >= tower::NEXUS_TURRET ? 0.f : 120.f - (tier == tower::OUTER ? 10.f * tower::decay_steps(now) : 0.f);
        bool first = turret_kill && !first_done;               // the first turret kill in unit order
        if (turret_kill) first_done = true;
        sev.valid[j] = gained > 0 || destroyed;
        sev.unit[j] = j;
        sev.local_gold[j] = (float)gained * plate_value + (first && !first0[0] ? 300.f : 0.f);
        sev.global_gold[j] = turret_kill ? TURRET_GLOBAL_GOLD[clampi(tier, 0, 3)] : 0.f;
        sev.is_turret[j] = destroyed && e.kind[j] == KIND_TURRET;
        sev.in_top_lane[j] = w.unit_lane[j] == 2;
        sev.is_structure[j] = e.kind[j] == KIND_TURRET || e.kind[j] == KIND_INHIBITOR || e.kind[j] == KIND_NEXUS;
    }
    dd.in_f.resize(c);
    for (int h = 0; h < c; ++h)
        dd.in_f[h] = api::E_in_fountain(e.x[h], e.y[h], cd.fountain[e.team[h] * 2], cd.fountain[e.team[h] * 2 + 1]);
    EconomyInputs in;
    Arr<float> zc(c, 0.f);
    in.now = now;
    in.unit.resize(c), in.x.resize(c), in.y.resize(c), in.team.resize(c), in.hp.resize(c), in.max_hp.resize(c);
    in.final_blow.resize(c), in.in_quest_lane.resize(c), in.recall_request.resize(c), in.cancel_action.resize(c);
    in.health_damage.resize(c), in.disabled.resize(c);
    for (int h = 0; h < c; ++h) {
        in.unit[h] = h, in.x[h] = e.x[h], in.y[h] = e.y[h], in.team[h] = e.team[h], in.hp[h] = d.hp[h];
        in.max_hp[h] = d.max_hp[h], in.final_blow[h] = killer[h] < c ? killer[h] : -1;
        in.in_quest_lane[h] = api::REG_in_quest_lane(e.x[h], e.y[h], 2);
        in.recall_request[h] = o.recall[h];
        in.cancel_action[h] = o.move[h] || o.attack[h] >= 0 || o.cast_slot[h] >= 0 || o.summoner_slot[h] >= 0;
        in.health_damage[h] = dd.took_health[h];
        in.disabled[h] = sc.caps.stunned[h] || sc.caps.silenced[h] || e.cc_root_until[h] > now;
    }
    in.report.packets = rp;
    in.cc = d.cc_items;
    in.minion_deaths = md;
    in.minion_in_lane.resize(n);
    for (int j = 0; j < n; ++j) in.minion_in_lane[j] = e.kind[j] == KIND_MINION && e.lane_ai_lane[j] == 2;
    in.structures = sev;
    in.last_champion_combat = d.out.state.clocks.last_champion_combat;
    in.in_fountain = dd.in_f;
    WorldUnits units = units_view(w, e);
    api::REG_homeguard_flags(in.x, in.y, in.team, now, units, arr(w.unit_lane.data(), n), arr(e.lane_ai_lane, n),
                             in.reached_endpoint, in.in_jungle);
    in.teleported = sc.s_out.teleport_arrive;
    in.extra_gold = zc, in.extra_xp = zc, in.epic = zc;
    dd.eco = api::E_economy_step(L.econ, in);
    L.econ = dd.eco.state;
    for (int h = 0; h < c; ++h) {
        float extra = d.out.effects.gold[h] + sc.s_eff.gold[h];
        L.econ.gold[h] = std::min(L.econ.gold[h] + extra, 100000.f);
        L.econ.gold_total[h] = L.econ.gold_total[h] + extra;
    }
    L.kits = api::K_on_takedown(L.kits, sc.kctx, sc.units, dd.eco.kills);
}

// --- phase 10, TIMERS (world/phases/timers.py) --------------------------------------------------------------------
void phase_timers(const World& w, Env& e, const ChampData& cd, Layer& L, const Orders& o, TS& sc, AttackOut& ao,
                  DamageOut& d, DeathOut& dd, const ActiveWorld& aw, const Arr<uint8_t>& vis0) {
    const int c = N_CHAMPIONS, n = w.n, ni = n_items();
    const float now = sc.now, dt = w.dt, t0 = *e.t;
    ChampionLayer& ch = L.champ;
    const RuneOutputs& ro = d.out.rune_outputs;
    for (int j = 0; j < n; ++j) e.alive[j] = e.alive[j] && !dd.died[j];
    const Arr<uint8_t>& respawn = dd.eco.respawned;
    const Arr<uint8_t>& recall = dd.eco.recalled;
    for (int h = 0; h < c; ++h) {
        if (respawn[h] || recall[h]) e.x[h] = cd.fountain[e.team[h] * 2], e.y[h] = cd.fountain[e.team[h] * 2 + 1];
        if (respawn[h]) e.alive[h] = 1, d.hp[h] = d.max_hp[h];
        // Cooldowns: decrement, kit starts (hasted), rune refunds.
        for (int s = 0; s < 4; ++s) {
            float rate = s < 3 ? aw.basic_cd_rate[h] : 1.f;
            float haste = s < 3 ? sc.st.basic_ability_haste[h] : sc.st.ultimate_haste[h];
            float cdv = std::max(ch.cooldowns[h * 4 + s] - dt * rate, 0.f);
            if (sc.kit_all.cooldown_start[h * 4 + s]) cdv = stats::cooldown(sc.kit_all.base_cooldown[h * 4 + s], haste);
            cdv = cdv * (s < 3 ? 1.f - ro.basic_cd_refund[h] : 1.f - ro.ult_cd_refund[h]);
            ch.cooldowns[h * 4 + s] = cdv;
        }
        float mana = std::min(std::max(ch.mana[h] - sc.kit_all.mana_cost[h] * aw.mana_cost_mult[h] + d.out.effects.mana[h]
                                           + sc.extra.mana[h] + sc.st.mana_regen[h] * dt, 0.f), sc.st.max_mana[h]);
        mana = respawn[h] ? sc.st.max_mana[h] : mana;
        // HP regen in 0.5 s pulses, then the fountain and Homeguard heal.
        float pulse = std::floor(now / .5f + 1e-6f) - std::floor(t0 / .5f + 1e-6f);
        float regen = (e.alive[h] && !respawn[h]) ? sc.st.hp_regen[h] * .5f * pulse : 0.f;
        float hp_c = std::min(d.hp[h] + regen, d.max_hp[h]);
        api::E_fountain_regen(hp_c, d.max_hp[h], mana, sc.st.max_mana[h], dd.in_f[h] && e.alive[h], t0, now,
                              L.econ.homeguard.active[h]);
        d.hp[h] = hp_c, ch.mana[h] = mana;
    }
    // Inventory: Tear-line transforms, Seeker's -> Shattered, consumed items, Control Wards, rune grants.
    for (int h = 0; h < c; ++h) {
        Inventory inv;
        inv.item.resize(7), inv.stack.resize(7);
        for (int s = 0; s < 7; ++s) inv.item[s] = ch.inventory.item[h * 7 + s], inv.stack[s] = ch.inventory.stack[h * 7 + s];
        api::I_replace_item(inv, d.out.transform_from[h], d.out.transform_to[h], d.out.transform_do[h]);
        api::I_replace_item(inv, aw.transform_from[h], aw.transform_to[h], aw.transform_do[h]);
        auto consume = [&](int r, bool ok) {
            int slot = 0;
            bool any = false;
            for (int s = 0; s < 7; ++s)
                if (inv.item[s] == r) { slot = s, any = true; break; }
            api::I_consume_one(inv, slot, ok && any);
        };
        consume(d.out.consume_row[h], d.out.consume_row[h] >= 0);
        for (int s = 0; s < 7; ++s) ch.inventory.item[h * 7 + s] = inv.item[s], ch.inventory.stack[h * 7 + s] = inv.stack[s];
    }
    // Wards and trinkets.
    const int w0 = w.ward0, wn = w.struct0 - w.ward0;
    static const int cw_row = item_row(2055);
    const auto& ids = data::table("catalog.ids");
    Arr<int32_t> trinket_id(c), control_count(c, 0);
    Arr<uint8_t> can_use(c);
    Arr<float> trinket_haste(c);
    for (int h = 0; h < c; ++h) {
        int trow = ch.inventory.item[h * 7 + 6];
        trinket_id[h] = trow >= 0 ? (int)ids[clampi(trow, 0, ni - 1)] : 0;
        for (int s = 0; s < 7; ++s)
            if (ch.inventory.item[h * 7 + s] == cw_row) control_count[h] += ch.inventory.stack[h * 7 + s];
        can_use[h] = e.alive[h] && !sc.caps.stunned[h] && !sc.in_stasis[h];
        trinket_haste[h] = sc.st.item_haste[h] + sc.st.trinket_haste[h];
    }
    WardRequest wreq;
    wreq.kind = arr(o.ward_kind, c), wreq.x = arr(o.ward_x, c), wreq.y = arr(o.ward_y, c);
    Arr<uint8_t> ward_visible((size_t)2 * wn);
    for (int t = 0; t < 2; ++t)
        for (int s = 0; s < wn; ++s) ward_visible[(size_t)t * wn + s] = vis0[(size_t)t * n + w0 + s];
    WardEvents wev;
    std::tie(L.wards, wev) = api::W_ward_step(L.wards, now, dt, wreq, arr(e.x, c), arr(e.y, c), arr(e.team, c),
                                              arr(e.alive, c), L.econ.level, trinket_id, control_count, can_use,
                                              trinket_haste, ao.ward_hits, ao.ward_hitter, cd.pages, ward_visible);
    for (int h = 0; h < c; ++h) {
        L.econ.gold[h] = L.econ.gold[h] + wev.gold[h], L.econ.gold_total[h] = L.econ.gold_total[h] + wev.gold[h];
        L.econ.xp[h] = L.econ.xp[h] + wev.xp[h];
        Inventory inv;
        inv.item.resize(7), inv.stack.resize(7);
        for (int s = 0; s < 7; ++s) inv.item[s] = ch.inventory.item[h * 7 + s], inv.stack[s] = ch.inventory.stack[h * 7 + s];
        int slot = 0;
        bool any = false;
        for (int s = 0; s < 7; ++s)
            if (inv.item[s] == cw_row) { slot = s, any = true; break; }
        api::I_consume_one(inv, slot, wev.consumed_control[h] && any);
        // Rune grants into the first free slot of 0..5.
        int grant_row = 0;
        for (int r = 0; r < ni; ++r)
            if ((int)ids[r] == ro.grant_item[h]) { grant_row = r; break; }
        int gslot = -1;
        for (int s = 0; s < 6 && gslot < 0; ++s)
            if (inv.item[s] < 0) gslot = s;
        bool can_grant = ro.grant_item[h] > 0 && gslot >= 0;
        if (can_grant) inv.item[gslot] = grant_row, inv.stack[gslot] = 1;
        for (int s = 0; s < 7; ++s) ch.inventory.item[h * 7 + s] = inv.item[s], ch.inventory.stack[h * 7 + s] = inv.stack[s];
        ch.granted[h] = can_grant ? ro.grant_item[h] : 0;
        // Locks and the champion layer's per-tick fields.
        float lock = std::max(ch.cast_lock_until[h], sc.kit_all.cast_started[h] ? now + sc.kit_all.cast_lockout[h] : 0.f);
        const ActiveOut& act = d.out.active;
        lock = std::max(lock, (act.used[h] && !act.can_move[h]) ? now + act.cast_time[h] : 0.f);
        float item_lock = std::max(ch.item_cast_until[h], (act.used[h] && act.can_move[h]) ? now + act.cast_time[h] : 0.f);
        ch.cast_lock_until[h] = lock, ch.item_cast_until[h] = item_lock;
        if (dd.took_health[h]) ch.last_damaged[h] = now;
        ch.homeguard_ms[h] = dd.eco.homeguard_ms[h];
        ch.blinked[h] = sc.s_out.blinked[h] || sc.dstart[h];
        ch.reset_next[h] = d.out.effects.attack_reset[h];
        ch.bonus_points[h] = ch.bonus_points[h] + ro.skill_points[h];
        ch.cs[h] = ch.cs[h] + (int32_t)dd.eco.kills.minion_kill[h];
        for (int s = 0; s < 4; ++s)
            if (sc.kit_all.cast_started[h] && sc.kit_all.cast_slot[h] == s) ch.last_cast[h * 4 + s] = now;
        bool reset_orders = respawn[h] || recall[h] || !e.alive[h];
        int kat = sc.kit_all.attack_target.size() ? sc.kit_all.attack_target[h] : -1;
        ch.moving[h] = reset_orders ? 0 : ch.moving[h];
        ch.attack_order[h] = reset_orders ? -1 : (kat >= 0 ? kat : ch.attack_order[h]);
    }
    ch.dyn = d.out.dynamic_stats;
    ch.forbid = ro.forbid_purchase;
    for (int j = 0; j < n; ++j)
        if (dd.minion_died[j]) e.kind[j] = KIND_NONE;                    // dead minions free their slot
    WardView view;
    Arr<float> oracle;
    std::tie(view, oracle) = api::W_ward_view(L.wards, now, arr(e.x, c), arr(e.y, c), arr(e.team, c), arr(e.alive, c),
                                              L.econ.level);
    for (int s = 0; s < wn; ++s) {
        int j = w0 + s;
        e.kind[j] = view.alive[s] ? KIND_WARD : KIND_NONE, e.alive[j] = view.alive[s];
        e.x[j] = view.x[s], e.y[j] = view.y[s], e.hp[j] = view.hp[s], e.max_hp[j] = view.max_hp[s];
        d.hp[j] = view.hp[s], d.max_hp[j] = view.max_hp[s];
        e.sub[j] = view.sub[s], e.team[j] = view.team[s], e.spawn_seq[j] = (1 << 24) + L.wards.slots.seq[s];
        e.radius[j] = 1.f, e.targetable[j] = view.alive[s];
    }
    // Units inside terrain that closed step out (dynamic_terrain.eject): the lane world's terrain is static, so a
    // mobile unit is only ever moved by this when the movement clamp let it in.
    int budget = 16;
    for (int i = 0; i < n && budget > 0; ++i) {
        if (!(e.alive[i] && (e.kind[i] == KIND_CHAMPION || e.kind[i] == KIND_MINION))) continue;
        int team = e.team[i] == 1 ? 1 : 0;
        float r = std::min(std::min(e.radius[i], w.routes.radius), 150.f);
        if (w.terrain[team].walkable(e.x[i], e.y[i], r, 3)) continue;
        --budget;
        int best = -1;
        float best_r = INF;
        for (size_t k = 0; k < w.eject_r.size(); ++k)
            if (w.eject_r[k] < best_r && w.terrain[team].walkable(e.x[i] + w.eject_dx[k], e.y[i] + w.eject_dy[k], r, 3))
                best_r = w.eject_r[k], best = (int)k;
        if (best >= 0) e.x[i] = e.x[i] + w.eject_dx[best], e.y[i] = e.y[i] + w.eject_dy[best];
    }
}

// --- phase 11, FOG (world/phases/fog.py, vision.visibility with ward inputs) ---------------------------------------
void phase_fog(const World& w, Env& e, Layer& L, TS& sc, AttackOut& ao, const Arr<uint8_t>& vis0,
               const Arr<uint8_t>& alive0, TickStats& st) {
    const int c = N_CHAMPIONS, n = w.n, nf = w.struct0, w0 = w.ward0, wn = w.struct0 - w.ward0;
    const float now = sc.now;
    Arr<uint8_t> hidden(c);
    for (int h = 0; h < c; ++h) {
        hidden[h] = !vis0[(size_t)(1 - (e.team[h] == 1 ? 1 : 0)) * n + h] && alive0[h];
        bool struck = ao.launched[h] || (sc.kit_all.cast_started[h] && sc.cast_order.target[h] >= 0);
        if (struck && hidden[h]) L.reveal.x[h] = e.x[h], L.reveal.y[h] = e.y[h], L.reveal.until[h] = now + REVEAL_DURATION;
    }
    WardView view;
    Arr<float> oracle;
    std::tie(view, oracle) = api::W_ward_view(L.wards, now, arr(e.x, c), arr(e.y, c), arr(e.team, c), arr(e.alive, c),
                                              L.econ.level);
    // vision.sight_radius with the ward rows' radius, stealth, true sight, unobstructed and exposed overrides.
    std::vector<float> r(n, 0.f), ts(n, 0.f);
    std::vector<uint8_t> live(n), stealthed(n, 0), unobstructed(n, 0), exposed(n, 0);
    for (int i = 0; i < n; ++i) {
        live[i] = e.alive[i] && e.kind[i] != KIND_NONE;
        int k = e.kind[i];
        float rad = k == KIND_CHAMPION ? 1350.f
                  : k == KIND_MINION ? (e.sub[i] == 3 ? 1350.f : 1200.f)
                  : k == KIND_TURRET ? 1350.f : k == KIND_NEXUS ? 1350.f : k == KIND_INHIBITOR ? 0.f
                  : k == KIND_WARD ? (e.sub[i] == 2 ? 500.f : 900.f) : 0.f;
        if (i >= w0 && i < w0 + wn) {
            int s = i - w0;
            rad = view.sight_radius[s], stealthed[i] = view.stealthed[s], unobstructed[i] = view.unobstructed[s];
            exposed[i] = view.exposed[s], ts[i] = view.true_sight[s];
        }
        if (i < c) ts[i] = oracle[i];
        r[i] = live[i] ? rad : 0.f;
        float tsi = k == KIND_TURRET ? TURRET_TRUE_SIGHT : 0.f;
        ts[i] = live[i] ? std::max(tsi, ts[i]) : 0.f;
    }
    auto d2 = [&](int i, int j) { return sq(e.x[i] - e.x[j]) + sq(e.y[i] - e.y[j]); };
    int pairs = 0;
    std::vector<int32_t> viewers;
    for (int i = 0; i < n; ++i)
        if (r[i] > 0.f || ts[i] > 0.f) viewers.push_back(i);
    // Champions' own sight rows (full: every in-range enemy gets its ray).
    for (int h = 0; h < c; ++h)
        for (int j = 0; j < n; ++j) {
            bool s;
            if (j < nf) {
                bool in_range = d2(h, j) <= r[h] * r[h] && r[h] > 0.f && live[j];
                bool enemy = e.team[h] != e.team[j];
                bool hidden_t = stealthed[j] && live[j];
                if (hidden_t && enemy) s = d2(h, j) <= ts[h] * ts[h] && ts[h] > 0.f && live[j];
                else s = (in_range && (!enemy || unobstructed[h] || w.vision.clear(e.x[h], e.y[h], e.x[j], e.y[j])))
                         || (h == j && live[j]);
            } else {
                bool structure = e.kind[j] == KIND_TURRET || e.kind[j] == KIND_INHIBITOR || e.kind[j] == KIND_NEXUS;
                s = structure && live[j] && d2(h, j) <= r[h] * r[h];
            }
            L.sight[(size_t)h * n + j] = s;
        }
    for (int j = 0; j < n; ++j)
        for (int t = 0; t < 2; ++t) {
            bool seen = j >= nf || e.team[j] == t;
            if (!seen && live[j]) {
                bool hidden_t = stealthed[j] && live[j];
                // Seen if a live viewer of team t sees it (enemy pair): true sight for stealthed units, else range
                // and a clear ray (unobstructed viewers skip the ray); nearest viewer first.
                static thread_local std::vector<std::pair<float, int>> cand;
                cand.clear();
                for (int i : viewers) {
                    if (e.team[i] != t || !live[i]) continue;
                    float dd = d2(i, j);
                    bool in_range = dd <= r[i] * r[i] && r[i] > 0.f;
                    if (in_range) ++pairs;
                    if (hidden_t) {
                        if (dd <= ts[i] * ts[i] && ts[i] > 0.f) seen = true;
                    } else if (in_range) {
                        cand.push_back({dd, i});
                    }
                }
                if (!seen && !hidden_t) {
                    std::sort(cand.begin(), cand.end());
                    for (const auto& [dd, i] : cand)
                        if (unobstructed[i] || !w.fog || w.vision.clear(e.x[i], e.y[i], e.x[j], e.y[j])) { seen = true; break; }
                }
                for (int h = 0; h < c && !seen; ++h)
                    seen = t != e.team[h] && now < L.reveal.until[h] && !hidden_t
                           && sq(e.x[j] - L.reveal.x[h]) + sq(e.y[j] - L.reveal.y[h]) <= REVEAL_RADIUS * REVEAL_RADIUS;
                seen = seen || (exposed[j] && e.team[j] != t);
            }
            e.visible[(size_t)t * n + j] = seen && live[j];
        }
    st.rays = pairs;
    st.ray_overflow = std::max(pairs - w.ray_capacity, 0);
    for (int h = 0; h < c; ++h) {
        bool witnessed = e.visible[(size_t)(1 - (e.team[h] == 1 ? 1 : 0)) * n + h] || !hidden[h];
        for (int s = 0; s < 4; ++s)
            if (sc.kit_all.cast_started[h] && sc.kit_all.cast_slot[h] == s && witnessed) L.champ.seen_cast[h * 4 + s] = now;
    }
}

}  // namespace

// --- one tick (world/tick.py: step, commit) ------------------------------------------------------------------------
TickStats step_full(const World& w, Env& e, Orders& o) {
    TickStats st;
    if (*e.game_over) return st;                        // a fallen Nexus freezes the world
    const int c = N_CHAMPIONS, n = w.n;
    const ChampData& cd = champ_data(w);
    slice::Scratch& ss = slice::scratch();
    ss.size(w);
    prof::Laps laps;
    DormancyScope dormancy(cd.has_initial ? &cd.initial_items : nullptr, cd.has_initial ? &cd.initial_runes : nullptr);
    Layer L;
    load(w, e, L);
    TS sc;
    sc.now = *e.t + w.dt;
    rng::Key key{e.key[0], e.key[1]};
    sc.key = rng::split(key, 0), sc.k_crit = rng::split(key, 1);
    Arr<uint8_t> vis0 = arr(e.visible, (size_t)2 * n), alive0 = arr(e.alive, n);
    Clock clk;
    LS_LAP(laps, "tick.0_load");
    phase_input(w, e, cd, L, o, sc);
    LS_LAP(laps, "tick.1_input");
    phase_stats(w, e, cd, L, sc);
    LS_LAP(laps, "tick.2_stats");
    phase_casts(w, e, cd, L, o, sc);
    LS_LAP(laps, "tick.3_casts");
    clk.lap(slice::P_CHAMP);
    // AI: structures, minion and turret targets, champion acquisition.
    slice::turret_tick(w, e, sc.now);
    clk.lap(slice::P_TURRET);
    sc.units = units_view(w, e);
    slice::select_targets(w, e, sc.now, ss);
    clk.lap(slice::P_SELECT);
    phase_ai_champions(w, e, cd, L, sc, ss.desired.data());
    LS_LAP(laps, "tick.4_ai");
    phase_move(w, e, cd, L, sc, ss);
    LS_LAP(laps, "tick.5_move");
    clk.lap(slice::P_ROUTE);
    AttackOut ao;
    phase_attack(w, e, cd, L, sc, ss, ao, st);
    LS_LAP(laps, "tick.6_attack");
    clk.lap(slice::P_ATTACK);
    DamageOut d;
    phase_damage(w, e, cd, L, o, sc, ao, d);
    LS_LAP(laps, "tick.7_damage");
    ActiveWorld aw = phase_cc_heal(w, e, L, sc, d);
    LS_LAP(laps, "tick.8_cc_heal");
    clk.lap(slice::P_DAMAGE);
    DeathOut dd;
    phase_death(w, e, cd, L, o, sc, d, dd);
    LS_LAP(laps, "tick.9_death");
    clk.lap(slice::P_DEATH);
    phase_timers(w, e, cd, L, o, sc, ao, d, dd, aw, vis0);
    LS_LAP(laps, "tick.10_timers");
    clk.lap(slice::P_TIMERS);
    phase_fog(w, e, L, sc, ao, vis0, alive0, st);
    LS_LAP(laps, "tick.11_fog");
    clk.lap(slice::P_FOG);
    // commit: CC of the dead cleared, champion unit columns from this tick's stats, the next state.
    for (int j = 0; j < n; ++j)
        if (dd.died[j])
            e.cc_stun_until[j] = e.cc_root_until[j] = e.cc_silence_until[j] = e.cc_knockup_until[j] = e.cc_slow[j]
                = e.cc_slow_until[j] = e.cc_champion_cc_until[j] = 0.f;
    for (int h = 0; h < c; ++h) {
        e.attack_damage[h] = sc.st.base_ad[h] + sc.st.bonus_ad[h];
        e.armor[h] = sc.st.base_armor[h] + sc.st.bonus_armor[h], e.magic_resist[h] = sc.st.base_mr[h] + sc.st.bonus_mr[h];
        e.attack_range[h] = sc.reach[h], e.attack_speed[h] = sc.st.attack_speed[h], e.move_speed[h] = ss.ms[h];
    }
    *e.t = sc.now;
    *e.tick = *e.tick + 1;
    e.key[0] = sc.key.k0, e.key[1] = sc.key.k1;
    for (int j = 0; j < n; ++j) {
        e.hp[j] = e.alive[j] ? d.hp[j] : std::min(d.hp[j], 0.f);
        e.max_hp[j] = d.max_hp[j];
    }
    L.combat = std::move(d.out.state);
    L.shields = d.shields, L.status = d.status;
    L.prev_kills = dd.eco.kills;
    L.prev_epic.assign(c, 0.f), L.prev_large.assign(c, 0.f);
    L.prev_pending_dash = aw.dash;
    store(e, L);
    bool lost[2] = {false, false};
    for (int i = 0; i < n; ++i)
        if (e.towers_is_structure[i] && e.towers_turret_tier[i] == tower::NEXUS_BUILDING && e.towers_turret_hp[i] <= 0.f
            && e.towers_team[i] >= 0 && e.towers_team[i] < 2)
            lost[e.towers_team[i]] = true;
    *e.game_over = lost[0] || lost[1];
    LS_LAP(laps, "tick.12_commit");
    *e.winner = (lost[0] && !lost[1]) ? 1 : ((lost[1] && !lost[0]) ? 0 : -1);
    st.packet_overflow = d.out.packet_overflow;
    return st;
}

}  // namespace lanesim::champ
