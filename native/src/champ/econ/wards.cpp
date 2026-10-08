// Wards, trinkets, stealth and true sight (lanerl_jax/modern/wards.py), with the Deep Ward / Sixth Sense kernels
// of runes/effects/domination.py it runs. The WardGrid (placement terrain, Deep Ward regions) is world data from
// native/python/consts/econ_wards.py. Slots: S = 16, team t owns [t*8, (t+1)*8).
#include <algorithm>
#include <cmath>

#include "../marshal.hpp"
#include "econ.hpp"

namespace lanesim::econ {

using namespace champ;

namespace {

enum WardType { TOTEM = 0, CONTROL = 1, FARSIGHT = 2 };
constexpr int TOTEM_ITEM = 3340, FARSIGHT_ITEM = 3363, ORACLE_ITEM = 3364;
constexpr int REQ_NONE = -1, REQ_TRINKET = 0, REQ_CONTROL = 1;
enum Code { W_OK, ERR_NONE, ERR_DEAD, ERR_NO_ITEM, ERR_NO_CHARGE, ERR_LOCKED, ERR_RANGE, ERR_TERRAIN };
constexpr float WARD_HP[3] = {3.f, 4.f, 1.f}, WARD_SIGHT[3] = {900.f, 900.f, 500.f};
constexpr float WARD_BOUNTY[3] = {10.f, 30.f, 15.f}, WARD_XP[3] = {0.f, 0.f, 0.f};
constexpr float WARD_RADIUS = 1.f, TOTEM_RANGE = 625.f, FARSIGHT_RANGE = 4000.f;
constexpr int TOTEM_CAP = 3, CONTROL_CAP = 1;
constexpr float TOTEM_STEALTH_DELAY = 2.f, TOTEM_LOCKOUT = 1.25f, ORACLE_LOCKOUT = 5.f, ORACLE_DURATION = 8.f;
constexpr float TOTEM_DURATION[2] = {90.f, 120.f}, TOTEM_RECHARGE[2] = {210.f, 90.f};
constexpr float ORACLE_RECHARGE[2] = {160.f, 100.f}, FARSIGHT_RECHARGE[2] = {198.f, 99.f};
constexpr float ORACLE_LINGER = 2.f, ORACLE_HIT_REVEAL = 2.f, FARSIGHT_REVEAL_SIGHT = 800.f, FARSIGHT_REVEAL_TIME = 2.f;
constexpr float FARSIGHT_TRIGGER_LIFE = 3.f, CONTROL_REGEN_DELAY = 6.f, CONTROL_REGEN_PERIOD = 3.f;
constexpr float EARLY_BONUS_WINDOW = 10.f, EARLY_BONUS_GOLD = 5.f;
enum Region { REGION_OTHER, REGION_BLUE_JUNGLE, REGION_RED_JUNGLE, REGION_RIVER };
constexpr int BLUE = 0;
constexpr int DEEP_WARD = 8141, SIXTH_SENSE = 8137;

struct Consts {
    std::vector<float> walkable, region, geom;
    float deep_level, deep_hp, deep_dur_start, deep_dur_span, sixth_range2, sixth_level, sixth_cd, sixth_reveal;
    std::vector<float> sight;
    Consts() {
        auto t = [](const char* n) { return data::table(std::string("econ.wards.") + n); };
        walkable = t("walkable"), region = t("region"), geom = t("geom"), sight = t("sight");
        deep_level = t("deep_level")[0], deep_hp = t("deep_hp")[0], deep_dur_start = t("deep_dur_start")[0];
        deep_dur_span = t("deep_dur_span")[0], sixth_range2 = t("sixth_range2")[0], sixth_level = t("sixth_level")[0];
        sixth_cd = t("sixth_cd")[0], sixth_reveal = t("sixth_reveal")[0];
    }
};
const Consts& K() {
    static const Consts k;
    return k;
}

int slot_team(int s) { return s / MAX_WARDS_PER_TEAM; }

// wards._lerp
float lerp(const float ab[2], float level) {
    float lv = std::min(std::max(level, 1.f), 18.f);
    return ab[0] + (ab[1] - ab[0]) * (lv - 1.f) / 17.f;
}

// wards.oracle_radius
float oracle_radius(int lv) {
    int steps = lv >= 5 ? 1 + (int)std::floor((lv - 5) / 3.0) : 0;
    return std::min(600.f + 30.f * (float)steps, 750.f);
}

// wards._lookup: (walkable & on grid, region)
void lookup(float x, float y, bool& walk, int& region) {
    const Consts& k = K();
    int h = (int)k.geom[0], w = (int)k.geom[1];
    float cell = k.geom[2], min_x = k.geom[3], min_z = k.geom[4];
    int cx = (int)std::floor((x - min_x) / cell), cz = (int)std::floor((y - min_z) / cell);
    bool ok = cx >= 0 && cz >= 0 && cx < w && cz < h;
    cx = clampi(cx, 0, w - 1), cz = clampi(cz, 0, h - 1);
    walk = ok && k.walkable[(size_t)cz * w + cx] != 0.f;
    region = ok ? (int)k.region[(size_t)cz * w + cx] : REGION_OTHER;
}

float recharge(int trinket, float avg_level, float haste) {
    float base = trinket == TOTEM_ITEM ? lerp(TOTEM_RECHARGE, avg_level)
                                       : (trinket == ORACLE_ITEM ? lerp(ORACLE_RECHARGE, avg_level)
                                                                 : lerp(FARSIGHT_RECHARGE, avg_level));
    return base * 100.f / (100.f + std::max(haste, 0.f));
}
int max_ammo(int t) { return t == FARSIGHT_ITEM ? 1 : ((t == TOTEM_ITEM || t == ORACLE_ITEM) ? 2 : 0); }
bool is_trinket(int t) { return t == TOTEM_ITEM || t == FARSIGHT_ITEM || t == ORACLE_ITEM; }

bool has_rune_page(const Arr<int32_t>& page, int perk, size_t c) {
    size_t r = page.size() / C;
    return page[c * r + rune_row(perk)] > 0;
}

bool stealthed(const WardState& sl, int j, float now) {
    return sl.alive[j] && sl.type_[j] == TOTEM && now >= sl.placed_at[j] + TOTEM_STEALTH_DELAY;
}

// wards._oracle_cover: (C, S)
std::vector<uint8_t> oracle_cover(const WardState& sl, const TrinketState& tr, float now, const Arr<float>& cx,
                                  const Arr<float>& cy, const Arr<int32_t>& cteam, const Arr<uint8_t>& calive,
                                  const Arr<int32_t>& level) {
    size_t c = cx.size();
    std::vector<uint8_t> out(c * S);
    for (size_t i = 0; i < c; ++i) {
        float r = oracle_radius(level[i]) + WARD_RADIUS;
        bool act = now < tr.oracle_until[i] && calive[i];
        for (int j = 0; j < S; ++j) {
            float d2 = sq(sl.x[j] - cx[i]) + sq(sl.y[j] - cy[i]);
            out[i * S + j] = act && sl.alive[j] && slot_team(j) != cteam[i] && d2 <= r * r;
        }
    }
    return out;
}

// wards._disable: (disabled (S,), Control Ward exposing a stealthed ward (S,))
void disable(const WardState& sl, const TrinketState& tr, float now, const Arr<float>& cx, const Arr<float>& cy,
             const Arr<int32_t>& cteam, const Arr<uint8_t>& calive, const Arr<int32_t>& level, uint8_t* disabled,
             uint8_t* exposing) {
    size_t c = cx.size();
    auto oc = oracle_cover(sl, tr, now, cx, cy, cteam, calive, level);
    bool cover[S][S];
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) {
            bool ctrl = sl.alive[i] && sl.type_[i] == CONTROL;
            float d2 = sq(sl.x[i] - sl.x[j]) + sq(sl.y[i] - sl.y[j]);
            cover[i][j] = ctrl && sl.alive[j] && slot_team(i) != slot_team(j) && d2 <= WARD_SIGHT[CONTROL] * WARD_SIGHT[CONTROL];
        }
    for (int j = 0; j < S; ++j) {
        bool by_c = false, by_o = false;
        for (int i = 0; i < S; ++i) by_c = by_c || cover[i][j];
        for (size_t i = 0; i < c; ++i) by_o = by_o || oc[i * S + j];
        bool st = stealthed(sl, j, now);
        by_c = by_c && sl.type_[j] != CONTROL;
        by_o = (by_o || now < sl.disabled_until[j]) && st;
        disabled[j] = sl.alive[j] && (by_c || by_o);
        bool ex = false;
        for (int k = 0; k < S; ++k) ex = ex || (cover[j][k] && stealthed(sl, k, now));
        exposing[j] = sl.alive[j] && sl.type_[j] == CONTROL && ex;
    }
}

}  // namespace

// wards.init_wards
Wards init_wards(int c, const Arr<int32_t>& trinket_ids) {
    Wards w;
    WardState& s = w.slots;
    const float inf = INF;
    s.alive.assign(S, 0), s.type_.assign(S, 0), s.owner.assign(S, -1), s.x.assign(S, 0.f), s.y.assign(S, 0.f);
    s.placed_at.assign(S, 0.f), s.expires_at.assign(S, inf), s.hp.assign(S, 0.f), s.max_hp.assign(S, 0.f);
    s.bounty.assign(S, 0.f), s.early_paid.assign(S, 0), s.last_damaged.assign(S, -inf), s.regen_at.assign(S, inf);
    s.revealed_until.assign(S, -inf), s.disabled_until.assign(S, -inf), s.triggered_at.assign(S, inf);
    s.tracked.assign(S, 0), s.deep.assign(S, 0), s.seq.assign(S, 0);
    TrinketState& t = w.trinket;
    t.trinket.assign(c, TOTEM_ITEM), t.charges.assign(c, 0);
    for (int i = 0; i < c; ++i) {
        int id = trinket_ids.size() ? trinket_ids[i] : TOTEM_ITEM;
        t.trinket[i] = id, t.charges[i] = (id == TOTEM_ITEM || id == ORACLE_ITEM) ? 1 : 0;
    }
    t.progress.assign(c, 0.f), t.lock_until.assign(c, 0.f), t.oracle_until.assign(c, -inf), t.sixth_cd_until.assign(c, 0.f);
    w.next_seq = 0;
    return w;
}

// wards.ward_step
std::tuple<Wards, WardEvents> ward_step(const Wards& w, float now, float dt, const WardRequest& request,
                                        const Arr<float>& x, const Arr<float>& y, const Arr<int32_t>& team,
                                        const Arr<uint8_t>& alive, const Arr<int32_t>& level,
                                        const Arr<int32_t>& trinket_id, const Arr<int32_t>& control_count,
                                        const Arr<uint8_t>& can_use_in, const Arr<float>& trinket_haste,
                                        const Arr<int32_t>& hits_in, const Arr<int32_t>& hitter_in,
                                        const Arr<int32_t>& rune_pages, const Arr<uint8_t>& ward_visible) {
    const Consts& kc = K();
    WardState sl = w.slots;
    const TrinketState& tr0 = w.trinket;
    const size_t c = x.size();
    float avg = 0.f;
    for (size_t i = 0; i < c; ++i) avg += (float)level[i];
    avg = avg / (float)c;
    Arr<uint8_t> can_use(c);
    for (size_t i = 0; i < c; ++i) can_use[i] = (can_use_in.size() ? can_use_in[i] : 1) && alive[i];
    Arr<float> haste = trinket_haste.size() ? trinket_haste : Arr<float>(c, 0.f);
    Arr<int32_t> hits = hits_in.size() ? hits_in : Arr<int32_t>(S, 0);
    Arr<int32_t> hitter = hitter_in.size() ? hitter_in : Arr<int32_t>(S, -1);

    // 1. Trinket swap: carry the time-equivalent of charges + progress. 2. Recharge.
    Arr<int32_t> tid(c), charges(c), amax(c);
    Arr<float> progress(c), oracle_until(c);
    for (size_t i = 0; i < c; ++i) {
        tid[i] = is_trinket(trinket_id[i]) ? trinket_id[i] : 0;
        bool swapped = tid[i] != tr0.trinket[i];
        float old_worth = ((float)tr0.charges[i] + tr0.progress[i]) * recharge(tr0.trinket[i], avg, haste[i]);
        float total = (is_trinket(tr0.trinket[i]) ? old_worth : 0.f) / recharge(tid[i], avg, haste[i]);
        amax[i] = max_ammo(tid[i]);
        int ch_sw = std::min((int)std::floor(total), amax[i]);
        float pr_sw = ch_sw < amax[i] ? total - std::floor(total) : 0.f;
        charges[i] = swapped ? ch_sw : tr0.charges[i];
        progress[i] = swapped ? pr_sw : tr0.progress[i];
        oracle_until[i] = swapped && tr0.trinket[i] == ORACLE_ITEM ? now : tr0.oracle_until[i];
        bool charging = charges[i] < amax[i] && tid[i] != 0;
        progress[i] = charging ? progress[i] + dt / recharge(tid[i], avg, haste[i]) : 0.f;
        int gain = (int)std::floor(progress[i]);
        charges[i] = std::min(charges[i] + (charging ? gain : 0), amax[i]);
        progress[i] = charges[i] < amax[i] ? progress[i] - (float)gain : 0.f;
    }

    // 3. Hits: 1 damage per champion basic attack; early-detection bonus; Oracle hit reveal.
    WardEvents ev;
    ev.killed.assign(S, 0), ev.killer.assign(S, -1), ev.expired.assign(S, 0), ev.replaced.assign(S, 0);
    ev.gold.assign(c, 0.f), ev.xp.assign(c, 0.f);
    Arr<uint8_t> early(S);
    Arr<float> early_gold(S), hp(S);
    for (int j = 0; j < S; ++j) {
        int hc = clampi(hitter[j], 0, (int)c - 1);
        bool hit = sl.alive[j] && hits[j] > 0 && hitter[j] >= 0 && team[hc] != slot_team(j);
        early[j] = hit && !sl.early_paid[j] && (now - sl.placed_at[j] <= EARLY_BONUS_WINDOW);
        early_gold[j] = early[j] ? std::min(EARLY_BONUS_GOLD, sl.bounty[j]) : 0.f;
        sl.bounty[j] = sl.bounty[j] - early_gold[j];
        hp[j] = hit ? sl.hp[j] - (float)hits[j] : sl.hp[j];
        ev.killed[j] = hit && hp[j] <= 0.f;
        ev.killer[j] = ev.killed[j] ? hitter[j] : -1;
        bool oracle_hit = hit && now < oracle_until[hc];
        if (oracle_hit) sl.revealed_until[j] = std::max(sl.revealed_until[j], now + ORACLE_HIT_REVEAL);
        if (hit) sl.last_damaged[j] = now, sl.regen_at[j] = now + CONTROL_REGEN_DELAY + CONTROL_REGEN_PERIOD;
    }
    for (int j = 0; j < S; ++j) ev.gold[clampi(hitter[j], 0, (int)c - 1)] += early[j] ? early_gold[j] : 0.f;
    for (int j = 0; j < S; ++j) ev.gold[clampi(hitter[j], 0, (int)c - 1)] += ev.killed[j] ? sl.bounty[j] : 0.f;
    for (int j = 0; j < S; ++j) {
        int t = sl.type_[j];
        float xp_s = t == CONTROL ? WARD_XP[1] : (t == TOTEM ? WARD_XP[0] : WARD_XP[2]);
        ev.xp[clampi(hitter[j], 0, (int)c - 1)] += ev.killed[j] ? xp_s : 0.f;
    }

    // 4. Expiry (Totem duration, Farsight 3 s after spotting) and Control Ward regen.
    for (int j = 0; j < S; ++j) {
        ev.expired[j] = sl.alive[j] && !ev.killed[j] &&
                        (now >= sl.expires_at[j] || now >= sl.triggered_at[j] + FARSIGHT_TRIGGER_LIFE);
        bool alive_s = sl.alive[j] && !ev.killed[j] && !ev.expired[j];
        bool regen = alive_s && sl.type_[j] == CONTROL && now >= sl.regen_at[j] && hp[j] < sl.max_hp[j];
        sl.hp[j] = regen ? hp[j] + 1.f : hp[j];
        if (regen) sl.regen_at[j] = sl.regen_at[j] + CONTROL_REGEN_PERIOD;
        sl.alive[j] = alive_s;
        sl.early_paid[j] = sl.early_paid[j] || early[j];
    }

    // 5. Requests: trinket (place Totem/Farsight, or Oracle sweep) and Control Ward.
    ev.code.assign(c, 0), ev.placed.assign(c, 0), ev.placed_slot.assign(c, -1), ev.placed_type.assign(c, 0);
    ev.sweep_started.assign(c, 0), ev.trinket_used.assign(c, 0), ev.consumed_control.assign(c, 0);
    ev.sensed.assign(c, 0);
    Arr<float> lock_until(c), hp_new(c), dur(c);
    Arr<uint8_t> deep(c);
    for (size_t i = 0; i < c; ++i) {
        int req = request.kind[i];
        float dist = std::sqrt(sq(request.x[i] - x[i]) + sq(request.y[i] - y[i]));
        bool walk;
        int region;
        lookup(request.x[i], request.y[i], walk, region);
        bool is_sweep = req == REQ_TRINKET && tid[i] == ORACLE_ITEM;
        int place_type = req == REQ_CONTROL ? CONTROL : (tid[i] == FARSIGHT_ITEM ? FARSIGHT : TOTEM);
        float rng = place_type == FARSIGHT ? FARSIGHT_RANGE : TOTEM_RANGE;
        bool has_item = req == REQ_CONTROL ? control_count[i] > 0 : tid[i] != 0;
        bool needs_charge = req == REQ_TRINKET;
        int code = req == REQ_NONE                         ? ERR_NONE
                   : !can_use[i]                           ? ERR_DEAD
                   : !has_item                             ? ERR_NO_ITEM
                   : needs_charge && charges[i] < 1        ? ERR_NO_CHARGE
                   : needs_charge && now < tr0.lock_until[i] ? ERR_LOCKED
                   : !is_sweep && dist > rng               ? ERR_RANGE
                   : !is_sweep && !walk                    ? ERR_TERRAIN
                                                           : W_OK;
        bool ok = code == W_OK, sweep = ok && is_sweep, place = ok && !is_sweep, used = ok && needs_charge;
        charges[i] = charges[i] - (int)used;
        lock_until[i] = used ? now + (is_sweep ? ORACLE_LOCKOUT : TOTEM_LOCKOUT) : tr0.lock_until[i];
        oracle_until[i] = sweep ? now + ORACLE_DURATION : oracle_until[i];
        // Deep Ward (domination.deep_ward) on Totem placements.
        float deep_hp = 0.f, deep_dur = 0.f;
        deep[i] = 0;
        if (rune_pages.size()) {
            bool enemy_jungle = team[i] == BLUE ? region == REGION_RED_JUNGLE : region == REGION_BLUE_JUNGLE;
            bool river_ok = region == REGION_RIVER && (float)level[i] >= kc.deep_level;
            deep[i] = has_rune_page(rune_pages, DEEP_WARD, i) && place_type == TOTEM && (enemy_jungle || river_ok);
            float lv = std::min(std::max(avg, 1.f), 18.f);
            float d = kc.deep_dur_start + kc.deep_dur_span * (lv - 1.f) / 17.f;
            deep_hp = deep[i] ? kc.deep_hp : 0.f;
            deep_dur = deep[i] ? d : 0.f;
        }
        hp_new[i] = WARD_HP[place_type] + deep_hp;
        dur[i] = place_type == TOTEM ? lerp(TOTEM_DURATION, avg) + deep_dur : INF;
        ev.code[i] = code, ev.placed[i] = place, ev.placed_type[i] = place_type, ev.sweep_started[i] = sweep;
        ev.trinket_used[i] = used, ev.consumed_control[i] = place && req == REQ_CONTROL;
    }
    int seq = w.next_seq;
    const int big = 1 << 30;
    for (size_t i = 0; i < c; ++i) {                     // static, C = 2
        int typ = ev.placed_type[i];
        int n_mine = 0, oldest_mine = 0, oldest_team = 0, first_free = -1;
        int best_mine = big, best_team = big;
        for (int j = 0; j < S; ++j) {
            bool mine = sl.alive[j] && sl.owner[j] == (int)i && sl.type_[j] == typ;
            bool in_team = slot_team(j) == team[i];
            n_mine += mine;
            int km = mine ? sl.seq[j] : big, kt = in_team && sl.alive[j] ? sl.seq[j] : big;
            if (km < best_mine) best_mine = km, oldest_mine = j;
            if (kt < best_team) best_team = kt, oldest_team = j;
            if (in_team && !sl.alive[j] && first_free < 0) first_free = j;
        }
        int cap = typ == TOTEM ? TOTEM_CAP : (typ == CONTROL ? CONTROL_CAP : S);
        bool over = n_mine >= cap;
        int k = over ? oldest_mine : (first_free >= 0 ? first_free : oldest_team);
        bool p = ev.placed[i];
        ev.replaced[k] = ev.replaced[k] || (p && sl.alive[k]);
        if (p) {
            sl.alive[k] = 1, sl.type_[k] = typ, sl.owner[k] = (int)i, sl.x[k] = request.x[i], sl.y[k] = request.y[i];
            sl.placed_at[k] = now, sl.expires_at[k] = now + dur[i], sl.hp[k] = hp_new[i], sl.max_hp[k] = hp_new[i];
            sl.bounty[k] = WARD_BOUNTY[typ], sl.early_paid[k] = 0, sl.last_damaged[k] = -INF, sl.regen_at[k] = INF;
            sl.revealed_until[k] = -INF, sl.disabled_until[k] = -INF, sl.triggered_at[k] = INF, sl.tracked[k] = 0;
            sl.deep[k] = deep[i], sl.seq[k] = seq;
        }
        ev.placed_slot[i] = p ? k : -1;
        seq += (int)p;
    }
    TrinketState tr;
    tr.trinket = tid, tr.charges = charges, tr.progress = progress, tr.lock_until = lock_until;
    tr.oracle_until = oracle_until, tr.sixth_cd_until = tr0.sixth_cd_until;

    // 6. Farsight trigger: a live enemy champion within its current (unobstructed) sight.
    uint8_t disabled_now[S], exposing[S];
    disable(sl, tr, now, x, y, team, alive, level, disabled_now, exposing);
    for (int j = 0; j < S; ++j) {
        float fr = now < sl.placed_at[j] + FARSIGHT_REVEAL_TIME ? FARSIGHT_REVEAL_SIGHT : WARD_SIGHT[2];
        bool spot = false;
        for (size_t i = 0; i < c; ++i) {
            float d2 = sq(sl.x[j] - x[i]) + sq(sl.y[j] - y[i]);
            spot = spot || (d2 <= fr * fr && alive[i] && team[i] != slot_team(j));
        }
        bool trig = sl.alive[j] && sl.type_[j] == FARSIGHT && spot && !disabled_now[j] && !std::isfinite(sl.triggered_at[j]);
        if (trig) sl.triggered_at[j] = now;
    }

    // 7. Oracle linger: wards inside an enemy sweep stay disabled 2 s after leaving it.
    {
        auto oc = oracle_cover(sl, tr, now, x, y, team, alive, level);
        for (int j = 0; j < S; ++j) {
            bool any = false;
            for (size_t i = 0; i < c; ++i) any = any || oc[i * S + j];
            if (any && stealthed(sl, j, now)) sl.disabled_until[j] = now + ORACLE_LINGER;
        }
    }

    // 8. Sixth Sense (domination.sixth_sense).
    if (rune_pages.size()) {
        Arr<uint8_t> picked(S, 0), rev_s(S, 0);
        for (size_t i = 0; i < c; ++i) {
            int vt = clampi(team[i], 0, 1);
            bool any = false;
            float best = INF;
            int near = 0;
            for (int j = 0; j < S; ++j) {
                bool unseen = ward_visible.size() ? !ward_visible[(size_t)vt * S + j] : true;
                float d2 = sq(sl.x[j] - x[i]) + sq(sl.y[j] - y[i]);
                bool cand = sl.alive[j] && slot_team(j) != team[i] && unseen && !sl.tracked[j] && d2 <= kc.sixth_range2;
                any = any || cand;
                float key = cand ? d2 : INF;
                if (key < best) best = key, near = j;
            }
            bool ready = has_rune_page(rune_pages, SIXTH_SENSE, i) && alive[i] && now >= tr.sixth_cd_until[i];
            bool go = ready && any;
            bool reveal = go && (float)level[i] >= kc.sixth_level;
            tr.sixth_cd_until[i] = go ? now + kc.sixth_cd : tr.sixth_cd_until[i];
            ev.sensed[i] = go;
            if (go) picked[near] = 1, rev_s[near] = rev_s[near] || reveal;
        }
        float reveal_at = now + kc.sixth_reveal;
        for (int j = 0; j < S; ++j) {
            sl.tracked[j] = sl.tracked[j] || picked[j];
            if (rev_s[j]) sl.revealed_until[j] = std::max(sl.revealed_until[j], reveal_at);
        }
    }
    Wards out;
    out.slots = sl, out.trinket = tr, out.next_seq = seq;
    return {out, ev};
}

// wards.ward_view
std::tuple<WardView, Arr<float>> ward_view(const Wards& w, float now, const Arr<float>& x, const Arr<float>& y,
                                           const Arr<int32_t>& team, const Arr<uint8_t>& alive,
                                           const Arr<int32_t>& level) {
    const WardState& sl = w.slots;
    const size_t c = x.size();
    uint8_t disabled[S], exposing[S];
    disable(sl, w.trinket, now, x, y, team, alive, level, disabled, exposing);
    WardView v;
    v.alive = sl.alive, v.x = sl.x, v.y = sl.y, v.sub = sl.type_, v.owner = sl.owner, v.hp = sl.hp;
    v.max_hp = sl.max_hp, v.expires_at = sl.expires_at;
    v.team.assign(S, 0), v.sight_radius.assign(S, 0.f), v.stealthed.assign(S, 0), v.true_sight.assign(S, 0.f);
    v.unobstructed.assign(S, 0), v.exposed.assign(S, 0), v.disabled.assign(S, 0), v.tracked.assign(S, 0);
    for (int j = 0; j < S; ++j) {
        bool far = sl.type_[j] == FARSIGHT;
        float far_r = (now < sl.placed_at[j] + FARSIGHT_REVEAL_TIME) || std::isfinite(sl.triggered_at[j])
                          ? FARSIGHT_REVEAL_SIGHT : WARD_SIGHT[2];
        float r = far ? far_r : WARD_SIGHT[0];
        v.sight_radius[j] = sl.alive[j] && !disabled[j] ? r : 0.f;
        v.team[j] = slot_team(j);
        v.stealthed[j] = stealthed(sl, j, now);
        v.true_sight[j] = sl.alive[j] && sl.type_[j] == CONTROL ? WARD_SIGHT[1] : 0.f;
        v.unobstructed[j] = sl.alive[j] && far;
        v.exposed[j] = sl.alive[j] && ((now < sl.revealed_until[j]) || exposing[j]);
        v.disabled[j] = disabled[j];
        v.tracked[j] = sl.tracked[j] && sl.alive[j];
    }
    Arr<float> oracle(c);
    for (size_t i = 0; i < c; ++i)
        oracle[i] = now < w.trinket.oracle_until[i] && alive[i] ? oracle_radius(level[i]) + WARD_RADIUS : 0.f;
    return {v, oracle};
}

// wards.vision_kwargs (vision.sight_radius for the base radius)
VisionKw vision_kwargs(const WardView& view, const Arr<float>& oracle, const Arr<int32_t>& kind,
                       const Arr<int32_t>& sub, const Arr<uint8_t>& alive, int ward_start) {
    const auto& s = K().sight;   // champion, minion, super, turret, nexus, inhibitor, ward, farsight, SUPER, FARSIGHT_SUB
    size_t n = kind.size(), c = oracle.size();
    VisionKw o;
    o.radius.assign(n, 0.f), o.stealthed.assign(n, 0), o.true_sight.assign(n, 0.f), o.unobstructed.assign(n, 0);
    o.exposed.assign(n, 0);
    for (size_t j = 0; j < n; ++j) {
        int k = kind[j];
        float r = k == KIND_CHAMPION ? s[0]
                  : k == KIND_MINION ? (sub[j] == (int)s[8] ? s[2] : s[1])
                  : k == KIND_TURRET ? s[3]
                  : k == KIND_NEXUS ? s[4]
                  : k == KIND_INHIBITOR ? s[5]
                  : k == KIND_WARD ? (sub[j] == (int)s[9] ? s[7] : s[6]) : 0.f;
        o.radius[j] = alive[j] ? r : 0.f;
    }
    for (int j = 0; j < S; ++j) {
        size_t u = (size_t)ward_start + j;
        o.radius[u] = view.sight_radius[j], o.stealthed[u] = view.stealthed[j], o.true_sight[u] = view.true_sight[j];
        o.unobstructed[u] = view.unobstructed[j], o.exposed[u] = view.exposed[j];
    }
    for (size_t i = 0; i < c; ++i) o.true_sight[i] = oracle[i];
    return o;
}

// ---- replay registrations ----------------------------------------------------------------------------------------
namespace {
// The captured ``grid`` kwarg (WardGrid leaves); the native port reads the same grid from the constants.
struct WardGridArg {
    Arr<uint8_t> walkable;
    Arr<int32_t> region;
    float cell_size, min_x, min_z;
    template <class F> void visit(F&& f) { f(walkable); f(region); f(cell_size); f(min_x); f(min_z); }
};
// Keyword arguments arrive in the captured (sorted) key order.
std::tuple<Wards, WardEvents> ward_step_test(Wards w, Arr<uint8_t> alive, Arr<uint8_t> can_use,
                                             Arr<int32_t> control_count, float dt, WardGridArg grid, Arr<int32_t> hits,
                                             Arr<int32_t> hitter, Arr<int32_t> level, float now, const WardRequest& request,
                                             Arr<int32_t> rune_pages, Arr<int32_t> team, Arr<float> trinket_haste,
                                             Arr<int32_t> trinket_id, Arr<uint8_t> ward_visible, Arr<float> x,
                                             Arr<float> y) {
    return ward_step(w, now, dt, request, x, y, team, alive, level, trinket_id, control_count, can_use, trinket_haste,
                     hits, hitter, rune_pages, ward_visible);
}
std::tuple<WardView, Arr<float>> ward_view_test(Wards w, Arr<uint8_t> alive, Arr<int32_t> level, float now,
                                                Arr<int32_t> team, Arr<float> x, Arr<float> y) {
    return ward_view(w, now, x, y, team, alive, level);
}
VisionKw vision_kwargs_test(WardView view, Arr<float> oracle, Arr<int32_t> kind, Arr<int32_t> sub, Arr<uint8_t> alive,
                            int32_t ward_start) {
    return vision_kwargs(view, oracle, kind, sub, alive, ward_start);
}
Wards init_wards_test(int32_t c, Arr<int32_t> trinket_ids) { return init_wards(c, trinket_ids); }
}  // namespace
LANESIM_TEST(wards_ward_step, "wards.ward_step", ward_step_test);
LANESIM_TEST(wards_ward_view, "wards.ward_view", ward_view_test);
LANESIM_TEST(wards_vision_kwargs, "wards.vision_kwargs", vision_kwargs_test);
LANESIM_TEST(wards_init_wards, "wards.init_wards", init_wards_test);

}  // namespace lanesim::econ
