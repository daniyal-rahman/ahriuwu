// Top role quest (lanerl_jax/modern/role_quest.py), per champion (C,).
#include <algorithm>

#include "../marshal.hpp"
#include "econ.hpp"

namespace lanesim::econ {

namespace {
constexpr float THRESHOLD = 1200.f, PASSIVE_START = 65.f, PASSIVE_IN_LANE = 1.5f, RECALL_LOCK = 12.f;
const float PASSIVE_BASE = (float)(1.0 / 3.0);
constexpr float ROAM_FILL = .5f, ROAM_CAP_EARLY = 5.f, ROAM_CAP = 60.f;
constexpr float P_MINION = 2.f, P_TURRET = 50.f, P_PLATE = 40.f, P_TAKEDOWN = 15.f, P_EPIC = 30.f;
constexpr float XP_BONUS = .11f, TAKEDOWN_XP = 80.f, EARLY_PENALTY = .25f, QUEST_TP_REDUCTION = 30.f;
constexpr int EARLY_PENALTY_LEVEL = 3;
constexpr float TP_SHIELD_FRACTION = .35f;
}  // namespace

QuestState init_quest(const Arr<int32_t>& roles) {
    size_t c = roles.size();
    QuestState q;
    q.role = roles, q.points.assign(c, 0.f), q.complete.assign(c, 0), q.roam_bank.assign(c, 0.f);
    q.recall_lock_until.assign(c, 0.f - 1e9f);
    return q;
}

// role_quest.out_of_lane_mult
float out_of_lane_mult(float points) { return .25f + .75f * std::min(std::max(points / THRESHOLD, 0.f), 1.f); }

// role_quest.quest_step
QuestStep quest_step(const QuestState& state, const QuestEvents& ev, float now, float dt, const Arr<uint8_t>& in_lane,
                     const Arr<uint8_t>& alive, const Arr<int32_t>& level, const Arr<uint8_t>& recalled) {
    size_t c = state.role.size();
    QuestStep out;
    out.state = state;
    out.completed_now.assign(c, 0), out.level_cap.assign(c, 0);
    for (size_t i = 0; i < c; ++i) {
        bool top = state.role[i] == ROLE_TOP;
        float pts = state.points[i];
        pts = pts + P_MINION * (ev.minions_in_lane[i] + ev.minions_out[i] * out_of_lane_mult(pts));
        pts = pts + P_PLATE * (ev.plates_in_lane[i] + ev.plates_out[i] * out_of_lane_mult(pts));
        pts = pts + P_TURRET * (ev.turrets_in_lane[i] + ev.turrets_out[i] * out_of_lane_mult(pts));
        pts = pts + P_TAKEDOWN * ev.takedowns[i] + P_EPIC * ev.epic[i];
        float cap = level[i] >= 3 ? ROAM_CAP : ROAM_CAP_EARLY;
        bool lane = in_lane[i] && alive[i];
        float bank = lane ? std::min(state.roam_bank[i] + ROAM_FILL * dt, cap) : state.roam_bank[i];
        bool spend = !lane && alive[i] && bank > 0.f;
        bank = spend ? std::max(bank - dt, 0.f) : bank;
        float lock_until = recalled[i] ? now + RECALL_LOCK : state.recall_lock_until[i];
        float rate = (lane || spend) ? PASSIVE_IN_LANE : PASSIVE_BASE;
        pts = pts + ((now >= PASSIVE_START && now >= lock_until) ? rate * dt : 0.f);
        bool open = top && !state.complete[i];
        pts = open ? pts : state.points[i];
        bool done = open && pts >= THRESHOLD;
        bool complete = state.complete[i] || done;
        out.state.points[i] = pts, out.state.complete[i] = complete;
        out.state.roam_bank[i] = top ? bank : state.roam_bank[i];
        out.state.recall_lock_until[i] = lock_until;
        out.completed_now[i] = done, out.level_cap[i] = complete ? 20 : 18;
    }
    return out;
}

// role_quest.xp_bonus
Arr<float> xp_bonus(const QuestState& state) {
    Arr<float> o(state.role.size());
    for (size_t i = 0; i < o.size(); ++i) o[i] = state.complete[i] && state.role[i] == ROLE_TOP ? XP_BONUS : 0.f;
    return o;
}

// role_quest.takedown_xp
Arr<float> takedown_xp(const QuestState& state) {
    Arr<float> o(state.role.size());
    for (size_t i = 0; i < o.size(); ++i) o[i] = state.complete[i] && state.role[i] == ROLE_TOP ? TAKEDOWN_XP : 0.f;
    return o;
}

// role_quest.minion_penalty: (C, M)
Arr<float> minion_penalty(const QuestState& state, const Arr<int32_t>& level, const Arr<uint8_t>& minion_in_lane) {
    size_t c = state.role.size(), m = minion_in_lane.size();
    Arr<float> o(c * m);
    for (size_t i = 0; i < c; ++i) {
        bool early = state.role[i] == ROLE_TOP && level[i] < EARLY_PENALTY_LEVEL;
        for (size_t j = 0; j < m; ++j) o[i * m + j] = early && !minion_in_lane[j] ? 1.f - EARLY_PENALTY : 1.f;
    }
    return o;
}

// role_quest.unleashed_tp_cooldown
float unleashed_tp_cooldown(float level, bool quest_complete) {
    float cd = 330.f - 10.f * (std::min(level, 9.f) - 1.f) - (level >= 10 ? 10.f : 0.f);
    return cd - (quest_complete ? QUEST_TP_REDUCTION : 0.f);
}

// role_quest.tp_arrival_shield
float tp_arrival_shield(float max_hp, bool quest_complete, bool has_teleport) {
    return quest_complete && has_teleport ? TP_SHIELD_FRACTION * max_hp : 0.f;
}

namespace {
QuestStep quest_step_test(QuestState state, QuestEvents ev, float now, float dt, Arr<uint8_t> in_lane,
                          Arr<uint8_t> alive, Arr<int32_t> level, Arr<uint8_t> recalled) {
    return quest_step(state, ev, now, dt, in_lane, alive, level, recalled);
}
Arr<float> minion_penalty_test(QuestState state, Arr<int32_t> level, Arr<uint8_t> in_lane) {
    return minion_penalty(state, level, in_lane);
}
}  // namespace
LANESIM_TEST(role_quest_quest_step, "role_quest.quest_step", quest_step_test);
LANESIM_TEST(role_quest_minion_penalty, "role_quest.minion_penalty", minion_penalty_test);

}  // namespace lanesim::econ
