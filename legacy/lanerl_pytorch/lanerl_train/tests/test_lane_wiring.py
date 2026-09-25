"""What the collector records, and what it used to record instead.

Every test below is one number that a run had no way to produce, and each one
cost something concrete:

* ``cs_at_10`` was the MEAN over both champions in the frame.  Correct in a
  mirror, where both are the same policy -- and wrong the moment red is a
  scripted bot, because it then averages the agent's farm with its opponent's.
  A bronze bot farming 48 CS puts a floor of 24 under the headline metric of a
  policy that farms nothing.
* the per-term reward breakdown was computed on every tick since the reward
  was written and thrown away on every tick since the reward was written, so
  "dying is net-positive" (``lanerl_rl.reward``: "A respawn is not a heal")
  survived weeks of ordinary-looking runs.
* kills and deaths did not exist on ``EpisodeResult`` at all, so under
  ``--no-end-on-death`` -- which is required for CS@10 to exist -- nothing in
  the metrics could answer "is the agent feeding?".
* ``ad``/``mhp`` on the first frame is the canary for an in-process reset that
  strips the rune page (mhp 672 -> 616), which makes every episode after the
  first a different game from the one being measured, invisibly.
* the per-button marginals are the only curve that shows a BC prior eroding.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from lanerl_rl import constants as C
from lanerl_train.tests.fakes import RED_TEAM, FakeInstance
from lanerl_train.vec import EpisodeSpec, SideAssignment, VecDriver, VecLaneEnv

SELF = "self"


class AsymmetricFake(FakeInstance):
    """A fake whose two champions are NOT interchangeable.

    ``fakes.make_obs`` gives blue and red the same CS, which is right for the
    boundary tests it was written for and useless here: a readout that averages
    the two teams and one that reports only the agent's are the same number on
    a symmetric frame, so the bug this file exists to pin would pass.

    It also puts ``ad`` on the champions (the server does; ``make_obs`` does
    not) and makes ``mhp`` depend on the game clock, so that a first-frame
    readout taken from any other frame reports 616 instead of 672 -- the exact
    pair of numbers the rune-page regression produced.
    """

    FIRST_FRAME_MHP = 672.0
    LATER_MHP = 616.0

    def __init__(self, index: int, red_cs_multiple: int = 1, ad: float = 78.14, **kw):
        super().__init__(index, **kw)
        self.red_cs_multiple = int(red_cs_multiple)
        self.ad = float(ad)

    def _obs(self):
        obs = super()._obs()
        for u in obs["u"]:
            if u.get("k") != "Champion":
                continue
            u["ad"] = self.ad
            u["mhp"] = self.FIRST_FRAME_MHP if self.t_ms == 0 else self.LATER_MHP
            if u["tm"] == RED_TEAM:
                u["cs"] = int(self.cs * self.red_cs_multiple)
        return obs


def _small_actor():
    # Imported inside the helpers, which is this file's existing convention:
    # it keeps collection independent of the module under test.
    from lanerl_rl.model import LanePolicy, ModelConfig
    from lanerl_train.lane_wiring import LanePolicyActor

    return LanePolicyActor(
        LanePolicy(
            ModelConfig(core_dim=32, d_model=32, ffn_dim=32, n_layers=1, n_heads=2, mlp_hidden=32)
        )
    )


def rollout_over(
    instances: List[FakeInstance],
    episode: EpisodeSpec,
    num_steps: int,
    red: Optional[str] = SELF,
    opponent_labels=None,
):
    """One real ``collect_rollout`` over fake servers.

    ``red=None`` is the phase-1 configuration: the action line omits the key
    and the in-server bot's own orders stand.
    """
    from lanerl_train.lane_wiring import collect_rollout, make_lane_adapters
    from lanerl_train.ports import InstancePorts

    n = len(instances)
    ports = [InstancePorts(i, 45000 + 2 * i, 45000 + 2 * i + 1) for i in range(n)]
    env = VecLaneEnv(
        n, ports=ports, factory=lambda i, p: instances[i],
        step_timeout_s=2.0, auto_restart=False,
    )
    actor = _small_actor()
    adapters = make_lane_adapters(train_step_source=lambda: 0)
    driver = VecDriver(
        env,
        policies={SELF: actor},
        adapter_factory=adapters.adapter_factory,
        encoder=adapters.encoder,
        assignments=[SideAssignment(blue=SELF, red=red) for _ in instances],
        episode=episode,
    )
    driver.start()
    return collect_rollout(
        driver, actor, adapters.reward_contexts, SELF,
        num_steps=num_steps, gamma=0.99, gae_lambda=0.95,
        opponent_labels=opponent_labels,
    )


def _timed(rollout):
    eps = [e for e in rollout.episodes if e.reason == "time"]
    assert eps, f"no episode ended on the clock: {[e.reason for e in rollout.episodes]}"
    return eps[0]


# -- whose CS is CS@10 -----------------------------------------------------


def test_cs_at_10_is_the_agents_own_team_when_red_is_the_scripted_bot():
    """The mean of both teams is the agent's CS only in a mirror.

    Against a bot it is ``(agent + bot) / 2``, so a bronze bot farming 48 puts
    a floor of 24 under the headline metric however badly the policy plays, and
    an agent that improved by 10 CS would show as improving by 5.
    """
    insts = [AsymmetricFake(0, step_ms=100_000, cs_per_step=3, red_cs_multiple=5)]
    ep = _timed(rollout_over(insts, EpisodeSpec(max_game_ms=600_000), 12, red=None))
    # 600_000 / 100_000 = 6 decisions, 3 CS each -> blue 18, red 90.
    assert ep.cs_at_10 == pytest.approx(18.0), (
        "cs_at_10 is not the agent's own farm; the mean of both teams would be 54"
    )
    assert ep.opponent_cs_at_10 == pytest.approx(90.0), (
        "the opponent's CS from the same game is the honest yardstick and was dropped"
    )


def test_cs_at_10_still_averages_both_champions_in_self_play():
    """The other half of the same rule, and the one that must not regress.

    Both champions ARE the agent here, so both are samples of its skill: the
    mean is unbiased where max() is upward-biased by ~0.56 sigma. There is no
    opponent to report.
    """
    insts = [AsymmetricFake(0, step_ms=100_000, cs_per_step=3, red_cs_multiple=5)]
    ep = _timed(rollout_over(insts, EpisodeSpec(max_game_ms=600_000), 12, red=SELF))
    assert ep.cs_at_10 == pytest.approx((18.0 + 90.0) / 2.0)
    assert ep.opponent_cs_at_10 is None, (
        "in a mirror there is no opponent to measure; reporting one invites the "
        "agent's own second champion to be read as a yardstick"
    )


def test_an_episode_against_the_bot_is_scored_and_named_not_called_a_self_match():
    """A real opponent means a real result; 0.5 was hardcoded.

    ``TrainingLoop.record_episode`` routes on ``opponent_id``: labelled as a
    self-match, a phase-1 game contributes nothing to any rating and its score
    is a constant. The label is namespaced (``train:``) so training games do
    not pool into the anchor ladder's win rate, which is a measurement with a
    chosen sample size.
    """
    insts = [AsymmetricFake(0, step_ms=1_000, champ_dies_at_ms=4_000)]
    rollout = rollout_over(
        insts, EpisodeSpec(max_game_ms=10 ** 9, end_on_death=True), 12,
        red=None, opponent_labels={0: "train:scripted_bronze"},
    )
    deaths = [e for e in rollout.episodes if e.reason.startswith("death")]
    assert deaths, f"no death-ended episode: {[e.reason for e in rollout.episodes]}"
    ep = deaths[0]
    assert ep.opponent_id == "train:scripted_bronze"
    assert ep.opponent_category == "scripted"
    assert ep.score == 0.0, "blue is the one that died; that is a loss, not a draw"


def test_a_self_play_episode_is_still_an_unrated_draw():
    insts = [AsymmetricFake(0, step_ms=100_000)]
    ep = _timed(rollout_over(insts, EpisodeSpec(max_game_ms=600_000), 12, red=SELF))
    assert (ep.opponent_id, ep.opponent_category, ep.score) == (SELF, "self", 0.5)


# -- the diagnostics that did not exist ------------------------------------


def test_every_reward_term_is_summed_over_the_episode():
    """``_AgentReward.terms`` was computed every tick and discarded every tick."""
    insts = [AsymmetricFake(0, step_ms=100_000, cs_per_step=3)]
    ep = _timed(rollout_over(insts, EpisodeSpec(max_game_ms=600_000), 12, red=SELF))
    terms = ep.reward_terms
    assert terms, "no per-term breakdown was recorded"
    # Every weight in the table has to appear, including the named zeros: a
    # term that is absent and a term that summed to 0.00 are different facts,
    # and `spend` is in this file's history precisely because they were
    # confused (see lanerl_rl.reward's "spend, and why it is a named zero").
    for name in ("hp_point", "tower_hp", "money", "exp", "mana", "death", "kill",
                 "last_hit", "spend", "shaping"):
        assert name in terms, f"{name} is missing from the term breakdown: {sorted(terms)}"
    assert terms["last_hit"] > 0.0, (
        "the fake farms 3 CS a decision and last_hit summed to zero"
    )


def test_kills_and_deaths_are_counted_per_episode():
    """Required by --no-end-on-death, which is required by CS@10.

    An episode that plays through deaths can contain several, and nothing in
    the metrics counted them: the server prints ``deaths=`` on its LANERL_CS
    rows and ``EpisodeResult`` had no field to put it in.
    """
    insts = [AsymmetricFake(0, step_ms=100_000, champ_dies_at_ms=200_000)]
    ep = _timed(rollout_over(
        insts, EpisodeSpec(max_game_ms=600_000, end_on_death=False), 12, red=SELF
    ))
    assert ep.deaths == 1, "blue's death was not counted"
    assert ep.kills == 0, "blue killed nobody; red is the one that survived"


def test_the_first_frame_of_every_episode_records_ad_and_mhp():
    """The rune-page canary.

    The fake reports mhp 672 on the first frame of an episode and 616 on every
    later one -- the numbers of the real regression, in which an in-process
    reset stripped the page. A readout taken from any frame but the first, or
    one that latches the first episode's value forever, reports 616.
    """
    insts = [AsymmetricFake(0, step_ms=100_000)]
    rollout = rollout_over(insts, EpisodeSpec(max_game_ms=600_000), 26, red=SELF)
    timed = [e for e in rollout.episodes if e.reason == "time"]
    assert len(timed) >= 2, "need a second episode to prove the readout is per-episode"
    for ep in timed:
        assert ep.first_frame_mhp == pytest.approx(AsymmetricFake.FIRST_FRAME_MHP), (
            "mhp was not read from the first frame of THIS episode"
        )
        assert ep.first_frame_ad == pytest.approx(78.14)


def test_action_marginals_are_reported_for_every_rollout():
    """The curve that shows a BC prior eroding; nothing recorded it."""
    insts = [AsymmetricFake(0, step_ms=1_000)]
    rollout = rollout_over(insts, EpisodeSpec(max_game_ms=10 ** 9), 12, red=SELF)
    m = rollout.action_marginals
    assert set(m) == set(C.BUTTONS), f"not one fraction per button: {sorted(m)}"
    assert sum(m.values()) == pytest.approx(1.0)


def test_button_marginals_count_every_slot_of_every_row():
    """(T, B) decisions, not T: a per-env marginal would hide one env going noop."""
    import torch

    from lanerl_train.lane_wiring import button_marginals

    actions = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)  # 3 noop, 1 move
    m = button_marginals(actions)
    assert m["noop"] == pytest.approx(0.75)
    assert m["move"] == pytest.approx(0.25)
    assert sum(m.values()) == pytest.approx(1.0)
    assert button_marginals(actions[:0]) == {}, "an empty rollout has no marginal to report"


def test_move_label_round_trips_and_both_sides_agree():
    """BC's move labels must decode back to the direction the bot walked.

    move_bins_for inverts decode_action. It first used LaneTransform.vector,
    which is lane-local -> WORLD, not its inverse -- an involution only for the
    old MirrorTransform. Blue's labels therefore pointed the wrong way while
    red's were right, BC averaged the two sides to the centre bin, and the
    prior learned to stand still: 288 units from spawn over 300 s, against a
    lane 11,866 units away.

    Both sides must map "toward the enemy" to the SAME bins -- that is what a
    canonical frame is for.
    """
    import math
    from lanerl_rl import constants as C
    from lanerl_train.lane_wiring import make_lane_adapters
    from lanerl_train.collect_demos import move_bins_for
    from lanerl_rl.scenarios import top_lane_scenario, encode_frame

    ad = make_lane_adapters(train_step_source=lambda: 0)
    raw = encode_frame(top_lane_scenario(t_ms=120_000, n_minions=4))
    seen = {}
    for side, team in (("blue", C.TEAM_BLUE), ("red", C.TEAM_RED)):
        a = ad.adapter_factory(0, side)
        a.build(raw, side)
        own = C.TOP_OUTER_TURRET[team]
        foe = C.TOP_OUTER_TURRET[C.TEAM_RED if team == C.TEAM_BLUE else C.TEAM_BLUE]
        ux, uy = foe[0] - own[0], foe[1] - own[1]
        ch = next(u for u in raw["u"] if u.get("k") == "Champion" and u.get("tm") == team)
        bx, bz = move_bins_for(a, raw, team, ch["x"] + ux, ch["y"] + uy)
        seen[side] = (bx, bz)
        # decode exactly as env.decode_action does
        tx, tz = float(C.SCREEN_X_VALUES[bx]), float(C.SCREEN_X_VALUES[bz])
        nn = math.hypot(tx, tz) or 1.0
        wx, wy = a.builder.transform.vector(tx / nn, tz / nn)
        d = math.hypot(ux, uy)
        assert (wx * ux + wy * uy) / d > 0.9, f"{side}: label does not decode toward the enemy"
    assert seen["blue"] == seen["red"], (
        f"blue and red disagree on 'toward the enemy': {seen} -- BC will average "
        f"them to the centre bin and stand still"
    )
